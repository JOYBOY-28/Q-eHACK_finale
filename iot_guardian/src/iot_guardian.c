/*
 * sensor_monitor.c — QNX Multi-Node Sensor Monitor + TinyLLaMA Health Manager
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * CORE / THREAD MAP  (RPi5 — BCM2712, 4× Cortex-A76)
 * ═══════════════════════════════════════════════════════════════════════════
 *
 *  Core 0  (runmask 0x1)  — RTOS sensor side
 *    main / watchdog            SCHED_FIFO  prio 20
 *    node_listener_thread[0]    SCHED_FIFO  prio 19   port 8080
 *    node_listener_thread[1]    SCHED_FIFO  prio 18   port 8081
 *    node_listener_thread[2]    SCHED_FIFO  prio 17   port 8082
 *
 *  Core 1  (runmask 0x2)  — GPIO / motor control
 *    motor_ctrl_thread[0]       SCHED_FIFO  prio 21   Node 1  BCM17
 *    motor_ctrl_thread[1]       SCHED_FIFO  prio 21   Node 2  BCM18
 *    motor_ctrl_thread[2]       SCHED_FIFO  prio 21   Node 3  BCM27
 *
 *  Core 2+3  (runmask 0xC)  — AI inference
 *    ai_manager_thread          SCHED_FIFO  prio 10
 *    llama-cli child process    SCHED_SPORADIC prio 5
 *                               budget 7ms / 10ms window  (70% cap)
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * SAFETY MECHANISMS
 * ═══════════════════════════════════════════════════════════════════════════
 *
 *  1. mlock()        — NodeState array + AI snapshot pinned in RAM.
 *                      AI matrix ops cannot trigger page faults on RT path.
 *
 *  2. Watchdog       — SIGEV_PULSE timer fires every WATCHDOG_MS on each
 *                      motor ctrl channel. Listener must advance heartbeat_seq
 *                      within that window or motor ctrl asserts GPIO fail-safe.
 *
 *  3. SCHED_SPORADIC — llama-cli child gets at most AI_BUDGET_NS per
 *                      AI_PERIOD_NS. Breaks sustained memory bus pressure
 *                      into controlled bursts.
 *
 *  4. RUNMASK_INHERIT — AI threads set both runmask AND inherit-mask so
 *                       llama.cpp internal thread pool stays on Core 2+3.
 *
 *  5. trylock snapshot — listener uses pthread_mutex_trylock (never blocks)
 *                        so RT path is never delayed by AI manager.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>
#include <stdint.h>
#include <errno.h>
#include <signal.h>
#include <time.h>
#include <sys/mman.h>

/* Networking */
#include <sys/socket.h>
#include <netinet/in.h>
#include <sys/select.h>
#include <fcntl.h>

/* QNX */
#include <sys/neutrino.h>
#include <sys/sched.h>
#include <pthread.h>
#include <sched.h>
#include <sys/wait.h>

/* GPIO via QNX /dev/gpio/N resource manager — no MMIO headers needed */

/* ═══════════════════════════════════════════════════════════════════════════
 * Constants
 * ═══════════════════════════════════════════════════════════════════════════ */
#define NUM_NODES           3
#define BASE_UDP_PORT       8080
#define BUFFER_SIZE         256
#define FFT_WINDOW          256
#define PI                  3.14159265358979323846

/* IPC pulse codes — all >= _PULSE_CODE_MINAVAIL */
#define PULSE_MOTOR_RESTART     (_PULSE_CODE_MINAVAIL + 0)
#define PULSE_TEMP_WARNING      (_PULSE_CODE_MINAVAIL + 1)
#define PULSE_VIB_WARNING       (_PULSE_CODE_MINAVAIL + 2)
#define PULSE_FFT_WARNING       (_PULSE_CODE_MINAVAIL + 3)
#define PULSE_HEARTBEAT         (_PULSE_CODE_MINAVAIL + 4)
#define PULSE_WATCHDOG_TICK     (_PULSE_CODE_MINAVAIL + 5)
#define PULSE_AI_SNAPSHOT_REQ   (_PULSE_CODE_MINAVAIL + 6)

/* Thread priorities */
#define PRIO_WATCHDOG       22
#define PRIO_MOTOR_CTRL     21
#define PRIO_MAIN           20
#define PRIO_NODE_1         19
#define PRIO_NODE_2         18
#define PRIO_NODE_3         17
#define PRIO_AI_MANAGER     10
#define PRIO_LLAMA_CHILD     5

/* CPU runmasks */
#define CORE0_MASK   0x1        /* Core 0 — RTOS sensors       */
#define CORE1_MASK   0x2        /* Core 1 — GPIO motor ctrl    */
#define CORE23_MASK  0xC        /* Core 2+3 — AI inference     */

/* Sensor thresholds */
#define TEMP_WARNING        110.0
#define TEMP_RESTART        190.0
#define VIB_WARNING         1.4
#define VIB_RESTART         4.5

/* Watchdog */
#define WATCHDOG_MS         100

/* AI manager */
#define AI_INTERVAL_S       30
#define AI_BUDGET_NS        7000000LL   /* 7 ms budget                */
#define AI_PERIOD_NS        10000000LL  /* per 10 ms → 70% cap        */
#define LLAMA_CLI_PATH      "/usr/local/bin/llama-cli"
#define LLAMA_MODEL_PATH    "/data/models/tinyllama-1.1b-chat-v1.0-q4_k_m.gguf"

/* ═══════════════════════════════════════════════════════════════════════════
 * GPIO — rpi_gpio resource manager  (/dev/gpio/N)
 *
 * Confirmed interface from strings /system/bin/rpi_gpio:
 *
 *   write "out"   — configure pin as output      (no newline — echo -n)
 *   write "in"    — configure pin as input
 *   write "on"    — drive HIGH                   (NOT "1")
 *   write "off"   — drive LOW                    (NOT "0")
 *   read          — returns current pin value
 *
 * "1"/"0" and newline-terminated strings are silently ignored by rpi_gpio.
 * That is why every previous attempt failed despite open() succeeding.
 *
 * Startup sequence (called once from main before any thread starts):
 *   gpio_init_all_high() — drives all three motor reset pins HIGH so
 *   motors are NOT in reset at boot. Without this the pins are floating
 *   and the motors may be stuck in reset from power-on.
 * ═══════════════════════════════════════════════════════════════════════════ */
#define GPIO_DEV_FMT        "/dev/gpio/%d"
static const int MOTOR_GPIO_PINS[NUM_NODES] = {17, 18, 27};

/* ═══════════════════════════════════════════════════════════════════════════
 * Data types
 * ═══════════════════════════════════════════════════════════════════════════ */
typedef struct {
    double real;
    double imag;
} complex_t;

/*
 * NodeState — one per sensor node, entire array is mlocked.
 *
 * Ownership:
 *   last_temp*, last_rms*, fft_peak, heartbeat_seq
 *                             written by listener_thread (Core 0)
 *   last_seen_seq             written by motor_ctrl_thread (Core 1)
 *   chid, coid                written once at init, read-only thereafter
 *   vib1_buffer, sample_idx   exclusive to listener_thread
 */
typedef struct {
    int     node_id;
    int     port;
    int     chid;
    int     coid;

    volatile double   last_temp1;
    volatile double   last_temp2;
    volatile double   last_rms1;
    volatile double   last_rms2;
    volatile double   last_fft_peak;

    /* Watchdog heartbeat */
    volatile uint32_t heartbeat_seq;
    volatile uint32_t last_seen_seq;

    /* FFT — exclusive to listener */
    complex_t vib1_buffer[FFT_WINDOW];
    int       sample_idx;
} NodeState;

/* Global sensor snapshot for AI manager */
typedef struct {
    double  temp1[NUM_NODES];
    double  temp2[NUM_NODES];
    double  rms1[NUM_NODES];
    double  rms2[NUM_NODES];
    double  fft_peak[NUM_NODES];
    int     restart_count[NUM_NODES];
    time_t  snapshot_time;
} AISnapshot;

static AISnapshot      g_snapshot;
static pthread_mutex_t g_snapshot_mutex = PTHREAD_MUTEX_INITIALIZER;

/* Thread arg bundles */
typedef struct { NodeState *ns; }  ListenerArg;
typedef struct { NodeState *ns; }  MotorCtrlArg;
typedef struct {
    NodeState *nodes;
    int        ai_chid;
} AIManagerArg;

/* ═══════════════════════════════════════════════════════════════════════════
 * FFT — Cooley-Tukey Radix-2 in-place
 * ═══════════════════════════════════════════════════════════════════════════ */
static void compute_fft(complex_t *x, int N) {
    int i, j, k, n1;
    complex_t t, e;
    j = 0;
    for (i = 0; i < N - 1; i++) {
        if (i < j) { t = x[i]; x[i] = x[j]; x[j] = t; }
        k = N / 2;
        while (k <= j) { j -= k; k /= 2; }
        j += k;
    }
    for (k = 1; k < N; k *= 2) {
        n1 = k * 2;
        for (i = 0; i < k; i++) {
            e.real = cos(-PI * i / k);
            e.imag = sin(-PI * i / k);
            for (j = i; j < N; j += n1) {
                t.real = e.real * x[j+k].real - e.imag * x[j+k].imag;
                t.imag = e.real * x[j+k].imag + e.imag * x[j+k].real;
                x[j+k].real = x[j].real - t.real;
                x[j+k].imag = x[j].imag - t.imag;
                x[j].real  += t.real;
                x[j].imag  += t.imag;
            }
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * gpio_fd_write — write a command string to an open /dev/gpio/N fd.
 * NO newline — rpi_gpio uses echo -n semantics, not line-buffered.
 * ═══════════════════════════════════════════════════════════════════════════ */
static int gpio_fd_write(int fd, const char *cmd) {
    ssize_t n = write(fd, cmd, strlen(cmd));
    if (n < 0) {
        perror("[GPIO] write");
        return -1;
    }
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * gpio_pin_set — open a pin fd, write one command, close.
 * Used for single-shot operations (init, direction set).
 * ═══════════════════════════════════════════════════════════════════════════ */
static int gpio_pin_set(int pin, const char *cmd) {
    char path[32];
    snprintf(path, sizeof(path), GPIO_DEV_FMT, pin);
    int fd = open(path, O_RDWR);
    if (fd < 0) {
        fprintf(stderr, "[GPIO] open(%s): %s\n", path, strerror(errno));
        return -1;
    }
    int rc = gpio_fd_write(fd, cmd);
    close(fd);
    return rc;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * gpio_init_all_high — called ONCE from main() before any thread starts.
 *
 * Configures all three motor reset pins as OUTPUT and drives them HIGH.
 * This releases the motors from reset at boot — without this the pins
 * are floating and motors may be stuck in reset from power-on.
 * ═══════════════════════════════════════════════════════════════════════════ */
static void gpio_init_all_high(void) {
    for (int i = 0; i < NUM_NODES; i++) {
        int pin = MOTOR_GPIO_PINS[i];
        if (gpio_pin_set(pin, "out") < 0) {
            fprintf(stderr, "[GPIO] Init: failed to set pin %d as output\n", pin);
            continue;
        }
        if (gpio_pin_set(pin, "on") < 0) {
            fprintf(stderr, "[GPIO] Init: failed to drive pin %d HIGH\n", pin);
            continue;
        }
        fprintf(stderr, "[GPIO] Init: BCM%d → output HIGH (motor released)\n", pin);
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * motor_gpio_reset — active-low 50ms reset pulse via /dev/gpio/N
 *
 * rpi_gpio command reference (confirmed from binary strings):
 *   "out" — set direction output   (no newline)
 *   "off" — drive LOW              (NOT "0", NOT "0\n")
 *   "on"  — drive HIGH             (NOT "1", NOT "1\n")
 *
 * Sequence:
 *   1. open /dev/gpio/<pin>
 *   2. write "out"   — ensure output direction
 *   3. write "off"   — assert LOW  (motor enters reset)
 *   4. delay(50)     — hold 50 ms
 *   5. write "on"    — release HIGH (motor reinitialises)
 *   6. close fd
 * ═══════════════════════════════════════════════════════════════════════════ */
static void motor_gpio_reset(int node_id) {
    int  pin = MOTOR_GPIO_PINS[node_id - 1];
    char path[32];
    snprintf(path, sizeof(path), GPIO_DEV_FMT, pin);

    int fd = open(path, O_RDWR);
    if (fd < 0) {
        fprintf(stderr, "[GPIO] Node %d open(%s): %s\n",
                node_id, path, strerror(errno));
        return;
    }

    gpio_fd_write(fd, "out");   /* ensure output direction               */
    gpio_fd_write(fd, "off");   /* assert LOW  — motor enters reset      */
    delay(50);                  /* hold 50 ms                            */
    gpio_fd_write(fd, "on");    /* release HIGH — motor reinitialises    */

    close(fd);

    fprintf(stderr, "[GPIO] Node %d BCM%d — reset pulse complete.\n",
            node_id, pin);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Pulse helper — MsgSendPulse_r is non-blocking and RT-safe
 * ═══════════════════════════════════════════════════════════════════════════ */
static inline void send_pulse(NodeState *ns, int code, int value) {
    if (MsgSendPulse_r(ns->coid, PRIO_MOTOR_CTRL, code, value) == -1)
        fprintf(stderr, "[Node %d] send_pulse(%d): %s\n",
                ns->node_id, code, strerror(errno));
}

/* ═══════════════════════════════════════════════════════════════════════════
 * motor_ctrl_thread — Core 1, SCHED_FIFO prio 21
 *
 * Owns the per-node IPC channel.  Blocks on MsgReceive_r().
 * Runs a 100ms watchdog timer — if heartbeat_seq stalls, GPIO fail-safe fires.
 * ═══════════════════════════════════════════════════════════════════════════ */
static void *motor_ctrl_thread(void *arg) {
    MotorCtrlArg *ma = (MotorCtrlArg *)arg;
    NodeState    *ns  = ma->ns;

    /* ── Core 1 affinity — both runmask and inherit-mask ────────────────── */
    ThreadCtl(_NTO_TCTL_RUNMASK,
              (void *)(uintptr_t)CORE1_MASK);
    ThreadCtl(_NTO_TCTL_RUNMASK_GET_AND_SET_INHERIT,
              (void *)(uintptr_t)CORE1_MASK);

    /* Note: _NTO_TCTL_IO_PRIV not needed — GPIO via /dev/gpio/N fd, not MMIO */

    /* ── Watchdog timer: PULSE_WATCHDOG_TICK every WATCHDOG_MS ─────────── */
    struct sigevent wdog_ev;
    SIGEV_PULSE_INIT(&wdog_ev, ns->chid, PRIO_WATCHDOG,
                     PULSE_WATCHDOG_TICK, ns->node_id);
    timer_t wdog_tid;
    timer_create(CLOCK_MONOTONIC, &wdog_ev, &wdog_tid);
    struct itimerspec wdog_its = {
        .it_value    = { .tv_sec = 0,
                         .tv_nsec = (long)WATCHDOG_MS * 1000000L },
        .it_interval = { .tv_sec = 0,
                         .tv_nsec = (long)WATCHDOG_MS * 1000000L }
    };
    timer_settime(wdog_tid, 0, &wdog_its, NULL);

    fprintf(stderr,
        "[MotorCtrl] Node %d  Core1  chid=%d  watchdog=%dms\n",
        ns->node_id, ns->chid, WATCHDOG_MS);

    struct _pulse pulse;
    while (1) {
        int rc = MsgReceive_r(ns->chid, &pulse, sizeof(pulse), NULL);
        if (rc < 0) {
            if (rc == -EINTR) continue;
            fprintf(stderr, "[MotorCtrl] Node %d recv: %s\n",
                    ns->node_id, strerror(-rc));
            continue;
        }

        switch (pulse.code) {

        case PULSE_MOTOR_RESTART:
            fprintf(stderr,
                "[MotorCtrl] Node %d CRITICAL reason=%d "
                "T=%.1f°C RMS=%.3f — GPIO reset\n",
                ns->node_id, pulse.value.sival_int,
                ns->last_temp1, ns->last_rms1);
            motor_gpio_reset(ns->node_id);
            /* Increment restart counter visible to AI snapshot */
            pthread_mutex_lock(&g_snapshot_mutex);
            g_snapshot.restart_count[ns->node_id - 1]++;
            pthread_mutex_unlock(&g_snapshot_mutex);
            break;

        case PULSE_HEARTBEAT:
            /* Listener alive — record sequence */
            ns->last_seen_seq = (uint32_t)pulse.value.sival_int;
            break;

        case PULSE_WATCHDOG_TICK:
            /*
             * heartbeat_seq is written by the listener (Core 0).
             * If it has not advanced since last tick, Core 0 is starved
             * or the listener thread has died.  Fail-safe the motor.
             */
            if (ns->heartbeat_seq == ns->last_seen_seq) {
                fprintf(stderr,
                    "[Watchdog] Node %d STALE heartbeat seq=%u "
                    "— Core 0 may be starved. GPIO fail-safe.\n",
                    ns->node_id, ns->heartbeat_seq);
                motor_gpio_reset(ns->node_id);
            }
            ns->last_seen_seq = ns->heartbeat_seq;
            break;

        case PULSE_TEMP_WARNING:
            fprintf(stderr, "[MotorCtrl] Node %d WARN temp %.1f°C\n",
                    ns->node_id, ns->last_temp1);
            break;

        case PULSE_VIB_WARNING:
            fprintf(stderr, "[MotorCtrl] Node %d WARN vib RMS=%.3f\n",
                    ns->node_id, ns->last_rms1);
            break;

        case PULSE_FFT_WARNING:
            fprintf(stderr,
                "[MotorCtrl] Node %d PREDICTIVE bearing fault "
                "FFT_peak=%.3f\n",
                ns->node_id, ns->last_fft_peak);
            break;

        default:
            break;
        }
    }

    timer_delete(wdog_tid);
    return NULL;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * process_node_data — runs on Core 0 listener thread
 * ═══════════════════════════════════════════════════════════════════════════ */
static void process_node_data(NodeState *ns, const char *buf) {
    double temp1, x1, y1, z1, temp2, x2, y2, z2;
    if (sscanf(buf, "[%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf]",
               &temp1, &x1, &y1, &z1,
               &temp2, &x2, &y2, &z2) != 8)
        return;

    double rms1 = sqrt(x1*x1 + y1*y1 + z1*z1);
    double rms2 = sqrt(x2*x2 + y2*y2 + z2*z2);

    ns->last_temp1 = temp1;
    ns->last_temp2 = temp2;
    ns->last_rms1  = rms1;
    ns->last_rms2  = rms2;

    /* ── Critical ───────────────────────────────────────────────────────── */
    if (temp1 > TEMP_RESTART || temp2 > TEMP_RESTART)
        send_pulse(ns, PULSE_MOTOR_RESTART, 1);

    if (rms1 > VIB_RESTART || rms2 > VIB_RESTART)
        send_pulse(ns, PULSE_MOTOR_RESTART, 2);

    /* ── Warning ────────────────────────────────────────────────────────── */
    if ((temp1 >= TEMP_WARNING && temp1 <= TEMP_RESTART) ||
        (temp2 >= TEMP_WARNING && temp2 <= TEMP_RESTART))
        send_pulse(ns, PULSE_TEMP_WARNING, 0);

    if ((rms1 >= VIB_WARNING && rms1 <= VIB_RESTART) ||
        (rms2 >= VIB_WARNING && rms2 <= VIB_RESTART))
        send_pulse(ns, PULSE_VIB_WARNING, 0);

    /* ── Heartbeat (watchdog proof-of-life) ─────────────────────────────── */
    ns->heartbeat_seq++;
    send_pulse(ns, PULSE_HEARTBEAT, (int)ns->heartbeat_seq);

    /* ── FFT ────────────────────────────────────────────────────────────── */
    ns->vib1_buffer[ns->sample_idx].real = rms1;
    ns->vib1_buffer[ns->sample_idx].imag = 0.0;
    ns->sample_idx++;

    if (ns->sample_idx >= FFT_WINDOW) {
        compute_fft(ns->vib1_buffer, FFT_WINDOW);
        double max_ac = 0.0;
        for (int i = 1; i < FFT_WINDOW / 2; i++) {
            double mag = hypot(ns->vib1_buffer[i].real,
                               ns->vib1_buffer[i].imag) * 2.0 / FFT_WINDOW;
            if (mag > max_ac) max_ac = mag;
        }
        ns->last_fft_peak = max_ac;
        if (max_ac > VIB_WARNING * 0.5)
            send_pulse(ns, PULSE_FFT_WARNING, 0);
        ns->sample_idx = 0;
    }

    /*
     * Update AI snapshot — trylock so RT path NEVER blocks waiting for
     * the AI manager to release the mutex.  If trylock fails this packet's
     * snapshot is skipped; the next packet will update it.
     */
    if (pthread_mutex_trylock(&g_snapshot_mutex) == 0) {
        int idx = ns->node_id - 1;
        g_snapshot.temp1[idx]    = temp1;
        g_snapshot.temp2[idx]    = temp2;
        g_snapshot.rms1[idx]     = rms1;
        g_snapshot.rms2[idx]     = rms2;
        g_snapshot.fft_peak[idx] = ns->last_fft_peak;
        g_snapshot.snapshot_time = time(NULL);
        pthread_mutex_unlock(&g_snapshot_mutex);
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * node_listener_thread — Core 0, SCHED_FIFO staggered prio
 * ═══════════════════════════════════════════════════════════════════════════ */
static void *node_listener_thread(void *arg) {
    ListenerArg *la = (ListenerArg *)arg;
    NodeState   *ns  = la->ns;

    ThreadCtl(_NTO_TCTL_RUNMASK, (void *)(uintptr_t)CORE0_MASK);
    ThreadCtl(_NTO_TCTL_RUNMASK_GET_AND_SET_INHERIT,
              (void *)(uintptr_t)CORE0_MASK);

    ns->coid = ConnectAttach_r(0, 0, ns->chid, _NTO_SIDE_CHANNEL, 0);
    if (ns->coid < 0) {
        fprintf(stderr, "[Listener] Node %d ConnectAttach: %s\n",
                ns->node_id, strerror(-ns->coid));
        return NULL;
    }

    int sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if (sockfd < 0) { perror("socket"); return NULL; }

    struct sockaddr_in addr = {0};
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port        = htons(ns->port);

    if (bind(sockfd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        perror("bind"); close(sockfd); return NULL;
    }
    int fl = fcntl(sockfd, F_GETFL, 0);
    fcntl(sockfd, F_SETFL, fl | O_NONBLOCK);

    fprintf(stderr, "[Listener] Node %d  Core0  UDP port %d\n",
            ns->node_id, ns->port);

    char buf[BUFFER_SIZE], latest[BUFFER_SIZE];
    struct sockaddr_in cli;
    socklen_t cli_len = sizeof(cli);

    while (1) {
        fd_set rfds; FD_ZERO(&rfds); FD_SET(sockfd, &rfds);
        if (select(sockfd + 1, &rfds, NULL, NULL, NULL) < 0) {
            if (errno == EINTR) continue;
            perror("select"); continue;
        }
        int got = 0, n;
        while ((n = recvfrom(sockfd, buf, BUFFER_SIZE - 1, 0,
                             (struct sockaddr *)&cli, &cli_len)) > 0) {
            buf[n] = '\0';
            memcpy(latest, buf, BUFFER_SIZE);
            got = 1;
        }
        if (got) process_node_data(ns, latest);
    }

    close(sockfd);
    ConnectDetach_r(ns->coid);
    return NULL;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * build_llama_prompt
 * ═══════════════════════════════════════════════════════════════════════════ */
static void build_llama_prompt(char *out, size_t sz,
                                const AISnapshot *s) {
    char ts[32];
    struct tm *tm_info = localtime(&s->snapshot_time);
    strftime(ts, sizeof(ts), "%Y-%m-%d %H:%M:%S", tm_info);

    snprintf(out, sz,
        "<|system|>\n"
        "You are an embedded RTOS health monitor on a Raspberry Pi 5. "
        "Analyse the sensor snapshot and give a concise 3-5 sentence status "
        "report. Flag nodes at risk. Note any abnormal restart counts. "
        "Be direct — no preamble, no markdown.\n"
        "<|user|>\n"
        "Sensor snapshot at %s:\n"
        "Node1: T1=%.1fC T2=%.1fC RMS1=%.3f RMS2=%.3f "
            "FFT_peak=%.3f motor_restarts=%d\n"
        "Node2: T1=%.1fC T2=%.1fC RMS1=%.3f RMS2=%.3f "
            "FFT_peak=%.3f motor_restarts=%d\n"
        "Node3: T1=%.1fC T2=%.1fC RMS1=%.3f RMS2=%.3f "
            "FFT_peak=%.3f motor_restarts=%d\n"
        "Thresholds: temp_warn=110C temp_crit=190C "
            "vib_warn=1.4 vib_crit=4.5\n"
        "<|assistant|>\n",
        ts,
        s->temp1[0], s->temp2[0], s->rms1[0], s->rms2[0],
            s->fft_peak[0], s->restart_count[0],
        s->temp1[1], s->temp2[1], s->rms1[1], s->rms2[1],
            s->fft_peak[1], s->restart_count[1],
        s->temp1[2], s->temp2[2], s->rms1[2], s->rms2[2],
            s->fft_peak[2], s->restart_count[2]);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * run_llama_inference
 *
 * fork() → child applies:
 *   - Core 2+3 affinity + inherit-mask
 *   - SCHED_SPORADIC prio 5, budget 7ms/10ms (70% cap)
 * Parent pipes prompt in via stdin, reads response from stdout.
 * Hard kill if child exceeds AI_INTERVAL_S/2 seconds.
 * ═══════════════════════════════════════════════════════════════════════════ */
static void run_llama_inference(const char *prompt) {
    int pin[2], pout[2];
    if (pipe(pin) < 0 || pipe(pout) < 0) {
        perror("[AI] pipe"); return;
    }

    pid_t pid = fork();
    if (pid < 0) { perror("[AI] fork"); return; }

    if (pid == 0) {
        /* ── Child ───────────────────────────────────────────────────────── */
        dup2(pin[0],  STDIN_FILENO);
        dup2(pout[1], STDOUT_FILENO);
        close(pin[1]); close(pout[0]);

        /* Pin child + all threads it spawns to Core 2+3 */
        ThreadCtl(_NTO_TCTL_RUNMASK,
                  (void *)(uintptr_t)CORE23_MASK);
        ThreadCtl(_NTO_TCTL_RUNMASK_GET_AND_SET_INHERIT,
                  (void *)(uintptr_t)CORE23_MASK);

        /*
         * SCHED_SPORADIC — caps CPU consumption to AI_BUDGET_NS out of
         * every AI_PERIOD_NS.  This prevents llama-cli from saturating
         * the memory bus continuously and evicting RT cache lines.
         */
        struct sched_param sp;
        memset(&sp, 0, sizeof(sp));
        sp.sched_priority               = PRIO_LLAMA_CHILD;
        sp.sched_ss_low_priority        = 1;
        sp.sched_ss_repl_period.tv_sec  = 0;
        sp.sched_ss_repl_period.tv_nsec = AI_PERIOD_NS;
        sp.sched_ss_init_budget.tv_sec  = 0;
        sp.sched_ss_init_budget.tv_nsec = AI_BUDGET_NS;
        sp.sched_ss_max_repl            = 4;
        sched_setscheduler(0, SCHED_SPORADIC, &sp);

        execl(LLAMA_CLI_PATH, "llama-cli",
              "-m",  LLAMA_MODEL_PATH,
              "--ctx-size",          "512",
              "--temp",              "0.3",
              "--threads",           "2",
              "--no-display-prompt",
              "-f",  "/dev/stdin",
              NULL);

        perror("[AI] execl"); _exit(EXIT_FAILURE);
    }

    /* ── Parent ──────────────────────────────────────────────────────────── */
    close(pin[0]); close(pout[1]);

    write(pin[1], prompt, strlen(prompt));
    close(pin[1]);  /* EOF → llama-cli starts inference */

    char   resp[2048] = {0};
    size_t rlen = 0;
    struct timeval tv = { .tv_sec = AI_INTERVAL_S / 2, .tv_usec = 0 };
    fd_set rfds; FD_ZERO(&rfds); FD_SET(pout[0], &rfds);

    if (select(pout[0] + 1, &rfds, NULL, NULL, &tv) > 0) {
        ssize_t n;
        while ((n = read(pout[0], resp + rlen,
                         sizeof(resp) - rlen - 1)) > 0)
            rlen += (size_t)n;
        resp[rlen] = '\0';
    } else {
        kill(pid, SIGKILL);
        snprintf(resp, sizeof(resp), "[timeout — llama-cli killed]");
    }

    close(pout[0]);
    waitpid(pid, NULL, 0);

    fprintf(stdout,
        "\n╔══════════════════════════════════════════════════╗\n"
        "║  TinyLLaMA 30s Health Report                     ║\n"
        "╠══════════════════════════════════════════════════╣\n"
        "%s\n"
        "╚══════════════════════════════════════════════════╝\n\n",
        resp);
    fflush(stdout);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * ai_manager_thread — Core 2+3, SCHED_FIFO prio 10
 *
 * Sleeps on its own IPC channel.  Every AI_INTERVAL_S seconds a
 * CLOCK_MONOTONIC timer fires PULSE_AI_SNAPSHOT_REQ to wake it.
 * It snapshots sensor data, builds a TinyLLaMA chat prompt, and calls
 * run_llama_inference() which spawns llama-cli with a sporadic budget.
 * ═══════════════════════════════════════════════════════════════════════════ */
static void *ai_manager_thread(void *arg) {
    AIManagerArg *aa = (AIManagerArg *)arg;

    ThreadCtl(_NTO_TCTL_RUNMASK, (void *)(uintptr_t)CORE23_MASK);
    ThreadCtl(_NTO_TCTL_RUNMASK_GET_AND_SET_INHERIT,
              (void *)(uintptr_t)CORE23_MASK);

    /* 30-second repeating timer → self-pulse */
    struct sigevent ev;
    SIGEV_PULSE_INIT(&ev, aa->ai_chid, PRIO_AI_MANAGER,
                     PULSE_AI_SNAPSHOT_REQ, 0);
    timer_t ai_timer;
    timer_create(CLOCK_MONOTONIC, &ev, &ai_timer);
    struct itimerspec its = {
        .it_value    = { .tv_sec = AI_INTERVAL_S, .tv_nsec = 0 },
        .it_interval = { .tv_sec = AI_INTERVAL_S, .tv_nsec = 0 }
    };
    timer_settime(ai_timer, 0, &its, NULL);

    fprintf(stderr,
        "[AI Manager] Core2+3  prio%d  interval=%ds  model=%s\n",
        PRIO_AI_MANAGER, AI_INTERVAL_S, LLAMA_MODEL_PATH);

    struct _pulse pulse;
    while (1) {
        int rc = MsgReceive_r(aa->ai_chid, &pulse, sizeof(pulse), NULL);
        if (rc < 0) {
            if (rc == -EINTR) continue;
            fprintf(stderr, "[AI Manager] recv: %s\n", strerror(-rc));
            continue;
        }
        if (pulse.code != PULSE_AI_SNAPSHOT_REQ) continue;

        /* Snapshot — full lock is fine here, we are low-priority */
        AISnapshot snap;
        pthread_mutex_lock(&g_snapshot_mutex);
        memcpy(&snap, &g_snapshot, sizeof(snap));
        pthread_mutex_unlock(&g_snapshot_mutex);

        if (snap.snapshot_time == 0) {
            fprintf(stderr, "[AI Manager] no sensor data yet — skipping\n");
            continue;
        }

        char prompt[1200];
        build_llama_prompt(prompt, sizeof(prompt), &snap);
        run_llama_inference(prompt);
    }

    timer_delete(ai_timer);
    return NULL;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Thread factory — SCHED_FIFO + explicit priority
 * ═══════════════════════════════════════════════════════════════════════════ */
static int create_rt_thread(pthread_t *tid,
                             void *(*fn)(void *), void *arg,
                             int prio) {
    pthread_attr_t     attr;
    struct sched_param sp = { .sched_priority = prio };
    pthread_attr_init(&attr);
    pthread_attr_setinheritsched(&attr, PTHREAD_EXPLICIT_SCHED);
    pthread_attr_setschedpolicy(&attr, SCHED_FIFO);
    pthread_attr_setschedparam(&attr, &sp);
    int rc = pthread_create(tid, &attr, fn, arg);
    pthread_attr_destroy(&attr);
    return rc;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * main
 * ═══════════════════════════════════════════════════════════════════════════ */
int main(void) {
    /* Elevate main, pin to Core 0 */
    struct sched_param sp = { .sched_priority = PRIO_MAIN };
    pthread_setschedparam(pthread_self(), SCHED_FIFO, &sp);
    ThreadCtl(_NTO_TCTL_RUNMASK, (void *)(uintptr_t)CORE0_MASK);

    static NodeState    nodes[NUM_NODES];
    static ListenerArg  largs[NUM_NODES];
    static MotorCtrlArg margs[NUM_NODES];

    memset(nodes,       0, sizeof(nodes));
    memset(&g_snapshot, 0, sizeof(g_snapshot));

    /*
     * Drive all motor reset pins HIGH before any thread starts.
     * Motors are released from reset and run normally until a
     * threshold triggers motor_gpio_reset() on a specific node.
     */
    gpio_init_all_high();

    /*
     * mlock — pin RT-critical data into physical pages.
     * Eliminates page-fault latency spikes caused by the AI workload
     * pushing these pages out under memory pressure.
     */
    if (mlock(nodes, sizeof(nodes)) != 0)
        perror("[Main] mlock(nodes)");
    if (mlock(&g_snapshot, sizeof(g_snapshot)) != 0)
        perror("[Main] mlock(g_snapshot)");

    static const int LPRIOS[NUM_NODES] = {
        PRIO_NODE_1, PRIO_NODE_2, PRIO_NODE_3
    };

    pthread_t listener_tid[NUM_NODES];
    pthread_t motor_tid[NUM_NODES];

    for (int i = 0; i < NUM_NODES; i++) {
        nodes[i].node_id = i + 1;
        nodes[i].port    = BASE_UDP_PORT + i;

        nodes[i].chid = ChannelCreate_r(0);
        if (nodes[i].chid < 0) {
            fprintf(stderr, "[Main] ChannelCreate node %d: %s\n",
                    i+1, strerror(-nodes[i].chid));
            return EXIT_FAILURE;
        }

        margs[i].ns = &nodes[i];
        if (create_rt_thread(&motor_tid[i], motor_ctrl_thread,
                             &margs[i], PRIO_MOTOR_CTRL) != 0) {
            perror("[Main] motor_ctrl_thread"); return EXIT_FAILURE;
        }

        largs[i].ns = &nodes[i];
        if (create_rt_thread(&listener_tid[i], node_listener_thread,
                             &largs[i], LPRIOS[i]) != 0) {
            perror("[Main] node_listener_thread"); return EXIT_FAILURE;
        }

        fprintf(stderr,
            "[Main] Node %d — listener Core0/prio%d  "
            "motor_ctrl Core1/prio%d  port=%d  chid=%d\n",
            i+1, LPRIOS[i], PRIO_MOTOR_CTRL,
            nodes[i].port, nodes[i].chid);
    }

    /* AI manager */
    static AIManagerArg ai_arg;
    ai_arg.nodes   = nodes;
    ai_arg.ai_chid = ChannelCreate_r(0);
    if (ai_arg.ai_chid < 0) {
        fprintf(stderr, "[Main] ChannelCreate AI: %s\n",
                strerror(-ai_arg.ai_chid));
        return EXIT_FAILURE;
    }

    pthread_t ai_tid;
    if (create_rt_thread(&ai_tid, ai_manager_thread,
                         &ai_arg, PRIO_AI_MANAGER) != 0) {
        perror("[Main] ai_manager_thread"); return EXIT_FAILURE;
    }

    fprintf(stderr,
        "\n[Main] ══ System online ══\n"
        "  Core 0  → sensor listeners    prio 17-19  (SCHED_FIFO)\n"
        "  Core 1  → motor controllers   prio 21     (SCHED_FIFO)\n"
        "            watchdog timers      prio 22     (pulse-driven)\n"
        "  Core2+3 → TinyLLaMA manager   prio 10     (SCHED_FIFO)\n"
        "            llama-cli child      prio  5     (SCHED_SPORADIC 70%%)\n"
        "  NodeState + AISnapshot mlocked in RAM\n"
        "  Health report every %d seconds\n\n",
        AI_INTERVAL_S);

    for (int i = 0; i < NUM_NODES; i++)
        pthread_join(listener_tid[i], NULL);

    return 0;
}
