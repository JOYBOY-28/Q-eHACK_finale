# QNX Multi-Node Sensor Monitor + TinyLLaMA Health Manager

**Authors:** Anirudha Jayaprakash, Adithya Y

A hard real-time sensor monitoring system running on **QNX Neutrino RTOS** on a **Raspberry Pi 5 (BCM2712)**, with per-node motor GPIO control, fault isolation, watchdog protection, and an on-device AI health analysis engine powered by **TinyLLaMA 1.1B**.

---

## System Architecture

### Core / Thread Map

| Core | Runmask | Role | Threads | Policy | Priority |
|------|---------|------|---------|--------|----------|
| Core 0 | `0x1` | RTOS sensor side | `main`, `node_listener_thread[0,1,2]` | `SCHED_FIFO` | 17–20 |
| Core 1 | `0x2` | GPIO motor control | `motor_ctrl_thread[0,1,2]` | `SCHED_FIFO` | 21 |
| Core 1 | `0x2` | Watchdog timers | Per-node `SIGEV_PULSE` timer | Pulse prio | 22 |
| Core 2+3 | `0xC` | AI inference | `ai_manager_thread` | `SCHED_FIFO` | 10 |
| Core 2+3 | `0xC` | LLM child process | `llama-cli` (forked) | `SCHED_SPORADIC` | 5 |

Each core domain is hard-isolated using `_NTO_TCTL_RUNMASK` **and** `_NTO_TCTL_RUNMASK_GET_AND_SET_INHERIT`, so any threads spawned internally by the llama.cpp runtime are also locked to Core 2+3 and cannot bleed into the RTOS cores.

---

## Hardware

| Component | Detail |
|-----------|--------|
| SBC | Raspberry Pi 5 |
| SoC | BCM2712 (4× Cortex-A76 @ 2.4 GHz) |
| RTOS | QNX Neutrino 7.x |
| Sensor nodes | 3 nodes, each with 2 temperature sensors + 3-axis accelerometer |
| Transport | UDP (ports 8080, 8081, 8082) |
| Motor control | GPIO via direct MMIO (BCM17, BCM18, BCM27) — active-low reset pulse |

---

## AI Engine — TinyLLaMA 1.1B

### Model

| Property | Value |
|----------|-------|
| Model name | TinyLLaMA 1.1B Chat v1.0 |
| Quantisation | Q4_K_M (4-bit, K-means grouped) |
| File | `tinyllama-1.1b-chat-v1.0-q4_k_m.gguf` |
| Format | GGUF (llama.cpp native) |
| Parameters | 1.1 billion |
| Context window (configured) | 512 tokens |
| Inference threads | 2 (one per core — Core 2 and Core 3) |
| Prompt throughput | ~20 tokens/s on BCM2712 |
| Generation throughput | ~10 tokens/s on BCM2712 |
| Temperature | 0.3 (low — deterministic health reports) |
| Runtime | `llama-cli` (llama.cpp CLI) |

### Why Q4_K_M

Q4_K_M strikes the best balance for this deployment:
- **Q4** — 4-bit integer weights fit the model comfortably in the RPi5's 8 GB LPDDR4X without touching swap
- **K_M** — K-means grouped quantisation (medium grouping) preserves output quality better than flat Q4_0, especially for structured factual output like fault reports
- At 1.1B parameters with Q4_K_M, the model file is approximately **670 MB** on disk and loads into roughly **800 MB** of RAM at runtime

### Prompt Format

The system uses TinyLLaMA's native ChatML format:

```
<|system|>
You are a predictive maintenance AI on an RTOS. Analyze the snapshot
and output a concise report. Correlate temperature and FFT peaks to
state a specific mechanical fault hypothesis (bearing wear, misalignment,
friction).
<|user|>
Sensor snapshot at 2025-05-09 14:30:00:
Node1: T1=95.2C T2=98.7C RMS1=0.821 RMS2=0.743 FFT_peak=0.412 motor_restarts=0
Node2: T1=OFF   T2=OFF   RMS1=OFF   RMS2=OFF   FFT_peak=OFF   motor_restarts=0
Node3: T1=72.1C T2=71.8C RMS1=0.312 RMS2=0.298 FFT_peak=0.091 motor_restarts=2
Thresholds: temp_warn=110C temp_crit=190C vib_warn=1.4 vib_crit=4.5
<|assistant|>
```

### AI Scheduling — SCHED_SPORADIC

The `llama-cli` child process runs under `SCHED_SPORADIC` with the following budget to prevent sustained memory-bus saturation from evicting RTOS cache lines:

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `sched_priority` | 5 | Active execution priority |
| `sched_ss_low_priority` | 1 | Priority after budget is exhausted |
| `sched_ss_init_budget` | 7 ms | CPU time granted per replenishment |
| `sched_ss_repl_period` | 10 ms | Replenishment window |
| **Effective CPU cap** | **70%** | Of Core 2+3 per 10 ms window |
| `sched_ss_max_repl` | 4 | Max consecutive full replenishments |

The remaining 30% headroom ensures the AI manager thread itself (prio 10) can always preempt the inference child when needed.

### AI Trigger Behaviour

The AI manager wakes on **every incoming sensor packet** (not just on a fixed timer). A `PULSE_AI_SNAPSHOT_REQ` is sent to the AI manager's private IPC channel whenever `process_node_data` completes a valid (non-zero) packet. The AI then takes a snapshot of all three nodes' latest values and runs inference.

### Zero / Node-Off Detection

All-zero packets (`[0,0,0,0,0,0,0,0]`) are interpreted as **node powered off**, not as a valid reading. When detected:
- Threshold checks, FFT, and heartbeat are **skipped** — the watchdog is not fed, which causes the motor controller to trigger the GPIO fail-safe after `WATCHDOG_MS` (100 ms)
- The node's slot in the AI snapshot is labelled `OFF` in the prompt so the LLM can reason about partial system states correctly
- No spurious motor restart is triggered for an off node

---

## Safety Mechanisms

### 1. Per-Node Fault Isolation

Each sensor node runs in its own `node_listener_thread` and each motor runs in its own `motor_ctrl_thread`. A crash, deadlock, or socket error in Node 2's thread cannot affect Node 1 or Node 3. Threads communicate exclusively via QNX IPC channels — no shared mutable state on the critical path.

### 2. Watchdog (100 ms)

Each `motor_ctrl_thread` owns a `CLOCK_MONOTONIC` timer that fires `PULSE_WATCHDOG_TICK` every 100 ms. The listener must advance `heartbeat_seq` within that window. If it stalls (Core 0 starved, thread died, or AI memory pressure caused a scheduling blackout), the motor controller asserts the GPIO fail-safe reset independently, without waiting for the listener.

### 3. mlock — Page Pinning

`NodeState[3]` and `AISnapshot` are pinned into physical RAM with `mlock()` at startup. TinyLLaMA's matrix multiplications push hundreds of megabytes through the memory subsystem. `mlock` prevents the OS from evicting these pages under memory pressure, eliminating page-fault latency spikes on the RT path.

### 4. trylock on Snapshot Updates

The listener thread (Core 0) updates the AI snapshot using `pthread_mutex_trylock`. If the AI manager holds the lock, the listener **silently skips** that update cycle and continues processing sensor data. The RT path never blocks on the AI side under any circumstance.

### 5. GPIO Reset — nanospin, Not delay

`motor_gpio_reset()` uses `nanospin_ns(50000000)` (50 ms busy-wait) rather than `delay()`. On a dedicated Core 1 with nothing else scheduled, this gives deterministic reset pulse width. `delay()` would yield the thread and introduce scheduler jitter into the hold time.

### 6. Cross-Core IPC Latency

`MsgSendPulse_r` from Core 0 → Core 1 crosses a core boundary and issues an IPI (inter-processor interrupt) on BCM2712. Typical IPI latency is 2–4 µs. This is accounted for in the motor restart timing — the 50 ms GPIO hold far exceeds any IPI delay.

---

## Sensor Thresholds

| Parameter | Warning | Critical |
|-----------|---------|----------|
| Temperature | 110 °C | 190 °C |
| Vibration RMS | 1.4 g | 4.5 g |
| FFT AC peak (predictive) | 0.7 (= `VIB_WARNING × 0.5`) | — |

On **warning**: a `PULSE_TEMP_WARNING` or `PULSE_VIB_WARNING` is sent to the motor controller (logged to stderr).

On **critical**: a `PULSE_MOTOR_RESTART` is sent. The motor controller asserts GPIO reset on the specific node's pin only. The whole RPi is never rebooted.

On **FFT peak** (predictive): after every 256-sample FFT window, if the maximum AC amplitude exceeds the predictive threshold, `PULSE_FFT_WARNING` is sent. This is an early bearing-fault indicator before hard thresholds are crossed.

---

## IPC Pulse Code Map

| Code constant | Value | Sender → Receiver | Meaning |
|---------------|-------|-------------------|---------|
| `PULSE_MOTOR_RESTART` | `MINAVAIL+0` | Listener → MotorCtrl | Critical threshold — GPIO reset |
| `PULSE_TEMP_WARNING` | `MINAVAIL+1` | Listener → MotorCtrl | Temperature in warning band |
| `PULSE_VIB_WARNING` | `MINAVAIL+2` | Listener → MotorCtrl | Vibration in warning band |
| `PULSE_FFT_WARNING` | `MINAVAIL+3` | Listener → MotorCtrl | Predictive bearing fault |
| `PULSE_HEARTBEAT` | `MINAVAIL+4` | Listener → MotorCtrl | Proof-of-life (watchdog feed) |
| `PULSE_WATCHDOG_TICK` | `MINAVAIL+5` | Kernel timer → MotorCtrl | Watchdog check tick |
| `PULSE_AI_SNAPSHOT_REQ` | `MINAVAIL+6` | Kernel timer → AI Manager | Trigger inference cycle |

---

## File Structure

```
/              
└── /tmp/ai_engine/
    ├── llama-cli             # llama.cpp CLI binary (compiled for aarch64)
    └── tinyllama-1.1b-chat-v1.0-q4_k_m.gguf   # Model file (~670 MB)
```
* This is where i stored stuff inside the qnx os
---

Required QNX libraries: `libm`, `libsocket`. No third-party dependencies. The `llama-cli` binary must be compiled separately from the llama.cpp repo targeting `aarch64-unknown-nto-qnx7.1.0`.

---

## Runtime Requirements

- Process must run as **root** (or with `io` capability) for `_NTO_TCTL_IO_PRIV` and `mmap_device_io`
- `llama-cli` and the model file must be present at the paths defined by `LLAMA_CLI_PATH` and `LLAMA_MODEL_PATH`
- UDP ports 8080–8082 must be available
- Minimum 1.5 GB free RAM recommended (model runtime + node state + OS overhead)

---

## Sample Output

```
[Main] ══ System online ══
  Core 0  → sensor listeners    prio 17-19  (SCHED_FIFO)
  Core 1  → motor controllers   prio 21     (SCHED_FIFO)
            watchdog timers      prio 22     (pulse-driven)
  Core2+3 → TinyLLaMA manager   prio 10     (SCHED_FIFO)
            llama-cli child      prio  5     (SCHED_SPORADIC 70%)
  NodeState + AISnapshot mlocked in RAM

[Listener] Node 1  Core0  UDP port 8080
[MotorCtrl] Node 1  Core1  chid=3  watchdog=100ms
...

[Node 2] All-zero packet — node is OFF. Skipping thresholds / FFT / heartbeat.
[Watchdog] Node 2 STALE heartbeat seq=47 — Core 0 may be starved. GPIO fail-safe.
[GPIO] Node 2 BCM18 reset pulse complete.

╔══════════════════════════════════════════════════╗
║  TinyLLaMA Health Report                         ║
╠══════════════════════════════════════════════════╣
Node 1 is operating within safe parameters with
temperatures and vibration well below warning levels.
Node 2 is currently offline. Node 3 shows elevated
motor restart count (2) alongside moderate FFT peaks
suggesting early-stage bearing wear on the driven
shaft. Recommend scheduled inspection of Node 3
bearings within the next maintenance window.
╚══════════════════════════════════════════════════╝
```

---

## References

- [TinyLLaMA on HuggingFace](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)
- [GGUF Q4_K_M quantised model](https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- QNX Neutrino RTOS Programmer's Guide — Interprocess Communication, Thread Scheduling
- BCM2712 TRM — RP1 peripheral GPIO register map
