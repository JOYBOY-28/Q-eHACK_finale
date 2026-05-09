"""
Assembly Line 1  |  2 sensor pairs  |  UDP port 5001
Raw JSON datagrams sent over UDP every 1 second.

HOW TO USE
----------
Run:   python3 scrp1_asmbly_lin1.py

Type one of these and press Enter at any time:
    n  →  NORMAL
    w  →  WARNING
    c  →  CRITICAL
    q  →  quit
"""

import json, time, random, socket, signal, sys, threading, select
from datetime import datetime, timezone
from enum import Enum

UDP_HOST          = "169.254.130.97" 
UDP_PORT          = 8082
TICK_INTERVAL_SEC = 0.05
LINE_ID           = "AL1"
NUM_PAIRS         = 2

# ── STATE MACHINE (MANUAL) ─────────────────────────────────────────────────────
class STATE_MODE(Enum):
    NORMAL   = "NORMAL"
    WARNING  = "WARNING"
    CRITICAL = "CRITICAL"

_state      = STATE_MODE.NORMAL
_state_lock = threading.Lock()
_running    = True

def set_state(s):
    global _state
    with _state_lock:
        _state = s

def get_state():
    with _state_lock:
        return _state

def _cmd_reader():
    global _running
    sys.stdout.write("[CMD] n=NORMAL  w=WARNING  c=CRITICAL  q=quit\n")
    sys.stdout.flush()
    while _running:
        ready, _, _ = select.select([sys.stdin], [], [], 0.2)
        if ready:
            cmd = input().strip().lower()
            if   cmd == "n": set_state(STATE_MODE.NORMAL);   sys.stdout.write(f"[STATE] NORMAL\n");   sys.stdout.flush()
            elif cmd == "w": set_state(STATE_MODE.WARNING);  sys.stdout.write(f"[STATE] WARNING\n");  sys.stdout.flush()
            elif cmd == "c": set_state(STATE_MODE.CRITICAL); sys.stdout.write(f"[STATE] CRITICAL\n"); sys.stdout.flush()
            elif cmd == "q": _running = False; break

# ── DISTRIBUTIONS ──────────────────────────────────────────────────────────────
def _vib_normal():
    return round(random.uniform(0.1,   0.2),   4)
def _th_normal():    return round(random.uniform(99,101), 4)
def _vib_warning():  return round(random.uniform(1.9,   2.1),   4)
def _th_warning():   return round(random.uniform(149.5,151.5),  4)
def _vib_critical(): return round(random.uniform(4.75,  5.25),  4)
def _th_critical():  return round(random.uniform(201.0,199.0),  4)

_VIB={STATE_MODE.NORMAL:_vib_normal,STATE_MODE.WARNING:_vib_warning,STATE_MODE.CRITICAL:_vib_critical}
_TH ={STATE_MODE.NORMAL:_th_normal, STATE_MODE.WARNING:_th_warning, STATE_MODE.CRITICAL:_th_critical}

# ── SENSORS ────────────────────────────────────────────────────────────────────
class VibrationSensor:
    def __init__(self,sid): self.sid=sid
    def read(self,s): 
        # Generate independent X, Y, and Z values based on the current stateSSSSSS
        x_val = _VIB[s]()
        y_val = _VIB[s]()
        z_val = _VIB[s]()
        return {"id":self.sid, "t":"VB", "v":[x_val, y_val, z_val], "u":"mm/s"}

class ThermalSensor:
    def __init__(self,sid): self.sid=sid
    def read(self,s): return {"id":self.sid,"t":"TH","v":_TH[s](),"u":"C"}

pairs=[(ThermalSensor(f"{LINE_ID}_P{i}_TH"),VibrationSensor(f"{LINE_ID}_P{i}_VB"))
       for i in range(1,NUM_PAIRS+1)]

# ── PAYLOAD ────────────────────────────────────────────────────────────────────
def build(state):
    r = []
    for th, vb in pairs:
        # Append the single thermal float
        r.append(th.read(state)["v"])   
        
        # Extend the list with the 3 vibration floats [X, Y, Z]
        r.extend(vb.read(state)["v"])   
        
    # Payload format will now be: [Temp1, X1, Y1, Z1, Temp2, X2, Y2, Z2]
    return (json.dumps(r, separators=(",", ":")) + "\n").encode("utf-8")
# ── MAIN ───────────────────────────────────────────────────────────────────────
def main():
    global _running
    sock=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET,socket.SO_SNDBUF,262144)
    LOG_FILE = f"log_{LINE_ID}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    with open(LOG_FILE, "a", buffering=1) as log_f:
        def _shutdown(sig,frame):
            global _running
            _running=False
            sock.close(); sys.exit(0)

        signal.signal(signal.SIGINT,  _shutdown)
        signal.signal(signal.SIGTERM, _shutdown)

        threading.Thread(target=_cmd_reader,daemon=True).start()

        sys.stdout.write(f"[{LINE_ID}] started → UDP {UDP_HOST}:{UDP_PORT}\n")
        sys.stdout.flush()

        tick=0
        while _running:
            state=get_state()
            data=build(state)
            sock.sendto(data,(UDP_HOST,UDP_PORT))
            log_f.write(data.decode("utf-8"))
            sys.stdout.write(f"[{LINE_ID}] tick={tick:05d} state={state.value:<8} bytes={len(data)}\n")
            sys.stdout.flush()
            tick+=1
            time.sleep(TICK_INTERVAL_SEC)
    def _shutdown(sig,frame):
        global _running
        _running=False
        sock.close(); sys.exit(0)

    signal.signal(signal.SIGINT,  _shutdown)

    threading.Thread(target=_cmd_reader,daemon=True).start()

    sys.stdout.write(f"[{LINE_ID}] started → UDP {UDP_HOST}:{UDP_PORT}\n")
    sys.stdout.flush()

    tick=0
    while _running:
        state=get_state()
        data=build(state)
        sock.sendto(data,(UDP_HOST,UDP_PORT))
        sys.stdout.write(f"[{LINE_ID}] tick={tick:05d} state={state.value:<8} bytes={len(data)}\n")
        sys.stdout.flush()
        tick+=1
        time.sleep(TICK_INTERVAL_SEC)

    sock.close()

if __name__=="__main__":
    main()
