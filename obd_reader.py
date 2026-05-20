"""
OBD-II reader. Polls ELM327 emulator (or real adapter) in a background thread.

For testing without a real car, run:
    pip install ELM327-emulator
    elm -s car  # creates a pseudo-tty, prints the device path

Then point OBD_PORT to that path (e.g. /dev/pts/3).
"""
from __future__ import annotations

import os
import time
import threading
from dataclasses import dataclass, asdict
from queue import Queue, Full, Empty
from typing import Optional

import obd

# Reduce python-OBD log noise
obd.logger.setLevel("WARNING")


@dataclass
class OBDState:
    ts: float = 0.0
    connected: bool = False
    speed_kmh: float = 0.0
    rpm: float = 0.0
    throttle: float = 0.0
    engine_load: float = 0.0
    coolant_temp: float = 0.0
    fuel_level: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


class OBDReader(threading.Thread):
    def __init__(self, out_queue: Queue, port: Optional[str] = None, poll_hz: float = 5.0):
        super().__init__(daemon=True)
        self.port = port or os.getenv("OBD_PORT")  # None = auto-detect
        self.poll_interval = 1.0 / poll_hz
        self.out_queue = out_queue
        self._stop = threading.Event()
        self.conn: Optional[obd.OBD] = None

    def _connect(self) -> bool:
        try:
            self.conn = obd.OBD(portstr=self.port, fast=False, timeout=2.0)
            return self.conn.is_connected()
        except Exception as e:
            print(f"[OBD] connect failed: {e}")
            return False

    def run(self):
        # Retry connection loop
        while not self._stop.is_set() and not self._connect():
            self._emit(OBDState(ts=time.time(), connected=False))
            time.sleep(2.0)

        if self._stop.is_set():
            return

        cmds = {
            "speed_kmh":    obd.commands.SPEED,
            "rpm":          obd.commands.RPM,
            "throttle":     obd.commands.THROTTLE_POS,
            "engine_load":  obd.commands.ENGINE_LOAD,
            "coolant_temp": obd.commands.COOLANT_TEMP,
            "fuel_level":   obd.commands.FUEL_LEVEL,
        }

        while not self._stop.is_set():
            t0 = time.time()
            state = OBDState(ts=t0, connected=self.conn.is_connected())

            for field, cmd in cmds.items():
                if not self.conn.supports(cmd):
                    continue
                try:
                    resp = self.conn.query(cmd)
                    if resp.value is not None:
                        # value is a Pint Quantity; convert to plain float
                        setattr(state, field, float(resp.value.magnitude))
                except Exception:
                    pass

            self._emit(state)
            sleep = self.poll_interval - (time.time() - t0)
            if sleep > 0:
                time.sleep(sleep)

    def _emit(self, state: OBDState):
        try:
            self.out_queue.put_nowait(state)
        except Full:
            try:
                self.out_queue.get_nowait()
            except Empty:
                pass
            self.out_queue.put_nowait(state)

    def stop(self):
        self._stop.set()
        if self.conn:
            try:
                self.conn.close()
            except Exception:
                pass
