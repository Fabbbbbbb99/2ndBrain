"""Windows Named Pipe IPC server and auto-spawning client for braind."""
from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import psutil
import pywintypes
import win32api
import win32con
import win32file
import win32pipe
import win32security

MAX_FRAME_BYTES = 16 * 1024 * 1024
PIPE_BUFFER_SIZE = 65536


class IpcError(RuntimeError):
    """Raised when daemon IPC communication fails."""


def _current_user_security_attributes() -> win32security.SECURITY_ATTRIBUTES:
    """Build a DACL granting pipe access exclusively to the current Windows user SID."""
    token = win32security.OpenProcessToken(win32api.GetCurrentProcess(), win32con.TOKEN_QUERY)
    user_sid = win32security.GetTokenInformation(token, win32security.TokenUser)[0]
    win32api.CloseHandle(token)

    dacl = win32security.ACL()
    dacl.AddAccessAllowedAce(win32security.ACL_REVISION, win32con.GENERIC_ALL, user_sid)

    sd = win32security.SECURITY_DESCRIPTOR()
    sd.SetSecurityDescriptorDacl(1, dacl, 0)

    sa = win32security.SECURITY_ATTRIBUTES()
    sa.SECURITY_DESCRIPTOR = sd
    sa.bInheritHandle = False
    return sa


def _read_exact(handle: int, n_bytes: int) -> bytes:
    buf = bytearray()
    while len(buf) < n_bytes:
        _, chunk = win32file.ReadFile(handle, n_bytes - len(buf))
        if not chunk:
            raise EOFError("Named pipe closed before reading complete frame.")
        buf.extend(chunk)
    return bytes(buf)


def _write_frame(handle: int, payload: dict[str, Any]) -> None:
    raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    if len(raw) > MAX_FRAME_BYTES:
        raise IpcError(f"IPC frame exceeds {MAX_FRAME_BYTES} bytes: {len(raw)}")
    header = struct.pack(">I", len(raw))
    data = header + raw
    offset = 0
    while offset < len(data):
        _, written = win32file.WriteFile(handle, data[offset:])
        offset += written


def _read_frame(handle: int) -> dict[str, Any] | None:
    try:
        header = _read_exact(handle, 4)
    except (EOFError, pywintypes.error) as exc:
        if isinstance(exc, pywintypes.error) and exc.winerror not in (109, 232, 233):
            raise
        return None
    (length,) = struct.unpack(">I", header)
    if length == 0 or length > MAX_FRAME_BYTES:
        raise IpcError(f"Invalid IPC frame length: {length}")
    raw = _read_exact(handle, length)
    return json.loads(raw.decode("utf-8"))


class NamedPipeIpcServer:
    """Concurrent multi-client Windows Named Pipe IPC server."""

    def __init__(
        self,
        pipe_name: str,
        lock_path: Path,
        handler: Callable[[str, dict[str, Any]], dict[str, Any]],
        idle_ttl_seconds: int = 1800,
    ) -> None:
        self.pipe_name = pipe_name
        self.lock_path = Path(lock_path)
        self.handler = handler
        self.idle_ttl_seconds = idle_ttl_seconds
        self._stop_event = threading.Event()
        self._last_activity = time.monotonic()
        self._activity_lock = threading.Lock()
        self._sa = _current_user_security_attributes()
        self._threads: list[threading.Thread] = []

    def touch(self) -> None:
        with self._activity_lock:
            self._last_activity = time.monotonic()

    def request_stop(self) -> None:
        self._stop_event.set()
        # Wake blocking ConnectNamedPipe with a short self-connect probe.
        try:
            h = win32file.CreateFile(
                self.pipe_name,
                win32file.GENERIC_READ | win32file.GENERIC_WRITE,
                0,
                None,
                win32file.OPEN_EXISTING,
                0,
                None,
            )
            win32file.CloseHandle(h)
        except pywintypes.error:
            pass

    def _write_lockfile(self) -> None:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "pid": os.getpid(),
            "pipe_name": self.pipe_name,
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        tmp = self.lock_path.with_suffix(".lock.tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp, self.lock_path)

    def _remove_lockfile(self) -> None:
        try:
            if self.lock_path.exists():
                data = json.loads(self.lock_path.read_text(encoding="utf-8"))
                if data.get("pid") == os.getpid():
                    self.lock_path.unlink(missing_ok=True)
        except Exception:
            pass

    def _handle_client(self, handle: int) -> None:
        try:
            while not self._stop_event.is_set():
                req = _read_frame(handle)
                if req is None:
                    break
                self.touch()
                req_id = req.get("id")
                method = req.get("method", "")
                params = req.get("params") or {}
                try:
                    result = self.handler(method, params)
                    _write_frame(handle, {"id": req_id, "ok": True, "result": result})
                except Exception as exc:
                    _write_frame(
                        handle,
                        {
                            "id": req_id,
                            "ok": False,
                            "error": {
                                "type": type(exc).__name__,
                                "message": str(exc),
                            },
                        },
                    )
        finally:
            try:
                win32pipe.DisconnectNamedPipe(handle)
            except pywintypes.error:
                pass
            try:
                win32file.CloseHandle(handle)
            except pywintypes.error:
                pass

    def _idle_watchdog(self) -> None:
        while not self._stop_event.wait(timeout=5.0):
            with self._activity_lock:
                elapsed = time.monotonic() - self._last_activity
            if elapsed >= self.idle_ttl_seconds:
                self.request_stop()
                return

    def serve_forever(self) -> None:
        self._write_lockfile()
        watchdog = threading.Thread(target=self._idle_watchdog, daemon=True)
        watchdog.start()
        try:
            while not self._stop_event.is_set():
                pipe = win32pipe.CreateNamedPipe(
                    self.pipe_name,
                    win32pipe.PIPE_ACCESS_DUPLEX,
                    win32pipe.PIPE_TYPE_BYTE | win32pipe.PIPE_READMODE_BYTE | win32pipe.PIPE_WAIT,
                    win32pipe.PIPE_UNLIMITED_INSTANCES,
                    PIPE_BUFFER_SIZE,
                    PIPE_BUFFER_SIZE,
                    0,
                    self._sa,
                )
                try:
                    win32pipe.ConnectNamedPipe(pipe, None)
                except pywintypes.error as exc:
                    # 535 = ERROR_PIPE_CONNECTED (client connected between Create and Connect)
                    if exc.winerror != 535:
                        win32file.CloseHandle(pipe)
                        if self._stop_event.is_set():
                            break
                        continue
                if self._stop_event.is_set():
                    try:
                        win32file.CloseHandle(pipe)
                    except pywintypes.error:
                        pass
                    break
                t = threading.Thread(target=self._handle_client, args=(pipe,), daemon=True)
                t.start()
                self._threads = [th for th in self._threads if th.is_alive()]
                self._threads.append(t)
        finally:
            self._remove_lockfile()


def is_daemon_running(pipe_name: str, lock_path: Path | None = None) -> tuple[bool, dict[str, Any] | None]:
    lock_info: dict[str, Any] | None = None
    if lock_path and lock_path.exists():
        try:
            lock_info = json.loads(lock_path.read_text(encoding="utf-8"))
            pid = lock_info.get("pid")
            if pid and not psutil.pid_exists(int(pid)):
                lock_path.unlink(missing_ok=True)
                lock_info = None
        except Exception:
            lock_info = None

    try:
        h = win32file.CreateFile(
            pipe_name,
            win32file.GENERIC_READ | win32file.GENERIC_WRITE,
            0,
            None,
            win32file.OPEN_EXISTING,
            0,
            None,
        )
        win32file.CloseHandle(h)
        return True, lock_info
    except pywintypes.error as exc:
        if exc.winerror == 231:  # ERROR_PIPE_BUSY means pipe exists and is active
            return True, lock_info
        return False, lock_info


def spawn_daemon_background(vault_path: Path, state_dir: Path) -> None:
    env = os.environ.copy()
    env["BRAIN_VAULT_PATH"] = str(vault_path)
    env["BRAIN_STATE_DIR"] = str(state_dir)
    env["ORT_LOG_SEVERITY_LEVEL"] = "3"
    env["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    env["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    creationflags = getattr(subprocess, "DETACHED_PROCESS", 0x00000008) | getattr(
        subprocess, "CREATE_NO_WINDOW", 0x08000000
    )
    subprocess.Popen(
        [sys.executable, "-m", "second_brain.daemon.server"],
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=creationflags,
        close_fds=True,
    )


class NamedPipeIpcClient:
    """Thin stateless client calling braind over Windows Named Pipe."""

    def __init__(
        self,
        pipe_name: str,
        vault_path: Path,
        state_dir: Path,
        auto_spawn: bool = True,
        startup_timeout_s: float = 10.0,
    ) -> None:
        self.pipe_name = pipe_name
        self.vault_path = Path(vault_path)
        self.state_dir = Path(state_dir)
        self.auto_spawn = auto_spawn
        self.startup_timeout_s = startup_timeout_s

    def _connect(self) -> int:
        deadline = time.monotonic() + self.startup_timeout_s
        spawned = False
        while True:
            try:
                return win32file.CreateFile(
                    self.pipe_name,
                    win32file.GENERIC_READ | win32file.GENERIC_WRITE,
                    0,
                    None,
                    win32file.OPEN_EXISTING,
                    0,
                    None,
                )
            except pywintypes.error as exc:
                if exc.winerror == 231:  # ERROR_PIPE_BUSY
                    try:
                        win32pipe.WaitNamedPipe(self.pipe_name, 1000)
                    except pywintypes.error:
                        time.sleep(0.05)
                elif exc.winerror == 2:  # ERROR_FILE_NOT_FOUND
                    if not self.auto_spawn:
                        raise IpcError(f"braind daemon is not running at {self.pipe_name}") from exc
                    if not spawned:
                        spawn_daemon_background(self.vault_path, self.state_dir)
                        spawned = True
                    time.sleep(0.05)
                else:
                    raise IpcError(f"Failed connecting to {self.pipe_name}: {exc}") from exc
            if time.monotonic() > deadline:
                raise IpcError(f"Timed out waiting for braind daemon at {self.pipe_name}")

    def call(self, method: str, params: dict[str, Any] | None = None) -> Any:
        handle = self._connect()
        try:
            _write_frame(handle, {"id": 1, "method": method, "params": params or {}})
            resp = _read_frame(handle)
            if resp is None:
                raise IpcError(f"braind closed pipe before answering {method}")
            if not resp.get("ok"):
                err = resp.get("error") or {}
                raise IpcError(f"{err.get('type', 'DaemonError')}: {err.get('message', 'Unknown error')}")
            return resp.get("result")
        finally:
            try:
                win32file.CloseHandle(handle)
            except pywintypes.error:
                pass
