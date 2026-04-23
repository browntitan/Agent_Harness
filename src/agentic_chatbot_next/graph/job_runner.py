from __future__ import annotations

import argparse
import json
import os
import queue
import signal
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_HEARTBEAT_SECONDS = 10
_STREAM_TAIL_LIMIT = 4000
_PHASE_PREFIX = "[graph-build-phase]"
_REPAIR_PREFIX = "[graph-repair-summary]"
_FALLBACK_PREFIX = "[graph-fallback-used]"
_FALLBACK_SUMMARY_PREFIX = "[graph-fallback-summary]"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_state(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _trim_tail(current: str, addition: str, *, limit: int = _STREAM_TAIL_LIMIT) -> str:
    return f"{current}{addition}"[-limit:]


def _extract_structured_updates(line: str) -> dict[str, Any]:
    clean = str(line or "").strip()
    if not clean:
        return {}
    if clean.startswith(_PHASE_PREFIX):
        return {"build_phase": clean[len(_PHASE_PREFIX) :].strip()}
    if clean.startswith(_FALLBACK_PREFIX):
        value = clean[len(_FALLBACK_PREFIX) :].strip().lower()
        return {"fallback_used": value in {"1", "true", "yes", "y", "on"}}
    if clean.startswith(_REPAIR_PREFIX):
        try:
            return {"repair_summary": json.loads(clean[len(_REPAIR_PREFIX) :].strip())}
        except Exception:
            return {}
    if clean.startswith(_FALLBACK_SUMMARY_PREFIX):
        try:
            payload = json.loads(clean[len(_FALLBACK_SUMMARY_PREFIX) :].strip())
        except Exception:
            return {}
        updates: dict[str, Any] = {"fallback_summary": payload}
        if isinstance(payload, dict) and "fallback_used" in payload:
            updates["fallback_used"] = bool(payload.get("fallback_used"))
        return updates
    return {}


def _terminate_process_group(pid: int) -> None:
    try:
        pgid = os.getpgid(pid)
    except OSError:
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
    except OSError:
        return


def _kill_process_group(pid: int) -> None:
    try:
        pgid = os.getpgid(pid)
    except OSError:
        return
    try:
        os.killpg(pgid, signal.SIGKILL)
    except OSError:
        return


def _reader_loop(pipe: Any, output_queue: "queue.Queue[tuple[str, str]]") -> None:
    if pipe is None:
        output_queue.put(("eof", ""))
        return
    try:
        for line in iter(pipe.readline, ""):
            output_queue.put(("line", line))
    except Exception as exc:  # pragma: no cover - defensive path
        output_queue.put(("reader_error", f"{type(exc).__name__}: {exc}"))
    finally:
        try:
            pipe.close()
        except Exception:
            pass
        output_queue.put(("eof", ""))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a GraphRAG CLI job in the background and persist state.")
    parser.add_argument("--state-path", required=True)
    parser.add_argument("--stream-log-path", required=True)
    parser.add_argument("--runner-log-path", default="")
    parser.add_argument("--cwd", required=True)
    parser.add_argument("--timeout-seconds", required=True, type=int)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)

    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise SystemExit("job_runner requires a command after --")

    state_path = Path(args.state_path)
    stream_log_path = Path(args.stream_log_path)
    cwd = Path(args.cwd)
    timeout_seconds = max(0, int(args.timeout_seconds or 0))
    started_at = _now_iso()
    state: dict[str, Any] = {
        "status": "running",
        "runner_pid": os.getpid(),
        "process_group_id": os.getpgrp(),
        "command": command,
        "cwd": str(cwd),
        "state_path": str(state_path),
        "stream_log_path": str(stream_log_path),
        "runner_log_path": str(args.runner_log_path or ""),
        "timeout_seconds": timeout_seconds,
        "started_at": started_at,
        "updated_at": started_at,
        "last_heartbeat_at": started_at,
        "last_output_at": "",
        "completed_at": "",
        "returncode": None,
        "timed_out": False,
        "failure_mode": "",
        "stream_tail": "",
    }
    _write_state(state_path, state)

    stream_log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[graph-job-runner] starting child command: {' '.join(command)}", flush=True)

    try:
        proc = subprocess.Popen(
            command,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
    except Exception as exc:
        failed_at = _now_iso()
        state.update(
            {
                "status": "failed",
                "completed_at": failed_at,
                "updated_at": failed_at,
                "last_heartbeat_at": failed_at,
                "returncode": 1,
                "failure_mode": "runner_crash",
                "detail": f"{type(exc).__name__}: {exc}",
            }
        )
        _write_state(state_path, state)
        print(f"[graph-job-runner] failed to launch child: {type(exc).__name__}: {exc}", flush=True)
        return 1

    state["child_pid"] = proc.pid
    state["process_group_id"] = proc.pid
    state["updated_at"] = _now_iso()
    state["last_heartbeat_at"] = state["updated_at"]
    _write_state(state_path, state)
    print(f"[graph-job-runner] child pid={proc.pid}", flush=True)

    output_queue: "queue.Queue[tuple[str, str]]" = queue.Queue()
    reader = threading.Thread(target=_reader_loop, args=(proc.stdout, output_queue), daemon=True)
    reader.start()

    timed_out = False
    reader_done = False
    tail_buffer = ""
    reader_error = ""
    start_monotonic = datetime.now(timezone.utc).timestamp()

    with stream_log_path.open("a", encoding="utf-8") as stream:
        while True:
            now = _now_iso()
            if timeout_seconds > 0 and proc.poll() is None:
                elapsed_seconds = datetime.now(timezone.utc).timestamp() - start_monotonic
                if elapsed_seconds >= timeout_seconds:
                    timed_out = True
                    state["timed_out"] = True
                    state["failure_mode"] = "timeout"
                    state["detail"] = f"GraphRAG child exceeded the configured timeout of {timeout_seconds} seconds."
                    print(f"[graph-job-runner] timeout after {timeout_seconds}s; terminating child pid={proc.pid}", flush=True)
                    _terminate_process_group(proc.pid)
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        _kill_process_group(proc.pid)
                        proc.wait(timeout=10)
                    break

            try:
                event, payload = output_queue.get(timeout=_HEARTBEAT_SECONDS)
            except queue.Empty:
                state["updated_at"] = now
                state["last_heartbeat_at"] = now
                state["stream_tail"] = tail_buffer
                _write_state(state_path, state)
                if proc.poll() is not None and reader_done and output_queue.empty():
                    break
                continue

            if event == "line":
                stream.write(payload)
                stream.flush()
                tail_buffer = _trim_tail(tail_buffer, payload)
                state.update(_extract_structured_updates(payload))
                now = _now_iso()
                state["updated_at"] = now
                state["last_heartbeat_at"] = now
                state["last_output_at"] = now
                state["stream_tail"] = tail_buffer
                _write_state(state_path, state)
            elif event == "reader_error":
                reader_error = payload
                tail_buffer = _trim_tail(tail_buffer, f"\n[runner-reader-error] {payload}\n")
                state["updated_at"] = now
                state["last_heartbeat_at"] = now
                state["stream_tail"] = tail_buffer
                state["detail"] = payload
                _write_state(state_path, state)
            elif event == "eof":
                reader_done = True

            if proc.poll() is not None and reader_done and output_queue.empty():
                break

    returncode = int(proc.returncode if proc.returncode is not None else (124 if timed_out else 1))
    completed_at = _now_iso()
    state["completed_at"] = completed_at
    state["updated_at"] = completed_at
    state["last_heartbeat_at"] = completed_at
    state["stream_tail"] = tail_buffer
    state["returncode"] = returncode
    if timed_out:
        state["status"] = "failed"
        state["failure_mode"] = "timeout"
    elif returncode != 0:
        state["status"] = "failed"
        state["failure_mode"] = state.get("failure_mode") or "nonzero_exit"
        if reader_error and not str(state.get("detail") or "").strip():
            state["detail"] = reader_error
    else:
        state["status"] = "completed"
        state["failure_mode"] = ""
    _write_state(state_path, state)
    print(
        f"[graph-job-runner] finished status={state['status']} returncode={returncode} failure_mode={state.get('failure_mode') or 'none'}",
        flush=True,
    )
    return returncode


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
