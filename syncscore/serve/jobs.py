"""Uploads and their analyses, for `syncscore serve`.

Each upload is a folder under the root (results/uploads/ by default, git-ignored): `video.mp4`,
`config.yaml` (the server's config with this upload's member count), `job.json` (what the page
shows: step, progress, error, the end of the log), everything `analyze` writes, and `viewer/`.

Jobs run one at a time on a worker thread, each step as its own `syncscore` process, exactly the
two commands of the command-line flow. Pose detection keeps the CPU busy for minutes; in another
process it never slows the server down, and a crash in it fails the job, not the server. Progress
comes from the lines those processes print.
"""
from __future__ import annotations

import json
import os
import queue
import re
import secrets
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import BinaryIO, Callable

import yaml

from ..config import DEFAULT_CONFIG, load_config

ID = re.compile(r"\d{8}-\d{6}-[0-9a-f]{6}")
VIDEO = "video.mp4"
VIDEO_URL = "../" + VIDEO  # from <upload>/viewer/index.html; also works when the folder is opened from disk
LOG_LINES = 40
MAX_MEMBERS = 50
UNFINISHED = ("queued", "running")
_POSE = re.compile(r"pose \+ tracking: frame (\d+)/(\d+)")
_REDETECT = re.compile(r"re-detecting missed frames: (\d+)/(\d+)")
_EXCEPTION = re.compile(r"[A-Za-z_][\w.]*(Error|Exception|Interrupt)\b")
_PACKAGE_PARENT = Path(__file__).resolve().parents[2]


class UploadError(ValueError):
    """A problem with what was sent; the message is shown on the page."""


def new_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(3)


def commands(folder: Path) -> list[tuple[str, list[str]]]:
    """(step, command) for one upload: `analyze`, then `view` with the video served next to the viewer."""
    base = [sys.executable, "-m", "syncscore", "--config", str(folder / "config.yaml")]
    video = str(folder / VIDEO)
    return [("pose", [*base, "analyze", video, "--out", str(folder)]),
            ("view", [*base, "view", video, "--results", str(folder), "--video-url", VIDEO_URL])]


def progress_update(line: str, newline: bool) -> dict | None:
    """What a line printed by `analyze` says about where it is, as job fields (None: nothing).

    Progress counters are rewritten in place with a carriage return; the line that finally ends
    with a newline means that loop is over."""
    for pattern, step, doing, after in ((_POSE, "pose", "Finding every dancer's pose", "Tracking the dancers and measuring sync"),
                                        (_REDETECT, "measure", "Looking again at frames where a dancer was missed",
                                         "Measuring sync and writing the report")):
        found = pattern.search(line)
        if found:
            if newline:
                return {"step": "measure", "detail": after, "progress": None}
            return {"step": step, "detail": doing, "progress": [int(found[1]), int(found[2])]}
    if "using cached detections" in line:
        return {"step": "measure", "detail": "Using the poses found last time; tracking and measuring sync", "progress": None}
    if "Downloading" in line and ".pt" in line:
        return {"detail": "Downloading the pose model (about 20 MB, first run only)"}
    return None


def failure_reason(log: list[str]) -> str | None:
    """The line of a failed step's output that says why: a Python exception's last line, else the
    last line that isn't FFmpeg's own chatter ("[mov,mp4,... @ 0x...] moov atom not found"), which
    reaches the pipe after Python's traceback."""
    lines = [line.strip() for line in log if line.strip()]
    for line in reversed(lines):
        if _EXCEPTION.match(line):
            return line
    return next((line for line in reversed(lines) if not line.startswith("[")), lines[-1] if lines else None)


def read_pieces(stream: BinaryIO):
    """Yield (text, ended_by_newline) for each piece of a process's output as it arrives, and
    (text, None) for an unfinished last piece (a progress counter waiting for its next update).

    Pieces end at \\r, \\n or \\r\\n. A trailing \\r is held back until the next byte shows
    whether it starts a \\r\\n."""
    pending = b""
    while True:
        chunk = stream.read1(65536) if hasattr(stream, "read1") else stream.read(65536)
        if not chunk:
            break
        pending += chunk
        held = pending.endswith(b"\r")
        pieces = re.split(rb"(\r\n|\r|\n)", pending[:-1] if held else pending)
        pending = pieces.pop() + (b"\r" if held else b"")
        for text, end in zip(pieces[0::2], pieces[1::2]):
            yield text.decode("utf-8", "replace"), end != b"\r"
        if pending.strip(b"\r"):
            yield pending.strip(b"\r").decode("utf-8", "replace"), None
    if pending.strip(b"\r"):
        yield pending.strip(b"\r").decode("utf-8", "replace"), True


class Uploads:
    """The uploads under `root`, kept in memory and in each folder's job.json."""

    def __init__(self, root: Path, config: str | Path | None = None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.config = Path(config) if config else DEFAULT_CONFIG
        load_config(self.config)  # a broken config fails now, not minutes into the first job
        self._lock = threading.Lock()
        self._jobs: dict[str, dict] = {}
        for path in sorted(self.root.glob("*/job.json")):
            try:
                job = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if job.get("id") != path.parent.name or not ID.fullmatch(job["id"]):
                continue
            if job.get("status") in UNFINISHED:
                job.update(status="failed", error="The server stopped before this finished. Run it again.", progress=None)
                self._save(job)
            self._jobs[job["id"]] = job

    def folder(self, upload_id: str) -> Path:
        return self.root / upload_id

    def get(self, upload_id: str) -> dict | None:
        with self._lock:
            job = self._jobs.get(upload_id)
            return dict(job) if job else None

    def list(self) -> list[dict]:
        with self._lock:
            return sorted((dict(j) for j in self._jobs.values()), key=lambda j: j["created"], reverse=True)

    def create(self, name: str, members: int, body: BinaryIO, length: int) -> dict:
        """Save `length` bytes of `body` as a new upload's video and queue it."""
        _check_members(members)
        upload_id = new_id()
        folder = self.folder(upload_id)
        folder.mkdir(parents=True)
        try:
            part = folder / (VIDEO + ".part")
            with open(part, "wb") as out:
                left = length
                while left > 0:
                    chunk = body.read(min(left, 1 << 20))
                    if not chunk:
                        raise UploadError("The upload stopped before the whole video arrived.")
                    out.write(chunk)
                    left -= len(chunk)
            with open(part, "rb") as saved:
                if saved.read(12)[4:8] != b"ftyp":
                    raise UploadError("That file isn't an MP4 video.")
            part.replace(folder / VIDEO)
            self._write_config(folder, members)
        except BaseException:
            shutil.rmtree(folder, ignore_errors=True)
            raise
        now = time.time()
        job = {"id": upload_id, "name": name, "size": length, "members": members, "created": now,
               "status": "queued", "step": "queued", "detail": "Waiting its turn", "progress": None,
               "step_started": now, "finished": None, "error": None, "log": []}
        with self._lock:
            self._jobs[upload_id] = job
            self._save(job)
        return dict(job)

    def again(self, upload_id: str, members: int) -> dict:
        """Queue an upload's analysis again, e.g. with the right member count. Poses found the
        first time are reused (`analyze` keeps them in detections.npz)."""
        _check_members(members)
        with self._lock:
            job = self._jobs[upload_id]
            if job["status"] in UNFINISHED:
                raise UploadError("This video is still being measured.")
            self._write_config(self.folder(upload_id), members)
            job.update(members=members, status="queued", step="queued", detail="Waiting its turn", progress=None,
                       step_started=time.time(), finished=None, error=None, log=[])
            self._save(job)
            return dict(job)

    def update(self, upload_id: str, **fields) -> dict:
        """Change a job's fields. Values are replaced, never changed in place, so copies handed
        out earlier stay consistent."""
        with self._lock:
            job = self._jobs[upload_id]
            if "step" in fields and fields["step"] != job["step"]:
                fields.setdefault("step_started", time.time())
            job.update(fields)
            self._save(job)
            return dict(job)

    def _write_config(self, folder: Path, members: int) -> None:
        data = yaml.safe_load(self.config.read_text(encoding="utf-8")) or {}
        if not isinstance(data.get("tracking"), dict):
            data["tracking"] = {}
        data["tracking"]["members"] = int(members)
        path = folder / "config.yaml"
        path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        load_config(path)

    def _save(self, job: dict) -> None:
        path = self.folder(job["id"]) / "job.json"
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(job, indent=1), encoding="utf-8")
        temp.replace(path)


def _check_members(members: int) -> None:
    if not 0 <= members <= MAX_MEMBERS:
        raise UploadError(f"Members must be between 0 (count them) and {MAX_MEMBERS}.")


class Runner:
    """Runs queued uploads one at a time on a worker thread."""

    def __init__(self, uploads: Uploads, steps: Callable[[Path], list[tuple[str, list[str]]]] = commands):
        self.uploads, self.steps = uploads, steps
        self._queue: queue.Queue = queue.Queue()
        self._process: subprocess.Popen | None = None
        self._stopping = False
        self._thread = threading.Thread(target=self._work, name="syncscore-jobs", daemon=True)
        self._thread.start()

    def submit(self, upload_id: str) -> None:
        self._queue.put(upload_id)

    def stop(self) -> None:
        """Stop the worker and the process it is running (its job is marked failed)."""
        self._stopping = True
        self._queue.put(None)
        process = self._process
        if process is not None and process.poll() is None:
            process.terminate()
        self._thread.join(timeout=10)

    def _work(self) -> None:
        while True:
            upload_id = self._queue.get()
            if upload_id is None or self._stopping:
                return
            try:
                self._run(upload_id)
            except Exception as error:  # one job going wrong must never stop the worker
                self.uploads.update(upload_id, status="failed", error=f"{type(error).__name__}: {error}", progress=None)

    def _run(self, upload_id: str) -> None:
        folder = self.uploads.folder(upload_id)
        log: list[str] = []
        self.uploads.update(upload_id, status="running", step="pose", detail="Reading the video and loading the pose model", progress=None)
        env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8",
               "PYTHONPATH": os.pathsep.join(filter(None, [str(_PACKAGE_PARENT), os.environ.get("PYTHONPATH")]))}
        for step, command in self.steps(folder):
            if step == "view":
                self.uploads.update(upload_id, step="view", detail="Building the viewer and finding the beat", progress=None)
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env)
            self._process = process
            for text, newline in read_pieces(process.stdout):
                update = progress_update(text, bool(newline))
                if newline and text.strip():
                    log = (log + [text.rstrip()])[-LOG_LINES:]
                    update = {**(update or {}), "log": log}
                if update:
                    self.uploads.update(upload_id, **update)
            code = process.wait()
            self._process = None
            if code != 0:
                reason = "Stopped with the server." if self._stopping else (failure_reason(log) or f"`{step}` stopped with exit code {code}.")
                self.uploads.update(upload_id, status="failed", error=reason, progress=None, finished=time.time())
                return
        self.uploads.update(upload_id, status="done", step="done", detail="Done", progress=None, finished=time.time())
