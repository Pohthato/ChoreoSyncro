"""The local web server behind `syncscore serve`.

It listens on 127.0.0.1 only and answers only requests addressed to this computer by name
(127.0.0.1 or localhost), so a web page elsewhere can't reach it through a renamed address.

Routes:
  GET  /                              the start page (page/)
  GET  /api/uploads                   every upload, newest first
  GET  /api/uploads/<id>              one upload: step, progress, error, end of the log
  POST /api/uploads?name=&members=    the request body is the video; starts finding its people
  GET  /api/uploads/<id>/people?frame=<n>   the people found in frame n (boxes, left to right)
  GET  /api/uploads/<id>/frame/<n>.jpg      frame n as a JPEG, the frame those boxes belong to
  POST /api/uploads/<id>/picks        the named members (JSON); measures with them
  POST /api/uploads/<id>/repick       name the members again
  POST /api/uploads/<id>/run?members= measure without names (members told apart left to right)
  GET  /uploads/<id>/viewer/          that upload's viewer
  GET  /uploads/<id>/video.mp4        the video, with byte ranges so the page can seek

The viewer plays the video from `../video.mp4`, which this server answers with byte ranges
(206 Partial Content): a browser's <video> needs them to seek, and Python's own http.server
always sends the whole file.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

from .jobs import ID, MAX_PICKS, VIDEO, NoSuchFrame, NotReady, Runner, UploadError, Uploads, commands

PAGE_DIR = Path(__file__).resolve().parent / "page"
TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8",
         ".json": "application/json", ".png": "image/png", ".mp4": "video/mp4"}
MAX_UPLOAD = 8 << 30
LOCAL_NAMES = {"127.0.0.1", "localhost"}
_FILE_NAME = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]*")
_CHUNK = 1 << 18


class Unsatisfiable(ValueError):
    """A byte range that starts past the end of the file (answered with 416)."""


def parse_range(header: str | None, size: int) -> tuple[int, int] | None:
    """The single byte range a Range header asks for, as (first, last) inclusive and clipped to the
    file, or None to send the whole file (no header, another unit, several ranges, or nonsense:
    RFC 9110 lets a server ignore those). Raises Unsatisfiable for a range wholly past the end."""
    if not header:
        return None
    found = re.fullmatch(r"\s*bytes\s*=\s*(\d*)\s*-\s*(\d*)\s*", header)
    if not found or not (found[1] or found[2]):
        return None
    if found[1]:
        first = int(found[1])
        if found[2] and int(found[2]) < first:
            return None
        if first >= size:
            raise Unsatisfiable(header)
        return first, min(int(found[2]) if found[2] else size - 1, size - 1)
    suffix = int(found[2])  # bytes=-N: the last N bytes
    if suffix == 0 or size == 0:
        raise Unsatisfiable(header)
    return max(0, size - suffix), size - 1


def describe(job: dict, with_log: bool = True) -> dict:
    """A job as the page reads it."""
    out = {k: v for k, v in job.items() if with_log or k != "log"}
    out["viewer"] = f"/uploads/{job['id']}/viewer/" if job["status"] == "done" else None
    out["step_elapsed_s"] = round(time.time() - job["step_started"], 1) if job["status"] == "running" else None
    return out


class HttpError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


class Handler(BaseHTTPRequestHandler):
    server_version = "SyncScore"

    # ---- reading
    def do_GET(self):
        self._answer(self._get, head=False)

    def do_HEAD(self):
        self._answer(self._get, head=True)

    def do_POST(self):
        self._answer(self._post, head=False)

    def _answer(self, route, head: bool) -> None:
        self._body_left = int(self.headers.get("Content-Length") or 0) if self.command == "POST" else 0
        try:
            self._check_local()
            route(head)
        except HttpError as error:
            self._drain()
            self._json(error.status, {"error": str(error)}, head)
        except (NotReady, NoSuchFrame) as error:
            self._drain()
            self._json(409 if isinstance(error, NotReady) else 404, {"error": str(error)}, head)
        except UploadError as error:
            self._drain()
            self._json(400, {"error": str(error)}, head)
        except ConnectionError:  # the browser went away (it drops video requests all the time)
            pass

    def _get(self, head: bool) -> None:
        uploads: Uploads = self.server.uploads
        path = unquote(urlsplit(self.path).path)
        if path == "/":
            return self._file(PAGE_DIR / "index.html", head)
        if path == "/favicon.ico":  # browsers ask for it on every page; there is none
            self.send_response(204)
            self.end_headers()
            return
        if found := re.fullmatch(r"/static/(.+)", path):
            return self._file(self._inside(PAGE_DIR, found[1]), head)
        if path == "/api/uploads":
            return self._json(200, {"uploads": [describe(j, with_log=False) for j in uploads.list()]}, head)
        if found := re.fullmatch(r"/api/uploads/([^/]+)", path):
            return self._json(200, describe(self._job(found[1])), head)
        if found := re.fullmatch(r"/api/uploads/([^/]+)/people", path):
            self._job(found[1])
            frame = (parse_qs(urlsplit(self.path).query).get("frame") or [""])[0]
            return self._json(200, uploads.people(found[1], frame), head)
        if found := re.fullmatch(r"/api/uploads/([^/]+)/frame/([^/]+)\.jpg", path):
            self._job(found[1])
            return self._bytes(uploads.frame_jpeg(found[1], found[2]), "image/jpeg", head)
        if found := re.fullmatch(r"/uploads/([^/]+)/video\.mp4", path):
            self._job(found[1])
            return self._video(uploads.folder(found[1]) / VIDEO, head)
        if found := re.fullmatch(r"/uploads/([^/]+)/viewer", path):
            self._job(found[1])
            return self._redirect(path + "/")
        if found := re.fullmatch(r"/uploads/([^/]+)/viewer/(.*)", path):
            self._job(found[1])
            return self._file(self._inside(uploads.folder(found[1]) / "viewer", found[2] or "index.html"), head)
        raise HttpError(404, "Not found.")

    def _post(self, head: bool) -> None:
        uploads: Uploads = self.server.uploads
        origin = self.headers.get("Origin")
        if origin is not None and origin != f"http://{self.headers.get('Host')}":
            raise HttpError(403, "Uploads are only taken from this server's own page.")
        url = urlsplit(self.path)
        query = parse_qs(url.query)
        try:
            members = int((query.get("members") or ["0"])[0])
        except ValueError:
            raise HttpError(400, "Members must be a whole number (0 = count them).") from None
        if url.path == "/api/uploads":
            name = Path((query.get("name") or [""])[0].replace("\\", "/")).name.strip()[:200]
            if not name.lower().endswith(".mp4"):
                raise HttpError(400, "Choose an .mp4 video.")
            if "Content-Length" not in self.headers or not self.headers["Content-Length"].isdigit():
                raise HttpError(411, "The upload needs a Content-Length.")
            if not 0 < self._body_left <= MAX_UPLOAD:
                self._body_left = 0  # too big to read through: just close the connection
                raise HttpError(413, "The video is empty or larger than 8 GB.")
            length, self._body_left = self._body_left, 0
            job = uploads.create(name, members, self.rfile, length)
            print(f"upload {job['id']}: {name} ({length / 1e6:.1f} MB), analysing", file=sys.stderr)
            self.server.runner.submit(job["id"])
            return self._json(201, describe(job), head)
        if found := re.fullmatch(r"/api/uploads/([^/]+)/picks", url.path):
            self._job(found[1])
            if (self.headers.get("Content-Type") or "").split(";")[0].strip().lower() != "application/json":
                raise HttpError(415, "Send the picks as JSON.")
            if self._body_left > MAX_PICKS:
                raise HttpError(413, "The picks are too large.")
            body, self._body_left = self.rfile.read(self._body_left), 0
            try:
                data = json.loads(body.decode("utf-8"))
            except (UnicodeDecodeError, ValueError):
                raise HttpError(400, "The picks aren't valid JSON.") from None
            job = uploads.save_picks(found[1], data)
            print(f"upload {job['id']}: members named, measuring", file=sys.stderr)
            self.server.runner.submit(job["id"])
            return self._json(202, describe(job), head)
        if found := re.fullmatch(r"/api/uploads/([^/]+)/repick", url.path):
            self._job(found[1])
            job = uploads.repick(found[1])
            if job["status"] == "queued":
                self.server.runner.submit(job["id"])
            return self._json(202, describe(job), head)
        if found := re.fullmatch(r"/api/uploads/([^/]+)/run", url.path):
            self._job(found[1])
            job = uploads.again(found[1], members)
            print(f"upload {job['id']}: analysing again with members = {members}", file=sys.stderr)
            self.server.runner.submit(job["id"])
            return self._json(202, describe(job), head)
        raise HttpError(404, "Not found.")

    # ---- helpers
    def _check_local(self) -> None:
        host = self.headers.get("Host", "")
        name = host[: host.find("]") + 1] if host.startswith("[") else host.rsplit(":", 1)[0]
        if name not in LOCAL_NAMES | {"[::1]"}:
            raise HttpError(403, f"Open this server as http://127.0.0.1:{self.server.server_address[1]}/")

    def _job(self, upload_id: str) -> dict:
        job = self.server.uploads.get(upload_id) if ID.fullmatch(upload_id) else None
        if job is None:
            raise HttpError(404, "No such upload.")
        return job

    @staticmethod
    def _inside(folder: Path, name: str) -> Path:
        """`name` as a plain file inside `folder` (no subfolders, no dot files)."""
        if not _FILE_NAME.fullmatch(name):
            raise HttpError(404, "Not found.")
        return folder / name

    def _drain(self) -> None:
        """Read what's left of a request body, so the browser gets the error, not a reset connection."""
        while self._body_left > 0:
            chunk = self.rfile.read(min(self._body_left, _CHUNK))
            if not chunk:
                break
            self._body_left -= len(chunk)

    def _json(self, status: int, data: dict, head: bool) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if not head:
            self.wfile.write(body)

    def _bytes(self, body: bytes, content_type: str, head: bool) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if not head:
            self.wfile.write(body)

    def _redirect(self, location: str) -> None:
        self.send_response(301)
        self.send_header("Location", location)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _file(self, path: Path, head: bool) -> None:
        if not path.is_file():
            raise HttpError(404, "Not found.")
        self._send(path, 200, 0, path.stat().st_size - 1, head, {"Cache-Control": "no-cache"})

    def _video(self, path: Path, head: bool) -> None:
        if not path.is_file():
            raise HttpError(404, "Not found.")
        size = path.stat().st_size
        try:
            wanted = parse_range(self.headers.get("Range"), size)
        except Unsatisfiable:
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        first, last = wanted or (0, size - 1)
        extra = {"Accept-Ranges": "bytes"}
        if wanted:
            extra["Content-Range"] = f"bytes {first}-{last}/{size}"
        self._send(path, 206 if wanted else 200, first, last, head, extra)

    def _send(self, path: Path, status: int, first: int, last: int, head: bool, extra: dict) -> None:
        self.send_response(status)
        self.send_header("Content-Type", TYPES.get(path.suffix.lower(), "application/octet-stream"))
        self.send_header("Content-Length", str(last - first + 1))
        for key, value in extra.items():
            self.send_header(key, value)
        self.end_headers()
        if head:
            return
        with open(path, "rb") as source:
            source.seek(first)
            left = last - first + 1
            while left > 0:
                chunk = source.read(min(left, _CHUNK))
                if not chunk:
                    break
                self.wfile.write(chunk)
                left -= len(chunk)

    def log_request(self, code="-", size="-"):
        if str(getattr(code, "value", code))[:1] in "45":  # quiet unless something went wrong
            super().log_request(code, size)


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = os.name != "nt"  # on Windows it would let a second server share the port


def make_server(root: Path, config: str | Path | None, port: int = 0, steps=commands) -> Server:
    """The server on 127.0.0.1:`port` (0 = any free port), with its uploads and a job runner that
    runs `steps(folder)` for each upload (tests pass stand-ins for the real commands)."""
    server = Server(("127.0.0.1", port), Handler)
    try:
        server.uploads = Uploads(root, config)
    except BaseException:
        server.server_close()
        raise
    server.runner = Runner(server.uploads, steps)
    return server


def serve(root: Path, config: str | Path | None, port: int, open_browser: bool = True) -> int:
    try:
        server = make_server(root, config, port)
    except OSError as error:
        print(f"Can't listen on 127.0.0.1:{port} ({error}). Pick another port: syncscore serve --port 8001", file=sys.stderr)
        return 2
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print(f"SyncScore is at {url} (this computer only). Uploads and their analyses go to {Path(root).resolve()}.")
    print("Press Ctrl+C to stop.", flush=True)
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.runner.stop()
        server.server_close()
    return 0
