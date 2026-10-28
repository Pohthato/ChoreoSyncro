"""`syncscore serve`: uploads, job status, the viewer's files and byte ranges for the video.

The server runs for real on a free local port; the analysis steps are small stand-in processes
that print what `analyze` prints, so a job takes a fraction of a second instead of minutes."""
import http.client
import io
import json
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest
import yaml

from syncscore.serve.app import Unsatisfiable, make_server, parse_range
from syncscore.serve.jobs import VIDEO_URL, Uploads, commands, failure_reason, progress_update, read_pieces

MP4 = b"\x00\x00\x00\x18ftypisom" + bytes(range(256)) * 40  # an MP4 header, then 10 kB of known bytes

ANALYZE = textwrap.dedent("""
    import sys, time
    from pathlib import Path
    folder = Path(sys.argv[1])
    members = int(next(l for l in (folder / "config.yaml").read_text().splitlines() if "members:" in l).split(":")[1])
    print("video.mp4: 100 frames, 30.00 fps, 64x48")
    for f in (0, 50):
        print(f"\\r  pose + tracking: frame {f}/100", end="", file=sys.stderr, flush=True)
        if f == 50 and (folder.parent / "hold").exists():  # wait here until the test opens the gate
            while not (folder.parent / "gate").exists():
                time.sleep(0.02)
    print(file=sys.stderr)
    if members == 0:
        print("Found 1 clearly visible people; need at least 2 (set tracking.members in the config).", file=sys.stderr)
        sys.exit(3)
    print("results in", folder)
""")
VIEW = textwrap.dedent("""
    import sys
    from pathlib import Path
    viewer = Path(sys.argv[1]) / "viewer"
    viewer.mkdir(exist_ok=True)
    (viewer / "index.html").write_text("<!doctype html><title>viewer</title>")
    (viewer / "viewer_data.js").write_text("window.SYNCSCORE = {video: '../video.mp4'};")
""")


def stand_in(folder: Path):
    return [("pose", [sys.executable, "-c", ANALYZE, str(folder)]), ("view", [sys.executable, "-c", VIEW, str(folder)])]


@pytest.fixture
def server(tmp_path):
    server = make_server(tmp_path / "uploads", None, 0, steps=stand_in)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.runner.stop()
    server.server_close()


def request(server, method, path, body=None, headers=None):
    connection = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=10)
    connection.request(method, path, body=body, headers=headers or {})
    response = connection.getresponse()
    data = response.read()
    connection.close()
    return response, data


def upload(server, name="practice clip.mp4", members=3, body=MP4, headers=None):
    from urllib.parse import quote

    response, data = request(server, "POST", f"/api/uploads?name={quote(name)}&members={members}", body,
                             {"Content-Type": "video/mp4", **(headers or {})})
    return response.status, json.loads(data)


def wait_for(server, upload_id, done=lambda job: job["status"] in ("done", "failed"), timeout=30.0):
    end = time.time() + timeout
    while time.time() < end:
        _, data = request(server, "GET", f"/api/uploads/{upload_id}")
        job = json.loads(data)
        if done(job):
            return job
        time.sleep(0.05)
    raise AssertionError(f"job never got there: {job}")


# ---- uploads and jobs

def test_an_upload_is_saved_analysed_and_ends_on_its_viewer(server):
    status, job = upload(server)
    assert status == 201 and job["status"] == "queued" and job["name"] == "practice clip.mp4"
    folder = server.uploads.folder(job["id"])
    assert (folder / "video.mp4").read_bytes() == MP4
    assert yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))["tracking"]["members"] == 3

    job = wait_for(server, job["id"])
    assert job["status"] == "done" and job["error"] is None
    assert job["viewer"] == f"/uploads/{job['id']}/viewer/"
    assert any("100 frames" in line for line in job["log"])
    response, page = request(server, "GET", job["viewer"])
    assert response.status == 200 and b"<title>viewer</title>" in page
    assert response.getheader("Content-Type").startswith("text/html")
    response, _ = request(server, "GET", f"/uploads/{job['id']}/viewer")
    assert response.status == 301 and response.getheader("Location") == job["viewer"]
    # The list on the start page has it, without the log.
    _, data = request(server, "GET", "/api/uploads")
    listed = json.loads(data)["uploads"]
    assert [j["id"] for j in listed] == [job["id"]] and "log" not in listed[0]
    assert json.loads((folder / "job.json").read_text(encoding="utf-8"))["status"] == "done"


def test_job_status_shows_the_step_and_frame_progress_while_it_runs(server):
    (server.uploads.root / "hold").touch()
    _, job = upload(server)
    running = wait_for(server, job["id"], lambda j: j["progress"] == [50, 100])
    assert running["status"] == "running" and running["step"] == "pose"
    assert running["detail"] == "Finding every dancer's pose" and running["step_elapsed_s"] >= 0
    (server.uploads.root / "gate").touch()
    assert wait_for(server, job["id"])["status"] == "done"


def test_jobs_run_one_at_a_time_in_upload_order(server):
    ids = [upload(server, name=f"clip {k}.mp4")[1]["id"] for k in range(3)]
    jobs = [wait_for(server, i) for i in ids]
    assert all(j["status"] == "done" for j in jobs)
    finished = [j["finished"] for j in jobs]
    assert finished == sorted(finished)


def test_a_failed_job_says_why_and_runs_again_with_a_member_count(server):
    _, job = upload(server, members=0)  # the stand-in, like analyze, can't count fewer than 2
    failed = wait_for(server, job["id"])
    assert failed["status"] == "failed" and "need at least 2" in failed["error"] and failed["viewer"] is None
    response, data = request(server, "POST", f"/api/uploads/{job['id']}/run?members=4")
    assert response.status == 202 and json.loads(data)["members"] == 4
    assert wait_for(server, job["id"])["status"] == "done"
    folder = server.uploads.folder(job["id"])
    assert yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))["tracking"]["members"] == 4


@pytest.mark.parametrize("name, members, body, message", [
    ("notes.txt", 0, MP4, "Choose an .mp4 video."),
    ("clip.mp4", 0, b"not a video at all", "That file isn't an MP4 video."),
    ("clip.mp4", 99, MP4, "Members must be between 0"),
    ("clip.mp4", "two", MP4, "Members must be a whole number"),
])
def test_bad_uploads_are_refused_and_leave_nothing_behind(server, name, members, body, message):
    status, data = upload(server, name=name, members=members, body=body)
    assert status == 400 and data["error"].startswith(message)
    assert list(server.uploads.root.iterdir()) == [] and server.uploads.list() == []


def test_only_this_computer_and_this_page_are_answered(server):
    response, _ = request(server, "GET", "/", headers={"Host": "attacker.example"})
    assert response.status == 403
    status, _ = upload(server, headers={"Origin": "https://attacker.example"})
    assert status == 403 and server.uploads.list() == []
    port = server.server_address[1]
    status, _ = upload(server, headers={"Origin": f"http://127.0.0.1:{port}"})
    assert status == 201
    assert server.server_address[0] == "127.0.0.1"


def test_the_start_page_and_its_files_are_served_and_nothing_else(server):
    response, page = request(server, "GET", "/")
    assert response.status == 200 and b"/static/start.js" in page
    response, _ = request(server, "GET", "/static/start.js")
    assert response.status == 200 and response.getheader("Content-Type").startswith("text/javascript")
    for path in ("/static/../app.py", "/static/%2e%2e/app.py", "/uploads/../job.json", "/api/uploads/nope", "/uploads/20260101-000000-abcdef/video.mp4"):
        assert request(server, "GET", path)[0].status == 404, path


def test_a_restart_marks_unfinished_jobs_as_stopped(tmp_path):
    uploads = Uploads(tmp_path)
    job = uploads.create("clip.mp4", 2, io.BytesIO(MP4), len(MP4))
    uploads.update(job["id"], status="running", step="pose")
    again = Uploads(tmp_path)
    assert again.get(job["id"])["status"] == "failed" and "server stopped" in again.get(job["id"])["error"]


def test_real_steps_are_analyze_then_view_with_the_served_video(tmp_path):
    (pose, analyze), (view, show) = commands(tmp_path)
    assert (pose, view) == ("pose", "view")
    assert analyze[1:] == ["-m", "syncscore", "--config", str(tmp_path / "config.yaml"), "analyze", str(tmp_path / "video.mp4"), "--out", str(tmp_path)]
    assert show[5:] == ["view", str(tmp_path / "video.mp4"), "--results", str(tmp_path), "--video-url", VIDEO_URL]
    assert VIDEO_URL == "../video.mp4"


# ---- progress from what analyze prints

class Trickle(io.RawIOBase):
    """A pipe that hands out its bytes in the given pieces."""

    def __init__(self, pieces):
        self.pieces = list(pieces)

    def read1(self, n=-1):
        return self.pieces.pop(0) if self.pieces else b""


def test_output_pieces_end_at_carriage_returns_and_newlines():
    pipe = Trickle([b"clip: 9 frames\r\n\r  pose + tracking: frame 0/9", b"\r  pose + tracking: frame 5/9\r", b"\n", b"done\r\n"])
    pieces = [p for p in read_pieces(pipe) if p[0].strip()]
    assert pieces == [("clip: 9 frames", True), ("  pose + tracking: frame 0/9", None), ("  pose + tracking: frame 0/9", False),
                      ("  pose + tracking: frame 5/9", None), ("  pose + tracking: frame 5/9", True), ("done", True)]


def test_progress_follows_the_pose_loop_and_its_end():
    assert progress_update("  pose + tracking: frame 50/5476", False) == {"step": "pose", "detail": "Finding every dancer's pose", "progress": [50, 5476]}
    assert progress_update("  pose + tracking: frame 5450/5476", True)["step"] == "measure"
    assert progress_update("  re-detecting missed frames: 3/40", False)["progress"] == [3, 40]
    assert progress_update("  using cached detections (pass --redo-pose to run the model again)", True)["step"] == "measure"
    assert progress_update("results in x", True) is None


def test_the_reason_for_a_failure_is_the_exception_not_ffmpegs_chatter_after_it():
    log = ["Traceback (most recent call last):", '  File "cli.py", line 31, in _analyze',
           "FileNotFoundError: cannot open video: video.mp4", "[mov,mp4,m4a,3gp,3g2,mj2 @ 0000021e19676d40] moov atom not found"]
    assert failure_reason(log) == "FileNotFoundError: cannot open video: video.mp4"
    assert failure_reason(["clip: 9 frames", "Found 1 clearly visible people; need at least 2 (set tracking.members in the config)."]).startswith("Found 1")
    assert failure_reason([]) is None


# ---- byte ranges for <video>

@pytest.mark.parametrize("header, expected", [
    (None, None), ("", None), ("bytes=0-9", (0, 9)), ("bytes=100-", (100, 999)), ("bytes=-5", (995, 999)),
    ("bytes=990-5000", (990, 999)), ("bytes=-5000", (0, 999)), ("bytes=9-0", None), ("items=0-9", None),
    ("bytes=0-1,5-6", None), ("bytes=-", None),
])
def test_parse_range(header, expected):
    assert parse_range(header, 1000) == expected


@pytest.mark.parametrize("header", ["bytes=1000-", "bytes=5000-6000", "bytes=-0"])
def test_ranges_past_the_end_are_unsatisfiable(header):
    with pytest.raises(Unsatisfiable):
        parse_range(header, 1000)


def test_the_video_is_served_whole_or_by_byte_range(server):
    _, job = upload(server)
    url = f"/uploads/{job['id']}/video.mp4"
    response, body = request(server, "GET", url)
    assert response.status == 200 and body == MP4
    assert response.getheader("Accept-Ranges") == "bytes" and response.getheader("Content-Type") == "video/mp4"
    response, body = request(server, "GET", url, headers={"Range": "bytes=4-11"})
    assert response.status == 206 and body == b"ftypisom"
    assert response.getheader("Content-Range") == f"bytes 4-11/{len(MP4)}" and response.getheader("Content-Length") == "8"
    response, body = request(server, "GET", url, headers={"Range": "bytes=10000-"})
    assert response.status == 206 and body == MP4[10000:]
    response, body = request(server, "GET", url, headers={"Range": "bytes=-3"})
    assert response.status == 206 and body == MP4[-3:]
    response, body = request(server, "GET", url, headers={"Range": f"bytes={len(MP4)}-"})
    assert response.status == 416 and response.getheader("Content-Range") == f"bytes */{len(MP4)}" and body == b""
    response, body = request(server, "HEAD", url, headers={"Range": "bytes=0-99"})
    assert response.status == 206 and response.getheader("Content-Length") == "100" and body == b""
