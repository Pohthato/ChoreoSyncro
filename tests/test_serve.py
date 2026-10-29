"""`syncscore serve`: uploads, job status, the viewer's files and byte ranges for the video.

The server runs for real on a free local port; the analysis steps are small stand-in processes
that print what `analyze` prints, so a job takes a fraction of a second instead of minutes. An
upload first finds people and stops for the members to be named (the PREPARE stand-in writes three
people per frame); naming them, or measuring without names, runs ANALYZE and VIEW."""
import http.client
import io
import json
import sys
import textwrap
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from syncscore.serve.app import Unsatisfiable, make_server, parse_range
from syncscore.serve.jobs import VIDEO_URL, Uploads, commands, failure_reason, progress_update, read_pieces

MP4 = b"\x00\x00\x00\x18ftypisom" + bytes(range(256)) * 40  # an MP4 header, then 10 kB of known bytes

PREPARE = textwrap.dedent("""
    import json, sys, time
    from pathlib import Path
    import numpy as np
    from syncscore.pose.tracks import Detections
    from syncscore.pose.video import probe
    folder = Path(sys.argv[1])
    with open(folder / "prepared.txt", "a") as runs:
        runs.write("run\\n")
    try:
        info = probe(folder / "video.mp4")
        n, times = info.n_frames, info.times_ms.tolist()
    except Exception:  # the stand-in MP4 is only a header
        n, times = 100, [k * 1000 / 30 for k in range(100)]
    print(f"video.mp4: {n} frames, 30.00 fps, 64x48")
    for f in (0, 50):
        print(f"\\r  pose + tracking: frame {f}/{n}", end="", file=sys.stderr, flush=True)
        if f == 50 and (folder.parent / "hold").exists():  # wait here until the test opens the gate
            while not (folder.parent / "gate").exists():
                time.sleep(0.02)
    print(file=sys.stderr)
    frame, x = np.repeat(np.arange(n), 3), np.tile([10.0, 30.0, 50.0], n)
    Detections(frame=frame, track_id=np.tile([1, 2, 3], n), box=np.stack([x - 6, np.full(3 * n, 6.0), x + 6, np.full(3 * n, 42.0)], 1),
               box_conf=np.full(3 * n, 0.9), xy=np.zeros((3 * n, 17, 2)), kconf=np.full((3 * n, 17), 0.9), hist=np.zeros((3 * n, 4)),
               fps=30.0, n_frames=n, image_size=(64, 48)).save(folder / "detections.npz")
    (folder / "camera.npz").write_bytes(b"")
    (folder / "fingerprints.npz").write_bytes(b"")
    (folder / "pick_frames.json").write_text(json.dumps({"n_frames": n, "fps": 30.0, "size": [64, 48], "suggested": [10, 60], "times_ms": times}))
    print("ready to name the members (suggested frames: 10, 60)")
""")
ANALYZE = textwrap.dedent("""
    import sys, time
    from pathlib import Path
    folder = Path(sys.argv[1])
    (folder / "args.txt").write_text(" ".join(sys.argv[2:]))
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


def stand_in(folder: Path, phase: str = "plain"):
    """The real steps for this phase, each run by a stand-in script that gets the real arguments."""
    steps = []
    for step, command in commands(folder, phase):
        script = VIEW if "view" in command else PREPARE if "--pose-only" in command else ANALYZE
        steps.append((step, [sys.executable, "-c", script, str(folder), *command[3:]]))
    return steps


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
    job = None
    end = time.time() + timeout
    while time.time() < end:
        _, data = request(server, "GET", f"/api/uploads/{upload_id}")
        job = json.loads(data)
        if done(job):
            return job
        time.sleep(0.05)
    raise AssertionError(f"job never got there: {job}")


def picking(job):
    return job["status"] in ("picking", "failed")


def post_json(server, path, data):
    response, body = request(server, "POST", path, json.dumps(data).encode("utf-8"), {"Content-Type": "application/json"})
    return response.status, json.loads(body)


def people(server, upload_id, frame):
    response, data = request(server, "GET", f"/api/uploads/{upload_id}/people?frame={frame}")
    return response.status, json.loads(data)


def grey_video(path: Path, frames: int = 25) -> bytes:
    """A real 64x48 MP4 whose frame k is a flat grey of 10 + 8k."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (64, 48))
    for k in range(frames):
        writer.write(np.full((48, 64, 3), 10 + 8 * k, np.uint8))
    writer.release()
    return path.read_bytes()


# ---- uploads and jobs

def test_an_upload_is_saved_analysed_and_ends_on_its_viewer(server):
    status, job = upload(server)
    assert status == 201 and job["status"] == "queued" and job["name"] == "practice clip.mp4"
    folder = server.uploads.folder(job["id"])
    assert (folder / "video.mp4").read_bytes() == MP4
    assert yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))["tracking"]["members"] == 3

    assert wait_for(server, job["id"], picking)["status"] == "picking"
    response, _ = request(server, "POST", f"/api/uploads/{job['id']}/run?members=3")  # measure without names
    assert response.status == 202
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
    assert wait_for(server, job["id"], picking)["status"] == "picking"


def test_jobs_run_one_at_a_time_in_upload_order(server):
    ids = [upload(server, name=f"clip {k}.mp4")[1]["id"] for k in range(3)]
    jobs = [wait_for(server, i, picking) for i in ids]
    assert all(j["status"] == "picking" for j in jobs)
    finished = [j["finished"] for j in jobs]
    assert finished == sorted(finished)


def test_a_failed_job_says_why_and_runs_again_with_a_member_count(server):
    _, job = upload(server)
    wait_for(server, job["id"], picking)
    request(server, "POST", f"/api/uploads/{job['id']}/run?members=0")  # the stand-in, like analyze, can't count fewer than 2
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
    analyze_args = ["-m", "syncscore", "--config", str(tmp_path / "config.yaml"), "analyze", str(tmp_path / "video.mp4"), "--out", str(tmp_path)]
    view_args = ["view", str(tmp_path / "video.mp4"), "--results", str(tmp_path), "--video-url", VIDEO_URL]
    (pose, analyze), (view, show) = commands(tmp_path, "plain")
    assert (pose, view) == ("pose", "view")
    assert analyze[1:] == analyze_args and show[5:] == view_args
    assert commands(tmp_path) == commands(tmp_path, "plain")
    [(pose, prepare)] = commands(tmp_path, "prepare")
    assert pose == "pose" and prepare[1:] == [*analyze_args, "--pose-only"]
    (measure, analyze), (view, show) = commands(tmp_path, "picked")
    assert (measure, view) == ("measure", "view")
    assert analyze[1:] == [*analyze_args, "--members", str(tmp_path / "picks.json")] and show[5:] == view_args
    assert VIDEO_URL == "../video.mp4"


# ---- naming the members

def _two_picks(found, frame=10, names=("Ann", "Bo")):
    return {"frame": frame, "members": [{"name": n, "box": p["box"]} for n, p in zip(names, found["people"])]}


def test_an_upload_pauses_for_picking_then_measures_with_the_picks(server):
    _, job = upload(server)
    job = wait_for(server, job["id"], picking)
    assert job["status"] == "picking" and job["step"] == "pick" and job["detail"] == "Name the members in one frame"
    assert job["picking"] == {"n_frames": 100, "fps": 30.0, "size": [64, 48], "suggested": [10, 60]}
    assert job["viewer"] is None
    status, found = people(server, job["id"], 10)
    assert status == 200 and found["frame"] == 10 and len(found["people"]) == 3
    assert [p["box"][0] for p in found["people"]] == [4.0, 24.0, 44.0] and all(p["clear"] for p in found["people"])
    status, queued = post_json(server, f"/api/uploads/{job['id']}/picks", _two_picks(found))
    assert status == 202 and queued["status"] == "queued" and queued["phase"] == "picked"
    done = wait_for(server, job["id"])
    assert done["status"] == "done" and done["viewer"]
    folder = server.uploads.folder(job["id"])
    saved = json.loads((folder / "picks.json").read_text(encoding="utf-8"))
    assert saved["frame"] == 10 and [m["name"] for m in saved["members"]] == ["Ann", "Bo"]
    assert f"--members {folder / 'picks.json'}" in (folder / "args.txt").read_text()


def test_people_and_frames_answer_only_for_frames_in_the_video(server):
    _, job = upload(server)
    wait_for(server, job["id"], picking)
    for frame in ("-1", "100", "abc"):
        assert people(server, job["id"], frame)[0] == 404, frame
        assert request(server, "GET", f"/api/uploads/{job['id']}/frame/{frame}.jpg")[0].status == 404, frame


def test_people_wait_until_the_poses_are_found(server):
    (server.uploads.root / "hold").touch()
    _, job = upload(server)
    wait_for(server, job["id"], lambda j: j["progress"] == [50, 100])
    status, data = people(server, job["id"], 10)
    assert status == 409 and data["error"]
    (server.uploads.root / "gate").touch()
    wait_for(server, job["id"], picking)


def test_the_frame_image_is_the_frame_the_boxes_belong_to(server, tmp_path):  # Review Focus 1
    _, job = upload(server, body=grey_video(tmp_path / "grey.mp4"))
    job = wait_for(server, job["id"], picking)
    assert job["picking"]["n_frames"] == 25
    for frame, grey in ((17, 146), (3, 34)):
        response, data = request(server, "GET", f"/api/uploads/{job['id']}/frame/{frame}.jpg")
        assert response.status == 200 and response.getheader("Content-Type") == "image/jpeg"
        assert response.getheader("Cache-Control") == "no-store"
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
        assert abs(float(image.mean()) - grey) <= 4, frame


def test_bad_picks_are_refused_with_the_reason_and_the_job_keeps_waiting(server):
    _, job = upload(server)
    wait_for(server, job["id"], picking)
    _, found = people(server, job["id"], 10)
    status, data = post_json(server, f"/api/uploads/{job['id']}/picks", {"frame": 10, "members": [{"name": "Ann", "box": found["people"][0]["box"]}]})
    assert status == 400 and data["error"] == "Name at least 2 members."
    picks = _two_picks(found)
    picks["members"][1]["box"] = [0, 0, 2, 2]
    status, data = post_json(server, f"/api/uploads/{job['id']}/picks", picks)
    assert status == 400 and data["error"] == "No person found where Bo was picked in frame 10."
    response, data = request(server, "POST", f"/api/uploads/{job['id']}/picks", b"{", {"Content-Type": "application/json"})
    assert response.status == 400
    response, _ = request(server, "POST", f"/api/uploads/{job['id']}/picks", json.dumps(_two_picks(found)).encode(), {"Content-Type": "text/plain"})
    assert response.status == 415
    response, _ = request(server, "POST", f"/api/uploads/{job['id']}/picks", b"x" * (65 << 10), {"Content-Type": "application/json"})
    assert response.status == 413
    assert server.uploads.get(job["id"])["status"] == "picking"
    assert not (server.uploads.folder(job["id"]) / "picks.json").exists()


def test_pick_again_returns_to_picking_without_measuring_again(server):
    _, job = upload(server)
    wait_for(server, job["id"], picking)
    _, found = people(server, job["id"], 10)
    post_json(server, f"/api/uploads/{job['id']}/picks", _two_picks(found))
    assert wait_for(server, job["id"])["status"] == "done"
    response, data = request(server, "POST", f"/api/uploads/{job['id']}/repick")
    again = json.loads(data)
    assert response.status == 202 and again["status"] == "picking" and again["picking"]["suggested"] == [10, 60]
    folder = server.uploads.folder(job["id"])
    assert (folder / "prepared.txt").read_text().count("run") == 1  # people were not found again
    status, _ = post_json(server, f"/api/uploads/{job['id']}/picks", _two_picks(found, names=("Cy", "Di")))
    assert status == 202 and wait_for(server, job["id"])["status"] == "done"
    assert [m["name"] for m in json.loads((folder / "picks.json").read_text(encoding="utf-8"))["members"]] == ["Cy", "Di"]


def test_pick_again_finds_people_again_when_their_files_are_gone(server):
    _, job = upload(server)
    wait_for(server, job["id"], picking)
    folder = server.uploads.folder(job["id"])
    (folder / "camera.npz").unlink()
    response, data = request(server, "POST", f"/api/uploads/{job['id']}/repick")
    assert response.status == 202 and json.loads(data)["status"] == "queued"
    assert wait_for(server, job["id"], picking)["status"] == "picking"
    assert (folder / "prepared.txt").read_text().count("run") == 2


def test_picks_are_taken_only_while_the_job_waits_for_them(server):
    (server.uploads.root / "hold").touch()
    _, job = upload(server)
    wait_for(server, job["id"], lambda j: j["progress"] == [50, 100])
    status, data = post_json(server, f"/api/uploads/{job['id']}/picks", {"frame": 10, "members": []})
    assert status == 400 and data["error"]
    response, _ = request(server, "POST", f"/api/uploads/{job['id']}/repick")
    assert response.status == 400
    (server.uploads.root / "gate").touch()
    wait_for(server, job["id"], picking)


def test_a_restart_keeps_a_job_waiting_for_picks(tmp_path):
    uploads = Uploads(tmp_path)
    job = uploads.create("clip.mp4", 2, io.BytesIO(MP4), len(MP4))
    assert job["phase"] == "prepare"
    uploads.update(job["id"], status="picking", step="pick")
    assert Uploads(tmp_path).get(job["id"])["status"] == "picking"


def test_an_old_job_without_a_phase_measures_as_before(tmp_path):
    uploads = Uploads(tmp_path)
    job = uploads.create("clip.mp4", 2, io.BytesIO(MP4), len(MP4))
    saved = json.loads((uploads.folder(job["id"]) / "job.json").read_text(encoding="utf-8"))
    del saved["phase"]
    saved["status"] = "done"
    (uploads.folder(job["id"]) / "job.json").write_text(json.dumps(saved), encoding="utf-8")
    assert Uploads(tmp_path).get(job["id"])["phase"] == "plain"


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


def test_progress_in_the_prepare_phase_stays_on_pose_until_it_ends():
    assert progress_update("  pose + tracking: frame 50/5476", False, "prepare")["step"] == "pose"
    assert progress_update("  pose + tracking: frame 5476/5476", True, "prepare") == {"detail": "Following the camera's movement", "progress": None}
    assert progress_update("  following the camera: frame 50/5476", False, "prepare") == {
        "step": "pose", "detail": "Following the camera's movement", "progress": [50, 5476]}
    assert "step" not in progress_update("  following the camera: frame 5450/5476", True, "prepare")
    assert progress_update("  fingerprints: frame 64/5476", False, "prepare") == {
        "step": "pose", "detail": "Making a fingerprint of each person", "progress": [64, 5476]}
    assert "step" not in progress_update("  fingerprints: frame 5450/5476", True, "prepare")
    assert "step" not in progress_update("  using cached detections (pass --redo-pose to run the model again)", True, "prepare")
    assert progress_update("Downloading youtu_reid.onnx (102 MB, first run only)", True, "prepare") == {
        "detail": "Downloading the re-identification model (first run only)"}
    assert progress_update("  pose + tracking: frame 5450/5476", True)["step"] == "measure"  # the other phases as before


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
