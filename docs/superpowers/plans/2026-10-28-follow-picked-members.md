# Follow the Picked Members Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The user names the members once, in one frame of the upload page, and SyncScore follows each named member for the whole clip, leaving her unseen rather than putting her name on someone else.

**Architecture:** `analyze --pose-only` adds two cached steps after pose detection: the camera's movement (`camera.npz`) and one re-ID fingerprint per detection (`fingerprints.npz`). The upload job then waits in `picking` until the page sends `picks.json`. `analyze --members picks.json` then splits tracklets where they change hands, names each piece with a kNN classifier seeded by the picks, and keeps names through gaps with a constant-velocity Kalman filter in stage coordinates. The result feeds the existing `assemble`, gap-filling and analysis, with members in pick order. Runs without picks take the old path untouched.

**Tech Stack:** Python 3.12, NumPy, OpenCV 4.13 (`cv2.dnn`, optical flow), the OpenCV Zoo YouTu person re-ID ONNX model, the stdlib HTTP server in `syncscore/serve`, plain JS in the page, pytest, `node --test`.

**Spec:** `docs/superpowers/specs/2026-10-28-follow-picked-members.md`

## Global Constraints

- Without `--members`, every output stays exactly as now: the BTS `summary.json` after Task 10 is byte-identical to the baseline below, and the full existing suite passes.
- Never commit footage, frames, model weights, `.npz` files, audio or anything under `results/`. Weights download to `~/.cache/syncscore/` and are checked against the pinned SHA-256.
- Fingerprints are numbers only. No pixels are saved, except the audit's frame images under `results/` (git-ignored, as now).
- Re-ID model: OpenCV Zoo YouTu ReID (Apache-2.0), run with `cv2.dnn.readNet`. Input 128x256 (w x h), RGB, ImageNet mean (0.485, 0.456, 0.406) and std (0.229, 0.224, 0.225); output 768 numbers.
- `picks.json` is exactly `{"frame": int, "members": [{"name": str, "box": [x1, y1, x2, y2]}, ...]}`: at least 2 members, names non-empty after trimming, distinct ignoring case, at most 40 characters, kept verbatim otherwise.
- Each pick resolves to the detection in its frame with IoU at least 0.5, or fails with an error that names the pick.
- Config section `identity:` has exactly `model`, `name_margin`, `learn_margin`, `rounds` (3), `split_distance`, `position_gate`, `free_after_s` (3.0).
- New routes: `GET /api/uploads/<id>/frame/<n>.jpg`, `GET /api/uploads/<id>/people?frame=<n>`, `POST /api/uploads/<id>/picks`, `POST /api/uploads/<id>/repick`. The server's local-only and Origin checks apply to all of them.
- The picking view uses the start page's and viewer's fonts, colours and spacing. Logic that needs no DOM lives in `picker_core.js` and is tested in Node.
- Commits: plain conventional messages (`feat(pose): ...`) with no AI attribution. Prose in docs follows the `clean-user-facing-text` rule in the user's CLAUDE.md.
- Commands run from the repo root in Git Bash: `.venv/Scripts/python.exe -m pytest ...` and `node --test "tests/js/*.test.mjs"`.

## Review Focus

1. **The frame image and its boxes come from different frames.** OpenCV seeks by index with the average frame rate, so on variable-rate phone video the picture can be a frame off from the boxes drawn on it. Expected: the served image is the decoded frame the detections were found in. Test: Task 11, `test_the_frame_image_is_the_frame_the_boxes_belong_to`.
2. **Stale caches after `--redo-pose`.** New detections with old `fingerprints.npz` or `camera.npz` would give every row someone else's fingerprint. Expected: both are recomputed whenever the detections change. Test: Task 10, `test_prepare_recomputes_when_the_detections_change`.
3. **A picked person with no tracker ID, or on a tracklet shorter than `min_frames`.** Expected: her name is still seeded from that one detection and spreads by fingerprint. Test: Task 8, `test_a_picked_person_with_no_tracker_id_still_seeds_her_name`.
4. **Two picks on the same person** (boxes that resolve to one detection). Expected: an error naming both picks, nothing saved. Test: Task 6, `test_two_picks_on_the_same_person_are_refused`.
5. **A plain backdrop with a fixed camera** (too few background corners to fit). Expected: one segment with identity transforms, so the motion model keeps working. Test: Task 5, `test_a_plain_backdrop_with_a_fixed_camera_stays_one_segment`.

## Baseline (before Task 1)

- [ ] Copy the BTS caches and record the current output, so Task 10 can show nothing changed:

```bash
mkdir -p results/bts_baseline && cp results/bts/detections.npz results/bts/recovered.npz results/bts_baseline/
.venv/Scripts/python.exe -m syncscore --config configs/bts_boy_with_luv.yaml analyze "tests/[CHOREOGRAPHY] BTS (방탄소년단) '작은 것들을 위한 시 (Boy With Luv)' Dance Practice.mp4" --out results/bts_baseline
```

Expected: `using cached detections` and `results in results/bts_baseline`. Keep `results/bts_baseline/summary.json`.

---

### Task 1: Fingerprints and the identity config

**Files:**
- Create: `syncscore/pose/fingerprint.py`
- Modify: `syncscore/pose/video.py` (gains `read_frames`, moved from `recover.py`), `syncscore/pose/recover.py` (imports it), `syncscore/pose/tracks.py`, `syncscore/config.py`, `configs/default.yaml`
- Test: `tests/test_fingerprint.py`, `tests/test_config.py`, `tests/test_recover.py` (import path only)

**Interfaces:**
- Produces:
  - `MODELS: dict[str, dict]` with keys `"youtu"` and `"youtu-int8"`, each `{"file", "url", "size", "sha256"}`.
  - `fetch(entry: dict, cache: Path = CACHE, opener=urllib.request.urlopen) -> Path`.
  - `crop(frame: np.ndarray, box: np.ndarray) -> np.ndarray | None`.
  - `Fingerprinter(net, batch: int = 32)`, `Fingerprinter.load(name: str) -> Fingerprinter`, `Fingerprinter.__call__(crops: list[np.ndarray]) -> np.ndarray` (K, D) float32 with unit rows.
  - `fingerprint_video(video, det: Detections, fingerprinter, progress: bool = True) -> np.ndarray` (N, D) float32, zeros where there is no usable crop.
  - `save_fingerprints(path, vectors, model: str, key: str)` and `load_fingerprints(path, model: str, key: str) -> np.ndarray | None`.
  - In `tracks.py`: `Detections.vectors: np.ndarray | None = None` (last field, never written by `save`), and `detections_key(det) -> str`.
  - In `video.py`: `read_frames(cap, frames) -> Iterator[tuple[int, np.ndarray]]`.
  - In `config.py`: `IdentityConfig` and `Config.identity`.

- [ ] **Step 1: Write the failing tests** in `tests/test_fingerprint.py`. Use a stand-in for the `cv2.dnn` net:

```python
class MeanNet:  # stands in for cv2.dnn: the "fingerprint" is the input blob's mean per channel
    def setInput(self, blob): self.blob = blob
    def forward(self): return self.blob.mean(axis=(2, 3))

def test_fetch_downloads_once_and_checks_the_hash(tmp_path):
    payload = b"model bytes"
    entry = {"file": "m.onnx", "url": "https://example.invalid/m.onnx", "size": len(payload),
             "sha256": hashlib.sha256(payload).hexdigest()}
    calls = []
    def opener(url, timeout=60):
        calls.append(url)
        return io.BytesIO(payload)
    path = fetch(entry, tmp_path, opener)
    assert path.read_bytes() == payload
    assert fetch(entry, tmp_path, opener) == path and len(calls) == 1

def test_a_download_with_the_wrong_hash_is_refused_and_not_kept(tmp_path):
    entry = {"file": "m.onnx", "url": "https://example.invalid/m.onnx", "size": 8, "sha256": "0" * 64}
    with pytest.raises(ValueError, match="SHA-256"):
        fetch(entry, tmp_path, lambda url, timeout=60: io.BytesIO(b"tampered"))
    assert list(tmp_path.iterdir()) == []

def test_crops_are_rgb_imagenet_normalised_and_vectors_unit_length():
    red = np.zeros((40, 20, 3), np.uint8); red[..., 2] = 255          # BGR red
    v = Fingerprinter(MeanNet())([red])
    expected = np.array([(1 - 0.485) / 0.229, -0.456 / 0.224, -0.406 / 0.225])
    assert np.allclose(v[0], expected / np.linalg.norm(expected), atol=1e-3)

def test_batches_give_the_same_vectors_as_one_at_a_time():
    # 5 random crops: Fingerprinter(MeanNet(), batch=2)(crops) equals stacking each crop alone

def test_crop_is_clipped_to_the_frame_and_none_when_tiny():
    frame = np.zeros((100, 200, 3), np.uint8)
    assert crop(frame, np.array([150.0, 10, 260, 90])).shape == (80, 50, 3)
    assert crop(frame, np.array([10.0, 10, 15, 90])) is None           # under 8 px wide

def test_fingerprint_video_gives_one_vector_per_detection_row(tmp_path):
    # 10 frames of 160x120 written with cv2.VideoWriter (mp4v): left half blue, right half red.
    # Detections: per frame a left box [5, 5, 75, 115], a right box [85, 5, 155, 115], and on frame 3
    # a 4 px box. Left rows ≈ the blue vector, right rows ≈ the red one (atol 0.05), tiny row all zeros.

def test_saved_fingerprints_load_only_for_the_same_detections_and_model(tmp_path):
    # save with ("youtu", key); load with the same -> equal (float16 tolerance 1e-3);
    # a different model or key -> None; a missing file -> None
```

Add to `tests/test_config.py`:

```python
def test_identity_settings_load_and_reject_unknown_keys(tmp_path):
    identity = load_config().identity
    assert identity.rounds == 3 and identity.free_after_s == 3.0 and identity.model in MODELS
    bad = tmp_path / "bad.yaml"
    bad.write_text("identity:\n  margin: 0.1\n")
    with pytest.raises(ValueError, match="IdentityConfig"):
        load_config(bad)
```

- [ ] **Step 2: Run them and see them fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_fingerprint.py tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: syncscore.pose.fingerprint` and `AttributeError: identity`.

- [ ] **Step 3: Implement**

  - `MODELS` pins the repo revision and the hashes:
    - URL: `https://huggingface.co/opencv/person_reid_youtureid/resolve/b7bb1eda63dd4035eb7fa48d13d31b818877fc1a/<file>`
    - `youtu`: `person_reid_youtu_2021nov.onnx`, size 106878407, sha256 `0579683334d4b9440221606dcb461656dd0dc64143b18f48faedaced9b4f580d`
    - `youtu-int8`: `person_reid_youtu_2021nov_int8.onnx`, size 26763574, sha256 `4757c4cb759b79030a9870abf29c064c2ee51e079a05700690800c81b16cf245`
    - `CACHE = Path.home() / ".cache" / "syncscore"`.
  - `fetch` streams to `<file>.part` while hashing, then compares the hash. On a match it renames the file; on a mismatch it deletes it and raises `ValueError` naming the SHA-256. Before downloading it prints `Downloading <file> (<MB> MB, first run only)` to stderr. That line is what the job page shows in Task 11.
  - `Fingerprinter.__call__`:
    - Build the blob with `cv2.dnn.blobFromImages(batch, 1 / 255, (128, 256), swapRB=True)`.
    - Then per channel `(x - mean) / std`, `setInput`, `forward`.
    - L2-normalise each row; a zero row stays zero.
  - `crop` clips the box to the frame and returns `None` when the result is under 8 px on either side.
  - `fingerprint_video` decodes in order with `read_frames` (the old `recover._read_frames`, moved to `video.py` with its docstring). It crops the rows of each wanted frame, batches the crops across frames, and prints `\r  fingerprints: frame {f}/{n_frames}` every 50 frames, then a newline.
  - `save_fingerprints` stores `vectors` as float16 plus `model` and `key`; `load_fingerprints` returns float32.
  - `detections_key` is the first 16 hex digits of a SHA-1 over `det.frame.astype(np.int64)` and `det.box.astype(np.float64)` bytes.
  - `IdentityConfig` defaults: `model="youtu"`, `name_margin=0.05`, `learn_margin=0.10`, `rounds=3`, `split_distance=0.35`, `position_gate=3.0`, `free_after_s=3.0`. Task 4 replaces the margins and model with measured values. Add the same section to `configs/default.yaml` with a one-line comment per key, in the file's style.

- [ ] **Step 4: Run the tests and the old recover tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_fingerprint.py tests/test_config.py tests/test_recover.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add syncscore/pose/fingerprint.py syncscore/pose/video.py syncscore/pose/recover.py syncscore/pose/tracks.py syncscore/config.py configs/default.yaml tests/test_fingerprint.py tests/test_config.py tests/test_recover.py
git commit -m "feat(pose): a re-ID fingerprint for every detection"
```

---

### Task 2: Naming pieces by fingerprint

**Files:**
- Create: `syncscore/pose/recognize.py`
- Test: `tests/test_recognize.py`

**Interfaces:**
- Consumes: `IdentityConfig` (Task 1).
- Produces:
  - `TOP_K = 5`, `MAX_ROWS = 64` (rows of a piece used for scoring, evenly spaced), `MAX_EXAMPLES = 4000` (per class, evenly spaced).
  - `@dataclass Naming: member: int | None; margin: float; scores: np.ndarray`. The scores are one per member, then not-a-member last.
  - `class_score(rows: np.ndarray, examples: np.ndarray) -> float`: the mean over rows of the mean of each row's top-k cosine similarities to the examples. Zero rows are ignored; NaN if nothing is left.
  - `@dataclass Recognizer: examples: list[np.ndarray]; others: np.ndarray; margin: float`, with `name(rows, exclude=()) -> Naming`.
  - `overlap(a_frames, b_frames) -> int` (shared frames).
  - `learn(pieces: list[np.ndarray], frames: list[np.ndarray], seeds: list[int], not_members: list[int], config: IdentityConfig) -> tuple[Recognizer, list[Naming]]`. `seeds[i]` is the piece index picked as member i. Seeds come back as `Naming(i, math.inf, ...)`, not-member seeds as `Naming(None, math.inf, ...)`.

- [ ] **Step 1: Write the failing tests.** Build fingerprints in 32 dimensions: `unit(rng)` gives a random unit vector, and `near(v, cos, rng)` gives a unit vector at that cosine to `v`. `piece(centre, n=30)` gives n rows of `centre + 0.05 * N(0, 1)`, re-normalised. Four members a, b, c, d and two dancers e, f come from `unit`. The picked frame gives seeds a..d and not-member seeds e and f.

```python
def test_pieces_take_the_name_of_the_member_they_look_like():
    # three new pieces of b, d, a -> members 1, 3, 0, each with margin > 0.3

def test_a_piece_between_two_members_stays_unassigned():
    # piece around normalise(a + b) -> member None

def test_a_dancer_not_in_the_picked_frame_is_not_named():
    # piece around a fresh unit vector g -> member None

def test_learning_rounds_teach_a_members_other_side():
    # a2 = normalise(0.3 a + 0.95 u); a dancer seed at d2 = near(a2, 0.55)
    # chain pieces at normalise((1 - t) a + t a2) for t in 0.25, 0.5, 0.75, 1.0
    # rounds=0: the t=1.0 piece -> None; rounds=3: it -> member 0

def test_learning_does_not_drift_onto_a_decoy_near_a_member():
    # decoy centre near(b, 0.8), with a not-member seed of it; 5 decoy pieces and 5 b pieces
    # after 3 rounds: every decoy piece None, every b piece member 1

def test_a_piece_seen_at_the_same_time_as_a_member_is_never_learned_as_her():
    # piece around near(a, 0.95) whose frames overlap the seed of a by 30 frames:
    # it does not join a's examples (recognizer.examples[0] keeps only a's rows)

def test_excluding_a_member_rescores_without_it():
    # piece of a: name(rows, exclude={0}).member is None

def test_rows_without_a_fingerprint_are_ignored():
    # a piece of a plus 10 zero rows -> member 0; a piece of only zero rows -> None, margin nan
```

- [ ] **Step 2: Run them and see them fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_recognize.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement.** The margin is the best non-excluded member's score minus the larger of the other members' scores and not-a-member. A piece is named when the margin is at least `name_margin`. Learning:

```text
for round in range(config.rounds):
    name every non-seed piece with margin = config.name_margin
    joiners: pieces with member m and margin >= config.learn_margin,
             strongest first, refused if they share more than 2 frames with any piece already in m's examples
    dancers: pieces whose not-a-member score beats every member by >= config.learn_margin
    stop early when nothing joins; otherwise add their rows (capped at MAX_EXAMPLES, evenly spaced)
final: name every non-seed piece once more with the learned examples
```

- [ ] **Step 4: Run the tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_recognize.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add syncscore/pose/recognize.py tests/test_recognize.py
git commit -m "feat(pose): name tracks from the picked frame by fingerprint, with learning rounds"
```

---

### Task 3: An audit that labels everyone and scores names

**Files:**
- Modify: `syncscore/cli.py` (`_audit`, the `audit` parser), `syncscore/pose/tracks.py` (`score_audit`)
- Test: `tests/test_audit.py` (new), `tests/test_tracks.py`

**Interfaces:**
- Produces:
  - `audit_frames(seen: np.ndarray, fps: float, count: int, windows: list[tuple[float, float]], step_s: float = 0.2) -> np.ndarray`. That is `count` frames evenly spaced over the seen frames, plus every `step_s` inside each window whether or not anyone is tracked, sorted and unique.
  - `parse_window(text: str) -> tuple[float, float]` for `"148-150"`; `ValueError` when the end comes before the start.
  - `ids.csv` columns become `frame,row,shown_label,true_name`:
    - Members are drawn as now.
    - Every other detection in that frame is drawn grey and labelled `?1`, `?2`... left to right, with `row` its detection row.
    - A member box matches a detection by IoU ≥ 0.9; `row` is -1 when none does.
  - `score_audit(csv_path, fps, step_s: float = 0.2) -> dict`. It keeps the old keys, computed on member labels only (rows whose label starts with `?` are left out). New keys:
    - `exact_accuracy`
    - `dancer_share`
    - `longest_wrong_s`
    - `tracked_share` (dict per true member name)

- [ ] **Step 1: Write the failing tests**

```python
def test_audit_frames_add_every_step_inside_each_window():
    frames = audit_frames(np.ones(3000, bool), fps=30.0, count=3, windows=[(10.0, 10.4)], step_s=0.2)
    assert list(frames) == [0, 300, 306, 312, 1499, 2999]

def test_a_window_must_end_after_it_starts():
    assert parse_window("148-150.5") == (148.0, 150.5)
    with pytest.raises(ValueError):
        parse_window("150-148")

def test_named_audit_scores(tmp_path):
    rows = ["frame,row,shown_label,true_name", "0,1,A,A", "0,2,B,dancer", "0,3,?1,B", "6,1,A,A", "6,2,B,B",
            "12,1,A,B", "18,1,A,B", "24,1,A,A"]
    path = tmp_path / "ids.csv"; path.write_text("\n".join(rows) + "\n")
    score = score_audit(path, fps=30.0)
    assert score["labelled"] == 7
    assert score["exact_accuracy"] == pytest.approx(4 / 7)
    assert score["dancer_share"] == pytest.approx(1 / 7)
    assert score["longest_wrong_s"] == pytest.approx(0.4)   # label A wrong at 12 and 18: 6 frames + one 0.2 s step
    assert score["tracked_share"] == {"A": 1.0, "B": 0.25}
```

Add `test_audit_draws_every_detection_and_writes_rows(tmp_path)` to `tests/test_audit.py`:
- Write a 30-frame video with cv2, a `keypoints_raw.npz` with one member, and a `detections.npz` with that member plus one other person.
- Run `main(["audit", video, "--results", dir, "--frames", "3", "--times", "0.2-0.4"])`.
- Assert that `ids.csv` has a `?1` row with the other person's detection row, and that `frame_*.jpg` exist.

- [ ] **Step 2: Run them and see them fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_audit.py tests/test_tracks.py -v`
Expected: the new tests FAIL; the old `score_audit` tests still pass.

- [ ] **Step 3: Implement.**
  - `--times` is `nargs="*"` of windows; `--step` defaults to 0.2.
  - `longest_wrong_s`: a run is consecutive audited frames of one label (frame gaps at most `round(step_s * fps) + 1`) where shown differs from true. Its length is `(last - first) / fps + step_s`.
  - `tracked_share[name]`: among the audited frames where `name` appears as a true name, the share where that row's shown label is `name`. The names counted are the true names other than `dancer`.
  - Print the same footage warning as now.

- [ ] **Step 4: Run the tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_audit.py tests/test_tracks.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add syncscore/cli.py syncscore/pose/tracks.py tests/test_audit.py tests/test_tracks.py
git commit -m "feat(audit): label every person, sample dense windows, score names directly"
```

---

### Task 4: The gate. Do fingerprints separate the people on aespa?

**Files:**
- Create: `syncscore/pose/gate.py`
- Modify: `syncscore/cli.py` (new `id-gate` command), `configs/default.yaml` (measured `model`, `name_margin`, `learn_margin`)
- Test: `tests/test_gate.py`

**Interfaces:**
- Consumes: `build_tracklets` (existing), `learn` (Task 2), `fingerprint_video`, `Fingerprinter`, `save_fingerprints`/`load_fingerprints` (Task 1), the audit CSV (Task 3).
- Produces:
  - `gate(det: Detections, vectors: np.ndarray, labels: dict[int, str], picked_frame: int, config: Config) -> dict`. `labels` maps detection row to its true name, with `dancer` for backup dancers. The pieces are tracklets (min_frames `max(3, int(0.3 * fps))`). Seeds are the labelled rows of the picked frame, members in order of first appearance in `labels`. Every other labelled row in the picked frame is not-a-member. The result:
    - `member_correct`: share of member rows outside the picked frame whose piece is named their member.
    - `member_unassigned`
    - `dancer_as_member`
    - `per_member`
    - `rows`
  - `syncscore id-gate VIDEO --results DIR --labels CSV --frame N --model youtu --model youtu-int8`:
    - Fingerprints go to `DIR/gate_fingerprints_<model>.npz`, reused when present.
    - For each model it reports `ms_per_crop`, `fingerprint_s`, and `gate()` for `name_margin` in 0.02, 0.04, 0.06, 0.08, 0.10, with `learn_margin = 2 * name_margin`.
    - It writes `DIR/gate.json`.

- [ ] **Step 1: Write the failing test** `test_gate_counts_right_names_unassigned_members_and_named_dancers`:
  - Use hand-built detections (the `tests/test_tracks.py` `_detections` style): two members and one dancer over 60 frames, each its own track, vectors from three unit centres.
  - Labels cover frames 0, 30 and 59; picked frame 0.
  - Assert `member_correct == 1.0` and `dancer_as_member == 0.0`.
  - A labelled member row with track_id -1 counts in `member_unassigned`.

- [ ] **Step 2: Run it and see it fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_gate.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `gate` and the `id-gate` command.** Then run the tests.

Run: `.venv/Scripts/python.exe -m pytest tests/test_gate.py -v`
Expected: PASS.

- [ ] **Step 4: Label 30 aespa frames**

```bash
.venv/Scripts/python.exe -m syncscore audit tests/aespa-whiplash-dance-practice-mirrored-ytmp4.savetube.vip.mp4 --results results/aespa --frames 30
```

- Look at each `results/aespa/audit/frame_*.jpg` and write `results/aespa/audit/legend.md`, one line per member outfit with a short tag (`A` to `D`). Members are told apart by outfit, never by face.
- Fill `true_name` for every row of `ids.csv`: the member's tag, or `dancer`.
- Choose as the picked frame the labelled frame where all four members are most clearly separate.

- [ ] **Step 5: Run the gate on both versions of the model**

```bash
.venv/Scripts/python.exe -m syncscore id-gate tests/aespa-whiplash-dance-practice-mirrored-ytmp4.savetube.vip.mp4 --results results/aespa --labels results/aespa/audit/ids.csv --frame <picked> --model youtu --model youtu-int8
```

Expected: `results/aespa/gate.json` with both models and the margin grid.

- [ ] **Step 6: Decide**
  - **Pass:** at least 95% `member_correct` and at most 2% `dancer_as_member`, including the member dressed in black.
  - Take int8 when it passes and its `member_correct` is within 1 point of fp32. Otherwise take fp32.
  - Take the smallest `name_margin` that passes. Write the model and the margins into `configs/default.yaml` and keep the numbers for Task 13's walkthrough stage.
  - **If neither model passes, STOP.** Report the numbers to the user and build nothing further.

- [ ] **Step 7: Commit** (code and config only; the CSV, frames and `gate.json` stay in `results/`)

```bash
git add syncscore/pose/gate.py syncscore/cli.py configs/default.yaml tests/test_gate.py
git commit -m "feat(pose): id-gate measures how well fingerprints name the members"
```

---

### Task 5: Following the camera

**Files:**
- Create: `syncscore/pose/camera.py`
- Test: `tests/test_camera.py`

**Interfaces:**
- Produces:
  - `@dataclass CameraPath: transform (T, 2, 3) float32; segment (T,) int32; inliers (T,) int32`, with `save(path, key: str)` and `CameraPath.load(path, key: str) -> CameraPath | None` (None when missing or the key differs).
  - `still_camera(n_frames) -> CameraPath` (identity, one segment). Tests in Tasks 7 and 8 use it.
  - `estimate_frames(frames: Iterable[np.ndarray], boxes: list[np.ndarray]) -> CameraPath`. `boxes[f]` are the person boxes of frame f, shape (k, 4).
  - `estimate_camera(video, det, progress: bool = True) -> CameraPath`. It decodes in order and prints `\r  following the camera: frame {f}/{T}` every 50 frames.
  - `to_stage(camera, frames: np.ndarray, points: np.ndarray) -> np.ndarray`: (n, 2) image points to (n, 2) stage points.
  - `scale_at(camera, frames) -> np.ndarray`: the transform's uniform scale.
  - Constants:
    - `WORK_WIDTH = 640`
    - `MAX_CORNERS = 400`
    - `MIN_POINTS = 40`
    - `MIN_INLIERS = 20`
    - `BOX_PAD = 0.1`
    - `CUT_FACTOR = 4.0`
    - `CUT_MIN = 12.0`

- [ ] **Step 1: Write the failing tests.**
  - The scene: a 1600x1200 background of Gaussian-blurred noise.
  - Frame f is `cv2.warpAffine(background, M_f, (640, 360))`, where `M_f` scales by `zoom ** f` about the centre and shifts by `pan * f` px.
  - For a background point p, its stage position is `M_0 p`, and in frame f it appears at `M_f p`.

```python
def test_a_pan_and_zoom_are_recovered():
    # 60 frames, pan 3 px/frame, zoom 1.003/frame, no people
    # for a 5x5 grid of background points and every f: |to_stage(cam, f, M_f p) - M_0 p| < 1.5 px
    # cam.segment is all 0

def test_a_fixed_camera_gives_identity_transforms():
    # 30 identical frames: every transform within 1e-3 of [[1, 0, 0], [0, 1, 0]]

def test_a_hard_cut_starts_a_new_segment():
    # 20 frames of one background, then 20 of another: segment[:20] == 0, segment[20:] == 1

def test_moving_people_do_not_pull_the_estimate():
    # fixed camera; a textured 200x300 block moving 5 px/frame inside a person box
    # covering 30% of the frame: every translation within 0.5 px of 0

def test_a_plain_backdrop_with_a_fixed_camera_stays_one_segment():   # Review Focus 5
    # flat grey frames with a moving textured person block: one segment, identity transforms
```

- [ ] **Step 2: Run them and see them fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_camera.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement.** Per frame, on grey images scaled to `WORK_WIDTH` (scale s):

```text
mask = everything outside the person boxes padded by BOX_PAD
diff = mean |gray - prev_gray| over the mask
cut  = diff > max(CUT_MIN, CUT_FACTOR * median of the last 30 diffs)
pts_prev = goodFeaturesToTrack(prev_gray, MAX_CORNERS, 0.01, 8, mask=mask)
pts_now, status = calcOpticalFlowPyrLK(prev_gray, gray, pts_prev)
if cut: new segment, identity
elif fewer than MIN_POINTS tracked points: same segment, A = identity   # plain backdrop
else:
    A, inliers = estimateAffinePartial2D(pts_now, pts_prev, RANSAC, ransacReprojThreshold=2.0)
    if A is None or inliers < MIN_INLIERS: new segment, identity
    else: A[:, 2] /= s                      # back to full-size pixels; the 2x2 part is unchanged
transform[f] = transform[f-1] @ A   (as 3x3, first row pair kept)    # image f -> stage
```

- [ ] **Step 4: Run the tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_camera.py -v`
Expected: PASS.

- [ ] **Step 5: Check it on aespa.**
  - Run `estimate_camera` on aespa with `results/aespa/detections.npz` in a scratch script.
  - Track 20 background corners from frame 0 through the first 600 frames with optical flow. Their stage positions should stay within 3 px.
  - Note the segment count (the spec expects no hard cuts, so 1 or very few) for the walkthrough.

- [ ] **Step 6: Commit**

```bash
git add syncscore/pose/camera.py tests/test_camera.py
git commit -m "feat(pose): follow the camera's pan and zoom into stage coordinates"
```

---

### Task 6: Picks and suggested frames

**Files:**
- Create: `syncscore/pose/picks.py`
- Test: `tests/test_picks.py`

**Interfaces:**
- Produces:
  - `class PickError(ValueError)`; `MAX_NAME = 40`.
  - `@dataclass Pick: name: str; box: np.ndarray`; `@dataclass Picks: frame: int; members: list[Pick]`.
  - `parse_picks(data: dict, n_frames: int) -> Picks`, `load_picks(path, n_frames) -> Picks`, `dump_picks(picks) -> dict`.
  - `resolve_picks(picks, det) -> list[int]` (detection rows, in pick order).
  - `people_in_frame(det, frame: int, min_conf: float) -> list[dict]`, each `{"row": int, "box": [4 floats, 1 decimal], "clear": bool}`, left to right. `clear` means 8+ keypoints at `min_conf`.
  - `suggest_frames(det, min_conf: float, count: int = 5) -> list[int]`.

- [ ] **Step 1: Write the failing tests**

```python
def test_picks_are_parsed_with_names_trimmed_and_kept_verbatim():
    picks = parse_picks({"frame": 5, "members": [{"name": "  Ning-ning 宁宁 ", "box": [1, 2, 3, 4]},
                                                 {"name": "<b>Winter</b>", "box": [5, 6, 7, 8]}]}, n_frames=10)
    assert [p.name for p in picks.members] == ["Ning-ning 宁宁", "<b>Winter</b>"]

@pytest.mark.parametrize("data, message", [
    ({"frame": 5, "members": [{"name": "A", "box": [1, 2, 3, 4]}]}, "at least 2"),
    ({"frame": 5, "members": [{"name": "A", "box": [1, 2, 3, 4]}, {"name": " ", "box": [1, 2, 3, 4]}]}, "Type a name"),
    ({"frame": 5, "members": [{"name": "Karina", "box": [1, 2, 3, 4]}, {"name": "karina", "box": [5, 6, 7, 8]}]}, "already called"),
    ({"frame": 5, "members": [{"name": "A" * 41, "box": [1, 2, 3, 4]}, {"name": "B", "box": [1, 2, 3, 4]}]}, "40 characters"),
    ({"frame": 10, "members": [{"name": "A", "box": [1, 2, 3, 4]}, {"name": "B", "box": [1, 2, 3, 4]}]}, "frame"),
    ({"frame": 5, "members": [{"name": "A", "box": [1, 2, 3]}, {"name": "B", "box": [1, 2, 3, 4]}]}, "box"),
])
def test_bad_picks_are_refused_with_the_reason(data, message): ...

def test_each_pick_resolves_to_the_detection_under_it():
    # frame 5 with three people; picks at their boxes shifted by 5 px -> their rows, in pick order

def test_a_pick_where_nobody_stands_is_refused_by_name():
    # PickError matching "Karina" when the box overlaps nobody by IoU >= 0.5

def test_two_picks_on_the_same_person_are_refused():   # Review Focus 4
    # two boxes over one detection -> PickError matching "Karina and Winter were picked on the same person"

def test_people_in_a_frame_are_listed_left_to_right_with_clear_flags(): ...

def test_suggested_frames_favour_clear_separate_people_away_from_the_ends():
    # 300 frames at 30 fps, 2 clear people everywhere except: frames 100-120: 4 clear, separate people;
    # 200-210: 4 people, two overlapping;
    # frame 5: 5 clear people. First suggestion in 100..120; none in 0..14 or 285..299;
    # suggestions at least 60 frames apart; at most 5
```

- [ ] **Step 2: Run them and see them fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_picks.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement.**
  - Messages:
    - "Name at least 2 members."
    - "Type a name for every member."
    - "Another member is already called {name}."
    - "Names can be up to 40 characters."
    - "Frame {n} is not in this video."
    - "Each pick needs a box of 4 numbers."
    - "No person found where {name} was picked in frame {frame}."
    - "{a} and {b} were picked on the same person."
  - Ranking in `suggest_frames`:
    - Leave out the first and last 5% of the clip.
    - Score each frame by the number of clear people, then by `-(max pairwise IoU)`. Frames with any pairwise IoU above 0.05 rank below every frame without.
    - Take the best frames greedily, at least 2 s apart.

- [ ] **Step 4: Run the tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_picks.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add syncscore/pose/picks.py tests/test_picks.py
git commit -m "feat(pose): read, check and resolve the picked members; suggest frames to pick in"
```

---

### Task 7: Motion and splitting tracks that change hands

**Files:**
- Create: `syncscore/pose/motion.py`, `tests/scenes.py` (shared test helper)
- Test: `tests/test_motion.py`

**Interfaces:**
- Consumes: `CameraPath`, `to_stage`, `scale_at`, `still_camera` (Task 5); `Tracklet`, `Detections` (existing).
- Produces:
  - Constants:
    - `ACCEL_STD = 2.0` (body heights per s²)
    - `MEASURE_STD = 0.05` (body heights)
    - `WINDOW_S = 0.5`
    - `HEADING_MARGIN = 0.5`
    - `MIN_SPEED = 0.3` (body heights per s)
  - `positions(det, rows, camera, min_conf) -> tuple[np.ndarray, np.ndarray]`. The first array is the stage hip centre (n, 2): the mean of keypoints 11 and 12 when both are at `min_conf`, else the box centre. The second is the body height (n,), box height times `scale_at`.
  - `class Track`:
    - `Track.follow(points, frames, height, fps) -> Track` runs the constant-velocity filter over one piece.
    - `.predict(frames: int)`
    - `.distance(point) -> float` (Mahalanobis)
    - `.position`, `.velocity`
  - `@dataclass Split: frame: int; track_id: int; reason: str` (`"fingerprint"` or `"heading"`).
  - `split_at_crossings(det, tracklets, camera, config: Config) -> tuple[list[Tracklet], list[Split]]`. It uses `det.vectors`, `config.tracking.crossing_iou`, `config.identity.split_distance` and `config.pose.min_keypoint_conf`. Pieces keep their `track_id`; each piece's `hist` is the mean of its rows.
- `tests/scenes.py` provides:
  - `unit(rng, dim=32)` and `near(v, cos, rng)`.
  - `@dataclass Person: look; x: Callable[[int], float]; track: Callable[[int], int | None]`. `track(f)` is the tracker ID at frame f, or None when the person isn't detected; -1 is allowed.
  - `scene(people, n_frames=300, fps=30.0, noise=0.05, seed=0) -> tuple[Detections, np.ndarray]`. Boxes are `[x-40, 200, x+40, 520]`, hips at y 400, keypoint conf 0.9, `det.vectors` set. The second value gives the true person per row.

- [ ] **Step 1: Write the failing tests**

```python
def test_a_steady_walk_is_predicted_and_doubt_grows_while_unseen():
    # 30 frames moving +10 px/frame (height 320): after predict(10) the position is within 5 px of last + 100;
    # distance(last + 100) < 1; distance(last - 100) > 3; after predict(60) more, distance(last - 100) is smaller

def test_a_track_handed_to_a_differently_dressed_dancer_is_split_at_the_crossing():
    # A walks right, D walks left, they cross at frame 150; track 1 is A before and D after
    # -> track 1 becomes two pieces split within 150 +- 5, reason "fingerprint"

def test_a_track_handed_to_a_lookalike_going_the_other_way_is_split_by_heading():
    # as above, with B = near(a, 0.85) (fingerprint distance under split_distance) -> split, reason "heading"

def test_a_dancer_turning_around_alone_is_not_split():
    # one person reverses direction at frame 150 and her look drifts to near(a, 0.8); nobody overlaps -> no split

def test_two_lookalikes_passing_without_a_handover_are_not_split():
    # A and B = near(a, 0.85) cross with the tracker keeping each on its own person -> no split
```

- [ ] **Step 2: Run them and see them fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_motion.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement.**
  - The filter works in stage pixels. Its noise is scaled by the piece's median body height: process noise from `ACCEL_STD`, measurement noise `MEASURE_STD`.
  - Crossing events and the split rule:

```text
for each tracklet: per frame, the largest IoU with any other detection in that frame
events: runs of frames with IoU > crossing_iou, merging runs at most 2 frames apart
for each event [a, b], mid = (a + b) // 2, W = round(WINDOW_S * fps):
    before = own rows in [a - W, a), after = own rows in (b, b + W]; skip unless both have >= 3
    fingerprint: 1 - cos(mean(before), mean(after)) > split_distance  -> split "fingerprint"
    heading: other = the track ID overlapping most during the event
             own_v, other_v = velocities over [a - W, a); after_v = velocity over after
             if |own_v|, |other_v| >= MIN_SPEED * height and cos(own_v, other_v) < 0
                and cos(after_v, other_v) - cos(after_v, own_v) > HEADING_MARGIN  -> split "heading"
    split: rows <= mid stay, rows > mid form a new piece; keep checking the later piece's events
    never split if either side would be shorter than min_frames
```

- [ ] **Step 4: Run the tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_motion.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add syncscore/pose/motion.py tests/scenes.py tests/test_motion.py
git commit -m "feat(pose): split tracks where they change hands at a crossing; a motion model with heading"
```

---

### Task 8: Following the members by name

**Files:**
- Create: `syncscore/pose/identity.py`
- Modify: `syncscore/pose/tracks.py` (`assemble(det, slots, keep_order: bool = False)`)
- Test: `tests/test_identity.py`, `tests/test_tracks.py`

**Interfaces:**
- Consumes:
  - `learn`, `Recognizer`, `Naming`, `overlap` (Task 2)
  - `split_at_crossings`, `Track`, `positions` (Task 7)
  - `CameraPath`, `still_camera` (Task 5)
  - rows from `resolve_picks` (Task 6)
  - `Slot` and `build_tracklets` (existing)
- Produces:
  - `@dataclass Following: slots: list[Slot]; recognizer: Recognizer; report: dict`.
  - `follow_members(det, tracklets, camera, picked_rows: list[int], config: Config) -> Following`. It needs `det.vectors`. The report has `picked_frame`, `splits` (`{"time_s", "track_id", "reason"}`), `close_calls` (`{"start_s", "end_s", "member", "margin", "named"}`) and `position_rejections` (`{"time_s", "member", "distance"}`). Members are indices in pick order; the CLI swaps in names.
  - `allow_for(recognizer) -> Callable[[Detections, int, int], bool]`. Its `(det, member, row)` is True when `recognizer.name(det.vectors[row:row + 1]).member == member`.
  - `assemble(..., keep_order=True)` keeps the slots' order and skips the left-to-right sort.

- [ ] **Step 1: Write the failing tests** (scenes from `tests/scenes.py`, `still_camera`, 300 frames at 30 fps, picks at frame 30):

```python
def test_lookalikes_crossing_keep_their_own_names():
    # A (look a) walks 200 -> 1400, B (near(a, 0.85)) walks 1400 -> 200; the tracker swaps their IDs at 150.
    # Every row in slot 0 is truly A, every row in slot 1 truly B; each tracked in >= 90% of frames

def test_someone_walking_in_front_of_a_member_does_not_take_her_name():
    # A still at x=600; dancer D walks 100 -> 1100 and is in the picked frame; A undetected 120-130;
    # the tracker puts A's ID on D from 120 on, and A returns as ID 3 at 131.
    # Slot 0 holds no D rows and includes ID 3's rows

def test_a_member_who_leaves_the_frame_is_found_again_by_fingerprint():
    # A seen 0-99 at x=300, gone 100-249, back at x=1200 as a new ID -> slot 0 includes the new ID's rows

def test_a_lookalike_far_from_where_a_hidden_member_is_heading_is_refused():
    # A walks right 4 px/frame, hidden 150-179, returns as ID 3 at 920; lookalike (look a) appears
    # still at x=100 as ID 5 from 160 -> slot 0 has ID 3, not ID 5; one position rejection at 160/30 s

def test_two_pieces_with_one_name_at_once_keep_the_stronger():
    # A seen throughout as ID 1; a piece near(a, 0.93) as ID 7 over 100-200 -> slot 0 has only ID 1 rows

def test_a_picked_person_with_no_tracker_id_still_seeds_her_name():   # Review Focus 3
    # A's rows have track_id -1 at the picked frame only and ID 4 elsewhere -> slot 0 includes ID 4's rows
```

Add to `tests/test_tracks.py` `test_assemble_keeps_the_given_order_when_asked`: two slots, right one first, `keep_order=True` returns them in that order.

- [ ] **Step 2: Run them and see them fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_identity.py tests/test_tracks.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement.**
  - Pieces come from `split_at_crossings`.
  - Seed pieces are the pieces that contain the picked rows. A row in no piece becomes a one-row piece.
  - The not-member seeds are the pieces, or one-row pieces, of the picked frame's other detections.
  - Then `learn`.
  - Then for each member, walk outward from her seed: forward over later pieces, and the same walk with time reversed over earlier ones.

```text
candidates: pieces named m (by fingerprint), sorted by start
end = seed piece
for P in candidates:
    Q = accepted pieces of m sharing more than 2 frames with P
    if Q:
        if P.margin > max margin in Q: drop Q from m (to re-score), accept P
        else: P goes to re-score
        continue
    gap = P.start - end.end - 1
    if gap <= free_after_s * fps and P.start and end.end are in the same camera segment:
        track = Track.follow(positions of end); track.predict(gap + 1)
        if track.distance(first position of P) > position_gate: record a rejection; continue
    accept P; if P.end > end.end: end = P
re-score: name each dropped piece again excluding the members it lost; if it now names m2,
          run the same checks against m2's accepted pieces once (no further rounds)
```

  - Close calls are pieces whose margin is within `name_margin / 2` of `name_margin`.

- [ ] **Step 4: Run the tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_identity.py tests/test_tracks.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add syncscore/pose/identity.py syncscore/pose/tracks.py tests/test_identity.py tests/test_tracks.py
git commit -m "feat(pose): follow each picked member by name through crossings and gaps"
```

---

### Task 9: Gap-filling checks fingerprints

**Files:**
- Modify: `syncscore/pose/fill.py`, `syncscore/pose/recover.py`
- Test: `tests/test_fill.py`, `tests/test_recover.py`

**Interfaces:**
- Consumes: `Detections.vectors` (Task 1), `allow_for` (Task 8), `Fingerprinter` and `crop` (Task 1).
- Produces:
  - `fill_gaps(det, tracks, max_gap_s, min_keypoint_conf, allow=None)`. A pair where `allow(det, member, row)` is False costs infinity. The existing `accept` stays the position test: it interpolates between the member's boxes on both sides of the gap.
  - `fill_occlusions(..., redetect=None, allow=None)` passes `allow` to both passes.
  - `merge(det, extra)` concatenates `vectors` when both have them, else leaves `vectors` None.
  - `redetect(..., fingerprinter=None)` fingerprints what it finds from the frames it already decodes. It sets `vectors` and saves them in `recovered.npz`. A cache made without vectors doesn't match when a fingerprinter is given.

- [ ] **Step 1: Write the failing tests**

```python
def test_fill_gaps_skips_candidates_the_acceptance_test_refuses():
    # the existing "near the expected spot" case: allow=lambda det, m, r: False -> nothing filled;
    # allow=None -> filled as before

def test_merge_keeps_fingerprints_only_when_both_sides_have_them(): ...

def test_a_recover_cache_without_fingerprints_is_rebuilt_when_they_are_needed(tmp_path): ...
```

- [ ] **Step 2: Run them and see them fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_fill.py tests/test_recover.py -v`
Expected: the new tests FAIL; the old ones pass.

- [ ] **Step 3: Implement** the parameters above. With `allow=None` and no fingerprinter, the code path is the current one.

- [ ] **Step 4: Run the tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_fill.py tests/test_recover.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add syncscore/pose/fill.py syncscore/pose/recover.py tests/test_fill.py tests/test_recover.py
git commit -m "feat(pose): gap-filling can require a member's fingerprint"
```

---

### Task 10: `analyze --pose-only` and `analyze --members`

**Files:**
- Create: `syncscore/pose/prepare.py`
- Modify: `syncscore/cli.py` (`_analyze`, the `analyze` parser)
- Test: `tests/test_follow_cli.py`

**Interfaces:**
- Consumes: everything above.
- Produces:
  - `prepare(video, out: Path, det, config, progress: bool = True) -> tuple[CameraPath, np.ndarray]`:
    - Loads `camera.npz` and `fingerprints.npz` when their key is `detections_key(det)` (and the model is `config.identity.model`); otherwise computes and saves them.
    - Writes `pick_frames.json` as `{"n_frames", "fps", "size": [w, h], "suggested": [...], "times_ms": [...]}`. `times_ms` comes from `probe` and is what the server seeks with in Task 11.
  - `analyze VIDEO --out DIR --pose-only`: detections (cached or found), then `prepare`, then it prints `ready to name the members (suggested frames: ...)` and stops.
  - `analyze VIDEO --out DIR --members PICKS`:
    1. `load_picks`, then `resolve_picks`, then `prepare`, then `det.vectors = vectors`.
    2. `build_tracklets`, then `follow_members`, then `assemble(keep_order=True)`.
    3. `fill_occlusions` with `allow_for(...)`, and a second look with the fingerprinter.
    4. The names are the picked names.
    5. In `tracking_report.json`, `identity` replaces `suspected_swaps`, `suspected_swaps_per_minute` and `note`. It holds `picked_frame`, `model`, `tracked_share` per name (frames with a row after filling, divided by n_frames), `splits`, `close_calls` and `position_rejections`, with member names in place of indices.
    6. A `PickError` prints its message and exits 2.
  - The two flags are mutually exclusive. Without either, `_analyze` runs exactly as now.

- [ ] **Step 1: Write the failing tests.** The synthetic scene:
  - Six people from `make_group(GroupSpec(n_members=6, duration_s=6, fps=30, seed=1))`, keypoints scaled into 640x360.
  - Four are members and two are dancers.
  - A 180-frame video: a textured still background with a filled rectangle per person in a distinct colour.
  - `detections.npz` written into the out folder, so `analyze` uses it as the cache.
  - A config with `tracking.recover: false`.
  - `Fingerprinter.load` monkeypatched to a fingerprinter whose vector is the crop's normalised mean BGR colour.

```python
def test_pose_only_prepares_for_picking_and_stops(scene_dir):
    # exit 0; camera.npz, fingerprints.npz and pick_frames.json exist; keypoints_raw.npz does not

def test_members_are_followed_by_name_in_pick_order(scene_dir):
    # picks.json names the four members right to left as "D", "C", "B", "A"
    # keypoints_raw names == ["D", "C", "B", "A"]; each member's box centre is within 5 px of her true
    # position wherever seen; no dancer box is ever a member's; tracking_report has "identity" and no "suspected_swaps"

def test_a_bad_picks_file_stops_with_the_reason(scene_dir, capsys):
    # one name -> exit 2, "Name at least 2 members." on stderr

def test_prepare_recomputes_when_the_detections_change(scene_dir):   # Review Focus 2
    # after --pose-only, shift every box by 1 px and save detections.npz -> prepare recomputes both files
    # (their stored key changes); with the same detections a second run reuses them (mtime unchanged)

def test_without_picks_analyze_is_unchanged(scene_dir):
    # plain analyze: names M1..M4 left to right, tracking_report has suspected_swaps and no identity
```

- [ ] **Step 2: Run them and see them fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_follow_cli.py -v`
Expected: FAIL (`--pose-only` unknown).

- [ ] **Step 3: Implement `prepare` and the two flags**, with `_analyze`'s existing body as the no-flag path.

- [ ] **Step 4: Run the new tests and the whole suite**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: all pass.

- [ ] **Step 5: Check that BTS is unchanged**

```bash
cp results/bts_baseline/summary.json results/bts_baseline/summary.json.base
.venv/Scripts/python.exe -m syncscore --config configs/bts_boy_with_luv.yaml analyze "tests/[CHOREOGRAPHY] BTS (방탄소년단) '작은 것들을 위한 시 (Boy With Luv)' Dance Practice.mp4" --out results/bts_baseline
cmp results/bts_baseline/summary.json results/bts_baseline/summary.json.base && echo identical
```

Expected: `identical`. Any difference is a bug in this task: find it before going on.

- [ ] **Step 6: Commit**

```bash
git add syncscore/pose/prepare.py syncscore/cli.py tests/test_follow_cli.py
git commit -m "feat(cli): analyze --pose-only prepares for picking; --members follows the picked members"
```

---

### Task 11: The upload server waits for picks

**Files:**
- Modify: `syncscore/serve/jobs.py`, `syncscore/serve/app.py` (routes and the module docstring's route list), `syncscore/pose/video.py`
- Test: `tests/test_serve.py`, `tests/test_video.py`

**Interfaces:**
- Consumes: `parse_picks`, `resolve_picks`, `dump_picks`, `people_in_frame`, `PickError` (Task 6); `Detections` (existing).
- Produces:
  - `read_frame(video, frame: int, times_ms: np.ndarray) -> np.ndarray | None` in `video.py`. It seeks by time and checks by timestamp:

```text
target = times_ms[frame]; half = half the median frame spacing
for preroll in (2000, 8000, None):                      # None: reopen and read from the start
    seek to max(0, target - preroll) ms (or reopen); grab; t = POS_MSEC
    if t > target + half: continue                     # landed past it: back off further
    while t < target - half: grab; t = POS_MSEC
    return retrieve()
```

  - `commands(folder, phase) -> list[tuple[str, list[str]]]`:
    - `"prepare"`: `[("pose", analyze ... --pose-only)]`
    - `"picked"`: `[("measure", analyze ... --members <folder>/picks.json), ("view", view ...)]`
    - `"plain"`: the current pair
  - Jobs gain a `phase` field. `create` sets `"prepare"`, `again` sets `"plain"`, and an old job without the field counts as `"plain"`.
  - When a `"prepare"` job's steps succeed, it becomes `status="picking", step="pick", detail="Name the members in one frame"`, with `picking = {n_frames, fps, size, suggested}` read from `pick_frames.json` (not `times_ms`).
  - `"picking"` is not in `UNFINISHED`, so it survives a restart.
  - `progress_update(line, newline, phase="plain")`:
    - Adds `following the camera: frame (\d+)/(\d+)` and `fingerprints: frame (\d+)/(\d+)`, both step `"pose"`, with the details "Following the camera's movement" and "Making a fingerprint of each person".
    - A `Downloading ... .onnx` line gives the detail "Downloading the re-identification model (first run only)".
    - In `"prepare"`, the end of a loop changes only the detail, never the step.
  - Methods on `Uploads`:
    - `people(upload_id, frame) -> dict` returns `{"frame", "people": [...]}`, keeping one loaded `Detections` per upload keyed by file mtime.
    - `frame_jpeg(upload_id, frame) -> bytes` (JPEG quality 85).
    - `save_picks(upload_id, data) -> dict`: only while `picking`. It parses and resolves, turns a `PickError` into an `UploadError` with the same message, writes `picks.json`, and sets `phase="picked"`, `status="queued"`.
    - `repick(upload_id) -> dict`: back to `picking` when `camera.npz`, `fingerprints.npz` and `pick_frames.json` exist; otherwise `phase="prepare"` and queued. Unfinished jobs are refused.
  - `class NotReady(UploadError)` (409) is raised when `detections.npz` or `pick_frames.json` is missing. A frame outside the video, or not a whole number, gets a 404 "No such frame."
  - Routes:
    - `GET /api/uploads/<id>/frame/<n>.jpg` (image/jpeg, `Cache-Control: no-store`)
    - `GET /api/uploads/<id>/people?frame=<n>`
    - `POST /api/uploads/<id>/picks` (JSON body, at most 64 kB, Content-Type `application/json`)
    - `POST /api/uploads/<id>/repick`
    - Measuring without names reuses `POST /api/uploads/<id>/run?members=`.

- [ ] **Step 1: Write the failing tests.** `stand_in(folder, phase)`:
  - It runs a `PREPARE` script for `"prepare"`. That script writes a `detections.npz` (three people per frame, 100 frames at 30 fps) using `syncscore.pose.tracks.Detections`, plus empty `camera.npz` and `fingerprints.npz` placeholders and a `pick_frames.json`.
  - For the other phases it runs the current ANALYZE and VIEW. ANALYZE also writes its argv to `args.txt`.
  - Uploads that need real frames use a 25-frame 64x48 mp4 written with cv2 (mp4v), where frame k is a flat grey of `10 + 8k`.

```python
def test_an_upload_pauses_for_picking_then_measures_with_the_picks(server): ...
    # wait for status "picking"; job["picking"]["suggested"] present; people?frame=10 -> 3 people;
    # POST picks for 2 of them -> queued -> done; picks.json saved; args.txt contains --members

def test_people_and_frames_answer_only_for_frames_in_the_video(server): ...
    # frame -1, 100 and "abc" -> 404 for both routes

def test_the_frame_image_is_the_frame_the_boxes_belong_to(server):   # Review Focus 1
    # real 25-frame video: GET frame/17.jpg decodes to a mean grey of 146 +- 4; frame/3.jpg to 34 +- 4

def test_bad_picks_are_refused_with_the_reason_and_the_job_keeps_waiting(server): ...
    # one name -> 400 "Name at least 2 members."; a box over nobody -> 400 naming the pick; status still "picking"

def test_pick_again_returns_to_picking_without_measuring_again(server): ...

def test_a_restart_keeps_a_job_waiting_for_picks(tmp_path): ...

def test_progress_in_the_prepare_phase_stays_on_pose_until_it_ends(): ...
```

Update `test_real_steps_are_analyze_then_view_with_the_served_video` to cover the three phases. Add a `read_frame` test to `tests/test_video.py`: on the 25-frame video, frames 0, 17 and 24 come back with grey levels 10, 146 and 202 (±4).

- [ ] **Step 2: Run them and see them fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_serve.py tests/test_video.py -v`
Expected: the new tests FAIL.

- [ ] **Step 3: Implement.**

- [ ] **Step 4: Run the tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_serve.py tests/test_video.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add syncscore/serve/jobs.py syncscore/serve/app.py syncscore/pose/video.py tests/test_serve.py tests/test_video.py
git commit -m "feat(serve): an upload waits for the members to be named, then measures with them"
```

---

### Task 12: The picking view

**Files:**
- Create: `syncscore/serve/page/picker_core.js`, `syncscore/serve/page/picker.js`
- Modify: `syncscore/serve/page/index.html`, `syncscore/serve/page/start.js`, `syncscore/serve/page/start.css`
- Test: `tests/js/picker_core.test.mjs`

**Interfaces:**
- Consumes: the Task 11 routes and the job's `picking` field.
- Produces: `picker_core.js` is a classic script exposing `window.PickerCore` and a CommonJS module, in the style of `viewer_core.js`:
  - `layout(people, imageW, imageH) -> [{index, left, top, width, height, z}]`, in percentages, sorted by box centre x. `z` is higher for smaller boxes, so a person in front stays clickable.
  - `nameProblem(named, person, name) -> string | null`, with the same messages as Task 6 for empty, too long and duplicate (ignoring case, other people only).
  - `setName(named, person, name) -> named`: a rename keeps its place; a new name goes last. `removeName(named, person) -> named`. `named` is `[{person, name}]` in pick order.
  - `picksBody(frame, people, named) -> {frame, members: [{name, box}]}` in pick order.
  - `canContinue(named) -> boolean` (2 or more).
  - `timeLabel(frame, fps) -> "m:ss.s"`.

- [ ] **Step 1: Write the failing Node tests**, one per function. Among them:
  - `layout` puts a small box inside a big one on top.
  - Renaming keeps the order.
  - "karina" is refused when another person is called "Karina".
  - `picksBody` follows pick order, not left-to-right.
  - `timeLabel(1215, 30) === "0:40.5"`.

- [ ] **Step 2: Run them and see them fail**

Run: `node --test "tests/js/*.test.mjs"`
Expected: FAIL (`Cannot find module`).

- [ ] **Step 3: Implement `picker_core.js`, then the view.**
  - In the job card of `index.html`:
    - `<ol class="steps">` gains `<li data-step="pick">Name the members</li>` between pose and measure. `STEPS` in `start.js` becomes `["upload", "pose", "pick", "measure", "view"]`.
    - A `<section id="picking" hidden>` holds:
      - A heading "Name the members".
      - The hint "Click a person, type their name, press Enter. Name at least two."
      - Suggested-frame buttons labelled with `timeLabel`.
      - A `<figure>` with the frame `<img>` and one absolutely positioned `<button>` per person. Each button's accessible name is "Person n" or the given name, and named boxes show the name as a tag.
      - A name form that opens under the chosen box (above it near the bottom edge): Enter confirms, Escape cancels.
      - A frame slider with its time.
      - The note "Names belong to this frame; picking another frame clears them." It is shown once a name exists, and changing the frame clears the names.
      - The list of named members with Remove buttons.
      - The error line.
      - **Continue** (primary, disabled until `canContinue`) and **Measure without names**, which posts to `run` with the upload's member count.
    - Delete or Backspace on a focused named box removes its name. Tab follows `layout` order.
  - Earlier uploads' rows gain a **Pick again** button that posts to `repick`.
  - Colours, type and spacing use `start.css`'s existing tokens and the viewer's look: 2 px outlines, the viewer's focus ring, no animation beyond the viewer's.

- [ ] **Step 4: Run the Node tests and the Python suite**

Run: `node --test "tests/js/*.test.mjs"` and `.venv/Scripts/python.exe -m pytest tests/test_viewer_js.py tests/test_serve.py -q`
Expected: PASS.

- [ ] **Step 5: See it work.**
  - With a scratch script, make an upload folder under `results/uploads/` from aespa: `video.mp4` and `detections.npz` copied from `results/aespa`, and a `job.json` with `status: "done"`, `phase: "plain"`.
  - Start `syncscore serve` through `.claude/launch.json` and click **Pick again**. `--pose-only` reuses the cached poses and computes the camera and the fingerprints.
  - When it shows the picking view, check:
    - The boxes sit on the people at the suggested frames and after moving the slider.
    - Naming, renaming and removing work, with the mouse and with Tab, Enter and Delete.
    - **Continue** is disabled at one name.
    - The view works at 375 px wide.
  - Name four members and continue: the job measures and opens the viewer with those names.
  - Take a screenshot of the picking view.

- [ ] **Step 6: Commit**

```bash
git add syncscore/serve/page/picker_core.js syncscore/serve/page/picker.js syncscore/serve/page/index.html syncscore/serve/page/start.js syncscore/serve/page/start.css tests/js/picker_core.test.mjs
git commit -m "feat(serve): name the members in one frame of the upload page"
```

---

### Task 13: Ground truth on aespa, walkthrough and README

**Files:**
- Modify: `docs/walkthrough.md` (new stage), `README.md`, `configs/default.yaml` (only if a measured change is needed)

- [ ] **Step 1: Run the whole flow on aespa from the command line**

```bash
mkdir -p results/aespa_named && cp results/aespa/detections.npz results/aespa_named/
.venv/Scripts/python.exe -m syncscore --config configs/aespa_whiplash.yaml analyze tests/aespa-whiplash-dance-practice-mirrored-ytmp4.savetube.vip.mp4 --out results/aespa_named --pose-only
```

Write `results/aespa_named/picks.json` from the first suggested frame where all four members show, using the gate's outfit tags (`A` to `D`) as names. Then:

```bash
.venv/Scripts/python.exe -m syncscore --config configs/aespa_whiplash.yaml analyze tests/aespa-whiplash-dance-practice-mirrored-ytmp4.savetube.vip.mp4 --out results/aespa_named --members results/aespa_named/picks.json
```

- [ ] **Step 2: Label about 80 frames plus the dense windows.** Take the windows from `tracking_report.json`'s splits and close calls: crossings, someone passing in front of a member, the solo around 150 s, and the re-entries when the camera widens around 170 s. Use about six 2 s windows:

```bash
.venv/Scripts/python.exe -m syncscore audit tests/aespa-whiplash-dance-practice-mirrored-ytmp4.savetube.vip.mp4 --results results/aespa_named --frames 80 --times <w1> <w2> <w3> <w4> <w5> <w6>
```

Fill `true_name` for every row (an outfit tag or `dancer`), then run `syncscore audit-score results/aespa_named/audit/ids.csv`.

- [ ] **Step 3: Compare with the targets**
  - At least 0.95 `exact_accuracy`.
  - At most 0.02 `dancer_share`.
  - At most 1.0 s `longest_wrong_s` in the windows.
  - `tracked_share` reported as measured.
  - **If a target is missed:** use superpowers:systematic-debugging on the failing frames. Change identity settings only with the evidence written down, and never change a target. If it is still missed, report the numbers to the user rather than calling it done.

- [ ] **Step 4: See it in the viewer.**
  - Build it with `syncscore view ... --results results/aespa_named`.
  - Check in the browser that each name stays on the same outfit at 40 s, 90 s and 150 s, and when the camera widens at 170 s.
  - Take screenshots for the walkthrough (`docs/media/`). The skeleton views only, as the existing media are.

- [ ] **Step 5: Write the docs.**
  - `docs/walkthrough.md` gets "Stage G: following the picked members", with:
    - the problem on aespa
    - the gate numbers for both versions of the model and the margins chosen
    - the camera segment count
    - the ground-truth numbers against the targets
    - the per-member tracked share
    - the limits found
  - `README.md` gets:
    - naming members in the upload page
    - `analyze --pose-only` and `--members`
    - `audit --times`
    - the model (OpenCV Zoo YouTu ReID: source URL, Apache-2.0, size, downloaded on first use to `~/.cache/syncscore/`)
  - Run `py ~/.claude/skills/clean-user-facing-text/scripts/inspect_text.py --stylometry --json docs/walkthrough.md`, and the same for `README.md`. Rewrite only if `density_tier` is `high`.

- [ ] **Step 6: Run everything and commit**

Run: `.venv/Scripts/python.exe -m pytest -q` and `node --test "tests/js/*.test.mjs"`
Expected: all pass.

```bash
git add docs/walkthrough.md README.md docs/media/<new images> configs/default.yaml
git commit -m "docs: following the picked members on aespa, with the measured accuracy"
```
