# Overlay and Ghost Viewer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A local web page that plays the practice video with every member's tracked skeleton drawn on them and a faint "ghost" of what the rest of the group is doing at that moment, with a choreographer view, a dancer view, a side-by-side lineup and slow motion.

**Architecture:** Python computes everything the page draws (ghost poses, halos, section status) from the existing analysis outputs and writes it to one `viewer_data.js` file next to a static page (`index.html`, `viewer.css`, `viewer_core.js`, `viewer.js`) copied from the package. The page loads with no server (works from `file://`), references the source video by its local path and never copies it. Pure drawing logic lives in `viewer_core.js`, unit-tested with Node's built-in test runner; DOM code lives in `viewer.js`.

**Tech Stack:** Python 3.12 + NumPy (data), plain HTML/CSS/ES2020 JavaScript with `<canvas>` and `<video>` (no libraries, no build step), Node 24 `node --test` for the JS logic, Edge headless for reproducible skeleton-only screenshots.

**Spec:** `docs/superpowers/specs/2026-10-18-group-coaching-design.md`, part 2.

## Global Constraints

- No source footage is committed: viewer output goes under `results/` (git-ignored); docs images are rendered with the video switched off (skeletons only).
- The page needs no network, server, third-party script or font; it must open from `file://`.
- Only measured data is drawn: unseen frames show no skeleton and no ghost; nothing is interpolated for display.
- The ghost is labelled in the page as "ghost = what the other members are doing at this moment".
- Status colours and wording come from `syncscore/viz/style.py` and `skeleton_video.status`: green "in sync" (≤ `render.in_sync_ms`), amber "drifting", red "off" (≥ `render.off_sync_ms`), grey "no reading"; colour is always paired with a text label.
- Pink halo on a bone whose pose error is more than `sync.shape_threshold_deg` beyond the group's typical error for that bone at that frame (same rule as the skeleton video).
- Sign convention shown on the page: "+ ms = behind the group, − = ahead".

## Decisions taken in this plan (arguing from the spec)

- **Medium: a local web page** (the spec left desktop window vs. local web page to this plan). It opens in any browser with nothing to install, `<video>` gives frame-accurate seeking and `playbackRate` for slow motion, and the same page serves both views.
- **Ghost = leave-one-out median of the other members' bone directions at the same video frame**, drawn from the member's own pelvis with the member's own bone lengths. The spec says "as `group_median`"; leave-one-out is used instead because a member must not drag their own ghost toward themselves (the reason `consensus.py` gives for timing), and "at the same video frame" means the ghost shows what the others are doing *now*, with no timing estimate in between, so a late member visibly trails it.
- **URL parameters** set the initial state (`t`, `view`, `member`, `video`, `ghost`, `lineup`, `speed`), so a moment can be shared as a link and docs screenshots are reproducible.

## Review Focus

1. The video was moved or deleted after `syncscore view`. Expected: the page still opens in skeleton-only mode with a visible note "video not found: <path>" (Task 4, browser check step `video missing`).
2. The video's duration or frame rate differs slightly from the analysis (e.g. a re-encoded copy). Expected: frame index clamps to the analysed range, no blank canvas or exception (Task 3 test `frameAt clamps to the analysed range`).
3. The followed member is unseen in the current frame in the dancer view. Expected: the crop holds the last known position and the label reads "not seen" (Task 3 test `followBox holds the last seen box`).
4. A section without a reliable reading. Expected: grey with the text "no reading", never a number (Task 3 test `statusOf never shows a number without a reading`).
5. A full 7-member, 3:50 clip. Expected: `viewer_data.js` stays under 15 MB and the page is interactive within a few seconds (Task 2 test `test_data_size_for_a_full_clip`).

---

### Task 1: Ghost poses

**Files:**
- Create: `syncscore/viz/ghost.py`
- Test: `tests/test_ghost.py`

**Interfaces:**
- Consumes: `bone_directions`, `mirror_keypoints`, `BONES` (`syncscore/features/bones.py`); `leave_one_out_median` (`syncscore/sync/consensus.py`).
- Produces: `pose_from_directions(u: np.ndarray, w: np.ndarray, xy: np.ndarray, conf: np.ndarray, min_conf: float) -> np.ndarray` — u (T, B, 2) unit directions, w (T, B); xy (T, 17, 2) / conf (T, 17) of the member whose body the pose is drawn on; returns (T, 17, 2) with joints 0 and 5-16 placed by chaining bones from the member's pelvis (hips and shoulders centred on pelvis and neck), lengths = the member's own length in that frame when both joints are confident, else their median length over the clip; joints 1-4 and any joint whose chain is unknown are NaN.
- Produces: `ghost_keypoints(xy: np.ndarray, conf: np.ndarray, mirrored: np.ndarray, min_conf: float) -> np.ndarray` — xy (M, T, 17, 2), conf (M, T, 17), mirrored (M, T) bool; returns (M, T, 17, 2). For each member: the leave-one-out median (min_members=2) of the others' directions, all expressed in the group's convention (a mirrored member's directions come from `mirror_keypoints`); drawn on a mirrored member by building the pose on their mirrored keypoints and mirroring the result back. NaN wherever the member is unseen (pelvis unknown).
- Produces: `best_lag(member_u: np.ndarray, ghost_u: np.ndarray, max_lag: int) -> int` — the shift in frames (−max_lag..max_lag) at which the member's directions best match the ghost's (highest mean cosine over bones and frames known in both); + = member behind.

- [ ] **Step 1: Write the failing tests** in `tests/test_ghost.py`:
  - `test_own_directions_rebuild_the_members_own_pose`: one synthetic member (`make_group`, 1 of 3 members, no noise); `pose_from_directions` with that member's own `bone_directions` reproduces their joints 0 and 5-16 within 1e-6 px.
  - `test_a_late_member_trails_the_ghost`: `make_group(GroupSpec(n_members=5, offsets_ms=[0, 0, 133, 0, 0], seed=51))` at 30 fps; `best_lag` between member 2's directions and the directions of `ghost_keypoints(...)[2]` is +4 frames; for member 0 it is 0.
  - `test_unseen_member_has_no_ghost`: occlusion of member 1 for 3-5 s → ghost of member 1 is all NaN in those frames, finite elsewhere.
  - `test_a_mirrored_member_gets_a_mirrored_ghost`: 4 members, member 3 `mirrored=True`, no offsets; with `mirrored[3] = True` everywhere the ghost of member 3 matches their own joints 5-16 within 2 px (median); with `mirrored` all False it is off by more than 20 px.
- [ ] **Step 2: Run** `pytest tests/test_ghost.py -v`. Expected: FAIL (module missing).
- [ ] **Step 3: Implement** the three functions in `syncscore/viz/ghost.py` (module docstring: what the ghost is and why leave-one-out at the same frame).
- [ ] **Step 4: Run** `pytest tests/test_ghost.py -v`. Expected: PASS.
- [ ] **Step 5: Commit** `feat(viewer): ghost poses, the other members' pose drawn on each member's body`.

### Task 2: Viewer data file

**Files:**
- Create: `syncscore/viz/viewer_data.py`
- Test: `tests/test_viewer_data.py`

**Interfaces:**
- Consumes: `ghost_keypoints` (Task 1); `make_sections` (`sync/summary.py`); `section_timing` (`sync/sections.py`); outputs written by `write_outputs`: `keypoints_smoothed.npz`, `sync_result.npz`, `summary.json`.
- Produces: `encode_int16(values: np.ndarray, scale: float = 4.0) -> str` and `decode_int16(text: str, shape: tuple, scale: float = 4.0) -> np.ndarray` — base64 of little-endian int16 of `round(values * scale)`, NaN ↔ -32768.
- Produces: `viewer_data(result_dir: Path, config: Config, image_size: tuple[int, int], video: str | None) -> dict` with keys: `version` (1), `fps`, `n_frames`, `image_size`, `names`, `in_sync_ms`, `off_sync_ms`, `shape_threshold_deg`, `video` (string or None), `sections` (list of `{start_s, end_s, name}`), `section_status` (per member, per section: `{offset_ms, low_ms, high_ms, reliable}`), `keypoints` and `ghost` (encoded (M, T, 17, 2)), `conf` (base64 uint8 of round(conf × 255), (M, T, 17)), `halo` (base64 uint16 bitmask (M, T), bit b set when bone b is more than the threshold beyond the group, mapped to the member's own limbs when mirrored), `moments` (per member: `worst_timing_moments` and `worst_shape_moments` from `summary.json` "overall").
- Produces: `video_url(path: Path) -> str` — absolute `file:///` URL, forward slashes, percent-encoded.
- Produces: `write_viewer(result_dir: Path, out_dir: Path, config: Config, image_size: tuple[int, int], video: Path | None) -> Path` — writes `out_dir/viewer_data.js` as `window.SYNCSCORE = <json>;` (with `video` = `video_url(video)` or None), copies the static files from `syncscore/viz/viewer/` (none exist until Task 3/4; copy whatever is there), returns the path of `index.html`.

- [ ] **Step 1: Write the failing tests** in `tests/test_viewer_data.py` (fixture: run `analyze_keypoints` + `write_outputs` on a 3-member, 16 s synthetic group into tmp, as `tests/test_outputs.py` does):
  - `test_int16_round_trip_keeps_quarter_pixels_and_nan`: values with NaN round-trip within 0.125 px and NaN stays NaN.
  - `test_viewer_data_has_every_part_the_page_reads`: the keys above exist; `decode_int16(data["keypoints"], (3, T, 17, 2))` equals `keypoints_smoothed.npz` within 0.125 px.
  - `test_halo_marks_the_planted_limb_error`: with `LimbError(member=1, joint="r_knee", degrees=30, t0=2, t1=14)` the right-shin bit is set for member 1 in most frames 3-13 s and in under 5% of member 0's frames.
  - `test_no_video_gives_null` and `test_video_path_becomes_a_file_url` (`video_url` of a path with spaces and Korean characters → `file:///` URL that `urllib.parse.unquote` turns back into the path).
  - `test_write_viewer_writes_the_data_script`: `viewer_data.js` starts with `window.SYNCSCORE = ` and the rest parses as JSON.
  - `test_data_size_for_a_full_clip`: arrays shaped (7, 6905, 17, 2) of random pixels encode to under 15 MB in total for keypoints + ghost + conf + halo.
- [ ] **Step 2: Run** `pytest tests/test_viewer_data.py -v`. Expected: FAIL (module missing).
- [ ] **Step 3: Implement** `syncscore/viz/viewer_data.py`. Mirrored flags per frame come from `sync_result.npz` `mirrored` (M, K) by nearest window centre, as `skeleton_video` does.
- [ ] **Step 4: Run** `pytest tests/test_viewer_data.py -v`. Expected: PASS.
- [ ] **Step 5: Commit** `feat(viewer): one data file with skeletons, ghosts, halos and section status`.

### Task 3: Viewer logic in JavaScript, unit-tested

**Files:**
- Create: `syncscore/viz/viewer/viewer_core.js` (works as a browser global `SyncCore` and as an ES module for Node via a small export shim)
- Create: `tests/js/viewer_core.test.mjs`
- Create: `tests/test_viewer_js.py` (runs `node --test tests/js`; skipped when `node` is not on PATH)

**Interfaces:**
- Produces (all pure, no DOM):
  - `decodeInt16(b64, scale) -> Float32Array` (NaN for -32768), `decodeUint8(b64) -> Uint8Array`, `decodeUint16(b64) -> Uint16Array`.
  - `frameAt(timeS, fps, nFrames) -> int` — `floor(timeS * fps + 1e-3)` clamped to `0..nFrames-1`.
  - `sectionAt(timeS, sections) -> int` — index of the section containing the time (last one at the end).
  - `statusOf(offsetMs, reliable, inSyncMs, offSyncMs) -> {color, label, text}` — `text` is "+12 ms in sync" style or "-- no reading"; colours `#0ca30c`, `#fab219`, `#d03b3b`, `#898781`.
  - `skeletonSegments(points, conf, minConf, haloMask) -> [{a, b, halo}]` — the 13 scored bones plus the two torso sides, with neck/pelvis virtual points, only where both ends are known.
  - `followBox(keypoints, member, frame, lastBox, pad) -> {x, y, w, h, seen}` — padded box around the member's confident joints; when unseen returns `lastBox` with `seen: false`.
  - `lineupTransform(points, conf, minConf, slotX, floorY, targetHeight) -> {scale, dx, dy}` — scales the member to `targetHeight` (their head-to-ankle extent) and puts their ankle midpoint at (`slotX`, `floorY`).

- [ ] **Step 1: Write the failing tests** in `tests/js/viewer_core.test.mjs` with `node:test` and `node:assert`:
  - `decodeInt16 round-trips quarter pixels and NaN` (base64 produced by the same encoding as Task 2; include one fixture string generated by Python in the test file).
  - `frameAt clamps to the analysed range` (negative time → 0, time past the end → nFrames-1, 1.0 s at 30 fps → 30).
  - `statusOf never shows a number without a reading` (reliable false → "-- no reading", grey).
  - `statusOf uses the in-sync and off thresholds` (35 → in sync, 36 → drifting, 135 → off; −40 → "-40 ms drifting").
  - `skeletonSegments skips bones with an unknown end`.
  - `followBox holds the last seen box` (unseen frame returns the previous box with `seen: false`).
  - `lineupTransform puts the feet on the floor line`.
  - `tests/test_viewer_js.py::test_viewer_core_js` runs `node --test tests/js` and asserts exit code 0.
- [ ] **Step 2: Run** `pytest tests/test_viewer_js.py -v`. Expected: FAIL (module file missing).
- [ ] **Step 3: Implement** `viewer_core.js`.
- [ ] **Step 4: Run** `pytest tests/test_viewer_js.py -v`. Expected: PASS.
- [ ] **Step 5: Commit** `feat(viewer): drawing logic for the viewer, unit-tested in Node`.

### Task 4: The page

**Files:**
- Create: `syncscore/viz/viewer/index.html`, `syncscore/viz/viewer/viewer.css`, `syncscore/viz/viewer/viewer.js`
- Modify: `pyproject.toml` (`[tool.setuptools.package-data] syncscore = ["viz/viewer/*"]`)
- Test: `tests/test_viewer_data.py` (contract test)

**Interfaces:**
- Consumes: `window.SYNCSCORE` (Task 2 keys), `SyncCore` (Task 3).
- Page behaviour (one screen, dark surface `#1a1a19` for the stage, light panels):
  - **Stage:** the video (when present and switched on) with a `<canvas>` overlay of the same size; with the video off or missing, the canvas alone on the dark surface. Each member: skeleton in their section-status colour, pink halos, name and status text above the head; ghost as a thin white line at 45% opacity. Drawn on `requestVideoFrameCallback` when available (exact presented frame), else `requestAnimationFrame`; without video an internal clock drives time.
  - **Controls:** play/pause, step ±1 frame, speed 1× / ½× / ¼×, "loop ±1 s" toggle, video on/off, ghost on/off, lineup on/off, view switch (Group / Dancer + member picker). Keyboard: space, ←/→.
  - **Group view:** below the stage, a timeline: one row per member, sections coloured by status with the status text in a tooltip; a cursor at the current time; click seeks.
  - **Dancer view:** the stage crops to `followBox` for the chosen member (scaled up), only that member and their ghost drawn; a side list of their worst timing and worst shape moments (time, value, part/bone, "likely label swap" when flagged); clicking a moment seeks to 1 s before it, sets ¼× and turns the loop on.
  - **Lineup:** a strip under the stage with every member side by side via `lineupTransform`, feet on one floor line, each with their ghost.
  - Legend with the status colours, the halo meaning, the ghost meaning and the sign convention. Missing video: note "video not found: <path>" and skeleton-only mode.
  - URL parameters `t`, `view`, `member`, `video`, `ghost`, `lineup`, `speed` set the initial state.

- [ ] **Step 1: Write the failing contract test** `test_page_reads_only_keys_the_data_has` in `tests/test_viewer_data.py`: every `D.<name>` read in `viewer.js` (regex `\bD\.(\w+)`) is a key of `viewer_data(...)`, and `index.html` loads exactly `viewer_data.js`, `viewer_core.js`, `viewer.js` and `viewer.css`, all present after `write_viewer`.
- [ ] **Step 2: Run** it. Expected: FAIL (page files missing).
- [ ] **Step 3: Implement** the three files and the package-data line.
- [ ] **Step 4: Run** `pytest tests/test_viewer_data.py -v`. Expected: PASS.
- [ ] **Step 5: Browser check** on the synthetic demo viewer (Task 5 Step 3 produces it; if not yet, write it with `write_viewer` from a demo results folder): serve `results/` on localhost, open it in the built-in browser, read console errors (expected: none), and look at group view, dancer view, lineup, ¼× playback and the `video missing` case (point `D.video` at a missing file). Expected: each behaves as described above.
- [ ] **Step 6: Commit** `feat(viewer): the page: group and dancer views, ghost, lineup, slow motion`.

### Task 5: Commands

**Files:**
- Modify: `syncscore/cli.py` (new `view` command; `demo` also writes `viewer/`)
- Test: `tests/test_viewer_data.py`

**Interfaces:**
- Consumes: `write_viewer` (Task 2), `probe` (`pose/video.py`).
- Produces: `syncscore view VIDEO --results DIR [--out DIR/viewer]` — probes the video for its size, writes the viewer, prints `open <path to index.html>` and the reminder that the folder references the source footage and stays out of the repository. `syncscore demo` writes `<out>/viewer/` with no video.

- [ ] **Step 1: Write the failing test** `test_view_command_writes_a_viewer`: write a 2 s, 64×48 synthetic mp4 with `cv2.VideoWriter` into tmp, reuse the Task 2 fixture's results folder, run `main(["view", str(video), "--results", str(results)])`, assert exit code 0, `results/viewer/index.html` exists and `viewer_data.js` contains the video's `file:///` URL.
- [ ] **Step 2: Run** it. Expected: FAIL (unknown command).
- [ ] **Step 3: Implement** `_view` and the `demo` addition.
- [ ] **Step 4: Run** `pytest -q`. Expected: all pass.
- [ ] **Step 5: Commit** `feat(viewer): syncscore view, and the demo writes a viewer`.

### Task 6: Proof on real footage, docs

**Files:**
- Create: `docs/media/viewer_group.png`, `docs/media/viewer_dancer.png`, `docs/media/viewer_ghost_original.png`, `docs/media/viewer_ghost_planted.png` (skeleton-only screenshots)
- Modify: `docs/walkthrough.md` ("Stage E: seeing it"), `README.md` (a "See it" section: how to open the viewer, one screenshot)

- [ ] **Step 1: Viewer for the real clip:** `python -m syncscore --config configs/bts_boy_with_luv.yaml view "<BTS video>" --results results/bts`. Open it in the built-in browser (served from localhost) and check the two views and the lineup on the real footage. Expected: skeletons sit on the dancers, ghosts close to their own skeletons (a professional group), no console errors.
- [ ] **Step 2: Planted delay:** delay one real member (M3, index 2) by 133 ms with `delay_member` on `keypoints_raw.npz`, run `analyze_keypoints` + `write_outputs` into `results/bts_planted/`, and write its viewer. Measure `best_lag` between M3's directions and their ghost over the clip in both runs. Expected: about 0 frames before, about +4 frames after.
- [ ] **Step 3: Screenshots** with Edge headless (`msedge --headless=new --screenshot=<png> --window-size=1400,900 --virtual-time-budget=4000 "<file URL>?t=<s>&video=0&..."`): the synthetic demo in group view and dancer view (M5, during its planted elbow error), and a pair from the BTS clip in dancer view for M3 at the same hit, video off: the original run and the planted run. Look at each. Expected: in the original M3's skeleton sits close to the ghost; in the planted one it visibly trails it.
- [ ] **Step 4: Document** Stage E (what the ghost is and why leave-one-out at the same frame, the views, the planted-delay result with `best_lag` numbers and the screenshot pair, the footage rule) and the README section. Run `pytest -q` (all pass) and check that no footage, frames or weights are on the branch.
- [ ] **Step 5: Commit** `docs: Stage E, the viewer; skeleton-only screenshots`.
