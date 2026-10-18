# Tracking Under Occlusion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Track every member in as many frames as the footage allows, using only real detections, and measure the gain.

**Architecture:** After the existing tracklet-to-member assignment, two passes fill short member gaps: first from detections the assignment left unused, then from a second, more sensitive pose-model pass on just the still-empty frames. Both use one shared acceptance rule (near the position interpolated across the gap, similar size, not another member, not a duplicate of the occluder's skeleton). A coverage report measures frames with all members before and after.

**Tech Stack:** Python 3.12, NumPy, Ultralytics YOLO11-pose (already used), OpenCV, pytest.

**Spec:** `docs/superpowers/specs/2026-10-18-group-coaching-design.md`, part 1.

## Global Constraints

- Only real detections fill a gap; nothing is interpolated or extrapolated into keypoints.
- Gaps at most `tracking.fill_max_gap_s` = 2.0 s are filled; gaps touching the clip's start or end are never filled.
- One detection fills at most one member in a frame.
- Nothing stores pixels; caches hold derived data only (`.npz` in `results/`, git-ignored).
- Existing behaviour is unchanged when `tracking.fill_max_gap_s: 0` and `tracking.recover: false`.
- Code style: match `syncscore/pose/tracks.py` (module docstring explaining the why, dataclasses, NumPy, short comments only where a choice needs it).

## Review Focus

1. A recovered box behind the front dancer carries the front dancer's keypoints. Expected: rejected as a duplicate, member stays unseen (Task 2 test `test_duplicate_of_the_occluder_is_rejected`).
2. A mirror reflection or crew member stands near a missing member's predicted spot but at a different size. Expected: rejected on size (Task 2 test `test_wrong_size_is_rejected`).
3. Two members missing in the same frame near one candidate. Expected: the candidate fills only the nearer one (Task 3 test `test_one_detection_fills_one_member`).
4. A member missing at the very start or end of the clip. Expected: not filled (Task 3 test `test_edge_gaps_are_not_filled`).
5. The re-detection cache from another video or config reused by mistake. Expected: ignored and rebuilt when the frame list or settings differ (Task 4 test `test_recover_cache_is_rebuilt_when_settings_change`).

---

### Task 1: Coverage report

**Files:**
- Create: `syncscore/pose/coverage.py`
- Modify: `syncscore/pose/tracks.py` (`MemberTracks`, `assemble`)
- Test: `tests/test_coverage.py`

**Interfaces:**
- Produces: `MemberTracks.rows: np.ndarray` (M, T) int, the `Detections` row used per member per frame, -1 where unseen (set by `assemble`, in the same left-to-right member order).
- Produces: `coverage(det: Detections, tracks: MemberTracks, min_keypoint_conf: float) -> dict` with keys `all_members_share` (float), `members_seen_share` (list of float per member), `gaps` (dict `"<0.5s"`, `"0.5-2s"`, `">=2s"` → int, summed over members), `missing_detector_share` (float: of frames with fewer than M members, share where the detector found fewer than M clear people), `missing_unassigned_share` (float: share where it found M or more).

- [ ] **Step 1: Write the failing tests** in `tests/test_coverage.py`, reusing the `_detections` helper pattern from `tests/test_tracks.py`:
  - `test_assemble_records_rows`: 2 people, 20 frames each; `tracks.rows[i, f]` equals the row index of that person's detection and is -1 where absent.
  - `test_coverage_counts_frames_with_everyone`: 3 members over 100 frames, member 2 absent in frames 40-49 (10 frames) → `all_members_share == 0.9`, `gaps == {"<0.5s": 1, "0.5-2s": 0, ">=2s": 0}` at 30 fps.
  - `test_coverage_splits_the_cause`: same setup but in frames 40-44 an unassigned detection exists (a fourth track id of 5 frames, shorter than the 0.3 s tracklet minimum) → `missing_unassigned_share == 0.5`, `missing_detector_share == 0.5`.
- [ ] **Step 2: Run** `pytest tests/test_coverage.py -v`. Expected: FAIL (import error / no `rows`).
- [ ] **Step 3: Implement** `rows` in `assemble` (record the row chosen per member per frame, reorder with the members) and `coverage(...)` in `syncscore/pose/coverage.py`. "Clear people" uses the same rule as `count_members` (box conf ≥ 0.5 and ≥ 8 keypoints at `min_keypoint_conf`).
- [ ] **Step 4: Run** `pytest tests/test_coverage.py tests/test_tracks.py -v`. Expected: PASS.
- [ ] **Step 5: Commit** `feat(tracks): coverage report; record which detection each member uses`.

### Task 2: One acceptance rule for a gap-filling candidate

**Files:**
- Create: `syncscore/pose/fill.py`
- Test: `tests/test_fill.py`

**Interfaces:**
- Produces: `expected_box(box: np.ndarray, member: int, frame: int, max_gap: int) -> np.ndarray | None` — box (4,) linearly interpolated between the member's last seen box before `frame` and first seen box after it, when both exist and the gap containing `frame` is at most `max_gap` frames; `None` otherwise (including gaps touching the clip's start or end). `box` is `MemberTracks.box` (M, T, 4).
- Produces: `accept(candidate_box: np.ndarray, candidate_xy: np.ndarray, candidate_conf: np.ndarray, expected: np.ndarray, others_box: np.ndarray, others_xy: np.ndarray, others_conf: np.ndarray, min_keypoint_conf: float) -> float` — a cost (lower is better) or `np.inf` when rejected. `others_*` are the present members' boxes (K, 4), keypoints (K, 17, 2) and confidences (K, 17) in that frame.
- Constants in `fill.py`: `MAX_JUMP = 0.5` (centre distance / expected box height), `SIZE_RATIO = 1.3` (height within ×/÷ 1.3 of expected), `MAX_IOU_OTHER = 0.7`, `DUPLICATE_PX = 0.05` (median distance of shared confident keypoints / expected height, below which it duplicates another member).

- [ ] **Step 1: Write the failing tests** in `tests/test_fill.py`:
  - `test_expected_box_interpolates_inside_a_short_gap`: member seen at frames 9 (x-centre 100) and 20 (x-centre 210), max_gap 60 → `expected_box(..., frame=15, ...)` has x-centre 160.
  - `test_edge_gaps_have_no_expected_box`: member first seen at frame 10 → `expected_box(..., frame=5, ...)` is `None`; likewise after the last sighting.
  - `test_long_gap_has_no_expected_box`: gap of 61 frames with max_gap 60 → `None`.
  - `test_candidate_near_the_expected_spot_is_accepted`: finite cost.
  - `test_far_candidate_is_rejected`: centre 0.6 × height away → `inf`.
  - `test_wrong_size_is_rejected`: height 1.5 × expected → `inf`.
  - `test_candidate_that_is_another_member_is_rejected`: IoU 0.8 with a present member's box → `inf`.
  - `test_duplicate_of_the_occluder_is_rejected`: candidate box overlaps the occluder by IoU 0.4 (allowed) but its keypoints equal the occluder's within 2 px → `inf`.
- [ ] **Step 2: Run** `pytest tests/test_fill.py -v`. Expected: FAIL (module missing).
- [ ] **Step 3: Implement** `expected_box` and `accept` in `syncscore/pose/fill.py`. Cost = centre distance / expected height + 0.5 × |log(height ratio)|. Module docstring states the anti-goal: only real detections fill gaps.
- [ ] **Step 4: Run** `pytest tests/test_fill.py -v`. Expected: PASS.
- [ ] **Step 5: Commit** `feat(tracks): acceptance rule for filling a member's short gap`.

### Task 3: Fill gaps from unassigned detections

**Files:**
- Modify: `syncscore/pose/fill.py`
- Modify: `syncscore/config.py` (`TrackingConfig.fill_max_gap_s: float = 2.0`), `configs/default.yaml` (same key, commented)
- Test: `tests/test_fill.py`

**Interfaces:**
- Consumes: `MemberTracks.rows` (Task 1), `expected_box`, `accept` (Task 2).
- Produces: `fill_gaps(det: Detections, tracks: MemberTracks, max_gap_s: float, min_keypoint_conf: float) -> tuple[MemberTracks, int]` — a new `MemberTracks` with filled frames (xy, conf, box, hist, rows updated) and the number of member-frames filled. Candidates are detection rows in that frame not used by any member.

- [ ] **Step 1: Write the failing tests**:
  - `test_leftover_detection_fills_a_short_gap`: member 1's tracklet broken by 6 frames that exist only as a separate 6-frame track id (below the 0.3 s tracklet minimum) → after `fill_gaps`, member 1 is seen in all 6 frames, count == 6, and `rows` points at those detections.
  - `test_one_detection_fills_one_member`: members 0 and 1 both missing in frame 50, one leftover detection nearer member 0's expected spot → filled for member 0 only.
  - `test_edge_gaps_are_not_filled`: a leftover detection exists before the member's first sighting → not used.
  - `test_fill_off_changes_nothing`: `max_gap_s=0` → returned tracks equal the input arrays.
- [ ] **Step 2: Run** `pytest tests/test_fill.py -v`. Expected: the new tests FAIL.
- [ ] **Step 3: Implement** `fill_gaps`: per frame, build the cost matrix (members missing there × unused candidates) with `accept`, assign greedily cheapest first so each candidate fills at most one member. Add the config key.
- [ ] **Step 4: Run** `pytest tests/test_fill.py tests/test_config.py -v`. Expected: PASS.
- [ ] **Step 5: Commit** `feat(tracks): fill short member gaps from detections the assignment left unused`.

### Task 4: Re-detect frames the detector missed

**Files:**
- Create: `syncscore/pose/recover.py`
- Modify: `syncscore/config.py` (`TrackingConfig.recover: bool = True`, `recover_conf: float = 0.05`, `recover_imgsz: int = 1280`), `configs/default.yaml`
- Test: `tests/test_recover.py`

**Interfaces:**
- Consumes: `expected_box` (Task 2), `fill_gaps` (Task 3), `Detections` (existing).
- Produces: `frames_to_recover(tracks: MemberTracks, fps: float, max_gap_s: float) -> np.ndarray` — sorted unique frame indices where some member has an `expected_box` and is unseen.
- Produces: `redetect(video: str | Path, frames: np.ndarray, config: Config, cache: Path) -> Detections` — runs `YOLO(config.pose.model).predict` on exactly those frames with `conf=config.tracking.recover_conf`, `imgsz=config.tracking.recover_imgsz`, `track_id = -1`, torso histograms as in `extract`. Cache file stores the frame list, model, conf and imgsz; reused only if all four match.
- Produces: `merge(det: Detections, extra: Detections) -> Detections` — rows concatenated, sorted by frame.

- [ ] **Step 1: Write the failing tests** (no model needed):
  - `test_frames_to_recover_are_the_unseen_frames_inside_short_gaps`: member 0 unseen 40-49 and before frame 5 → returns 40..49 only.
  - `test_merge_keeps_every_row_sorted_by_frame`.
  - `test_recover_cache_is_rebuilt_when_settings_change`: write a cache with conf 0.05 via a helper `_cache_matches(cache, frames, config) -> bool`; it returns True for the same settings and False when `recover_conf`, the model name or the frame list differs.
- [ ] **Step 2: Run** `pytest tests/test_recover.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** the three functions and `_cache_matches`. Read frames with `cv2.VideoCapture` seeking to each index (frames are sparse); progress line like `extract`.
- [ ] **Step 4: Run** `pytest tests/test_recover.py -v`. Expected: PASS.
- [ ] **Step 5: Commit** `feat(tracks): re-detect only the frames where a member is missing inside a short gap`.

### Task 5: Wire into `analyze`, measure on the real clip, document

**Files:**
- Modify: `syncscore/cli.py` (`_analyze`)
- Modify: `docs/walkthrough.md` (new "Stage D: everyone tracked"), `README.md` (tracking numbers in Limitations and the method table's tracking row)

**Interfaces:**
- Consumes: `coverage` (Task 1), `fill_gaps` (Task 3), `frames_to_recover`, `redetect`, `merge` (Task 4).

- [ ] **Step 1: Wire it** in `_analyze` after `assemble`: coverage "before"; `fill_gaps`; if `config.tracking.recover`: `frames_to_recover` → `redetect` (cache `out / "recovered.npz"`, ignored with `--redo-pose`) → `merge` → `fill_gaps` again on the merged detections; coverage "after". `tracking_report.json` gains `"coverage": {"before": ..., "after": ..., "filled_from_leftovers": n1, "filled_from_redetection": n2}`; print one line: `everyone tracked in X% of frames (was Y%)`.
- [ ] **Step 2: Run** `pytest -q`. Expected: all tests pass (70 existing plus the new ones).
- [ ] **Step 3: Run on the real clip**: `python -m syncscore --config configs/bts_boy_with_luv.yaml analyze "<BTS video>" --out results/bts`. Expected: the new line prints, and `all_members_share` after is above the 46.5% baseline. Record before/after, both fill counts, and the gap table.
- [ ] **Step 4: Look**: render 10 recovered frames with the existing audit drawing (boxes and labels) into `results/bts/recovered_check/` and inspect them. Expected: each recovered box is the hidden member, not the occluder or a reflection. If any is wrong, tighten the constant that let it through and repeat Steps 3-4.
- [ ] **Step 5: Re-run the planted-shift test** (about 48 minutes): `python -m syncscore --config configs/bts_boy_with_luv.yaml shift-test --results results/bts --jobs 6`. Expected: section errors stay around 12 ms with 0% more than 30 ms wrong; record section coverage (22% before).
- [ ] **Step 6: Document** Stage D in the walkthrough (why, the two causes with the 73% / 27% split, the acceptance rule, the before/after numbers, what Step 4 showed) and update the README tracking numbers. Recommend the hand audit for identity accuracy.
- [ ] **Step 7: Commit** `feat(tracks): fill occlusion gaps with real detections; everyone tracked X% -> Y%` (after the user has seen the diff).
