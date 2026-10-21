# The Beat and Dancing Together Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Use the group as a reference only while the group is actually dancing together (no ghost and no group timing during solos or split parts), and bring the music's beat into the viewer: a beat lane, a beat pulse, a sound toggle and each dancer's clean landings marked against the beat.

**Architecture:** A new `sync/unison.py` decides, per member and frame, whether the other members agree on their pose and the member is in the same moving/still state as them; `measure_sync` stores it and drops group-timing windows where it fails, and the viewer hides the ghost there. A new `audio/` package decodes the soundtrack with PyAV, builds an onset curve, tracks beats and fits a steady half-beat grid; `sync/landings.py` finds clean landings (big moves into stillness). `syncscore view` adds both to the viewer data; the page draws a beat lane under the stage.

**Tech Stack:** Python 3.12, NumPy/SciPy, PyAV (`av`, optional: without it, beat features are skipped with a note), plain JS (no libraries), Node `node --test`.

**Spec:** `docs/superpowers/specs/2026-10-18-group-coaching-design.md`, part 2b.

## Global Constraints

- "Dancing together" for member i at frame t: the other members' limb directions agree (median limb angle to their median ≤ `unison.max_spread_deg` = 25, over 1 s), member i's own limbs are within `unison.max_own_deg` = 40 of the others' median (over 2 s), and i is in the same state as the others (both moving or both still; moving = speed above the member's typical, a state = moving in at least half of the 1 s window; the others' state = at least half of them moving). At least 2 other members seen.
- Group timing never uses a window in which member i is together for less than half of its frames.
- The ghost is drawn only where member i is together.
- Beat: on/off-beat ticks come from a steady grid only (`steady` = at least 70% of tracked beats within 40 ms of the half-beat grid); otherwise no lane, and the page says the tempo wasn't steady enough.
- Landings: speed peak ≥ `beat.landing_peak` = 2.5 × the member's typical, falling to ≤ `beat.landing_hold` = 0.5 within `beat.landing_within_s` = 0.2 s; the landing time is where speed falls through half of (peak − hold); measured joints only.
- Landing offsets are to the nearest half beat (± 125 ms); "on the beat" = |offset| ≤ `render.in_sync_ms` (35).
- The page states: the beat comes from the video's soundtrack, a delay between sound and picture shifts every dancer equally, and a deliberately syncopated landing reads as off the beat.
- No per-section beat score anywhere (the evidence on real footage doesn't support one).
- Nothing pixel-based or audio is stored; only the grid numbers and landing times go in the viewer data.

## Review Focus

1. A video without a soundtrack, or PyAV not installed. Expected: `view` still writes the viewer, no beat lane, and the page says why (Task 7 test `test_no_audio_means_no_beat_data`).
2. Music with a tempo change or silence in the middle. Expected: `steady` false, no grid drawn, the page says the tempo wasn't steady (Task 5 test `test_a_tempo_change_is_not_steady`).
3. The whole group holds still (an intro or ending pose). Expected: together (all still, poses agree), so the ghost stays (Task 2 test `test_a_group_holding_still_is_together`).
4. A member is unseen while the others dance. Expected: not together for that member, nothing drawn (Task 2 test `test_unseen_member_is_not_together`).
5. A dancer lands more than 125 ms off. Expected: reads as the neighbouring half beat; the page's limits note says so (Task 6 test `test_offsets_wrap_at_half_a_half_beat` pins the wrap).

---

### Task 1: Synthetic solos and split parts

**Files:**
- Modify: `syncscore/synth/generate.py` (`GroupSpec.solos: list = []` of `Interval`; `GroupSpec.parts: list = []` of `(members: list[int], t0: float, t1: float)`)
- Test: `tests/test_generate.py`

**Interfaces:**
- Produces: in a solo `Interval(member, t0, t1)` every *other* member holds the pose they had at t0; the soloist keeps dancing. In a part `([j, k], t0, t1)` the listed members dance a second choreography (one shared by them, drawn from `np.random.default_rng(spec.seed + 1000)` so existing seeds are unchanged).

- [ ] **Step 1: Write the failing tests:** `test_in_a_solo_the_others_hold_still` (5 members, `solos=[Interval(2, 8, 14)]`: keypoints of members 0, 1, 3, 4 change by under 0.5 px between frames inside 8.2-13.8 s; member 2's change by more than 5 px somewhere), `test_a_part_dances_other_moves` (`parts=[([3, 4], 8, 14)]`: members 3 and 4 agree with each other and differ from member 0 by more than 10° median limb angle inside 8.5-13.5 s), `test_existing_seeds_are_unchanged` (a spec without solos/parts gives identical keypoints to before: compare against `make_group` output saved in the test as a few checksums computed before the change).
- [ ] **Step 2: Run** `pytest tests/test_generate.py -v`. Expected: FAIL (unknown fields).
- [ ] **Step 3: Implement** the two fields in `make_group`.
- [ ] **Step 4: Run** `pytest tests/test_generate.py -v`. Expected: PASS.
- [ ] **Step 5: Commit** `feat(synth): solos and split parts with known timing`.

### Task 2: Dancing together

**Files:**
- Create: `syncscore/sync/unison.py`
- Modify: `syncscore/config.py` (`UnisonConfig`: `max_spread_deg 25.0`, `max_own_deg 40.0`, `moving_speed 1.0`, `min_moving_share 0.5`, `smooth_s 1.0`, `own_smooth_s 2.0`, `min_others 2`), `configs/default.yaml`
- Test: `tests/test_unison.py`

**Interfaces:**
- Consumes: `motion` (`sync/hits.py`) for normalised speed; `bone_directions` and `velocity`.
- Produces: `together(u: np.ndarray, w: np.ndarray, speed: np.ndarray, fps: float, cfg) -> np.ndarray` — u (M, T, B, 2), w (M, T, B), speed (M, T) normalised (NaN unseen); returns (M, T) bool per the Global Constraints. Limb bones only (upper arms, forearms, thighs, shins).
- Produces: `apart_intervals(together: np.ndarray, fps: float, min_s: float = 1.0) -> list[list[tuple[float, float]]]` — per member, stretches of at least `min_s` where they were seen but not together.

- [ ] **Step 1: Write the failing tests** (velocities and speed computed from synthetic keypoints the way `measure_sync` does): `test_a_group_in_unison_is_together` (≥ 95% of frames), `test_a_group_holding_still_is_together` (a 6 s hold for everyone: all together), `test_a_soloist_is_not_together_during_the_solo` (soloist false in ≥ 90% of 9-13 s, true in ≥ 90% outside 7-15 s), `test_split_parts_are_not_together` (7 members, `parts=[([4, 5, 6], 8, 14)]`: every member false in ≥ 80% of 9-13 s), `test_unseen_member_is_not_together`, `test_apart_intervals_merge_runs` (a bool row with a 2 s gap gives one interval of 2 s; a 0.5 s gap gives none).
- [ ] **Step 2: Run** `pytest tests/test_unison.py -v`. Expected: FAIL (module missing).
- [ ] **Step 3: Implement** `unison.py` and the config. Smoothing is a moving average over known samples (as `hits._smooth_known`).
- [ ] **Step 4: Run** `pytest tests/test_unison.py tests/test_config.py -v`. Expected: PASS.
- [ ] **Step 5: Commit** `feat(sync): know when the group is dancing together`.

### Task 3: No group timing when not together

**Files:**
- Modify: `syncscore/sync/measure.py` (`SyncResult.together: np.ndarray` (M, T) bool; computed after the alignment loop from each member's chosen-orientation directions and velocities; windows and part-windows where the member is together for less than half of the frames become not reliable)
- Test: `tests/test_measure.py`

**Interfaces:**
- Consumes: `together` (Task 2). Produces: `SyncResult.together`.

- [ ] **Step 1: Write the failing tests:** `test_no_timing_while_a_member_solos` (5 members, `solos=[Interval(2, 10, 18)]`, 30 s: no reliable window of member 2 has its centre inside 11-17 s; reliable coverage outside 9-19 s stays above 0.7), and `test_unison_results_are_unchanged` (`test_known_constant_offsets_are_recovered`'s group gives the same offsets, and `result.together.mean() > 0.95`).
- [ ] **Step 2: Run** them. Expected: FAIL (no `together`).
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `pytest -q`. Expected: all pass.
- [ ] **Step 5: Commit** `feat(sync): group timing only while the group dances together`.

### Task 4: Ghost only when together

**Files:**
- Modify: `syncscore/viz/viewer_data.py` (data keys `together`: base64 uint8 (M, T); `apart`: `apart_intervals` per member), `syncscore/viz/viewer/viewer.js`, `viewer.css`, `index.html`
- Test: `tests/test_viewer_data.py`

**Interfaces:**
- Consumes: `SyncResult.together` via `sync_result.npz`; `apart_intervals` (Task 2).
- Page: the ghost of member i at frame f is drawn only if `together[i, f]`; dancer view shows "no group reference here (solo or different parts)" when seen but not together; "Right now" adds "(solo or different part)" after the status; each timeline row gets a thin bar under it over the `apart` stretches, with a legend entry.

- [ ] **Step 1: Write the failing test** `test_viewer_marks_where_the_group_isnt_together` (a 5-member synthetic solo analysed into a tmp results folder: the soloist's `together` is 0 for most of the solo, `apart` holds one interval covering it within 1 s).
- [ ] **Step 2: Run** it. Expected: FAIL.
- [ ] **Step 3: Implement** the data keys and the page changes.
- [ ] **Step 4: Run** `pytest tests/test_viewer_data.py tests/test_viewer_js.py -v`. Expected: PASS (contract test included).
- [ ] **Step 5: Commit** `feat(viewer): ghost only while the group dances together`.

### Task 5: The beat grid from the soundtrack

**Files:**
- Create: `syncscore/audio/__init__.py`, `syncscore/audio/beats.py`
- Modify: `pyproject.toml` (optional extra `audio = ["av>=12"]`), `requirements.txt` (`av>=12`)
- Test: `tests/test_beats.py`

**Interfaces:**
- Produces: `load_audio(path: Path, sr: int = 22050) -> np.ndarray | None` (mono float32; None when the file has no audio stream or PyAV is missing).
- Produces: `onset_envelope(x: np.ndarray, sr: int, hop: int = 256) -> tuple[np.ndarray, np.ndarray]` (log-magnitude spectral flux, local mean removed, half-wave rectified; times in s).
- Produces: `@dataclass BeatGrid(tempo_bpm: float, half_beat_s: float, beat_phase_s: float, steady: bool, beat_share: float)` with `offset_ms(times_s: np.ndarray) -> np.ndarray` (signed ms to the nearest half beat, in ±half_beat/2) and `beats_between(t0: float, t1: float) -> list[tuple[float, bool]]` (time, is on-beat).
- Produces: `beat_grid(x: np.ndarray, sr: int) -> BeatGrid | None` — tempo by autocorrelation of the envelope (60-200 BPM, log-normal prior around 120), beats by dynamic programming, the period refined by maximising the tracked beats' concentration on a half-beat grid (±1%), phase from their circular mean, on-beat vs off-beat chosen by the larger mean envelope at the two candidate phases. None when the audio is shorter than 10 s.

- [ ] **Step 1: Write the failing tests** with generated click tracks (60 s; kick = 60 Hz burst on beats, hat = noise burst at half the level on off-beats): `test_tempo_and_phase_of_a_click_track` (120 BPM, first kick at 0.31 s: tempo within 0.3 BPM, on-beat phase within 10 ms of 0.31 mod 0.5, steady), `test_the_beat_not_the_off_beat` (on-beat phase is the kicks', not the hats'), `test_another_tempo` (96 BPM), `test_a_tempo_change_is_not_steady` (30 s at 120 then 30 s at 100: `steady` false), `test_silence_gives_no_grid` (None or not steady), `test_load_audio_reads_an_encoded_file` (PyAV writes a 2 s AAC .mp4 into tmp; `load_audio` returns about 2 s; skipped without PyAV).
- [ ] **Step 2: Run** `pytest tests/test_beats.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** `beats.py`: STFT window 1024, hop 256 at 22 050 Hz; envelope = positive part of the frame-to-frame difference of `log1p(100·|Z|)` summed over frequency, minus its 40-frame moving average, floored at 0; tempo prior `exp(-0.5·(log2(bpm/120))²)`; Ellis dynamic-programming beats with transition cost `-100·log(gap/period)²` over gaps of 0.5-2 periods.
- [ ] **Step 4: Run** `pytest tests/test_beats.py -v`. Expected: PASS.
- [ ] **Step 5: Commit** `feat(audio): the beat grid from the soundtrack`.

### Task 6: Clean landings

**Files:**
- Create: `syncscore/sync/landings.py`
- Modify: `syncscore/config.py` (`BeatConfig`: `landing_peak 2.5`, `landing_hold 0.5`, `landing_within_s 0.2`), `configs/default.yaml`
- Test: `tests/test_landings.py`

**Interfaces:**
- Produces: `clean_landings(speed: np.ndarray, fps: float, cfg) -> np.ndarray` — fractional frames per the Global Constraints.

- [ ] **Step 1: Write the failing tests:** `test_moves_into_a_hold_are_found` (synthetic member, 30 s at 120 BPM: at least 20 landings, each within 2 frames of a frame where the choreography's move ends), `test_a_planted_delay_moves_every_landing` (delay a member 40 ms with `delay_member`: matched landings move by 40 ± 3 ms), `test_continuous_motion_has_no_landings` (a sine-wave speed that never drops below 1: none), `test_offsets_wrap_at_half_a_half_beat` (`BeatGrid` at 120 BPM: a landing 130 ms after a half beat reads -120 ms).
- [ ] **Step 2: Run** `pytest tests/test_landings.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `pytest tests/test_landings.py -v`. Expected: PASS.
- [ ] **Step 5: Commit** `feat(sync): clean landings of big moves`.

### Task 7: The beat in the viewer

**Files:**
- Modify: `syncscore/viz/viewer_data.py` (`viewer_data(..., video)` computes the grid from the video when given: data keys `beats` = `{tempo_bpm, half_beat_s, beat_phase_s, steady}` or `null`, `beat_note` (why not, when null), `landings` = per member `[{time_s, offset_ms}]` or `null`), `syncscore/viz/viewer/viewer_core.js` (`beatTicks(t0, t1, beats)`, `pulseAt(t, beats, decayS)`, `landingStatus(offsetMs, inSyncMs)`), `viewer.js`, `index.html`, `viewer.css`
- Test: `tests/test_viewer_data.py`, `tests/js/viewer_core.test.mjs`

**Interfaces:**
- Consumes: `load_audio`, `beat_grid` (Task 5), `clean_landings`, `BeatGrid.offset_ms` (Task 6).
- Page: a **beat lane** under the controls (when `beats` is set and steady): 3 s around the current time, on-beat ticks strong and off-beat ("and") ticks light, a centre cursor, one row per member in group view (one, larger, in dancer view) with landing dots coloured by `landingStatus` and their offset as text on hover; a **beat pulse** dot at the stage's top right that flashes on each on-beat; a **Sound** toggle (unmutes the video; disabled without video); in dancer view, "Clean landings against the beat": count, median offset when there are at least 5, and the list (click: replay at ¼ speed in a 2 s loop). The legend gains the limits sentence from the Global Constraints. No beat data: the lane is replaced by the `beat_note`.

- [ ] **Step 1: Write the failing tests:** JS: `beatTicks lists on- and off-beats in a window`, `pulseAt peaks on the beat and fades`, `landingStatus uses the in-sync band`. Python: `test_no_audio_means_no_beat_data` (`viewer_data` with `video=None` and with a silent mp4: `beats` null and a `beat_note`), `test_beat_data_for_a_click_track_video` (PyAV writes a 20 s mp4 with a 120 BPM click track and a blank picture; with the Task 4 fixture's results: `beats.tempo_bpm` within 0.5 of 120 and `landings` has one list per member), and the existing contract test.
- [ ] **Step 2: Run** them. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `pytest -q`. Expected: all pass.
- [ ] **Step 5: Browser check** (built-in browser over localhost and headless Edge): the lane scrolls with time, ticks line up with the pulse, sound toggles, landing dots are clickable, no console errors; the demo (no audio) shows the beat note.
- [ ] **Step 6: Commit** `feat(viewer): the beat lane, a beat pulse, sound, and landings against the beat`.

### Task 8: Real clip, proof, docs

**Files:**
- Modify: `docs/walkthrough.md` ("Stage F: the beat, and dancing together"), `README.md` ("See it", Limitations, Running it), `docs/media/` (skeleton-only screenshots: the beat lane, a solo stretch without ghost)

- [ ] **Step 1:** Re-run `analyze` on the BTS clip (tracking cached) and `view`. Record: tempo and `steady`, landings per member and the share within 35 ms, the share of each member's frames not together, and the `apart` stretches.
- [ ] **Step 2:** Look at the two longest `apart` stretches in the viewer (video off) and at 175-185 s. Expected: a solo, a held formation with different poses, or split parts; if a stretch is ordinary unison dancing, adjust `max_spread_deg` / `max_own_deg` and repeat Steps 1-2, recording the change.
- [ ] **Step 3:** Planted delays 33 and 67 ms on M3: M3's matched landing offsets move by about the delay; the planted-shift test (`shift-test`) still shows sections with 0% more than 30 ms wrong; record section coverage (33% before).
- [ ] **Step 4:** Screenshots (Edge headless, video off) for the docs; document Stage F (the five beat tests and their numbers, what was built instead, the limits) and update the README. Run `pytest -q`; check no footage or audio files are on the branch.
- [ ] **Step 5: Commit** `docs: Stage F, the beat and dancing together`.
