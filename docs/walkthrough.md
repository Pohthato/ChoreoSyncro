# Walkthrough: how SyncScore is built, stage by stage

Written to be explained out loud. Each stage says what was built, why, and what question an
interviewer is likely to ask about it.

## Stage 1: scaffold

**What:** a Python package (`syncscore/`), one YAML config (`configs/default.yaml`), and a CLI
(`syncscore analyze <video> --out results/`, `syncscore validate`).

**Why it is shaped this way:**
- The package is split by pipeline step: `pose/` (video to keypoints), `features/` (keypoints
  to body-invariant features), `sync/` (the measurement), `synth/` (fake groups with known
  answers), `viz/` (pictures). Each folder can be tested on its own.
- Every tunable number lives in the config, not in the code. `config.py` loads it into typed
  dataclasses and **fails on unknown keys**: a typo like `windw_s` raises an error instead of
  silently running with the default.
- `mirror: "off"` is quoted on purpose. YAML 1.1 reads a bare `off` as the boolean `False`, a
  classic YAML trap.
- `.gitignore` blocks every video format plus `data/`, `frames/` and `results/`, so source
  footage cannot be committed by accident. Only `docs/media/` (derived skeleton renders and
  plots) is published.

## Stage 2: body-invariant features and a synthetic dancer

**The problem:** raw pixel coordinates can't be compared across members. A tall member near the
camera and a short member at the back doing the identical move have completely different
coordinates.

**The fix: bone directions** (`features/bones.py`). For each of 13 bones (upper arms, forearms,
thighs, shins, spine, shoulder line, hip line, neck to nose, ear line) we keep only the **unit
vector** from one joint to the next. A unit vector has no position (translation drops out), no
size (scale drops out) and no length (body proportions drop out). What is left is how the
joints are bent, which is the choreography. Bones are grouped into four body parts: arms,
legs, torso, head.

- **Confidence:** a bone is unknown (NaN) if either end keypoint is below `min_keypoint_conf`.
  Missing is better than wrong.
- **Foreshortening weight:** when a limb points at the camera its 2D length shrinks and its
  direction becomes noise. Each bone's current length is compared with *its own* typical length
  for that member (90th percentile over time), so proportions still don't matter, and the bone
  is trusted less as it gets shorter.
- **Mirroring:** a member doing the mirror image of the move (left arm where the others use the
  right) is handled by swapping left/right keypoints and reflecting x. Applying it twice gives
  back the original, which a test checks.

**The synthetic dancer** (`synth/generate.py`). To prove the method works we need groups where
the truth is known exactly, so we build them:
- A choreography is a list of key poses (12 joint values) on beats and half beats. Between key
  poses joints move with a **minimum-jerk** curve (the smooth S-curve human movements follow),
  then hold. Move then hold is what hits look like.
- **Forward kinematics** turns joint angles into the 17 COCO keypoints in pixels, so the output
  looks exactly like what YOLO pose gives us.
- Each member gets a different height (up to ±18%), limb proportions (±12% per segment) and
  stage position, and performs the choreography at `t − offset`. **The offset is therefore an
  exact time shift, even a fraction of a frame.**
- Known corruptions can be switched on: a limb held N degrees off, keypoint noise, low-confidence
  dropouts, occlusion, spins (the body's horizontal extent shrinks by cos(yaw); optionally the
  pose model swaps left/right while the back faces the camera), mirrored members.

**Tests that matter:**
- `test_tall_and_short_member_doing_the_same_move_are_identical`: different bodies and
  positions, the same move, features equal to 1e-9.
- `test_mirrored_member_matches_after_mirroring`: a mirrored member does **not** match until
  mirrored, then matches exactly.
- `test_limb_error_rotates_only_that_bone`: +20° at the left elbow changes the left forearm
  direction by exactly 20° and no other bone.

**Likely interview question:** *"Why not joint angles?"* Bone directions carry the same
information plus each limb's absolute orientation (an arm raised forward and the same arm
raised sideways have the same elbow angle but different directions). They are also simpler to
compare: one angle between two unit vectors. **Limit:** they are 2D. A member turned 30° away
from the camera shows foreshortened, slightly different directions for the same 3D pose. 3D
lifting would fix this, at the cost of a second model and its own errors.

## Stage 3: the alignment math

**Goal:** for every member, at every moment: how many ms ahead or behind the group
(**timing**), and how wrong the pose is once timing is taken out (**shape**), per body part.

**Step 1: the consensus** (`sync/consensus.py`). "The group" at each moment is the **median**
direction of each bone across the **other** members (leave-one-out).
- Leave-one-out: if member i were in their own reference, they'd pull it toward themselves
  and look better than they are. With 5 members that hides about a fifth of their error.
- Median, not mean: if one other member holds their arm wrong, a mean drags the reference and
  everyone else looks slightly wrong. A median ignores one outlier (a test checks this).

**Step 2: timing** (`sync/timing.py`). Take 2-second windows every 0.25 s. In each one, slide the
member's signal against the consensus by -400…+400 ms and score how well they line up with
**normalised cross-correlation** (1 = identical motion). The best-scoring shift is the offset.
- **The signal is bone velocity, not position.** Being in sync means starting and stopping each
  movement together, and velocity peaks exactly there. Velocity is also blind to a
  constant pose error (an arm held 20° off moves at the same speed as one held correctly), which
  is what keeps timing and shape apart.
- **Sub-frame precision:** fit a parabola through the best score and its two neighbours and
  take the top. At 30 fps one frame is 33 ms, but this recovers a 17 ms (half-frame) offset
  to within 2 ms.
- **Why not DTW:** DTW stretches time freely to make poses match, so part of a shape error leaks
  into timing. It returns a warping path, not "one offset for this stretch", and it has no
  natural sub-frame answer. Cross-correlation answers exactly the question we ask.
- **Periodic moves:** a move repeated every beat gives several correlation peaks. A **Viterbi**
  pass picks the sequence of offsets over all windows that maximises total score minus a
  penalty for jumping (0.1 per 100 ms), so one ambiguous window follows its neighbours.
- **Honesty gates:** a window gets "no reliable offset" when the group barely moves (nothing to
  time), the peak is low (the motions don't match), or the member is missing for most of it.

**Step 3: iterate to a common clock** (`sync/measure.py`). The first pass compares everyone
against an unaligned consensus, which is blurry because members are at different times. So:
shift each member by their current offset, rebuild the consensus from the shifted members,
measure again. The group clock is the **median member**: offsets are re-centred so the median
member is 0 ms.
- **A bug found and fixed here:** the first version did 3 plain passes and every member came
  out about 2.5 ms too far from zero. Printing the error per pass showed it flipping sign each
  pass (9 → −4.9 → +2.7 ms). Each member was measured against others whose own estimates were
  still wrong in the opposite direction. With 2 members it would flip forever. The fix is a
  **damped update** (move halfway toward each new estimate) and stopping once nothing moves by
  more than 0.1 ms. Error on clean data went from 2.7 ms to under 0.4 ms for groups of 2 to 7.
  A regression test with 2 members guards it.

**Step 4: shape** (`sync/shape.py`). Move the consensus onto the member's own clock using their
offset, then measure the angle between each of the member's bones and the consensus bone.
Average per body part, weighted by trust. The synthetic test injects +20° at one member's left
elbow and checks that the measurement says: that member, **left forearm, 20 ± 2°**, arms are the
worst part, timing unchanged, and **no other member picks up the error** through the consensus.

**Mirroring** (`features.mirror`): `"off"`, a list of member numbers, or `"auto"`, where each
window keeps whichever of "as is" or "mirrored" is closer to the consensus, smoothed over 5
windows. Both the fixed and auto modes are tested on a mirrored member who is also 100 ms late.

**Likely interview question:** *"What does an offset of +40 ms mean?"* That member performs
each movement 40 ms after the group's median member during that window. Negative means early.

## Stage 4: proving it works (`syncscore validate`)

**Smoothing first** (`pose/smooth.py`). Real keypoints jitter, so the benchmark runs the same
smoother the video path uses.
- **One Euro filter:** a low-pass filter whose cut-off rises with speed. Heavy smoothing when a
  joint is still (all you see there is jitter), light smoothing in fast moves (where lag hurts).
- **It must add no lag.** A normal (causal) filter delays its output. The test measured
  **67 ms** of delay with the first settings, as big as the offsets being measured, and the
  delay depends on how fast each member moves. So the filter runs forward and backward and the
  two are averaged: the delays cancel (under 1 ms, tested).
- **Tuned, not guessed.** The first defaults (min_cutoff 1, beta 0.05) kept only 31% of a hit's
  peak speed. A grid search against clean synthetic keypoints picked min_cutoff=1, beta=1:
  keypoint error 3.8 → 1.9 px, bone-angle noise 3.5° → 1.4°, 93% of peak speed kept.
  Caveat: tuned on Gaussian noise, and real pose-model jitter is not Gaussian.

**The benchmark** (`synth/benchmark.py`). Synthetic groups with planted errors go through the
full pipeline. It reports three numbers, and the third is the important one:
- *window error*: how far off the measured offset is, on windows the method trusted,
- *coverage*: how many windows it was willing to measure,
- *confidently wrong*: trusted windows off by more than 30 ms. Saying "I don't know" is fine;
  being wrong and sure is the failure that matters.

**Two things the benchmark caught and that were fixed:**
1. **Repeated fast grooves.** When a move repeats faster than the ±400 ms search (every
   0.36 s), the correlation has several equally high peaks (measured: +33, −327 and +393 ms
   all scored 0.98-1.0). The method was choosing one and calling it reliable: 15% confidently
   wrong. No smoothing can resolve a truly periodic signal, so the fix is honesty: if a separate
   peak scores within 0.15 of the chosen one, the window is marked *ambiguous* and not
   reported. Confidently wrong 15% → 1%.
2. **Spins looked too good** because the first simulation swapped left/right labels for
   every member identically, so the errors cancelled. Real pose models are inconsistent, so a
   harsher case (labels swap for 2 of 5 members) was added and is reported as a real failure:
   69% coverage, 2% confidently wrong.

## Stage 5: outputs (`syncscore demo`, and the same writer for `analyze`)

- `summary.json` / `summary.csv`: per member, overall and per section (8 s at the time, 32 s after Stage A): mean offset,
  mean |offset|, spread, share of time in sync (|offset| ≤ 35 ms), timing and pose error per
  body part, the body part most often off, the 3 worst timing moments and the 3 worst pose
  moments (one per second, named in the member's own left/right even if mirrored).
- `heatmap.png`: members by time, blue = ahead, red = behind, grey = in sync. **Windows that
  could not be measured are hatched**, so "no reading" never looks like "in sync".
- `report_cards/M*.png`: one page per member.
- `skeletons.mp4`: skeletons only, dark background, green / amber / red / grey status with a
  text label, and a pink halo on any bone whose pose is off by more than 15°. No source pixels
  are drawn, so it can be published.

**Two more design fixes found by tests here:**
- *"Body part most often off" averaged a part's bones*, so one shin 30° off counted as legs
  7.5° off and went unreported. A part is now judged by its **worst bone**.
- *With 3 members, leave-one-out leaks errors.* Each member's reference was the median of 2
  others, which is their average, so one member's 30° error showed up as 15° on the other
  two. Shape now uses the median of the **whole group including the member**. That works like a
  vote: in a trio the two who agree outvote the third. Timing keeps leave-one-out, where the
  iteration makes it exact. With 2 members no method can say who is wrong, only that they differ.

**Likely interview question:** *"How do you know the visual isn't lying?"* Every colour has a
text label next to it, "no reading" has its own look, and the tests read back
`summary.json` to check it names the planted errors (`tests/test_outputs.py`).

## Stage 6: from video to member tracks (`syncscore analyze`)

**Pose** (`pose/extract.py`). Ultralytics **YOLO11-pose**: one network finds every person and
their 17 keypoints in one pass and runs on a laptop CPU. ByteTrack comes in the same `track()`
call. Optiqen's badminton worker uses the same model family, so it's the same front end. RTMPose
is a bit more accurate in crowds but needs a separate detector and MMCV, which is painful to
install on a Windows CPU. Per detection we keep the box, keypoints and a 32-number torso
colour histogram, **never pixels**.

**Real timestamps** (`pose/video.py`). Phone video is often variable frame rate. Assuming
frame k sits at k/fps would add the recording's own jitter to every offset, so each frame's
presentation time is read and keypoints are resampled onto an even clock when spacing is
uneven.

**One track per member** (`pose/tracks.py`). ByteTrack fails in two ways:
- *Fragmentation:* after an occlusion a member returns with a new ID. Fixed by stitching. Seed N
  member slots with the tracklets visible together the longest, then attach the rest (longest
  first) to the slot they fit: no time overlap, similar torso colour, nearby position if the gap
  is short. Leftovers (crew, backup dancers) are dropped. Gaps are left empty, never invented.
- *Swaps:* two members cross and their IDs exchange. Flagged when, after a crossing, each track
  looks more like the *other* member's clothes than its own. With matching outfits this misses
  swaps, so `tracking_report.json` says so, and the **true** swap rate comes from
  `syncscore audit` (about 30 sampled frames with labels drawn: you write who each label really
  is) and `syncscore audit-score` (accuracy, switches, switches per minute).

Members are named M1…MN left to right by their median position, so names are readable and stable.

**Likely interview question:** *"How often do IDs swap?"* Two numbers, kept apart on purpose:
the automatic *suspected* swaps (cheap, misses look-alikes) and the hand-audited rate (the
real one). The tool never presents the heuristic as the truth.

## Stage 6b: what the real clip taught

The first real run (BTS dance practice, 3:50, 7 members) gave **3% reliable windows and a 0.19
median correlation**. A professional group doesn't look like that, so this was debugged, not
tuned.

- **Pose glitches.** About 1% of frames carried about 90% of the velocity "energy" of the
  shoulder line, hip line and ears: the pose model swapping left and right for a frame when a
  dancer turns side-on, or briefly putting a wrist somewhere else, at high confidence.
  Correlation squares velocity, so those few frames drowned the real motion. Fixes: undo swaps
  that **revert within 2 frames**, and mark bone turns faster than 1800°/s as unknown.
  - A trap caught by a test: the first version also "repaired" a real turn (arms cross in the
    image and stay crossed), flipping every later frame. Requiring the swap to revert fixed it.
  - Proof: the synthetic generator now injects the same glitches. Without the fixes, 10% of
    trusted windows were confidently wrong; with them, 0%. The headline numbers didn't change.
- **But that wasn't the main cause.** The fixes didn't raise the real clip's reliability. Plotting
  four dancers' bone angles showed why: they clearly move together on the big hits, but most
  of the time is individually styled grooving, so short windows rarely correlate strongly.
- **Planted shifts in real motion** (`synth/real_shift.py`). Real footage has no answer key, so
  plant one: delay one member's real track by 33 / 67 / 133 ms, re-measure, and check that their
  offset relative to the others changes by exactly that. Medians over the clip came back at
  about 30 / 66 / 133 ms in every setting tried; single windows were too noisy (6-13% confidently
  wrong once thresholds were loose enough to measure anything). That's why the next step reports
  timing per section, as a robust median with an uncertainty, instead of per window.

## Stage A: timing per section (`sync/sections.py`)

**Why:** on the real clip single windows are rarely trustworthy alone, but many of them
together are. So the main answer becomes one offset per member per section, with a 90%
interval.

**How:**
1. Pool every window that passes the hard gates (group moving, member visible, no ambiguous
   second peak) and a loose correlation floor (0.2). The median is robust to the bad ones.
2. **Keep "0 ms" meaning the same thing.** The first version re-centred each section on whoever
   was visible. A test hid one member for 10 s and everyone else's offsets moved by 20 ms,
   because the "median member" had silently become someone else. Now each section's clock
   drift is estimated by comparing each visible member with *their own* typical offset, and
   "0 ms" stays the clip's median member.
3. **Honest intervals:** a **joint block bootstrap**. Overlapping windows aren't independent (each
   frame is in 8 of them), so resample non-overlapping blocks, the same blocks for all members,
   and redo the re-centring in each draw so the reference's own noise is included. The first
   version resampled one member at a time and claimed 90% intervals that held the truth only 75%
   of the time. A section needs at least 5 blocks: with 2 the "interval" held the truth only 41%
   of the time.

**What the real clip says** (shifts planted into real members' tracks, `syncscore shift-test`):
- Per window, the **measurement noise is about ±12-15 ms**: the change between the run with and
  without a planted shift is tight. But window offsets themselves spread about ±60 ms, and that
  spread is the same in both runs, so it comes from the dancing: either real moment-to-moment
  timing variation, or styling the method reads as timing. The planted test can't tell those two
  apart.
- Because of that spread, a trustworthy section needs **about 30 s**. At 8 s no member-section
  qualified; at 32 s about a third did, with errors of about 10 ms (from only 12 planted
  comparisons, so treat it as indicative). Default sections are 32 s; better, mark the song's
  real parts in the config (`sections.explicit`).
- Planted shifts come back about **10% high** (+74 for +67). Delaying one member also moves the
  group reference the others are measured against, so the shift partly counts twice. That's a
  property of measuring against the group, not noise.

**Likely interview question:** *"Why not just lower the threshold?"* It was tried. Every looser
threshold measured more windows and was confidently wrong more often (8% → 13%). Pooling gives
coverage without pretending a single noisy window is an answer.

## Stage C: tracking that doesn't throw dancers away

**Found by looking, not by metrics.** Drawing the tracked skeletons on one real frame showed a
dancer with no skeleton at all. The detector had found all 7 people in that frame. The
stitching had thrown one away: a 10-second track that matched no member slot's clothing colour
well enough, so it was treated as a passer-by. Across the clip **15% of clear detections** were
dropped this way, in stretches up to 25 s, while only 19 frames in the whole video had more than
7 people.

**Fix:** in a dance video nobody long-lived is a passer-by. A tracklet of 1 s or more is always
attached when a member slot is free for its whole span, however poor the colour match (lighting
changes as dancers move around the room). Only short tracklets can be rejected for a poor fit.
Result: **96%** of clear detections kept, every member visible 72-93% of the time (was 55-93%).

**Code that was deleted:** a "make room" repair (move shorter tracklets out of the way when all
slots are busy). On the real clip it was tried 14 times and never succeeded, because tracklets
are placed longest first. Its test had also passed for the wrong reason. Code that can't be
shown to matter goes.

**What it bought** (planted shifts on the re-tracked clip): single windows 11 → 7 ms error,
sections 0% confidently wrong, with 83% of truths inside the claimed 90% intervals.

**One more thing the real report cards showed:** "worst pose moments" of exactly 180° on the
shoulders, hips and a forearm. Those aren't poses: the member faced a different way from the
group, or the model swapped their left and right. A 2D comparison is meaningless there, so frames
where *both* the shoulder line and the hip line are reversed (more than 120°) relative to the
group are left out of pose errors and reported as "facing a different way: N% of frames".

**Pose errors judged against the group, not against perfection.** On the real clip every member
showed about 50° of arm "error" and every member's arms were "most often off". That's 2D camera
perspective and personal styling, not seven people dancing wrong. Two changes:
- a body part is off in a frame when the member's worst bone is more than 15° beyond the
  *group's typical error* for that part in that frame (the report card shows this excess),
- "off more than the others" names a part only if this member is off there at least 10
  percentage points more often than the group's median member.

On the real clip no member's pose now stands out (a professional group). On the synthetic demo
only the member with the planted elbow error is named. Bone errors of 150° or more are labelled
"likely a left/right label swap": usually the pose model, possibly real, so flagged rather than
hidden.

## Stage B: timing at the group's hits (`sync/hits.py`), tried and measured, not in the report

**Why:** sections need about 30 s on the real clip because most of the dancing is individually
styled grooving. Plotting the clip showed moments where the whole group clearly moves at once.
If those hits are cleaner timing cues, each one could be read to a fraction of a frame, and
sections would need far less time.

**How:**
1. Speed per member: the RMS angular speed of their bones (the same glitch-gated velocities the
   windows use), divided by their own typical speed, lightly smoothed.
2. Group hits: peaks of the members' median speed that stand clearly above their surroundings.
   74 on the clip, about one every 3 s.
3. Each member's peak within ±400 ms of the hit, sub-frame with a parabola.
4. Offset against the hit's moment as the *other* members give it, each corrected by their own
   typical offset (a median polish), so "0 ms" is the median member as everywhere else.

**The tie problem.** At this tempo an arm going up and the same arm coming back down 250 ms
later are two speed peaks, both inside ±400 ms: half of all matches had a second peak nearly as
high. Four ways to choose, checked quickly by planting shifts in real members' tracks (with the
plain median of the others as the reference, before the fixes below):

| how the member's peak is chosen | member-hits read | error | > 30 ms wrong |
|---|---|---|---|
| speed peak, ties skipped | 9% | 2 ms | 2% |
| speed peak, movement direction must match at every hit | 16-24% | 6-11 ms | 1-8% |
| motion projected onto the group's direction | 30-49% | 10-24 ms | 5-13% |
| **speed peak, direction only breaks ties** | **26%** | **2 ms** | **1.5%** |

Direction costs precision because the reference direction comes from the other members, so it
moves when they do. Used only for ties, it tripled the readings without losing precision.

**Two traps, both caught by tests:**
- The median of the *other* members isn't the median member. With offsets 0, 40, -30, 90, 0 ms,
  the -30 member's four others have median 20, so they read -50. Each member's typical offset
  is now taken out first.
- When the others split between two moves (two peaking on one, two on the next, 490 ms apart),
  their median falls in between and every reading at that hit is nonsense. A hit's moment now
  counts only if at least 3 of the others peak within 100 ms of it.

**What the real clip says** (`syncscore shift-test`, 3 members × 3 planted shifts):
- **Single hits:** each member's own peak moves correctly for 99% of plants, and the typical
  error is about 3 ms. But rare pick flips (the neighbouring move, half a beat away) push the
  mean to 22 ms, and 3% are more than 30 ms wrong.
- **Sections from hits: 0%.** At the same hit, members' peaks spread with an IQR of 45-100 ms,
  and a 32 s section holds only 0-8 matched hits per member. No interval gets under 50 ms.
- **Windows or hits, whichever is more certain:** 24% of sections instead of 22%. That is one
  member-section, within noise.
- **Whole clip from hits:** 13-23 hits per member, but intervals 45-75 ms wide. Planted shifts
  mostly came back short (by 13-25 ms at +133 ms), and planting a shift in one member moved others'
  readings by up to ±30 ms.

So the hits aren't cleaner than the grooving on this clip: the spread is in the dancing, at the
big moments too. The hypothesis was wrong here, and the report stays on window sections.

**Why keep the code, when Stage C deleted code that didn't matter?** This code *is* the
measurement: `syncscore shift-test` reproduces every number above, and it is tested (sub-frame
recovery, planted delays, the return-stroke tie, an unbreakable tie as "no reading", a hidden
member). It changes no output. It might pay off on footage with sharper, sparser hits or a
slower tempo, where ±400 ms holds only one move, and that is now cheap to check.

**Why not show single hits as "worst moments"?** Because the largest per-hit offsets are
exactly where pick flips live. A list of the worst hits would mostly show the neighbouring move.

## Stage D: everyone tracked (`pose/coverage.py`, `pose/fill.py`, `pose/recover.py`)

**Why:** with 7 dancers, the skeleton view often showed 6. Measured on the BTS clip, all 7
members were tracked in only **46.5%** of frames. Everything downstream (timing, pose, the
viewer still to come) can only be as complete as this.

**Measure first.** A coverage report (`tracking_report.json` → `coverage`) splits each missing
member-frame by cause. In **80%** the detector itself found fewer than 7 clear people: a dancer
half hidden behind another. In **20%** it found them, but the member assignment left the piece
out (shorter than 0.3 s, or no tracker ID). Most gaps were short: 266 of 339 under 0.5 s.

**Two fixes, one rule.** Only a real detection may fill a gap; nothing is interpolated into the
keypoints, because an invented pose would be scored as measured.
1. **Unused detections:** inside a member's gap of at most 2 s, a detection no member uses,
   near the position interpolated across the gap and of similar size, fills that frame.
2. **A second look:** on just the frames still missing someone inside such a gap (1,583 of
   6,905), the pose model runs again at a lower box threshold (0.05) and larger input (1280 px).
   Its detections go through the same rule. Cached in `recovered.npz`.

Gaps touching the start or end of the clip are never filled (nothing to anchor them), and one
detection fills at most one member.

**Found by looking.** The first version's recovered boxes were usually on the right person, the
dancer half hidden behind another, but often with some joints on the front dancer's body. One
was a thin slice of the dancer in front. Borrowed limbs would make the hidden member's timing
partly the front dancer's. Now any joint lying on another present member's joint is dropped,
and a candidate left with fewer than 6 joints of its own is rejected. In a fresh sample after
the fix, every recovered box checked was the hidden dancer, with joints on their own body.

**Result on the BTS clip:** all 7 tracked in **57.1%** of frames (was 46.5%); each member seen
76-96% of the time (was 72-93%). 517 member-frames came from unused detections and 936 from the
second look. Gaps of 0.5-2 s went from 47 to 13. The 26 gaps of 2 s or more are untouched by
design.

**Caught by the final review: the same dancer given to two members.** The second look's low
threshold finds near-copies of one person (boxes overlapping 0.4-0.7). Every pair in a frame was
scored once, before any fill, so two missing members could each take a copy of the same dancer
(about 1-2% of fills). Now each fill counts as present before the next member is scored. Two
more fixes from the same review: the "after" cause split is counted on the original detections
(re-detection finds everyone again, which made detector misses look like assignment losses), and
re-detection reads frames in decode order instead of seeking, which can land on the wrong frame
in variable-frame-rate phone video.

**What it did to timing** (planted shifts, same 3 members × 3 shifts):

| | before Stage D | after |
|---|---|---|
| sections measured | 22% | **33%** |
| error | 11.7 ms | 9.7 ms |
| > 30 ms wrong | 0% | 0% |
| truth inside the 90% interval | 83% | **100%** |

The gain is spread over all three planted members, and **M2, who never got a section reading
before, now gets one in 14-29% of sections** (errors 3.6-10.5 ms). Before the duplicate fix the
same test read 22%: two members carrying one dancer's pose blurred the group reference. Nine
plants is a small sample, so read 33% as "about a third", not a precise figure.

**Suspected ID swaps went from 19 to 39, and that is mostly a measurement change.** The swap
check compares clothing colour before and after each crossing. Filled boxes now sit exactly in
the overlaps that used to be gaps, so the check sees crossings it couldn't see before: leaving
the filled boxes out of the crossing check gives exactly the original 19, so all 20 new ones are
at those crossings. A filled frame's colour is set to unknown, since in a half-hidden box the
torso is mostly the front dancer's clothes. Filling can't
create a lasting swap (a gap is only filled between the member's own sightings, at most 2 s
apart), but the newly visible crossings can expose stitching errors that were hidden before.
The true rate needs the hand audit (`syncscore audit`).
