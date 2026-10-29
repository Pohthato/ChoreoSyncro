# Follow the picked members: design

Date: 2026-10-28. Status: agreed in conversation, not yet built.

## Intent

On the aespa "Whiplash" practice clip, SyncScore tracked the wrong people. The clip has about 7
dancers in view (the four members plus backup dancers in black), and tracking fills its 4 member
slots with whichever tracks are longest. Any track of a second or more joins a free slot whatever
it looks like, and the only appearance cue is a torso hue/saturation histogram that can't tell
black outfits apart. The labels move between people: at 1:30, three of the four labels sit on
backup dancers while two members go untracked.

This design lets the user name the members once, in one frame of the upload page. From then on
the tool follows each named member for the whole clip and keeps her name on her. A doubtful
moment leaves her unseen rather than putting her name on someone else.

## Decided in conversation

- **Goal:** each member by name, with her own timing. Names must stay on the right person for the
  whole clip.
- **Where:** in the upload page (`syncscore serve`). The CLI gets the same option so the page can
  call it.
- **Effort:** the user names the members in one frame and nothing more. No review step after.
- **Recognition:** a person re-identification model makes a fingerprint of every detected person,
  and a classifier trained from the picked frame decides who each track is.
- **Moving camera:** the footage's camera pans and zooms to follow the group or a soloist (the
  aespa clip is wide at 0 s, tight on the four at 40 s, on one dancer at 150 s, wide at 170 s, with
  no hard cuts). Tracking must cope with this.
- **Someone passing in front:** skeletons must not switch. A motion model with heading keeps each
  member's name through the occlusion.
- **Solos:** members who are off screen are not seen, which is neither apart nor an error.
- **Auto-framing in the viewer** (pan and zoom to keep the group, or one chosen dancer, centred)
  was also asked for. It does not depend on this work and gets its own design and branch.
- **Professional:** the picking view matches the viewer's look; nothing jittery or makeshift.

## Non-goals

- Viewer auto-framing (separate design).
- Videos with hard cuts between camera angles. A cut is detected and handled by resetting the
  motion model (see Camera), but nothing is designed around multi-angle edits.
- Naming members automatically (face recognition or any lookup of who someone is). Names are
  whatever the user types.
- Any change to runs without picks. Without picks, tracking and every output stay exactly as now.

## Flow

The upload job splits in two with a pause:

1. **Upload, pose, fingerprints.** `syncscore analyze <video> --out <dir> --pose-only` runs pose
   detection (cached as now), the camera step and the fingerprint step, writes
   `detections.npz`, `camera.npz`, `fingerprints.npz` and `pick_frames.json`, then stops.
2. **Pick.** The job waits in a new state, `picking`. The start page shows the picking view.
3. **Track and measure.** `syncscore analyze <video> --out <dir> --members <dir>/picks.json`
   reuses the cached files, runs identity tracking and the rest of the analysis, then `view`
   builds the viewer as now.

Earlier uploads get **Pick again**, which goes back to step 2 with the cached poses and
fingerprints. The member-count box stays for runs without picks.

### picks.json

```json
{"frame": 1210, "members": [{"name": "Karina", "box": [412.0, 96.5, 498.2, 355.0]}, ...]}
```

Each pick is resolved to the detection in that frame with the highest box overlap (IoU at least
0.5, else an error naming the pick). Members are ordered as picked. At least 2 picks; names must
be non-empty and distinct.

### The picking view

- One frame with a box around every detected person. It starts on the suggested frame: the one
  with the most clearly visible people (8+ confident keypoints) whose boxes don't overlap, and
  prefers frames away from the very start and end. The five best are offered as quick buttons.
- A slider moves to any frame, for when a member is hidden in the suggested one.
- Click a box, type a name (Enter confirms). Named boxes show the name; clicking again renames or
  removes. **Continue** is enabled once 2 or more are named. Keyboard use works: Tab moves between
  boxes, Enter names, Delete removes.
- The frame image comes from the server reading the uploaded video on request
  (`GET /api/uploads/<id>/frame/<n>.jpg`); nothing is stored. Boxes come from
  `GET /api/uploads/<id>/people?frame=<n>`. Picks are sent with `POST /api/uploads/<id>/picks`,
  which writes `picks.json` and starts step 3.
- Same fonts, colours and spacing as the viewer. The picking logic that doesn't need the DOM
  (where each box sits and which one is on top, name validation, the picks sent) lives in a
  module tested in Node. The suggested frames are ranked in Python by `--pose-only`.

## Camera

New module `syncscore/pose/camera.py`.

- Between consecutive frames, find background points (corners via `cv2.goodFeaturesToTrack`)
  outside every person box (boxes padded by 10%), follow them with `cv2.calcOpticalFlowPyrLK`,
  and fit a similarity transform (translation, uniform scale, small rotation) with
  `cv2.estimateAffinePartial2D` and RANSAC.
- Chain the transforms into one per frame that maps image pixels to **stage coordinates**: pixels
  of the first frame of that frame's segment (see below). The camera step runs before picking, so
  it can't use the picked frame as reference, and it doesn't need to: positions are only ever
  compared within a segment. A pan no longer looks like everyone moving and a zoom no longer looks
  like everyone growing.
- A frame where the fit fails (too few inliers, or a hard cut: the mean frame difference jumps)
  starts a new **segment**. Positions are not compared across segments; the motion model treats a
  new segment like a long absence.
- Output `camera.npz`: per-frame 2x3 transform, segment id, inlier count.
- Timing is untouched: it uses limb angles, which pans and zooms don't change.

## Fingerprints

New module `syncscore/pose/fingerprint.py`.

- After pose detection, decode the video once more, cut out each detection's box, resize to the
  model's input, and run the re-ID model in batches. Save one L2-normalised vector per detection
  row in `fingerprints.npz`. Only numbers are saved, never pixels (as with the colour histograms).
- Works on old cached `detections.npz`, so a clip never needs pose detection again for this.
- The model is OpenCV Zoo's person re-identification model (YouTu ReID, Apache-2.0), run on CPU
  through OpenCV's DNN module. It comes in a full version (fp32, 107 MB) and an 8-bit one (int8,
  27 MB); the gate (below) picks one. OSNet was the other candidate, but its authors publish no
  ONNX file, only weights that need the torchreid code, so it was dropped. Weights download on
  first use to a cache folder, are checked against a pinned SHA-256, and are never committed. The
  README names the model, its source and its licence.
- The second look in gap-filling (`recover.py`) computes fingerprints for the detections it adds,
  from the frames it already decodes.

## Identity tracking

New module `syncscore/pose/identity.py`, used only when `--members` is given. It replaces
`assign_members` for that run and feeds the same `assemble`, gap-filling and analysis.

### Examples from the picked frame

- Each picked detection's tracklet gives every one of its rows as examples of that member.
- Each unpicked person in the picked frame gives their tracklet's rows as examples of **not a
  member**.

### Splitting tracks that change hands

ByteTrack sometimes hands one ID from one dancer to another at a crossing. At every crossing (the
tracklet's box overlaps another with IoU above `tracking.crossing_iou`), compare the mean
fingerprint over 0.5 s before and after, and the heading before and after (from the tracklet's own
path in stage coordinates). Split the tracklet there if the fingerprints differ by more than
`identity.split_distance`, or if the path after continues the *other* dancer's heading better
than its own. Only crossings are checked, so a dancer turning around isn't split.

### Naming a piece

- A piece's score for each class (each member, and not-a-member) is the mean of its rows' top-k
  cosine similarities to that class's examples.
- The piece takes member X's name when X's score beats every other member and not-a-member by at
  least `identity.name_margin`. Otherwise it stays unassigned.
- **Learning rounds:** pieces that beat the competition by the stricter `identity.learn_margin`
  join their member's examples, and naming runs again, up to `identity.rounds` (3). This teaches
  how each member looks from other sides and under other light. Not-a-member examples grow the
  same way.

### The motion model

- Each named member's hip centre in stage coordinates is followed by a constant-velocity Kalman
  filter (position and velocity, so heading is part of the state).
- **One person per name:** if two pieces carry the same name at overlapping frames, the one with
  the larger margin keeps it, and the other is re-scored without that member, which usually leaves
  it unassigned.
- **Hidden members keep their name.** While a member is unseen, the filter predicts where she is
  going and its uncertainty grows. A piece that starts after the gap takes her name only if its
  fingerprint passes *and* its first position falls inside the predicted region (Mahalanobis
  distance under `identity.position_gate`). A dancer crossing in front, going the other way, fails
  the position test even when her clothes are close.
- After `identity.free_after_s` (3 s) unseen, or across a camera segment boundary, the position
  test is dropped and the fingerprint alone decides. This covers members leaving the frame during
  a solo and coming back when the camera widens.

### Gap-filling

`fill.py` (spare detections) and `recover.py` (second look) take an optional acceptance test. With
picks, a detection may fill member X's gap only if its fingerprint names it X under the same
margin and its position passes the motion model. Without picks, they behave as now.

### Names, order and report

- Members keep the picked order and names in the viewer, the report cards and `summary.json`.
- `tracking_report.json` gains an `identity` block: the picked frame, each member's tracked share
  of the clip, the splits made, and the close calls (pieces that passed or failed by less than half
  the margin, with times). The colour-based `suspected_swaps` stays for runs without picks.

### Config

A new `identity:` section in `configs/default.yaml` holds `model`, `name_margin`, `learn_margin`,
`rounds`, `split_distance`, `position_gate` and `free_after_s`. Values come from the gate test on
aespa and are recorded with the evidence in the walkthrough.

## Testing and proof

### Gate: do fingerprints separate the people?

Runs first, before anything else is built. On aespa, hand-label about 30 frames by outfit. Using
the picked frame's tracklets as examples (with learning rounds), name the labelled detections in
the other frames. A model passes if at least 95% of member detections get the right name and at
most 2% of backup-dancer detections get a member's name, including the member dressed in black.
If neither version of the model passes, stop and report back before building the rest.

### Ground truth on aespa

Hand-label about 80 frames spread across the clip, plus dense windows (every 0.2 s for 2 s) at the
hard moments: crossings, someone passing in front of a member, the solo around 150 s, and
re-entries after the camera widens. `syncscore audit` gains a `--times` option for the dense
windows; a true name of `dancer` marks a backup dancer. Targets:

- at least 95% of member sightings carry the right name
- at most 2% of sightings put a member's name on a backup dancer
- in the dense windows, no wrong name lasts longer than 1 s
- each member's tracked share of her on-screen time, reported as measured (no target yet)

### Unit and synthetic tests

- Camera: a synthetic frame sequence with a known pan and zoom is recovered; a hard cut starts a
  new segment; a fixed camera gives identity transforms. On aespa, background points stay put in
  stage coordinates.
- Recognition: synthetic fingerprint groups. Names go to the right group, close cases stay
  unassigned, and learning rounds don't drift when a decoy group sits near one member.
- Tracking scenes, synthetic: two dancers in near-identical clothes crossing (heading keeps them
  apart); someone walking in front of a member (no switch, name returns to her after); a member
  leaving the frame and coming back (re-acquired by fingerprint); a tracklet handed from one dancer
  to another (split at the crossing).
- `picks.json` resolution and validation; `--pose-only` then `--members` round trip on a small
  synthetic video.
- Upload page: the job pauses in `picking`, `people` and `frame` answer for a valid frame and
  refuse others, picks are validated and saved, Continue runs step 3, Pick again returns to
  `picking`. Node tests for the picker module.

### Nothing else changes

A fresh BTS run without picks gives a `summary.json` identical to the current one, and the full
existing suite passes.

### Docs

A walkthrough stage with the gate results, the aespa ground-truth numbers, and the limits found.
README notes on picking members in the upload page and the `--members` option.

## Branch

`feat/follow-members`, cut from the upload page branch (`claude/beautiful-joliot-92c3cc`), which
sits on `feat/beat-together`. It merges after both.
