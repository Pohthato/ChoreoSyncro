# Group coaching from practice video: design

Date: 2026-10-18. Status: agreed in conversation, not yet built.

## Intent

SyncScore today measures timing and pose per member and says when it can't. This design turns
it into a tool a choreographer and a dancer can both use after practice: see every member,
see the difference against the group directly on the video, measure arm angles and floor
positions in 3D, and sort each difference into **timing**, **occasional difference** or
**personal signature**.

The bar is the project's own: every number is checked against a planted truth before it is
shown, and "not seen" or "not sure" is shown as such. No claim about what any company uses
internally; the claim is only that each feature is measured.

## Users and setting (decided)

- **Both** choreographer and dancer use it, reading the same analysis through two views.
  - Choreographer: the whole group, a song timeline of who is off where and which kind, click a
    moment to jump to it in slow motion with every member's ghost.
  - Dancer: one member, followed; their skeleton and the group's ghost, their own list of moments,
    and their personal signature on its own page, never marked as mistakes.
- **Footage:** fixed-camera practice-room video. No cuts, no camera motion.
- **Cameras:** one phone is the default. A second phone at the same time is optional and gives
  measured 3D; a few two-phone recordings also serve as the real-footage answer key for
  one-phone 3D.
- **No labeller available.** The tool describes differences against the group; the choreographer
  judges. Labels come from use (see part 4).
- Runs locally. Anything showing source footage (overlays, slow-motion clips) stays in the
  git-ignored `results/` and is never committed, as with `syncscore audit` today.

## Goals

1. All members tracked in as many frames as the footage allows, with each recovered frame a real
   detection (no invented poses).
2. A viewer: each member's skeleton on the video, the group's pose as a faint ghost scaled to
   their body, slow motion around chosen moments, choreographer and dancer views.
3. 3D arm and leg angles that don't change with which way a dancer faces; floor positions in
   metres; formation accuracy as a third measure next to timing and pose.
4. Every flagged difference sorted into timing / occasional difference / personal signature,
   with the sorting proven on planted cases.

## Non-goals

- Moving cameras, stage footage, broadcast edits.
- Judging "correct" choreography without a reference (the reference is the group).
- Audio beat alignment (still a listed limitation).
- Real-time use during practice.

## Anti-goals (failure even if it "works")

- Invented data shown as real: an interpolated or guessed pose drawn or scored as a detection.
- A dancer's personal style shown as a list of mistakes.
- A number on screen that no planted test backs.
- Source footage committed to the repository.

## Part 1: everyone tracked (first)

Measured on the BTS clip (`results/bts`): all 7 members are tracked in **46.5%** of frames. In
frames with fewer, the detector itself found fewer people **73%** of the time (occlusion), and in
**27%** it found 7 or more but the member assignment left one out (tracklets under 0.3 s, or
without a tracker ID, or short and overlapping every member). 266 of 339 member gaps are under
0.5 s; a few last 14-22 s.

Decided:
- **Measure first:** a coverage report (frames with all N members; gaps by length; cause split
  into "detector missed" and "detected but not assigned"), before and after each fix.
- **Fill from leftovers:** inside a member's gap of at most 2 s, an unassigned detection near the
  position interpolated between the gap's two edges, of similar size, not overlapping another
  member, fills that frame. One detection fills at most one member.
- **Re-detect missed frames:** for frames still empty inside such a gap, run the pose model again
  on that frame only, at a lower box threshold and a larger input size, and accept a detection
  only under the same rules, plus: its keypoints must not duplicate the front member's (an
  occluded person's box often carries the occluder's skeleton). Cached like `detections.npz`.
- Gaps at the start or end of the clip (no position on one side) are not filled. Nothing is
  extrapolated.
- **Proof:** coverage before/after on the real clip, the planted-shift test re-run, and the hand
  audit (`syncscore audit`) recommended to the user for identity accuracy.

Left to the builder: thresholds within the rules above, tuned on the real clip and stated.

## Part 2: the viewer (second)

- Overlay: tracked skeleton on each member; "not seen" frames drawn as nothing, recovered frames
  drawn the same as others (they are real detections).
- Ghost: the group's median pose (bone directions, as `group_median`), drawn from the member's
  own pelvis with the member's own bone lengths, faint. Timing is shown by drawing the ghost at
  the group's clock and the member at theirs: a late member trails it.
- Slow motion: any time range at 1/4 speed, all members side by side, aligned by their feet.
- Two views as above. A local app; the medium (desktop window vs. local web page) is decided in
  part 2's own plan.
- Proof: a planted 133 ms delay (4 frames) on one real member must be visible as trailing the
  ghost; a screenshot pair goes in the walkthrough.

## Part 3: 3D (third)

- One phone: monocular 3D lifting of each member's 2D keypoints (a published lifting model; the
  choice is made in part 3's plan on Windows-CPU installability and accuracy). Joint angles
  (elbow flexion, shoulder elevation, knee flexion...) computed in 3D.
- Floor: one calibration per room (floor tape or known room width) maps feet to metres;
  formation error = distance from each member's planned spot, where "planned" is the group's
  formation shape (relative spacing), not an absolute map.
- Two phones (optional): synced from the music audio, triangulated 3D.
- Proof: synthetic groups rendered from several camera angles (a planted 20° elbow error must
  read about 20° from any facing); then two-phone recordings as the answer key for one-phone
  angles, reported in degrees.

## Part 4: diagnosis (fourth)

Definitions, against the group's median member:
- **Timing:** shifting the member in time makes them match (existing timing measure).
- **Occasional difference:** a pose difference no time shift explains, at specific moments,
  larger than this member's own usual difference.
- **Personal signature:** a small, steady difference across the song (this member's baseline).

- Proof on synthetic groups with each kind planted separately (time shift, one-off limb error,
  steady style): each must land in its own category. This proves the sorting, not that a
  choreographer would agree; the docs say so.
- Labels from use: in the choreographer view each flagged moment has "mistake" / "style"
  buttons; clicks are saved locally per group and later used to calibrate.

## Order and sizing

Four separate plans, built in order 1 → 2 → 3 → 4, each shipping something usable on its own.
Each is weeks, not a session. This document is the shared reference; each plan cites it.

## Part 2b: the beat, and dancing together (decided 2026-10-21)

Feedback after using the viewer: most sections say "no reading", the beat of the music should be
the reference for an individual, and the ghost makes no sense during a solo.

**Tested before deciding** (BTS clip, 2D pose, 30 fps): the song's beat grid is found exactly
(120.0 BPM; a half-beat grid every 250 ms holds for the whole song; a beat tracker alone flips
between beat and off-beat). Dancers' movements barely lock to it: landings of all moves fall
within 50 ms of a half beat 39% of the time (40% by chance), move starts 43%, wrist stops 38%;
the beat-frequency pulse in the motion is 1.1x background; motion vs the music's accents
correlates at r = 0.07. Only big moves out of stillness lock somewhat (55%, about 10 per dancer
in 3:50). So per-dancer, per-section timing against the beat would give *fewer* readings than
the group method on this footage. Decided instead:

1. **Dancing together.** At each moment, member i has a group reference only when the other
   members agree on their pose and i is in the same state as them (moving while they move, still
   while they are still). A solo (i moves, the others hold) and split parts (the others disagree)
   both fail. Where it fails: no ghost, and no group timing for i (those windows are not used).
2. **The beat in the viewer.** The beat grid from the video's soundtrack (PyAV to decode, our own
   onset/tempo/grid code), shown as a lane of beat and off-beat ticks around the current time,
   a pulse on each beat, a sound toggle, and each dancer's landings of clean big moves marked
   against the beat (the user chose "the moment the move lands" as on the beat). Markers only,
   no per-section beat score: the evidence doesn't support one on this footage.
3. **Limits stated in the page:** the soundtrack's own sync with the picture shifts every dancer
   equally; a deliberately syncopated landing reads as off the beat; landings are measured on the
   half-beat grid, so anything more than 125 ms off reads as the neighbouring half beat.
4. **Retest later:** per-dancer beat timing on hit-heavy choreography, 60 fps footage, or 3D.
