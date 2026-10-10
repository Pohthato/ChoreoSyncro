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
