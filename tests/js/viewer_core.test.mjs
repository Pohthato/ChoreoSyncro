// The viewer's drawing logic, without a browser: `node --test tests/js`
import test from "node:test";
import assert from "node:assert/strict";
import core from "../../syncscore/viz/viewer/viewer_core.js";

const NAN17 = () => Array.from({ length: 17 }, () => [NaN, NaN]);

function person(cx, top, height) {
  // 17 COCO joints of a standing person, head at `top`, ankles at `top + height`.
  const y = (f) => top + f * height;
  return [
    [cx, y(0.05)], [cx - 3, y(0.03)], [cx + 3, y(0.03)], [cx - 6, y(0.04)], [cx + 6, y(0.04)],
    [cx - 20, y(0.2)], [cx + 20, y(0.2)], [cx - 25, y(0.35)], [cx + 25, y(0.35)], [cx - 28, y(0.5)], [cx + 28, y(0.5)],
    [cx - 12, y(0.52)], [cx + 12, y(0.52)], [cx - 12, y(0.75)], [cx + 12, y(0.75)], [cx - 12, y(1.0)], [cx + 12, y(1.0)],
  ];
}
const CONF = Array(17).fill(0.9);

test("decodeInt16 round-trips quarter pixels and NaN", () => {
  // encode_int16([0, 12.3456, nan, -731.9, 1919.99]) from syncscore/viz/viewer_data.py
  const v = core.decodeInt16("AAAxAACAkPQAHg==", 4);
  assert.equal(v.length, 5);
  assert.ok(Math.abs(v[1] - 12.3456) <= 0.125);
  assert.ok(Number.isNaN(v[2]));
  assert.ok(Math.abs(v[3] + 731.9) <= 0.125);
  assert.ok(Math.abs(v[4] - 1919.99) <= 0.125);
});

test("frameAt clamps to the analysed range", () => {
  assert.equal(core.frameAt(-1, 30, 100), 0);
  assert.equal(core.frameAt(1.0, 30, 100), 30);
  assert.equal(core.frameAt(99, 30, 100), 99);
});

test("sectionAt finds the section of a time", () => {
  const sections = [{ start_s: 0, end_s: 32 }, { start_s: 32, end_s: 64 }];
  assert.equal(core.sectionAt(10, sections), 0);
  assert.equal(core.sectionAt(40, sections), 1);
  assert.equal(core.sectionAt(80, sections), 1);
});

test("statusOf never shows a number without a reading", () => {
  const s = core.statusOf(12, false, 35, 135);
  assert.equal(s.label, "no reading");
  assert.equal(s.text, "-- no reading");
  assert.equal(s.color, "#898781");
  assert.equal(core.statusOf(null, true, 35, 135).label, "no reading");
});

test("statusOf uses the in-sync and off thresholds", () => {
  assert.equal(core.statusOf(35, true, 35, 135).label, "in sync");
  assert.equal(core.statusOf(36, true, 35, 135).label, "drifting");
  assert.equal(core.statusOf(135, true, 35, 135).label, "off");
  assert.equal(core.statusOf(-40, true, 35, 135).text, "-40 ms drifting");
  assert.equal(core.statusOf(-0.3, true, 35, 135).text, "0 ms in sync");
});

test("skeletonSegments skips bones with an unknown end", () => {
  const pts = person(100, 0, 200);
  const all = core.skeletonSegments(pts, CONF, 0.3, 0);
  assert.equal(all.length, 15); // 13 scored bones + 2 torso sides
  const conf = CONF.slice();
  conf[9] = 0.1; // left wrist unseen: left forearm gone
  assert.equal(core.skeletonSegments(pts, conf, 0.3, 0).length, 14);
  const halo = core.skeletonSegments(pts, CONF, 0.3, 1 << 1); // bit 1 = left_forearm
  assert.equal(halo.filter((s) => s.halo).length, 1);
  assert.equal(core.skeletonSegments(NAN17(), null, 0.3, 0).length, 0);
});

test("followBox holds the last seen box", () => {
  const seen = core.followBox(person(500, 100, 300), CONF, 0.3, null, 0.2);
  assert.equal(seen.seen, true);
  assert.ok(seen.x < 470 && seen.x + seen.w > 530 && seen.h > 300);
  const lost = core.followBox(NAN17(), CONF, 0.3, seen, 0.2);
  assert.equal(lost.seen, false);
  assert.equal(lost.x, seen.x);
  assert.equal(core.followBox(NAN17(), CONF, 0.3, null, 0.2), null);
});

test("lineupTransform puts the feet on the floor line", () => {
  const pts = person(700, 50, 300);
  const t = core.lineupTransform(pts, CONF, 0.3, 120, 400, 150);
  const feetX = (pts[15][0] + pts[16][0]) / 2, feetY = pts[15][1];
  assert.ok(Math.abs(feetX * t.scale + t.dx - 120) < 1e-9);
  assert.ok(Math.abs(feetY * t.scale + t.dy - 400) < 1e-9);
  assert.ok(Math.abs(t.scale - 150 / (feetY - pts[0][1])) < 1e-9);
  assert.equal(core.lineupTransform(NAN17(), CONF, 0.3, 120, 400, 150), null);
});

test("pointsAt reads one member's frame from the flat array", () => {
  const nFrames = 2, flat = new Float32Array(2 * nFrames * 17 * 2);
  flat[((1 * nFrames + 1) * 17 + 9) * 2] = 42; // member 1, frame 1, joint 9, x
  assert.equal(core.pointsAt(flat, nFrames, 1, 1)[9][0], 42);
});

test("advanceClock moves at the playback speed", () => {
  assert.equal(core.advanceClock(6.3, 0.1, 0.25, null, 32, 30).time, 6.3 + 0.025);
  assert.equal(core.advanceClock(6.3, 0.1, 1, null, 32, 30).ended, false);
});

test("advanceClock never steps backwards or jumps", () => {
  assert.equal(core.advanceClock(6.3, -0.02, 1, null, 32, 30).time, 6.3);
  assert.equal(core.advanceClock(6.3, 5, 1, null, 32, 30).time, 6.3 + 0.25); // a hidden tab came back
});

test("advanceClock wraps inside the loop", () => {
  const loop = [5.3, 7.3];
  assert.equal(core.advanceClock(7.25, 0.1, 1, loop, 32, 30).time, 5.3);
  assert.ok(Math.abs(core.advanceClock(6.0, 0.1, 1, loop, 32, 30).time - 6.1) < 1e-12);
  assert.equal(core.wrapLoop(4.0, loop), 5.3); // seeked before the loop
  assert.equal(core.wrapLoop(6.0, loop), null);
  assert.equal(core.wrapLoop(6.0, null), null);
});

test("advanceClock stops at the end", () => {
  const r = core.advanceClock(31.99, 0.1, 1, null, 32, 30);
  assert.equal(r.ended, true);
  assert.ok(r.time < 32);
});

test("nearestBox frames a member from the nearest frame they were seen", () => {
  const seenAt = { 12: person(800, 100, 300), 30: person(200, 100, 300) };
  const frame = (f) => ({ points: seenAt[f] || NAN17(), conf: CONF });
  const box = core.nearestBox(frame, 5, 40, 0.3, 0.25, 15);
  assert.equal(box.seen, false);
  assert.ok(box.x < 800 && box.x + box.w > 800); // from frame 12, the nearer one
  assert.equal(core.nearestBox(frame, 5, 40, 0.3, 0.25, 3), null); // nothing within reach
  assert.equal(core.nearestBox(frame, 12, 40, 0.3, 0.25, 3).seen, true);
});

test("trackTime turns a click on the timeline into the time under it", () => {
  assert.equal(core.trackTime(150, 100, 400, 230), 230 * 50 / 400);
  assert.equal(core.trackTime(50, 100, 400, 230), 0);
  assert.equal(core.trackTime(900, 100, 400, 230), 230);
});

test("lineupTransform keeps a member's own scale, so a squat stays low", () => {
  const standing = person(300, 50, 300), squat = person(300, 140, 210);
  const a = core.lineupTransform(standing, CONF, 0.3, 100, 400, 150, 285);
  const b = core.lineupTransform(squat, CONF, 0.3, 100, 400, 150, 285);
  assert.equal(a.scale, b.scale);
  const top = (pts, t) => pts[0][1] * t.scale + t.dy;
  assert.ok(400 - top(squat, b) < 0.8 * (400 - top(standing, a)));
});

const BEATS = { tempo_bpm: 120, half_beat_s: 0.25, beat_phase_s: 0.1, steady: true };

test("beatTicks lists on- and off-beats in a window", () => {
  const ticks = core.beatTicks(10.0, 11.0, BEATS);
  assert.deepEqual(ticks.map((k) => Math.round(k.t * 1000)), [10100, 10350, 10600, 10850]);
  assert.deepEqual(ticks.map((k) => k.on), [true, false, true, false]);
});

test("beatTicks gives no half beats outside the clip", () => {
  assert.deepEqual(core.beatTicks(-1.0, 1.0, BEATS, 10).map((k) => Math.round(k.t * 1000)), [100, 350, 600, 850]);
  assert.deepEqual(core.beatTicks(9.0, 11.0, BEATS, 10).map((k) => Math.round(k.t * 1000)), [9100, 9350, 9600, 9850]);
});

test("beatTicks gives no half beats where the soundtrack is quiet", () => {
  const quiet = { ...BEATS, quiet: [[4.0, 6.0]] };
  assert.deepEqual(core.beatTicks(3.0, 7.0, quiet, 10).map((k) => Math.round(k.t * 1000)),
    [3100, 3350, 3600, 3850, 6100, 6350, 6600, 6850]);
});

test("beatHeard: inside the clip and not in a quiet stretch", () => {
  const quiet = { ...BEATS, quiet: [[4.0, 6.0]] };
  assert.equal(core.beatHeard(3.9, quiet, 10), true);
  assert.equal(core.beatHeard(5.0, quiet, 10), false);
  assert.equal(core.beatHeard(-0.1, quiet, 10), false);
  assert.equal(core.beatHeard(10.0, quiet, 10), false);
  assert.equal(core.beatHeard(5.0, BEATS, 10), true); // data written before quiet stretches were found
});

test("pulseAt peaks on the beat and fades", () => {
  assert.ok(Math.abs(core.pulseAt(10.1, BEATS, 0.12) - 1) < 1e-9);
  assert.ok(core.pulseAt(10.2, BEATS, 0.12) < 0.5);
  assert.ok(core.pulseAt(10.59, BEATS, 0.12) < 0.05);
  assert.ok(Math.abs(core.pulseAt(10.35, BEATS, 0.12) - Math.exp(-0.25 / 0.12)) < 1e-9); // an off-beat doesn't pulse
});

test("landingStatus uses the in-sync band", () => {
  // Offsets are to the nearest half beat, which may be the "and": the text never says "the beat".
  assert.equal(core.landingStatus(12, 35).color, "#0ca30c");
  assert.equal(core.landingStatus(12, 35).text, "on a half beat (+12 ms)");
  assert.equal(core.landingStatus(-48, 35).text, "48 ms before the nearest half beat");
  assert.equal(core.landingStatus(60, 35).text, "60 ms after the nearest half beat");
  assert.equal(core.landingStatus(60, 35).color, "#fab219");
});

test("landingOffsetText states the offset without a verdict", () => {
  assert.equal(core.landingOffsetText(96), "96 ms after the nearest half beat");
  assert.equal(core.landingOffsetText(-43), "43 ms before the nearest half beat");
  assert.equal(core.landingOffsetText(0.3), "on a half beat");
});

test("nearestDot finds the landing dot under the pointer", () => {
  const dots = [{ x: 100, y: 20, id: "a" }, { x: 112, y: 20, id: "b" }, { x: 100, y: 42, id: "c" }];
  assert.equal(core.nearestDot(dots, 103, 21, 8).id, "a");
  assert.equal(core.nearestDot(dots, 108, 20, 8).id, "b");
  assert.equal(core.nearestDot(dots, 100, 40, 8).id, "c");
  assert.equal(core.nearestDot(dots, 150, 20, 8), null);
  assert.equal(core.nearestDot([], 0, 0, 8), null);
});

test("landingCaveat never states a share it doesn't have", () => {
  assert.equal(core.landingCaveat({ n: 39, within_50ms: 0.385, chance: 0.4, p: 0.13, locked: false, min_n: 20 }),
    "On this video the group's landings don't clearly line up with the beat (39% within 50 ms of a half beat; 40% by chance), so they aren't marked on or off the beat. Watch them with sound.");
  assert.equal(core.landingCaveat({ n: 0, within_50ms: null, chance: 0.4, p: null, locked: false, min_n: 20 }),
    "This video has no clean landings to compare with the beat.");
  assert.equal(core.landingCaveat({ n: 7, within_50ms: 0.57, chance: 0.4, p: 0.2, locked: false, min_n: 20 }),
    "Only 7 clean landings in this video: too few to tell whether they line up with the beat, so they aren't marked on or off it. Watch them with sound.");
  assert.equal(core.landingCaveat({ n: 40, within_50ms: 0.8, chance: 0.4, p: 0.001, locked: true, min_n: 20 }), "");
});

test("landingOrder lists every landing in time order", () => {
  const mine = [{ time_s: 30, offset_ms: 5 }, { time_s: 10, offset_ms: 120 }, { time_s: 20, offset_ms: -60 }];
  assert.deepEqual(core.landingOrder(mine).map((x) => x.time_s), [10, 20, 30]);
  assert.equal(mine[0].time_s, 30);  // the data isn't reordered in place
});
