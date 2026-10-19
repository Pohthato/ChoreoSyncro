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
