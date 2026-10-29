// Naming the members in one frame, without a browser: `node --test tests/js`
import test from "node:test";
import assert from "node:assert/strict";
import core from "../../syncscore/serve/page/picker_core.js";

const person = (x1, y1, x2, y2) => ({ row: 0, box: [x1, y1, x2, y2], clear: true });

test("layout gives each box in percent of the frame, left to right", () => {
  const people = [person(320, 90, 480, 360), person(0, 0, 160, 180)];
  const boxes = core.layout(people, 640, 360);
  assert.deepEqual(boxes.map((b) => b.index), [1, 0]);
  assert.deepEqual([boxes[1].left, boxes[1].top, boxes[1].width, boxes[1].height], [50, 25, 25, 75]);
});

test("layout puts a small box inside a big one on top", () => {
  const boxes = core.layout([person(100, 0, 500, 360), person(250, 100, 300, 200)], 640, 360);
  const big = boxes.find((b) => b.index === 0), small = boxes.find((b) => b.index === 1);
  assert.ok(small.z > big.z);
});

test("a name must be typed, short enough, and not another person's", () => {
  const named = [{ person: 0, name: "Karina" }];
  assert.equal(core.nameProblem(named, 1, "  "), "Type a name for every member.");
  assert.equal(core.nameProblem(named, 1, "x".repeat(41)), "Names can be up to 40 characters.");
  assert.equal(core.nameProblem(named, 1, "x".repeat(40)), null);
  assert.equal(core.nameProblem(named, 1, "karina"), "Another member is already called karina.");
  assert.equal(core.nameProblem(named, 0, "KARINA"), null);  // renaming her own box
});

test("renaming keeps the order and a new name goes last", () => {
  let named = core.setName([], 2, " Winter ");
  named = core.setName(named, 0, "Karina");
  named = core.setName(named, 2, "Ningning");
  assert.deepEqual(named, [{ person: 2, name: "Ningning" }, { person: 0, name: "Karina" }]);
});

test("removing a name leaves the others in order", () => {
  const named = [{ person: 2, name: "A" }, { person: 0, name: "B" }, { person: 1, name: "C" }];
  assert.deepEqual(core.removeName(named, 0), [{ person: 2, name: "A" }, { person: 1, name: "C" }]);
  assert.deepEqual(core.removeName(named, 5), named);
});

test("picksBody follows pick order, not left to right", () => {
  const people = [person(0, 0, 10, 10), person(20, 0, 30, 10), person(40, 0, 50, 10)];
  const body = core.picksBody(77, people, [{ person: 2, name: "Giselle" }, { person: 0, name: "Karina" }]);
  assert.deepEqual(body, { frame: 77, members: [{ name: "Giselle", box: [40, 0, 50, 10] }, { name: "Karina", box: [0, 0, 10, 10] }] });
});

test("continuing needs two names", () => {
  assert.equal(core.canContinue([{ person: 0, name: "A" }]), false);
  assert.equal(core.canContinue([{ person: 0, name: "A" }, { person: 1, name: "B" }]), true);
});

test("timeLabel shows minutes, seconds and tenths", () => {
  assert.equal(core.timeLabel(1215, 30), "0:40.5");
  assert.equal(core.timeLabel(0, 30), "0:00.0");
  assert.equal(core.timeLabel(3899, 30), "2:10.0");
  assert.equal(core.timeLabel(1799, 30), "1:00.0");  // 59.97 s, not "0:60.0"
});
