// Naming the members in one frame: the logic with no DOM, so it can be unit-tested in Node (tests/js).
// Loaded in the page as a classic script (window.PickerCore) and in Node as a CommonJS module.
// `people` are the boxes the server found in a frame ({row, box: [x1, y1, x2, y2], clear}); `named`
// is [{person, name}] in the order the members were named, `person` an index into `people`.
(function (root) {
  "use strict";

  const MAX_NAME = 40;  // as syncscore/pose/picks.py

  // Each box in percent of the frame, left to right by its centre. Smaller boxes get a higher z,
  // so a person standing in front of (inside the box of) someone bigger can still be clicked.
  function layout(people, imageW, imageH) {
    const area = (p) => (p.box[2] - p.box[0]) * (p.box[3] - p.box[1]);
    const bySize = people.map((p, i) => i).sort((a, b) => area(people[b]) - area(people[a]));
    const z = new Array(people.length);
    bySize.forEach((index, rank) => { z[index] = rank + 1; });
    return people
      .map((p, index) => ({
        index,
        left: (100 * p.box[0]) / imageW,
        top: (100 * p.box[1]) / imageH,
        width: (100 * (p.box[2] - p.box[0])) / imageW,
        height: (100 * (p.box[3] - p.box[1])) / imageH,
        z: z[index],
        centre: p.box[0] + p.box[2],
      }))
      .sort((a, b) => a.centre - b.centre || a.index - b.index)
      .map(({ centre, ...box }) => box);
  }

  // Why `name` can't be given to `person`, or null when it can.
  function nameProblem(named, person, name) {
    const clean = String(name).trim();
    if (!clean) return "Type a name for every member.";
    if (clean.length > MAX_NAME) return `Names can be up to ${MAX_NAME} characters.`;
    const lower = clean.toLowerCase();
    if (named.some((n) => n.person !== person && n.name.toLowerCase() === lower)) return `Another member is already called ${clean}.`;
    return null;
  }

  function setName(named, person, name) {
    const clean = String(name).trim();
    if (named.some((n) => n.person === person)) return named.map((n) => (n.person === person ? { person, name: clean } : n));
    return [...named, { person, name: clean }];
  }

  function removeName(named, person) {
    return named.filter((n) => n.person !== person);
  }

  function picksBody(frame, people, named) {
    return { frame, members: named.map((n) => ({ name: n.name, box: people[n.person].box })) };
  }

  function canContinue(named) {
    return named.length >= 2;
  }

  function timeLabel(frame, fps) {
    const tenths = Math.round((10 * frame) / fps);
    const minutes = Math.floor(tenths / 600);
    const seconds = (tenths - 600 * minutes) / 10;
    return `${minutes}:${seconds.toFixed(1).padStart(4, "0")}`;
  }

  const PickerCore = { MAX_NAME, layout, nameProblem, setName, removeName, picksBody, canContinue, timeLabel };
  if (typeof module !== "undefined" && module.exports) module.exports = PickerCore;
  else root.PickerCore = PickerCore;
})(typeof window !== "undefined" ? window : globalThis);
