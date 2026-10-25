// The page script (viewer.js) on a stand-in DOM, without a browser: `node --test tests/js`
import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import core from "../../syncscore/viz/viewer/viewer_core.js";

const PAGE = readFileSync(new URL("../../syncscore/viz/viewer/viewer.js", import.meta.url), "utf8");

// A 2D context that records every drawing call; style properties are plain fields.
function context() {
  const calls = [];
  return new Proxy({ calls }, {
    get: (target, key) => (key in target ? target[key] : key === "measureText" ? () => ({ width: 40 }) : (...args) => calls.push([key, ...args])),
  });
}

function element() {
  const listeners = {}, ctx = context();
  return {
    listeners, ctx, style: { setProperty() {} }, dataset: {}, classList: { toggle() {} }, attributes: {},
    hidden: false, disabled: false, textContent: "", innerHTML: "", title: "", value: "", width: 0, height: 0,
    currentTime: 0, readyState: 0, muted: true, playbackRate: 1,
    addEventListener(type, fn) { (listeners[type] ||= []).push(fn); },
    setAttribute(name, value) { this.attributes[name] = value; },
    getContext: () => ctx,
    getBoundingClientRect: () => ({ left: 0, top: 0, width: 800, height: 400 }),
    querySelector: () => element(),
    appendChild() {}, add() {}, play: () => Promise.resolve(), pause() {},
  };
}

// Open the page on `data` (as viewer_data.py writes it) with the URL query `search`; returns $(id).
function openPage(data, search = "") {
  const ids = new Map();
  const $ = (id) => (ids.has(id) ? ids.get(id) : ids.set(id, element()).get(id));
  const document = { getElementById: $, querySelectorAll: () => [], createElement: () => element(), addEventListener() {}, body: {} };
  const window = { SYNCSCORE: data, SyncCore: core, devicePixelRatio: 1, addEventListener() {} };
  new Function("window", "document", "location", "getComputedStyle", "Option", "requestAnimationFrame", PAGE)(
    window, document, { search }, () => ({ fontFamily: "sans-serif" }), function Option() {}, () => {});
  return $;
}

// One member, never seen (no skeleton is drawn), `seconds` long at 30 fps.
function analysis({ seconds = 10, video = null, beats = null } = {}) {
  const fps = 30, frames = seconds * fps;
  const missing = Buffer.alloc(frames * 17 * 2 * 2);
  for (let i = 0; i < missing.length; i += 2) missing.writeInt16LE(-32768, i);
  return {
    version: 1, fps, n_frames: frames, image_size: [1920, 1080], names: ["M1"], in_sync_ms: 35, off_sync_ms: 135,
    shape_threshold_deg: 15, video, sections: [{ name: "Section 1", start_s: 0, end_s: seconds }],
    section_status: [[{ offset_ms: null, low_ms: null, high_ms: null, reliable: false }]],
    keypoints: missing.toString("base64"), ghost: missing.toString("base64"),
    conf: Buffer.alloc(frames * 17).toString("base64"), halo: Buffer.alloc(frames * 2).toString("base64"),
    heights: [0], together: Buffer.alloc(frames, 1).toString("base64"), apart: [[]],
    beats, beat_note: beats ? "Beat: 120 BPM from the soundtrack." : "No beat: no video was given, so there is no soundtrack.",
    landings: beats ? [[]] : null,
    landing_lock: beats ? { n: 0, within_50ms: null, chance: 0.4, p: null, locked: false, min_n: 20 } : null,
    moments: [{ worst_timing_moments: [], worst_shape_moments: [], likely_label_swaps: 0 }],
  };
}

const BEATS = { tempo_bpm: 120, half_beat_s: 0.25, beat_phase_s: 0.1, steady: true };
const count = (ctx, name) => ctx.calls.filter((call) => call[0] === name).length;
// The lane strokes each half beat on its own, then the line marking now.
const halfBeatsDrawn = ($) => count($("beat-lane").ctx, "stroke") - 1;
// With no skeleton on the stage, its only circle is the beat pulse.
const pulseDrawn = ($) => count($("canvas").ctx, "arc") === 1;

test("the beat lane shows half beats only inside the clip", () => {
  // 1.5 s either side of now: at 0.2 s the half beats from 0.1 to 1.6 s, at 9.9 s from 8.6 to 9.85 s.
  assert.equal(halfBeatsDrawn(openPage(analysis({ beats: BEATS }), "?t=0.2")), 7);
  assert.equal(halfBeatsDrawn(openPage(analysis({ beats: BEATS }), "?t=9.9")), 6);
});

test("no beat lane ticks and no pulse where the soundtrack is quiet", () => {
  const data = analysis({ beats: { ...BEATS, quiet: [[6.0, 10.0]] } });
  const quiet = openPage(data, "?t=8.0");
  assert.equal(halfBeatsDrawn(quiet), 0);
  assert.equal(pulseDrawn(quiet), false);
  const music = openPage(data, "?t=3.0");
  assert.equal(halfBeatsDrawn(music), 12);
  assert.equal(pulseDrawn(music), true);
});

test("Sound is disabled whenever there is no video to hear", () => {
  assert.equal(openPage(analysis())("sound").disabled, true);
  const $ = openPage(analysis({ video: "file:///C:/clips/missing.mp4" }));
  assert.equal($("sound").disabled, false);
  $("video").listeners.error.forEach((fn) => fn());
  assert.equal($("note").hidden, false);
  assert.equal($("sound").disabled, true);
});
