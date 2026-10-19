// SyncScore viewer: drawing logic with no DOM, so it can be unit-tested in Node (tests/js).
// Loaded in the page as a classic script (window.SyncCore) and in Node as a CommonJS module.
(function (root) {
  "use strict";

  // The 13 scored bones in syncscore/features/bones.py order (halo bit b = bone b), then the two
  // torso sides that are drawn but not scored. 17 = neck, 18 = pelvis (virtual points).
  const BONES = [
    [5, 7], [7, 9], [6, 8], [8, 10], [11, 13], [13, 15], [12, 14], [14, 16],
    [18, 17], [6, 5], [12, 11], [17, 0], [4, 3],
  ];
  const TORSO_SIDES = [[5, 11], [6, 12]];
  const MISSING = -32768;
  const COLORS = { good: "#0ca30c", warning: "#fab219", critical: "#d03b3b", muted: "#898781" };

  function bytes(b64) {
    const text = atob(b64);
    const out = new Uint8Array(text.length);
    for (let i = 0; i < text.length; i++) out[i] = text.charCodeAt(i);
    return out;
  }

  function decodeInt16(b64, scale) {
    const raw = bytes(b64);
    const view = new DataView(raw.buffer);
    const out = new Float32Array(raw.length / 2);
    for (let i = 0; i < out.length; i++) {
      const v = view.getInt16(2 * i, true);
      out[i] = v === MISSING ? NaN : v / scale;
    }
    return out;
  }

  function decodeUint8(b64) {
    return bytes(b64);
  }

  function decodeUint16(b64) {
    const raw = bytes(b64);
    const view = new DataView(raw.buffer);
    const out = new Uint16Array(raw.length / 2);
    for (let i = 0; i < out.length; i++) out[i] = view.getUint16(2 * i, true);
    return out;
  }

  function frameAt(timeS, fps, nFrames) {
    const f = Math.floor(timeS * fps + 1e-3);
    return Math.min(Math.max(f, 0), nFrames - 1);
  }

  function sectionAt(timeS, sections) {
    let index = 0;
    for (let i = 0; i < sections.length; i++) if (sections[i].start_s <= timeS) index = i;
    return index;
  }

  function signedMs(value) {
    const r = Math.round(value);
    return r === 0 ? "0 ms" : (r > 0 ? "+" : "-") + Math.abs(r) + " ms";
  }

  // Same rule and words as syncscore/viz/skeleton_video.status: a colour never comes without its label.
  function statusOf(offsetMs, reliable, inSyncMs, offSyncMs) {
    if (!reliable || offsetMs === null || offsetMs === undefined || !Number.isFinite(offsetMs)) {
      return { color: COLORS.muted, label: "no reading", text: "-- no reading" };
    }
    const size = Math.abs(offsetMs);
    let color = COLORS.good, label = "in sync";
    if (size > inSyncMs) [color, label] = size < offSyncMs ? [COLORS.warning, "drifting"] : [COLORS.critical, "off"];
    return { color, label, text: signedMs(offsetMs) + " " + label };
  }

  function pointsAt(flat, nFrames, member, frame) {
    const base = (member * nFrames + frame) * 34;
    const out = new Array(17);
    for (let j = 0; j < 17; j++) out[j] = [flat[base + 2 * j], flat[base + 2 * j + 1]];
    return out;
  }

  function confAt(flat, nFrames, member, frame) {
    const base = (member * nFrames + frame) * 17;
    const out = new Array(17);
    for (let j = 0; j < 17; j++) out[j] = flat[base + j] / 255;
    return out;
  }

  function known(points, conf, minConf, j) {
    const p = points[j];
    return Number.isFinite(p[0]) && Number.isFinite(p[1]) && (conf === null || conf[j] >= minConf);
  }

  // Points 0-16 plus neck (17) and pelvis (18), each [x, y] or null when unknown.
  function extended(points, conf, minConf) {
    const ext = points.map((p, j) => (known(points, conf, minConf, j) ? p : null));
    const mid = (a, b) => (ext[a] && ext[b] ? [(ext[a][0] + ext[b][0]) / 2, (ext[a][1] + ext[b][1]) / 2] : null);
    ext.push(mid(5, 6), mid(11, 12));
    return ext;
  }

  function skeletonSegments(points, conf, minConf, haloMask) {
    const ext = extended(points, conf, minConf);
    const out = [];
    BONES.forEach(([a, b], i) => {
      if (ext[a] && ext[b]) out.push({ a: ext[a], b: ext[b], halo: ((haloMask >> i) & 1) === 1 });
    });
    for (const [a, b] of TORSO_SIDES) if (ext[a] && ext[b]) out.push({ a: ext[a], b: ext[b], halo: false });
    return out;
  }

  function followBox(points, conf, minConf, lastBox, pad) {
    const seen = points.filter((p, j) => known(points, conf, minConf, j));
    if (seen.length < 4) return lastBox ? { ...lastBox, seen: false } : null;
    const xs = seen.map((p) => p[0]), ys = seen.map((p) => p[1]);
    const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
    const margin = pad * (y1 - y0);
    return { x: x0 - margin, y: y0 - margin, w: x1 - x0 + 2 * margin, h: y1 - y0 + 2 * margin, seen: true };
  }

  // Scale a member to `targetHeight` (nose to feet) with their feet at (slotX, floorY).
  function lineupTransform(points, conf, minConf, slotX, floorY, targetHeight) {
    const ext = extended(points, conf, minConf);
    const present = ext.slice(0, 17).filter(Boolean);
    if (present.length < 4) return null;
    const ankles = [ext[15], ext[16]].filter(Boolean);
    const feet = ankles.length
      ? [ankles.reduce((s, p) => s + p[0], 0) / ankles.length, ankles.reduce((s, p) => s + p[1], 0) / ankles.length]
      : present.reduce((low, p) => (p[1] > low[1] ? p : low));
    const top = ext[0] ? ext[0][1] : Math.min(...present.map((p) => p[1]));
    const height = feet[1] - top;
    if (!(height > 1)) return null;
    const scale = targetHeight / height;
    return { scale, dx: slotX - scale * feet[0], dy: floorY - scale * feet[1] };
  }

  const SyncCore = {
    BONES, TORSO_SIDES, COLORS, decodeInt16, decodeUint8, decodeUint16, frameAt, sectionAt, signedMs, statusOf,
    pointsAt, confAt, skeletonSegments, followBox, lineupTransform,
  };
  if (typeof module !== "undefined" && module.exports) module.exports = SyncCore;
  else root.SyncCore = SyncCore;
})(typeof window !== "undefined" ? window : globalThis);
