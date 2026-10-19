// SyncScore viewer: the page. Everything it draws was computed by the analysis (viewer_data.js);
// this file only decides what to show and where. Pure drawing logic is in viewer_core.js.
(function () {
  "use strict";
  const D = window.SYNCSCORE;
  const C = window.SyncCore;
  const $ = (id) => document.getElementById(id);

  const M = D.names.length, T = D.n_frames, FPS = D.fps;
  const [IW, IH] = D.image_size;
  // The data marks undrawn joints (undetected, or filled in by smoothing) with confidence 0.
  const MIN_CONF = 0.5 / 255;
  const KP = C.decodeInt16(D.keypoints, 4);
  const GH = C.decodeInt16(D.ghost, 4);
  const CF = C.decodeUint8(D.conf);
  const HALO = C.decodeUint16(D.halo);
  const DURATION = T / FPS;

  const state = {
    view: "group", member: 0, video: true, ghost: true, lineup: false, speed: 1,
    playing: false, time: 0, loop: null, videoOk: false, box: null,
  };

  // ---- initial state from the URL: ?t=12.5&view=dancer&member=M3&video=0&ghost=1&lineup=1&speed=0.25&loop=1&play=1
  const q = new URLSearchParams(location.search);
  if (q.has("t")) state.time = Math.min(Math.max(parseFloat(q.get("t")) || 0, 0), DURATION);
  if (q.get("view") === "dancer") state.view = "dancer";
  if (q.has("member")) {
    const m = q.get("member"), byName = D.names.indexOf(m);
    state.member = byName >= 0 ? byName : Math.min(Math.max(parseInt(m, 10) || 0, 0), M - 1);
  }
  if (q.get("video") === "0") state.video = false;
  if (q.get("ghost") === "0") state.ghost = false;
  if (q.get("lineup") === "1") state.lineup = true;
  if (["1", "0.5", "0.25"].includes(q.get("speed"))) state.speed = parseFloat(q.get("speed"));
  const startLooping = q.get("loop") === "1", startPlaying = q.get("play") === "1";

  const video = $("video"), canvas = $("canvas"), ctx = canvas.getContext("2d");
  const lineup = $("lineup"), lctx = lineup.getContext("2d");

  // ---- the clock: the video drives time when it can play, otherwise an internal clock does
  function now() {
    return state.videoOk ? video.currentTime : state.time;
  }
  function seek(t) {
    t = Math.min(Math.max(t, 0), DURATION - 1 / FPS);
    state.time = t;
    if (state.videoOk) video.currentTime = t;
    draw();
  }
  function setPlaying(on) {
    state.playing = on;
    $("play").textContent = on ? "Pause" : "Play";
    if (state.videoOk) on ? video.play().catch(() => {}) : video.pause();
  }
  function setSpeed(s) {
    state.speed = s;
    video.playbackRate = s;
    document.querySelectorAll("[data-speed]").forEach((b) => b.setAttribute("aria-pressed", String(parseFloat(b.dataset.speed) === s)));
  }
  function setLoop(range) {
    state.loop = range;
    $("loop").setAttribute("aria-pressed", String(!!range));
  }

  if (D.video) {
    video.src = D.video;
    video.addEventListener("loadeddata", () => {
      state.videoOk = true;
      video.playbackRate = state.speed;
      video.currentTime = state.time;
      draw();
    });
    video.addEventListener("error", () => showNote("Video not found: " + decodeURI(D.video.replace(/^file:\/\/\//, "")) + ". Showing skeletons only."));
    video.addEventListener("seeked", draw);
  } else {
    showNote("No video for this analysis. Showing skeletons only.");
  }
  function showNote(text) {
    state.videoOk = false;
    $("note").textContent = text;
    $("note").hidden = false;
  }

  // ---- geometry
  function fit(box, w, h) {
    // Image rectangle `box` -> canvas of size w x h, centred, aspect kept.
    const s = Math.min(w / box.w, h / box.h);
    return { s, ox: (w - box.w * s) / 2 - box.x * s, oy: (h - box.h * s) / 2 - box.y * s };
  }
  function toStage(view) {
    if (state.view === "group") return fit({ x: 0, y: 0, w: IW, h: IH }, view.w, view.h);
    const f = C.frameAt(now(), FPS, T);
    const memberAt = (k) => ({ points: C.pointsAt(KP, T, state.member, k), conf: C.confAt(CF, T, state.member, k) });
    // Unseen with no box yet (the view just opened on a hidden member): frame them where they were last or next seen.
    const target = C.followBox(memberAt(f).points, memberAt(f).conf, MIN_CONF, state.box, 0.25)
      || C.nearestBox(memberAt, f, T, MIN_CONF, 0.25, Math.round(10 * FPS));
    if (target) {
      // Grow the box to the stage's shape, then ease toward it so the camera doesn't shake.
      const aspect = view.w / view.h;
      let { x, y, w, h } = target;
      if (w / h < aspect) { const nw = h * aspect; x -= (nw - w) / 2; w = nw; } else { const nh = w / aspect; y -= (nh - h) / 2; h = nh; }
      const prev = state.box && state.box.w ? state.box : { x, y, w, h };
      const k = state.playing ? 0.15 : 1;
      state.box = { x: prev.x + k * (x - prev.x), y: prev.y + k * (y - prev.y), w: prev.w + k * (w - prev.w), h: prev.h + k * (h - prev.h), seen: target.seen };
    }
    return fit(state.box || { x: 0, y: 0, w: IW, h: IH }, view.w, view.h);
  }

  function sizeCanvas(c) {
    const r = c.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
    const w = Math.max(1, Math.round(r.width * dpr)), h = Math.max(1, Math.round(r.height * dpr));
    if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
    return { w, h, dpr };
  }

  function statusAt(member, t) {
    const s = D.section_status[member][C.sectionAt(t, D.sections)];
    return C.statusOf(s.offset_ms, s.reliable, D.in_sync_ms, D.off_sync_ms);
  }

  function drawSegments(g, segments, map, color, width, alpha) {
    g.globalAlpha = alpha;
    g.strokeStyle = color;
    g.lineWidth = width;
    g.lineCap = "round";
    g.beginPath();
    for (const s of segments) {
      g.moveTo(s.a[0] * map.s + map.ox, s.a[1] * map.s + map.oy);
      g.lineTo(s.b[0] * map.s + map.ox, s.b[1] * map.s + map.oy);
    }
    g.stroke();
    g.globalAlpha = 1;
  }

  function drawMember(g, member, f, map, unit, status, withLabel) {
    const pts = C.pointsAt(KP, T, member, f), conf = C.confAt(CF, T, member, f);
    const segs = C.skeletonSegments(pts, conf, MIN_CONF, HALO[member * T + f]);
    if (!segs.length) return null;
    drawSegments(g, segs.filter((s) => s.halo), map, "#e87ba4", 9 * unit, 0.85);
    drawSegments(g, segs, map, status.color, 3.5 * unit, 1);
    // The ghost goes on top, thin: along a limb it reads as "matching", beside it as the gap.
    if (state.ghost) drawSegments(g, C.skeletonSegments(C.pointsAt(GH, T, member, f), null, 0, 0), map, "#ffffff", 1.6 * unit, 0.8);
    if (conf[0] >= MIN_CONF && Number.isFinite(pts[0][0])) {
      g.fillStyle = status.color;
      g.beginPath();
      g.arc(pts[0][0] * map.s + map.ox, pts[0][1] * map.s + map.oy, 5 * unit, 0, 2 * Math.PI);
      g.fill();
    }
    if (!withLabel) return null;
    const ys = pts.filter((p, j) => conf[j] >= MIN_CONF && Number.isFinite(p[1])).map((p) => p[1]);
    const xs = pts.filter((p, j) => conf[j] >= MIN_CONF && Number.isFinite(p[0])).map((p) => p[0]);
    return { x: (xs.reduce((a, b) => a + b, 0) / xs.length) * map.s + map.ox, y: Math.min(...ys) * map.s + map.oy, name: D.names[member], text: status.text };
  }

  function drawLabels(g, labels, unit) {
    // Name and status above each head; a label that would overlap one already placed moves up a line.
    const placed = [];
    g.textAlign = "center";
    for (const l of labels.sort((a, b) => a.x - b.x)) {
      g.font = `600 ${14 * unit}px ${getComputedStyle(document.body).fontFamily}`;
      const half = Math.max(g.measureText(l.name).width, g.measureText(l.text).width) / 2 + 6 * unit;
      let line = 0;
      while (placed.some((p) => p.line === line && l.x - half < p.r && l.x + half > p.l)) line++;
      placed.push({ l: l.x - half, r: l.x + half, line });
      const y = Math.max(34 * unit, l.y - 16 * unit - line * 40 * unit);
      g.fillStyle = "#ffffff";
      g.fillText(l.name, l.x, y - 17 * unit);
      g.font = `${13 * unit}px ${getComputedStyle(document.body).fontFamily}`;
      g.fillStyle = "#c3c2b7";
      g.fillText(l.text, l.x, y);
    }
  }

  function draw() {
    const t = now(), f = C.frameAt(t, FPS, T);
    const view = sizeCanvas(canvas), unit = view.dpr;
    ctx.fillStyle = "#1a1a19";
    ctx.fillRect(0, 0, view.w, view.h);
    const map = toStage(view);
    if (state.video && state.videoOk && video.readyState >= 2) {
      ctx.drawImage(video, map.ox, map.oy, IW * map.s, IH * map.s);
    }
    const labels = [];
    const members = state.view === "group" ? [...Array(M).keys()] : [state.member];
    const scale = state.view === "group" ? unit : unit * 1.6;
    for (const i of members) {
      const label = drawMember(ctx, i, f, map, scale, statusAt(i, t), true);
      if (label) labels.push(label);
    }
    drawLabels(ctx, labels, unit);
    if (state.view === "dancer" && !C.followBox(C.pointsAt(KP, T, state.member, f), C.confAt(CF, T, state.member, f), MIN_CONF, null, 0)) {
      ctx.fillStyle = "#c3c2b7";
      ctx.font = `${14 * unit}px ${getComputedStyle(document.body).fontFamily}`;
      ctx.textAlign = "left";
      ctx.fillText(`${D.names[state.member]}: not seen`, 14 * unit, 26 * unit);
    }
    if (state.lineup) drawLineup(f, t);
    updatePanels(t);
  }

  function drawLineup(f, t) {
    const view = sizeCanvas(lineup), unit = view.dpr;
    lctx.fillStyle = "#1a1a19";
    lctx.fillRect(0, 0, view.w, view.h);
    const floor = view.h - 34 * unit, height = view.h - 70 * unit, slot = view.w / M;
    lctx.strokeStyle = "#52514e";
    lctx.lineWidth = unit;
    lctx.beginPath();
    lctx.moveTo(8 * unit, floor);
    lctx.lineTo(view.w - 8 * unit, floor);
    lctx.stroke();
    lctx.textAlign = "center";
    for (let i = 0; i < M; i++) {
      const cx = slot * (i + 0.5);
      const pts = C.pointsAt(KP, T, i, f), conf = C.confAt(CF, T, i, f);
      const tr = C.lineupTransform(pts, conf, MIN_CONF, cx, floor, height);
      const status = statusAt(i, t);
      lctx.font = `${13 * unit}px ${getComputedStyle(document.body).fontFamily}`;
      lctx.fillStyle = "#c3c2b7";
      lctx.fillText(tr ? D.names[i] : `${D.names[i]} not seen`, cx, view.h - 12 * unit);
      if (!tr) continue;
      const map = { s: tr.scale, ox: tr.dx, oy: tr.dy };
      const segs = C.skeletonSegments(pts, conf, MIN_CONF, HALO[i * T + f]);
      drawSegments(lctx, segs.filter((s) => s.halo), map, "#e87ba4", 7 * unit, 0.85);
      drawSegments(lctx, segs, map, status.color, 3 * unit, 1);
      if (state.ghost) drawSegments(lctx, C.skeletonSegments(C.pointsAt(GH, T, i, f), null, 0, 0), map, "#ffffff", 1.4 * unit, 0.8);
    }
  }

  // ---- side panel, timeline cursor, clock
  function clockText(t) {
    const m = Math.floor(t / 60), s = t - 60 * m;
    return `${m}:${s.toFixed(2).padStart(5, "0")}`;
  }
  function updatePanels(t) {
    $("clock").textContent = `${clockText(t)} / ${clockText(DURATION)}, frame ${C.frameAt(t, FPS, T)}`;
    cursor.style.left = `calc(54px + (100% - 54px) * ${t / DURATION})`;
    if (state.view === "group") {
      D.names.forEach((name, i) => {
        const s = statusAt(i, t);
        nowItems[i].querySelector(".status").innerHTML = `<span class="chip" style="--c:${s.color}"></span>${s.text}`;
      });
    } else {
      $("dancer-now").textContent = `This section: ${statusAt(state.member, t).text}`;
    }
  }

  // ---- building the panels once
  const nowList = $("now");
  const nowItems = D.names.map((name, i) => {
    const li = document.createElement("li");
    li.innerHTML = `<button type="button"><span>${name}</span><span class="status"></span></button>`;
    li.querySelector("button").addEventListener("click", () => setView("dancer", i));
    nowList.appendChild(li);
    return li;
  });

  const select = $("member");
  D.names.forEach((name, i) => select.add(new Option(name, String(i))));
  select.addEventListener("change", () => setView("dancer", parseInt(select.value, 10)));

  const timeline = $("timeline");
  const rows = D.names.map((name, i) => {
    const row = document.createElement("div");
    row.className = "row";
    row.innerHTML = `<span class="name">${name}</span><div class="track"></div>`;
    const track = row.querySelector(".track");
    D.sections.forEach((sec, k) => {
      const st = D.section_status[i][k];
      const s = C.statusOf(st.offset_ms, st.reliable, D.in_sync_ms, D.off_sync_ms);
      const tape = document.createElement("button");
      tape.type = "button";
      tape.className = "tape" + (st.reliable ? "" : " none");
      tape.style.setProperty("--c", s.color);
      tape.style.left = `calc(${(100 * sec.start_s) / DURATION}% + 1px)`;
      tape.style.width = `calc(${(100 * (sec.end_s - sec.start_s)) / DURATION}% - 2px)`;
      const range = `${clockText(sec.start_s)}-${clockText(sec.end_s)}`;
      tape.title = `${name}, ${range}: ${s.text}`;
      tape.setAttribute("aria-label", tape.title);
      // A click lands where it was aimed; Enter or Space on a strip (no pointer) goes to the section's start.
      tape.addEventListener("click", (e) => {
        e.stopPropagation();
        if (e.detail === 0) seek(sec.start_s);
        else seek(C.trackTime(e.clientX, track.getBoundingClientRect().left, track.getBoundingClientRect().width, DURATION));
      });
      track.appendChild(tape);
    });
    track.addEventListener("click", (e) => {
      const r = track.getBoundingClientRect();
      seek(C.trackTime(e.clientX, r.left, r.width, DURATION));
    });
    timeline.appendChild(row);
    return row;
  });
  const cursor = document.createElement("div");
  cursor.className = "cursor";
  timeline.appendChild(cursor);

  function moment(list, item, describe) {
    const li = document.createElement("li");
    const b = document.createElement("button");
    b.type = "button";
    b.innerHTML = describe(item);
    b.addEventListener("click", () => {
      // One second either side, slowed down and looping, so the moment can be studied.
      setLoop([Math.max(item.time_s - 1, 0), Math.min(item.time_s + 1, DURATION)]);
      setSpeed(0.25);
      seek(Math.max(item.time_s - 1, 0));
      setPlaying(true);
    });
    li.appendChild(b);
    list.appendChild(li);
  }
  function fillMoments() {
    const m = D.moments[state.member];
    $("dancer-name").textContent = D.names[state.member];
    const timing = $("timing-moments"), shape = $("shape-moments");
    timing.innerHTML = shape.innerHTML = "";
    if (!m.worst_timing_moments.length) timing.innerHTML = `<li class="empty">No timing moments: no window measured reliably.</li>`;
    m.worst_timing_moments.forEach((it) => moment(timing, it, (x) =>
      `${clockText(x.time_s)}  ${C.signedMs(x.offset_ms)}${x.part_furthest_off ? `<small>${x.part_furthest_off} furthest off</small>` : ""}`));
    if (!m.worst_shape_moments.length) shape.innerHTML = `<li class="empty">No pose difference held beyond the group.</li>`;
    m.worst_shape_moments.forEach((it) => moment(shape, it, (x) =>
      `${clockText(x.time_s)}  ${x.bone.replace(/_/g, " ")} ${Math.round(x.error_deg)}°<small>held ${Math.round(x.above_group_deg)}° beyond the group's usual difference</small>`));
    if (m.likely_label_swaps) {
      const li = document.createElement("li");
      li.className = "empty";
      li.textContent = `Not listed: ${m.likely_label_swaps} s where a bone points 90° or more away from the group's. That is usually the pose model swapping left and right, or a turn; check those by eye.`;
      shape.appendChild(li);
    }
  }

  function setView(view, member) {
    state.view = view;
    if (member !== undefined) state.member = member;
    state.box = null;
    select.value = String(state.member);
    $("view-group").setAttribute("aria-pressed", String(view === "group"));
    $("view-dancer").setAttribute("aria-pressed", String(view === "dancer"));
    $("side-group").hidden = view !== "group";
    $("side-dancer").hidden = view !== "dancer";
    rows.forEach((r, i) => r.classList.toggle("selected", view === "dancer" && i === state.member));
    if (view === "dancer") fillMoments();
    draw();
  }

  function toggle(id, key) {
    const b = $(id);
    b.setAttribute("aria-pressed", String(state[key]));
    b.addEventListener("click", () => {
      state[key] = !state[key];
      b.setAttribute("aria-pressed", String(state[key]));
      if (key === "lineup") lineup.hidden = !state.lineup;
      draw();
    });
  }

  // ---- controls
  $("legend-in").textContent = `within ${D.in_sync_ms} ms of the group`;
  $("legend-off").textContent = `${D.off_sync_ms} ms or more`;
  $("legend-halo").textContent = `pose more than ${D.shape_threshold_deg}° beyond the group's usual difference`;
  $("view-group").addEventListener("click", () => setView("group"));
  $("view-dancer").addEventListener("click", () => setView("dancer"));
  $("play").addEventListener("click", () => setPlaying(!state.playing));
  const step = (d) => {
    setPlaying(false);
    seek((C.frameAt(now(), FPS, T) + d) / FPS);
  };
  $("back").addEventListener("click", () => step(-1));
  $("forward").addEventListener("click", () => step(1));
  document.querySelectorAll("[data-speed]").forEach((b) => b.addEventListener("click", () => setSpeed(parseFloat(b.dataset.speed))));
  $("loop").addEventListener("click", () => {
    const t = now();
    setLoop(state.loop ? null : [Math.max(t - 1, 0), Math.min(t + 1, DURATION)]);
  });
  toggle("show-video", "video");
  toggle("show-ghost", "ghost");
  toggle("show-lineup", "lineup");
  lineup.hidden = !state.lineup;
  document.addEventListener("keydown", (e) => {
    if (e.target.tagName === "SELECT") return;
    if (e.key === " ") { e.preventDefault(); setPlaying(!state.playing); }
    else if (e.key === "ArrowLeft") { e.preventDefault(); step(-1); }
    else if (e.key === "ArrowRight") { e.preventDefault(); step(1); }
  });
  window.addEventListener("resize", draw);

  // ---- the loop: advance the internal clock when there's no video, honour the loop range, draw
  let last = performance.now();
  function tick(ms) {
    const dt = (ms - last) / 1000;
    last = ms;
    if (state.playing) {
      if (state.videoOk) {
        const wrapped = C.wrapLoop(video.currentTime, state.loop);
        if (wrapped !== null) seek(wrapped);
      } else {
        const step = C.advanceClock(state.time, dt, state.speed, state.loop, DURATION, FPS);
        state.time = step.time;
        if (step.ended) setPlaying(false);
      }
      draw();
    }
    requestAnimationFrame(tick);
  }

  setSpeed(state.speed);
  setView(state.view, state.member);
  if (startLooping) setLoop([Math.max(state.time - 1, 0), Math.min(state.time + 1, DURATION)]);
  if (startPlaying) setPlaying(true);
  requestAnimationFrame(tick);
})();
