// SyncScore start page: naming the members in one frame of an upload (the "picking" step).
// start.js calls Picker.show(job, sent) while a job waits for names and Picker.hide() otherwise;
// `sent(id)` is called once the names (or "measure without names") went to the server.
(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const core = window.PickerCore;
  let job = null;        // the upload being named
  let sent = null;
  let frame = 0;
  let people = [];
  let boxes = [];        // core.layout(people), left to right
  let named = [];        // [{person, name}] in the order they were named
  let editing = null;    // the person whose name is being typed
  let loading = 0;       // the latest frame request; older answers are dropped

  async function getJson(url, options) {
    const response = await fetch(url, { cache: "no-store", ...options });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.error || `The server answered ${response.status}.`);
    return data;
  }

  function error(text) {
    $("pick-error").hidden = !text;
    $("pick-error").textContent = text || "";
  }

  function label(person) {
    const n = named.find((x) => x.person === person);
    return n ? n.name : `Person ${boxes.findIndex((b) => b.index === person) + 1}`;
  }

  // ---- drawing
  function render() {
    const buttons = boxes.map((b) => {
      const n = named.find((x) => x.person === b.index);
      const button = document.createElement("button");
      button.type = "button";
      button.className = "person" + (n ? " named" : "") + (editing === b.index ? " editing" : "");
      button.dataset.person = b.index;
      Object.assign(button.style, { left: `${b.left}%`, top: `${b.top}%`, width: `${b.width}%`, height: `${b.height}%`, zIndex: b.z });
      button.setAttribute("aria-label", label(b.index));
      if (n) {
        const tag = document.createElement("span");
        tag.className = "tag";
        tag.textContent = n.name;
        button.append(tag);
      }
      button.addEventListener("click", () => openForm(b.index));
      button.addEventListener("keydown", (event) => {
        if ((event.key === "Delete" || event.key === "Backspace") && n) {
          event.preventDefault();
          remove(b.index);
        }
      });
      return button;
    });
    $("boxes").replaceChildren(...buttons);
    $("named").replaceChildren(...named.map((n) => {
      const li = document.createElement("li");
      const name = document.createElement("span");
      name.textContent = n.name;
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = "Remove";
      button.setAttribute("aria-label", `Remove ${n.name}`);
      button.addEventListener("click", () => remove(n.person));
      li.append(name, button);
      return li;
    }));
    $("named").hidden = named.length === 0;
    $("frame-note").hidden = named.length === 0;
    $("continue").disabled = !core.canContinue(named);
  }

  function focusPerson(person) {
    const button = $("boxes").querySelector(`[data-person="${person}"]`);
    if (button) button.focus();
  }

  // ---- the name form, under the chosen box (above it near the bottom of the frame)
  function openForm(person) {
    const b = boxes.find((x) => x.index === person);
    if (!b) return;
    editing = person;
    error("");
    render();
    const form = $("name-form");
    form.hidden = false;
    const below = b.top + b.height < 75, onRight = b.left + b.width > 60;
    Object.assign(form.style, {
      left: onRight ? "" : `${Math.max(b.left, 0)}%`,
      right: onRight ? `${Math.max(100 - b.left - b.width, 0)}%` : "",
      top: below ? `${b.top + b.height}%` : "",
      bottom: below ? "" : `${Math.max(100 - b.top, 0)}%`,
    });
    const input = $("name-input");
    const n = named.find((x) => x.person === person);
    input.value = n ? n.name : "";
    input.setAttribute("aria-label", `Name for ${label(person)}`);
    input.focus();
    input.select();
  }

  function closeForm() {
    const person = editing;
    editing = null;
    $("name-form").hidden = true;
    render();
    if (person != null) focusPerson(person);
  }

  function remove(person) {
    named = core.removeName(named, person);
    if (editing === person) editing = null;
    $("name-form").hidden = editing == null;
    render();
    const next = $("boxes").querySelector(`[data-person="${person}"]`);
    if (next) next.focus();
  }

  $("name-form").addEventListener("submit", (event) => {
    event.preventDefault();
    const problem = core.nameProblem(named, editing, $("name-input").value);
    if (problem) return error(problem);
    named = core.setName(named, editing, $("name-input").value);
    closeForm();
  });
  $("name-input").addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      event.preventDefault();
      error("");
      closeForm();
    }
  });

  // ---- frames
  async function load(f) {
    const ask = ++loading;
    frame = f;
    named = [];
    editing = null;
    $("name-form").hidden = true;
    error("");
    $("frame-slider").value = f;
    $("frame-time").textContent = core.timeLabel(f, job.picking.fps);
    $("frame-img").src = `/api/uploads/${job.id}/frame/${f}.jpg`;
    $("frame-img").alt = `Frame at ${core.timeLabel(f, job.picking.fps)}`;
    people = [];
    boxes = [];
    render();
    try {
      const found = await getJson(`/api/uploads/${job.id}/people?frame=${f}`);
      if (ask !== loading) return;
      people = found.people;
      boxes = core.layout(people, job.picking.size[0], job.picking.size[1]);
      if (!people.length) error("Nobody was found in this frame. Pick another one.");
    } catch (e) {
      if (ask === loading) error(e.message);
    }
    if (ask === loading) render();
  }

  $("frame-slider").addEventListener("input", () => {
    $("frame-time").textContent = core.timeLabel(Number($("frame-slider").value), job.picking.fps);
  });
  $("frame-slider").addEventListener("change", () => load(Number($("frame-slider").value)));

  // ---- sending
  $("continue").addEventListener("click", async () => {
    if (!core.canContinue(named)) return;
    $("continue").disabled = true;
    try {
      await getJson(`/api/uploads/${job.id}/picks`, {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(core.picksBody(frame, people, named)),
      });
      const id = job.id;
      hide();
      sent(id);
    } catch (e) {
      error(e.message);
      $("continue").disabled = !core.canContinue(named);
    }
  });

  $("no-names").addEventListener("click", async () => {
    try {
      await getJson(`/api/uploads/${job.id}/run?members=${encodeURIComponent(job.members || 0)}`, { method: "POST" });
      const id = job.id;
      hide();
      sent(id);
    } catch (e) {
      error(e.message);
    }
  });

  // ---- shown and hidden by start.js
  function show(waiting, onSent) {
    sent = onSent;
    if (job && job.id === waiting.id && !$("picking").hidden) return; // already naming this one
    job = waiting;
    const p = job.picking;
    $("frame-slider").max = p.n_frames - 1;
    $("suggested").replaceChildren(...p.suggested.map((f) => {
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = core.timeLabel(f, p.fps);
      button.setAttribute("aria-label", `Suggested frame at ${core.timeLabel(f, p.fps)}`);
      button.addEventListener("click", () => load(f));
      return button;
    }));
    $("picking").hidden = false;
    load(p.suggested.length ? p.suggested[0] : Math.floor(p.n_frames / 2));
  }

  function hide() {
    job = null;
    loading++;
    $("picking").hidden = true;
  }

  window.Picker = { show, hide };
})();
