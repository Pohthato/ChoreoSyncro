// SyncScore start page: upload a video, follow its analysis, open its viewer when it is done.
(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const STEPS = ["upload", "pose", "measure", "view"];
  let watching = null;       // the upload shown in the job panel
  let sawUnfinished = false; // go to the viewer only after watching it finish, never on a reload
  let timer = null;

  // ---- the job panel
  function minutes(seconds) {
    if (seconds < 60) return "under a minute";
    const m = Math.round(seconds / 60);
    return m === 1 ? "about 1 min" : `about ${m} min`;
  }

  function setSteps(current, allDone) {
    const at = STEPS.indexOf(current);
    for (const li of $("steps").children) {
      const i = STEPS.indexOf(li.dataset.step);
      li.classList.toggle("done", allDone || (at >= 0 && i < at));
      li.classList.toggle("now", !allDone && i === at);
    }
  }

  function setBar(fraction) {
    const bar = $("bar");
    bar.hidden = false;
    if (fraction == null) bar.removeAttribute("value");
    else bar.value = fraction;
  }

  function showError(text) {
    $("job-error").hidden = !text;
    $("job-error").textContent = text || "";
  }

  function showUploading(name, sent, total) {
    $("job").hidden = false;
    $("job-name").textContent = name;
    setSteps("upload", false);
    setBar(total ? sent / total : null);
    $("detail").textContent = total ? `Uploading: ${(sent / 1e6).toFixed(0)} of ${(total / 1e6).toFixed(0)} MB` : "Uploading";
    showError("");
    $("log-box").hidden = true;
  }

  function showJob(job) {
    $("job").hidden = false;
    $("job-name").textContent = job.name;
    const done = job.status === "done", failed = job.status === "failed";
    setSteps(job.status === "queued" ? "pose" : job.step, done);
    if (job.status === "queued") $("steps").querySelector('[data-step="pose"]').classList.remove("now");
    if (done) {
      $("bar").hidden = true;
      const link = document.createElement("a");
      link.href = job.viewer;
      link.textContent = "Open the viewer";
      $("detail").replaceChildren(link);
    } else if (failed) {
      $("bar").hidden = true;
      $("detail").textContent = "";
    } else {
      const p = job.progress;
      const fraction = p && p[1] > 0 ? Math.min(1, p[0] / p[1]) : null;
      setBar(fraction);
      let detail = job.detail || "";
      if (p) {
        detail += `: frame ${p[0].toLocaleString()} of ${p[1].toLocaleString()}`;
        const elapsed = job.step_elapsed_s;
        if (fraction > 0.02 && elapsed > 5) detail += `, ${minutes(elapsed * (1 - fraction) / fraction)} left`;
      }
      $("detail").textContent = detail;
    }
    let error = failed ? job.error || "It stopped." : "";
    if (failed && /need at least 2|tracking\.members/.test(error)) error += " Set Members in its row below and run it again.";
    else if (failed) error += " You can run it again from its row below.";
    showError(error);
    const log = job.log || [];
    $("log-box").hidden = log.length === 0;
    if (failed && log.length) $("log-box").open = true;
    $("log").textContent = log.join("\n");
  }

  // ---- earlier uploads
  function statusText(job) {
    if (job.status === "done") return `Done, ${job.members ? job.members + " members" : "members counted from the video"}`;
    if (job.status === "failed") return "Failed: " + (job.error || "it stopped");
    if (job.status === "queued") return "Waiting its turn";
    const p = job.progress;
    return job.detail + (p && p[1] ? ` (${Math.floor((100 * p[0]) / p[1])}%)` : "");
  }

  function showList(jobs) {
    $("empty").hidden = jobs.length > 0;
    const rows = jobs.map((job) => {
      const row = $("row").content.firstElementChild.cloneNode(true);
      const name = row.querySelector(".name");
      name.textContent = job.name;
      if (job.viewer) name.href = job.viewer;
      else {
        name.href = "#" + job.id;
        name.addEventListener("click", (event) => { event.preventDefault(); watch(job.id); });
      }
      row.querySelector(".when").textContent = new Date(job.created * 1000).toLocaleString();
      const status = row.querySelector(".status");
      status.textContent = statusText(job);
      status.classList.toggle("failed", job.status === "failed");
      const again = row.querySelector(".again");
      if (job.status === "done" || job.status === "failed") {
        const input = again.querySelector("input");
        input.value = job.members;
        input.setAttribute("aria-label", `Members for ${job.name}`);
        again.addEventListener("submit", (event) => { event.preventDefault(); runAgain(job.id, input.value); });
      } else again.remove();
      return row;
    });
    // Leave the list alone while someone is typing a member count in it.
    if (!document.activeElement || !document.activeElement.matches("#list input")) $("list").replaceChildren(...rows);
  }

  async function getJson(url, options) {
    const response = await fetch(url, { cache: "no-store", ...options });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.error || `The server answered ${response.status}.`);
    return data;
  }

  // ---- polling: the watched job every second, the list with it
  async function tick() {
    clearTimeout(timer);
    let busy = false;
    try {
      const { uploads } = await getJson("/api/uploads");
      showList(uploads);
      busy = uploads.some((j) => j.status === "running" || j.status === "queued");
      if (watching) {
        const job = await getJson("/api/uploads/" + watching);
        showJob(job);
        if (job.status === "done" && sawUnfinished) {
          watching = null;
          history.replaceState(null, "", location.pathname); // Back from the viewer must not come straight here again
          location.assign(job.viewer);
          return;
        }
        if (job.status === "running" || job.status === "queued") sawUnfinished = true;
      }
    } catch (error) {
      showError(`Lost touch with the server (${error.message}). Is \`syncscore serve\` still running?`);
    }
    timer = setTimeout(tick, watching || busy ? 1000 : 5000);
  }

  function watch(id) {
    watching = id;
    sawUnfinished = false;
    history.replaceState(null, "", "#" + id);
    tick();
  }

  async function runAgain(id, members) {
    try {
      await getJson(`/api/uploads/${id}/run?members=${encodeURIComponent(members)}`, { method: "POST" });
      watch(id);
    } catch (error) {
      $("job").hidden = false;
      showError(error.message);
    }
  }

  // ---- the upload form
  function formError(text) {
    $("form-error").hidden = !text;
    $("form-error").textContent = text || "";
  }

  $("upload").addEventListener("submit", (event) => {
    event.preventDefault();
    const file = $("file").files[0];
    const members = Number($("members").value || 0);
    if (!file) return formError("Choose a video first.");
    if (!/\.mp4$/i.test(file.name)) return formError("Choose an .mp4 video.");
    if (!Number.isInteger(members) || members < 0 || members > 50) return formError("Members must be a whole number from 0 to 50.");
    formError("");
    watching = null;
    $("go").disabled = true;
    showUploading(file.name, 0, file.size);
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `/api/uploads?name=${encodeURIComponent(file.name)}&members=${members}`);
    xhr.setRequestHeader("Content-Type", "video/mp4");
    xhr.upload.addEventListener("progress", (e) => { if (e.lengthComputable) showUploading(file.name, e.loaded, e.total); });
    xhr.addEventListener("load", () => {
      $("go").disabled = false;
      let data = {};
      try { data = JSON.parse(xhr.responseText); } catch (_) { /* not JSON: use the status below */ }
      if (xhr.status === 201) {
        $("upload").reset();
        watch(data.id);
      } else showError(data.error || `The upload failed (the server answered ${xhr.status}).`);
    });
    xhr.addEventListener("error", () => {
      $("go").disabled = false;
      showError("The upload failed. Is `syncscore serve` still running?");
    });
    xhr.send(file);
  });

  const start = location.hash.slice(1);
  if (/^\d{8}-\d{6}-[0-9a-f]{6}$/.test(start)) watch(start);
  else tick();
})();
