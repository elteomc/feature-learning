// Headless smoke test for the weighted-law explorer.
//
// Loads index.html in a JSDOM environment, runs app.js, then exercises every
// control we expose. Fails (non-zero exit) on uncaught errors, missing DOM
// hooks, or empty critical regions after a render.

import { JSDOM, ResourceLoader, VirtualConsole } from "jsdom";
import { readFileSync } from "node:fs";
import { fileURLToPath, pathToFileURL } from "node:url";
import { dirname, resolve } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const appRoot = resolve(here, "..");
const indexPath = resolve(appRoot, "index.html");
const html = readFileSync(indexPath, "utf-8");

const failures = [];
function check(condition, message) {
  if (!condition) failures.push(message);
}

class LocalLoader extends ResourceLoader {
  fetch(url, options) {
    if (url.startsWith("file:")) {
      return super.fetch(url, options);
    }
    return super.fetch(url, options);
  }
}

const consoleErrors = [];
const virtualConsole = new VirtualConsole();
virtualConsole.on("error", (err) => consoleErrors.push(String(err && err.stack || err)));
virtualConsole.on("jsdomError", (err) => consoleErrors.push(String(err && err.stack || err)));
virtualConsole.on("warn", () => {});
virtualConsole.on("info", () => {});
virtualConsole.on("log", () => {});

const dom = new JSDOM(html, {
  url: pathToFileURL(indexPath).href,
  runScripts: "dangerously",
  resources: new LocalLoader(),
  pretendToBeVisual: true,
  virtualConsole
});

const { window } = dom;

// app.js uses fetch("data/<file>.json"). Polyfill from disk so JSDOM does not
// need an HTTP server.
const trajectoryPayload = readFileSync(resolve(appRoot, "data", "trajectories.json"), "utf-8");
const matrixPayload = readFileSync(resolve(appRoot, "data", "matrix_snapshots.json"), "utf-8");
window.fetch = async (url) => {
  if (typeof url === "string" && url.endsWith("trajectories.json")) {
    return { ok: true, status: 200, json: async () => JSON.parse(trajectoryPayload) };
  }
  if (typeof url === "string" && url.endsWith("matrix_snapshots.json")) {
    return { ok: true, status: 200, json: async () => JSON.parse(matrixPayload) };
  }
  return { ok: false, status: 404, json: async () => ({}) };
};

// JSDOM 22 does not implement HTMLDialogElement.showModal. Polyfill the bits
// we use so the open-figures dialog can be exercised in tests.
const DialogProto = window.HTMLDialogElement && window.HTMLDialogElement.prototype;
if (DialogProto && typeof DialogProto.showModal !== "function") {
  DialogProto.showModal = function () {
    this.setAttribute("open", "");
    this.open = true;
  };
  DialogProto.close = function () {
    this.removeAttribute("open");
    this.open = false;
  };
}

// Patch localStorage to not throw under file:// in some configs.
let storage = {};
const fakeStorage = {
  getItem: (k) => Object.prototype.hasOwnProperty.call(storage, k) ? storage[k] : null,
  setItem: (k, v) => { storage[k] = String(v); },
  removeItem: (k) => { delete storage[k]; },
  clear: () => { storage = {}; }
};
try {
  Object.defineProperty(window, "localStorage", { configurable: true, value: fakeStorage });
} catch (err) {
  console.warn("Could not redefine localStorage:", err.message);
}

// JSDOM loads app.js automatically through the <script src="app.js"> tag
// and the LocalLoader. Wait for window load before exercising controls.
await new Promise((resolve) => {
  if (window.document.readyState === "complete") {
    resolve();
  } else {
    window.addEventListener("load", () => resolve());
  }
});

// Allow trajectories.json fetch and a tick for renderLive to run.
await new Promise((r) => setTimeout(r, 80));
await new Promise((r) => setTimeout(r, 80));

const { document } = window;

function $(id) {
  return document.getElementById(id);
}

// 1. Critical DOM nodes the app reads/writes.
const criticalIds = [
  "family-select", "seed-input", "sweep-select",
  "residual-slider", "stationarity-slider",
  "leverage-cv-slider", "resid-cv-slider", "corr-slider",
  "pair-defect-slider", "pair-gain-slider", "gain-corr-slider",
  "residual-output", "stationarity-output", "leverage-cv-output",
  "resid-cv-output", "corr-output", "pair-defect-output",
  "pair-gain-output", "gain-corr-output",
  "sweep-title", "sweep-description", "sweep-active-controls",
  "live-plot", "plot-legend", "insight-cards",
  "sweep-table-head", "sweep-table-body", "sweep-table-note",
  "summary-table", "family-title", "family-summary",
  "trajectory-plot", "trajectory-metric-select", "trajectory-legend",
  "trajectory-config", "trajectory-scrubber", "trajectory-scrubber-output",
  "trajectory-readout",
  "alignment-heatmap", "alignment-meta",
  "spectrum-plot", "spectrum-legend",
  "regime-plot", "regime-status",
  "figure-select", "figure-image", "figure-caption", "figure-detail",
  "figures-dialog", "open-figures",
  "help-dialog", "open-help",
  "theme-button", "reset-button"
];
criticalIds.forEach((id) => check($(id), `missing #${id}`));

// 2. Selectors are populated.
check($("family-select").options.length === 3, "family-select should have 3 options");
check($("sweep-select").options.length === 3, "sweep-select should have 3 options");
check($("trajectory-metric-select").options.length === 7, "trajectory-metric-select should have 7 options");

const figureOptionCount = $("figure-select").querySelectorAll("option").length;
check(figureOptionCount === 14, `figure-select should have 14 options, has ${figureOptionCount}`);
const optgroups = $("figure-select").querySelectorAll("optgroup");
check(optgroups.length >= 4, `figure-select should have >=4 optgroups, has ${optgroups.length}`);

const snapshotButtons = document.querySelectorAll(".snapshot-button");
check(snapshotButtons.length === 3, `expected 3 snapshot buttons, got ${snapshotButtons.length}`);

// 3. After init, every key plot region has rendered.
check($("live-plot").innerHTML.length > 0, "live-plot is empty after init");
check($("trajectory-plot").innerHTML.length > 0, "trajectory-plot is empty after init");
check($("regime-plot").innerHTML.length > 0, "regime-plot is empty after init");
check($("alignment-heatmap").innerHTML.length > 0, "alignment-heatmap is empty after init");
check($("spectrum-plot").innerHTML.length > 0, "spectrum-plot is empty after init");
check($("trajectory-readout").children.length >= 4, "trajectory readout should have at least 4 cards");
check($("alignment-meta").textContent.includes("max"), "alignment meta should report max value");
check($("spectrum-legend").children.length >= 2, "spectrum legend should have entries");
check($("insight-cards").children.length >= 4, "expected at least 4 insight cards");
check($("summary-table").children.length === 3, "summary-table should have 3 family rows");
check($("plot-legend").children.length >= 2, "plot legend should have entries");
check($("sweep-table-body").children.length > 0, "sweep table is empty");
check($("regime-status").textContent.trim().length > 0, "regime-status has no text");

// 4. Each sweep produces a non-empty plot and table.
function fireInput(id) {
  const el = $(id);
  el.dispatchEvent(new window.Event("input", { bubbles: true }));
  el.dispatchEvent(new window.Event("change", { bubbles: true }));
}

const sweepSnapshots = {};
["regime", "beta", "pair"].forEach((sweepKey) => {
  $("sweep-select").value = sweepKey;
  fireInput("sweep-select");
  const plotHtml = $("live-plot").innerHTML;
  const tableRows = $("sweep-table-body").children.length;
  check(plotHtml.length > 0, `live-plot empty for sweep ${sweepKey}`);
  check(tableRows > 0, `sweep table empty for sweep ${sweepKey}`);
  sweepSnapshots[sweepKey] = { plotLen: plotHtml.length, tableRows };
});

// 5. Each slider should change *something* visible in the live plot for at
// least one sweep. We assert the plot innerHTML differs after twiddling each
// slider while at least one sweep configuration is active.
const sliderIds = [
  "residual-slider", "stationarity-slider",
  "leverage-cv-slider", "resid-cv-slider", "corr-slider",
  "pair-defect-slider", "pair-gain-slider", "gain-corr-slider"
];
const sliderResponses = {};
sliderIds.forEach((id) => sliderResponses[id] = false);

for (const sweepKey of ["regime", "beta", "pair"]) {
  $("sweep-select").value = sweepKey;
  fireInput("sweep-select");
  for (const id of sliderIds) {
    const el = $(id);
    const original = el.value;
    const before = $("live-plot").innerHTML + "::" + $("regime-plot").innerHTML;
    const num = Number(original);
    const min = Number(el.min);
    const max = Number(el.max);
    const target = num + (max - num > num - min ? 0.5 : -0.5);
    el.value = String(Math.max(min, Math.min(max, target)));
    fireInput(id);
    const after = $("live-plot").innerHTML + "::" + $("regime-plot").innerHTML;
    if (before !== after) sliderResponses[id] = true;
    el.value = original;
    fireInput(id);
  }
}

Object.entries(sliderResponses).forEach(([id, responded]) => {
  check(responded, `slider ${id} did not change any plot in any sweep`);
});

// 6. Family change drives the trajectory plot.
const trajectoryBefore = $("trajectory-plot").innerHTML;
$("family-select").value = "low_rank_signal";
fireInput("family-select");
const trajectoryAfter = $("trajectory-plot").innerHTML;
check(trajectoryBefore !== trajectoryAfter, "family change did not affect trajectory-plot");
check($("trajectory-config").textContent.includes("n="), "trajectory-config should show config string");
$("family-select").value = "isotropic";
fireInput("family-select");

// 7. Trajectory metric change should update the legend text.
$("trajectory-metric-select").value = "beta_fit";
fireInput("trajectory-metric-select");
check($("trajectory-legend").textContent.toLowerCase().includes("beta"), "trajectory legend should mention beta_fit after switch");
$("trajectory-metric-select").value = "gamma_tilde_eff_rel_h2";
fireInput("trajectory-metric-select");

// 8. Theme toggle.
const themeBefore = document.body.dataset.theme;
$("theme-button").dispatchEvent(new window.Event("click", { bubbles: true }));
const themeAfter = document.body.dataset.theme;
check(themeBefore !== themeAfter, "theme toggle did not change data-theme");
$("theme-button").dispatchEvent(new window.Event("click", { bubbles: true }));

// 9. Static figures dialog opens via the sidebar button.
const dialog = $("figures-dialog");
check(!dialog.open, "figures dialog should start closed");
$("open-figures").dispatchEvent(new window.Event("click", { bubbles: true }));
check(dialog.open === true || dialog.hasAttribute("open"), "figures dialog should open after click");

// 9a. Help dialog opens via the sidebar button and contains explanatory copy.
const helpDialog = $("help-dialog");
check(!helpDialog.open, "help dialog should start closed");
$("open-help").dispatchEvent(new window.Event("click", { bubbles: true }));
check(helpDialog.open === true || helpDialog.hasAttribute("open"), "help dialog should open after click");
check(helpDialog.querySelectorAll("section").length >= 4, "help dialog should have at least 4 sections");
check(helpDialog.querySelector(".help-table"), "help dialog should include the slider reference table");

// 10. Switching the figure select updates image src and caption.
const figureSelect = $("figure-select");
figureSelect.value = "5";
fireInput("figure-select");
check($("figure-image").src.includes(".png"), "figure-image src should reference a .png");
check($("figure-caption").textContent.length > 0, "figure-caption should have text");

// 10a. Snapshot buttons drive the heatmap and spectrum.
const heatBefore = $("alignment-heatmap").innerHTML;
document.querySelector('.snapshot-button[data-snapshot="init"]').dispatchEvent(new window.Event("click", { bubbles: true }));
check($("alignment-heatmap").innerHTML !== heatBefore, "snapshot toggle should change the alignment heatmap");
check($("alignment-heatmap").innerHTML.includes("init"), "alignment heatmap title should reference init snapshot");
document.querySelector('.snapshot-button[data-snapshot="late"]').dispatchEvent(new window.Event("click", { bubbles: true }));
check($("alignment-heatmap").innerHTML.includes("late"), "alignment heatmap title should reference late snapshot after switching back");

// 10b. Step scrubber updates readout and trajectory plot.
const trajectoryBeforeScrub = $("trajectory-plot").innerHTML;
const scrubber = $("trajectory-scrubber");
scrubber.value = "5";
scrubber.dispatchEvent(new window.Event("input", { bubbles: true }));
check($("trajectory-plot").innerHTML !== trajectoryBeforeScrub, "scrubber should re-render trajectory plot");
check($("trajectory-scrubber-output").textContent.includes("step"), "scrubber output should mention step");

// 10c. R^2 metric option is present and produces values.
$("trajectory-metric-select").value = "r2";
fireInput("trajectory-metric-select");
const r2Found = Array.from($("trajectory-readout").children).some((c) => c.textContent.includes("R^2"));
check(r2Found, "trajectory readout should include an R^2 card when R^2 is selected");

// 10d. R^2 card should also be present even when R^2 is NOT the selected
// overlay; it is a permanent diagnostic, not metric-gated.
$("trajectory-metric-select").value = "gamma_tilde_eff_rel_h2";
fireInput("trajectory-metric-select");
const r2AlsoFound = Array.from($("trajectory-readout").children).some((c) => c.textContent.includes("R^2"));
check(r2AlsoFound, "R^2 readout card should be present regardless of overlay metric");

// 10e. raw_vs_weighted overlay draws two right-axis lines on the trajectory.
$("trajectory-metric-select").value = "raw_vs_weighted";
fireInput("trajectory-metric-select");
const trajectoryPaths = ($("trajectory-plot").querySelectorAll("path") || []).length;
check(trajectoryPaths >= 3, "raw_vs_weighted should draw the loss path plus two overlay paths (got " + trajectoryPaths + ")");
const trajectoryLegendText = $("trajectory-legend").textContent.toLowerCase();
check(trajectoryLegendText.includes("raw"), "trajectory legend should mention raw residual");
check(trajectoryLegendText.includes("weighted"), "trajectory legend should mention weighted residual");

// 11. Active-controls hint changes per sweep.
$("sweep-select").value = "regime";
fireInput("sweep-select");
const regimeText = $("sweep-active-controls").textContent;
$("sweep-select").value = "beta";
fireInput("sweep-select");
const betaText = $("sweep-active-controls").textContent;
check(regimeText !== betaText && regimeText.length > 0 && betaText.length > 0, "active-controls hint should change between sweeps");

// 12. Reset.
$("reset-button").dispatchEvent(new window.Event("click", { bubbles: true }));
check($("residual-slider").value === "-3", "reset should restore residual-slider");

// Console errors check.
if (consoleErrors.length > 0) {
  failures.push(`Console errors:\n${consoleErrors.join("\n---\n")}`);
}

if (failures.length > 0) {
  console.error("FAIL");
  failures.forEach((f) => console.error("  - " + f));
  process.exit(1);
}

console.log("PASS, exercised", criticalIds.length, "DOM hooks and 3 sweeps x " + sliderIds.length + " sliders.");
console.log("Slider responsiveness:");
Object.entries(sliderResponses).forEach(([id, ok]) => {
  console.log("  " + id + ": " + (ok ? "responds" : "INERT"));
});
process.exit(0);
