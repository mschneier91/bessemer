#!/usr/bin/env python3
"""DFG 2D-3 scheme comparison: the HTML report (work-precision charts).

Usage: report.py OUT.html FINDINGS.html SUMMARY [SUMMARY ...]

Reads run_queue.sh summary files (kind=run lines only), embeds the runs as
JSON, and writes a self-contained page: the findings (an HTML fragment written
by hand from the analysis), log-log work-precision charts (error vs step wall
time) for the lift peak, the drag peak and the lift-peak time, error vs CFL
target per mesh, cost per step, and the full run table.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import parse  # noqa: E402

KEEP = ["name", "form", "ext", "mref", "cfl_target", "status", "steps", "ne",
        "step_wall", "wall", "outer_mean", "vel_inner_mean", "cd_max", "t_cd",
        "cl_max", "t_cl", "dp8", "err_cd", "err_tcd", "err_cl", "err_tcl",
        "err_dp", "t", "cfl_max"]

PAGE = r"""<title>Rotational vs IMEX on DFG 2D-3</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+Condensed:wght@500;600&family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;1,400&display=swap">
<style>
/* Layout: one reading column (~68ch) for prose; charts and tables break out
   to a wider measure. Two scheme hues carry through every chart and table. */
:root {
  --paper: #f6f7f9;
  --panel: #ffffff;
  --ink: #1b2330;
  --muted: #5b6676;
  --rule: #d9dee6;
  --grid: #e7ebf0;
  --imex: #1f5fa8;
  --rot: #b0412e;
  --imex-soft: #dce7f5;
  --rot-soft: #f5e0db;
  --bad: #8a1c1c;
  --font-display: "IBM Plex Sans Condensed", "Arial Narrow", system-ui, sans-serif;
  --font-body: "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
  --font-data: "IBM Plex Mono", ui-monospace, "SFMono-Regular", Menlo, monospace;
  --step--1: 0.8125rem;
  --step-0: 1rem;
  --step-1: 1.25rem;
  --step-2: 1.6rem;
  --step-3: 2.2rem;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --paper: #12161c;
    --panel: #181d25;
    --ink: #e4e8ee;
    --muted: #9aa5b4;
    --rule: #2c3440;
    --grid: #232a35;
    --imex: #6fa6e8;
    --rot: #e8866f;
    --imex-soft: #1d2c40;
    --rot-soft: #3a2420;
    --bad: #f08a8a;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --paper: #12161c;
  --panel: #181d25;
  --ink: #e4e8ee;
  --muted: #9aa5b4;
  --rule: #2c3440;
  --grid: #232a35;
  --imex: #6fa6e8;
  --rot: #e8866f;
  --imex-soft: #1d2c40;
  --rot-soft: #3a2420;
  --bad: #f08a8a;
}
* { box-sizing: border-box; }
body {
  background: var(--paper);
  color: var(--ink);
  font-family: var(--font-body);
  font-size: var(--step-0);
  line-height: 1.6;
  margin: 0;
  padding-inline: 20px;
}
main { max-width: 1040px; margin: 0 auto; padding-block: 40px 72px; display: grid; gap: 28px; }
.prose { max-width: 68ch; }
h1, h2, h3 { font-family: var(--font-display); line-height: 1.15; text-wrap: balance; margin: 0; }
h1 { font-size: var(--step-3); font-weight: 600; letter-spacing: -0.01em; }
h2 { font-size: var(--step-2); font-weight: 600; }
h3 { font-size: var(--step-1); font-weight: 500; }
p { margin: 0; }
.stack { display: grid; gap: 12px; }
.eyebrow { font-family: var(--font-data); font-size: var(--step--1); color: var(--muted); letter-spacing: 0.06em; text-transform: uppercase; }
.imex { color: var(--imex); font-weight: 600; }
.rot { color: var(--rot); font-weight: 600; }
code, .num { font-family: var(--font-data); font-size: 0.92em; }
section { display: grid; gap: 16px; padding-top: 8px; border-top: 1px solid var(--rule); }
.answer { background: var(--panel); border: 1px solid var(--rule); border-radius: 6px; padding: 20px 22px; display: grid; gap: 12px; }
.answer ul { margin: 0; padding-left: 1.2em; display: grid; gap: 6px; }
.charts { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
figure { margin: 0; background: var(--panel); border: 1px solid var(--rule); border-radius: 6px; padding: 14px 14px 10px; display: grid; gap: 6px; }
figcaption { font-size: var(--step--1); color: var(--muted); }
figure .title { font-family: var(--font-display); font-weight: 500; font-size: 1.05rem; color: var(--ink); }
svg { width: 100%; height: auto; display: block; overflow: visible; }
svg text { font-family: var(--font-data); font-size: 10.5px; fill: var(--muted); }
svg .gridline { stroke: var(--grid); stroke-width: 1; fill: none; }
svg .axis { stroke: var(--rule); stroke-width: 1; fill: none; }
svg .s-imex { stroke: var(--imex); fill: none; }
svg .s-rot { stroke: var(--rot); fill: none; }
svg .f-imex { fill: var(--imex); stroke: var(--panel); }
svg .f-rot { fill: var(--rot); stroke: var(--panel); }
svg .o-imex { fill: var(--panel); stroke: var(--imex); }
svg .o-rot { fill: var(--panel); stroke: var(--rot); }
svg .lbl { fill: var(--ink); }
.legend { display: flex; flex-wrap: wrap; gap: 8px 20px; font-size: var(--step--1); color: var(--muted); }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.sw { width: 18px; height: 3px; border-radius: 2px; display: inline-block; }
.table-wrap { overflow-x: auto; background: var(--panel); border: 1px solid var(--rule); border-radius: 6px; }
table { border-collapse: collapse; width: 100%; font-size: var(--step--1); font-variant-numeric: tabular-nums; }
th, td { padding: 6px 10px; text-align: right; white-space: nowrap; border-bottom: 1px solid var(--grid); }
th { font-family: var(--font-body); font-weight: 600; color: var(--muted); position: sticky; top: 0; background: var(--panel); }
td { font-family: var(--font-data); }
td:first-child, th:first-child { text-align: left; }
tr.row-imex td:first-child { color: var(--imex); }
tr.row-rot td:first-child { color: var(--rot); }
td.bad { color: var(--bad); }
.note { font-size: var(--step--1); color: var(--muted); }
@media (max-width: 480px) { h1 { font-size: 1.8rem; } .answer { padding: 16px; } }
</style>

<main>
  <header class="stack prose">
    <p class="eyebrow">bessemer · DFG 2D-3 · Q3/Q2 · CFL-controlled steps · 2026-10-08</p>
    <h1>Rotational vs IMEX on DFG 2D-3</h1>
    <p>What it costs each Navier–Stokes scheme to reach a given accuracy on the time-dependent flow around a cylinder (Re up to 100, t ∈ [0, 8]). Errors are against John's reference values. Cost is the wall time spent in time stepping on 4 MPI ranks.</p>
  </header>

  __FINDINGS__

  <section>
    <div class="stack prose">
      <h2>Work-precision</h2>
      <p>Each point is one run. Lines join the CFL targets of one scheme, extrapolation order and mesh; lower-left is better. Open markers ran to t = 8 with an error above 10 %, the silent failure mode; a cross at the top edge is a run that blew up.</p>
    </div>
    <div class="legend" id="legend"></div>
    <div class="charts" id="wp"></div>
  </section>

  <section>
    <div class="stack prose">
      <h2>Accuracy against the CFL target</h2>
      <p>Same runs, plotted against the CFL target (Nek5000's definition) instead of cost. Where a curve turns up, that scheme has stopped resolving the flow at that step size.</p>
    </div>
    <div class="charts" id="vscfl"></div>
  </section>

  <section>
    <div class="stack prose">
      <h2>Cost per step</h2>
      <p>Wall time per time step and outer FGMRES iterations per step, against the CFL target.</p>
    </div>
    <div class="charts" id="cost"></div>
  </section>

  <section>
    <div class="stack prose"><h2>All runs</h2>
    <p class="note">Errors relative for the peak values and Δp(8), absolute (time units) for the peak times. Δp(8) is published to four digits (−0.1116), so its error cannot resolve below about 1e-3.</p></div>
    <div class="table-wrap"><table id="runs"></table></div>
  </section>
</main>

<script type="application/json" id="data">__DATA__</script>
<script>
(function () {
  const runs = JSON.parse(document.getElementById("data").textContent);
  const SCH = { imex: "IMEX convective", rot: "semi-implicit rotational" };
  const MESH = { 0: "M0 · 208 cells", 1: "M1 · 832 cells", 2: "M2 · 3328 cells" };
  const DASH = { 2: "", 3: "5 3" }; // by EXT order
  const WIDTH = { 0: 1.25, 1: 1.75, 2: 2.5 };
  const sch = r => (r.form === "convective" ? "imex" : "rot");
  const ok = r => r.status === "ok";
  const NS = "http://www.w3.org/2000/svg";
  const el = (tag, attrs, parent) => {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  };
  const meshes = [...new Set(runs.map(r => r.mref))].sort();
  const exts = [...new Set(runs.map(r => r.ext))].sort();

  // Legend: scheme hue, mesh line weight / dash.
  const lg = document.getElementById("legend");
  for (const s of ["imex", "rot"]) {
    lg.insertAdjacentHTML("beforeend", `<span><i class="sw" style="background:var(--${s})"></i>${SCH[s]}</span>`);
  }
  for (const e of exts) {
    lg.insertAdjacentHTML("beforeend", `<span><svg width="26" height="8" style="width:26px"><line x1="1" y1="4" x2="25" y2="4" stroke="currentColor" stroke-width="1.75" stroke-dasharray="${DASH[e]}"/></svg>BDF2/EXT${e}</span>`);
  }
  if (meshes.length > 1) for (const m of meshes) {
    lg.insertAdjacentHTML("beforeend", `<span><svg width="26" height="8" style="width:26px"><line x1="1" y1="4" x2="25" y2="4" stroke="currentColor" stroke-width="${WIDTH[m]}"/></svg>${MESH[m]}</span>`);
  }

  function chart(host, opt) {
    const W = 460, H = 300, L = 58, R = 14, T = 12, B = 42;
    const fig = document.createElement("figure");
    fig.innerHTML = `<div class="title">${opt.title}</div>`;
    host.appendChild(fig);
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": opt.title }, fig);
    const pts = opt.series.flatMap(s => s.pts).filter(p => isFinite(p.x) && isFinite(p.y) && (!opt.ylog || p.y > 0));
    if (!pts.length) return;
    const lx = v => (opt.xlog ? Math.log10(v) : v), ly = v => (opt.ylog ? Math.log10(v) : v);
    let x0 = Math.min(...pts.map(p => lx(p.x))), x1 = Math.max(...pts.map(p => lx(p.x)));
    let y0 = Math.min(...pts.map(p => ly(p.y))), y1 = Math.max(...pts.map(p => ly(p.y)));
    if (opt.xlog) { x0 = Math.floor(x0 * 2) / 2; x1 = Math.ceil(x1 * 2) / 2; } else { x0 = 0; x1 = x1 * 1.05; }
    // Linear axes get round ticks: a 1-2-5 step, top rounded up to a tick.
    const nice = r => { const p = Math.pow(10, Math.floor(Math.log10(r))), f = r / p; return (f <= 1 ? 1 : f <= 2 ? 2 : f <= 5 ? 5 : 10) * p; };
    let ystep = 1;
    if (opt.ylog) { y0 = Math.floor(y0); y1 = Math.ceil(y1); }
    else { y0 = 0; ystep = nice(y1 * 1.1 / 4); y1 = Math.ceil(y1 * 1.1 / ystep) * ystep; }
    if (opt.ycap !== undefined && opt.ylog) y1 = Math.min(y1, Math.log10(opt.ycap));
    if (x1 === x0) x1 = x0 + 1;
    if (y1 === y0) y1 = y0 + 1;
    const X = v => L + (lx(v) - x0) / (x1 - x0) * (W - L - R);
    const Y = v => T + (1 - (Math.min(ly(v), y1) - y0) / (y1 - y0)) * (H - T - B);
    // grid + ticks
    const fmt = v => (Math.abs(v) >= 1e4 || (Math.abs(v) < 1e-2 && v !== 0)) ? v.toExponential(0).replace("e+", "e") : String(+v.toPrecision(3));
    if (opt.ylog) {
      for (let k = Math.ceil(y0); k <= y1; k++) {
        const y = T + (1 - (k - y0) / (y1 - y0)) * (H - T - B);
        el("line", { x1: L, x2: W - R, y1: y, y2: y, class: "gridline" }, svg);
        el("text", { x: L - 6, y: y + 3.5, "text-anchor": "end" }, svg).textContent = "1e" + k;
      }
    } else {
      for (let v = y0; v <= y1 + ystep * 1e-9; v += ystep) {
        const y = T + (1 - (v - y0) / (y1 - y0)) * (H - T - B);
        el("line", { x1: L, x2: W - R, y1: y, y2: y, class: "gridline" }, svg);
        el("text", { x: L - 6, y: y + 3.5, "text-anchor": "end" }, svg).textContent = fmt(+v.toPrecision(6));
      }
    }
    if (opt.xlog) {
      for (let k = Math.floor(x0); k <= Math.floor(x1); k++) {
        for (const m of [1, 2, 5]) {
          const v = Math.log10(m) + k;
          if (v < x0 - 1e-9 || v > x1 + 1e-9) continue;
          const x = L + (v - x0) / (x1 - x0) * (W - L - R);
          el("line", { x1: x, x2: x, y1: T, y2: H - B, class: "gridline" }, svg);
          el("text", { x: x, y: H - B + 14, "text-anchor": "middle" }, svg).textContent = fmt(m * Math.pow(10, k));
        }
      }
    } else {
      for (const v of opt.xticks) {
        if (v < x0 || v > x1) continue;
        const x = X(v);
        el("line", { x1: x, x2: x, y1: T, y2: H - B, class: "gridline" }, svg);
        el("text", { x: x, y: H - B + 14, "text-anchor": "middle" }, svg).textContent = String(v);
      }
    }
    el("path", { d: `M${L},${T}V${H - B}H${W - R}`, class: "axis" }, svg);
    el("text", { x: (L + W - R) / 2, y: H - 6, "text-anchor": "middle", class: "lbl" }, svg).textContent = opt.xlabel;
    el("text", { x: 12, y: (T + H - B) / 2, "text-anchor": "middle", class: "lbl", transform: `rotate(-90 12 ${(T + H - B) / 2})` }, svg).textContent = opt.ylabel;
    for (const s of opt.series) {
      const good = s.pts.filter(p => isFinite(p.x) && isFinite(p.y) && (!opt.ylog || p.y > 0));
      if (good.length > 1) {
        el("polyline", { points: good.map(p => `${X(p.x)},${Y(p.y)}`).join(" "), class: "s-" + s.sch, "stroke-width": WIDTH[s.mref], "stroke-dasharray": DASH[s.ext], "stroke-linejoin": "round" }, svg);
      }
      for (const p of s.pts) {
        if (p.blown) {
          if (!isFinite(p.x)) continue;
          const x = X(p.x), y = T + 4;
          el("path", { d: `M${x - 4},${y - 4}L${x + 4},${y + 4}M${x - 4},${y + 4}L${x + 4},${y - 4}`, class: "s-" + s.sch, "stroke-width": 1.75 }, svg);
          continue;
        }
        if (!(isFinite(p.x) && isFinite(p.y) && (!opt.ylog || p.y > 0))) continue;
        const c = el("circle", { cx: X(p.x), cy: Y(p.y), r: 3.6, class: (p.garbage ? "o-" : "f-") + s.sch, "stroke-width": p.garbage ? 1.5 : 1 }, svg);
        el("title", {}, c).textContent = p.tip;
      }
    }
    if (opt.caption) {
      const cap = document.createElement("figcaption");
      cap.textContent = opt.caption;
      fig.appendChild(cap);
    }
  }

  function series(metric, xkey) {
    const out = [];
    for (const s of ["imex", "rot"]) for (const m of meshes) for (const e of exts) {
      const rs = runs.filter(r => sch(r) === s && r.mref === m && r.ext === e).sort((a, b) => a.cfl_target - b.cfl_target);
      if (!rs.length) continue;
      out.push({ sch: s, mref: m, ext: e, pts: rs.map(r => ({
        x: xkey(r), y: ok(r) ? r[metric] : NaN, blown: !ok(r), garbage: ok(r) && r.err_cl > 0.1,
        tip: `${SCH[s]}/EXT${e}, M${m}, CFL ${r.cfl_target}: ${metric} = ${ok(r) ? r[metric].toExponential(2) : r.status}, ${r.step_wall.toFixed(0)} s`
      })) });
    }
    return out;
  }

  const wp = document.getElementById("wp");
  const xs = r => r.step_wall;
  chart(wp, { title: "Lift peak c_L,max", series: series("err_cl", xs), xlog: true, ylog: true, xlabel: "wall time in time stepping [s]", ylabel: "relative error", caption: "Reference 0.47795 at t = 5.693125 (John 2004)." });
  chart(wp, { title: "Drag peak c_D,max", series: series("err_cd", xs), xlog: true, ylog: true, xlabel: "wall time in time stepping [s]", ylabel: "relative error", caption: "Reference 2.950921575 at t = 3.93625." });
  chart(wp, { title: "Time of the lift peak", series: series("err_tcl", xs), xlog: true, ylog: true, xlabel: "wall time in time stepping [s]", ylabel: "absolute error [time units]", caption: "Phase error of the shedding: reference t = 5.693125." });

  const cflTicks = [0, 0.3, 0.6, 0.9, 1.2, 1.5];
  const vc = document.getElementById("vscfl");
  chart(vc, { title: "Lift peak c_L,max", series: series("err_cl", r => r.cfl_target), xlog: false, xticks: cflTicks, ylog: true, xlabel: "CFL target", ylabel: "relative error" });
  chart(vc, { title: "Drag peak c_D,max", series: series("err_cd", r => r.cfl_target), xlog: false, xticks: cflTicks, ylog: true, xlabel: "CFL target", ylabel: "relative error" });

  const co = document.getElementById("cost");
  const perStep = runs.map(r => Object.assign({}, r, { s_per_step: r.step_wall / Math.max(r.steps, 1), status: "ok", err_cl: 0 }));
  const seriesOf = (rows, key) => {
    const out = [];
    for (const s of ["imex", "rot"]) for (const m of meshes) for (const e of exts) {
      const rs = rows.filter(r => sch(r) === s && r.mref === m && r.ext === e).sort((a, b) => a.cfl_target - b.cfl_target);
      if (rs.length) out.push({ sch: s, mref: m, ext: e, pts: rs.map(r => ({ x: r.cfl_target, y: r[key], tip: `${SCH[s]}/EXT${e}, M${m}, CFL ${r.cfl_target}: ${(+r[key]).toPrecision(3)}` })) });
    }
    return out;
  };
  chart(co, { title: "Wall time per step [s]", series: seriesOf(perStep, "s_per_step"), xlog: false, xticks: cflTicks, ylog: false, xlabel: "CFL target", ylabel: "seconds per step" });
  chart(co, { title: "Outer iterations per step", series: seriesOf(perStep, "outer_mean"), xlog: false, xticks: cflTicks, ylog: false, xlabel: "CFL target", ylabel: "FGMRES iterations" });

  // Table
  const t = document.getElementById("runs");
  const e = v => (isFinite(v) ? v.toExponential(1) : "–");
  t.innerHTML = "<thead><tr><th>scheme</th><th>EXT</th><th>mesh</th><th>CFL</th><th>status</th><th>steps</th><th>step wall [s]</th><th>s/step</th><th>outer/step</th><th>c_L,max err</th><th>c_D,max err</th><th>t(c_L) err</th><th>Δp(8) err</th></tr></thead>";
  const tb = document.createElement("tbody");
  const sorted = runs.slice().sort((a, b) => sch(a).localeCompare(sch(b)) || a.mref - b.mref || a.ext - b.ext || a.cfl_target - b.cfl_target);
  for (const r of sorted) {
    const s = sch(r), good = ok(r);
    const garbage = good && r.err_cl > 0.1;
    tb.insertAdjacentHTML("beforeend", `<tr class="row-${s}"><td>${s === "imex" ? "IMEX" : "rotational"}</td><td>${r.ext}</td><td>M${r.mref}</td><td>${r.cfl_target}</td><td class="${good && !garbage ? "" : "bad"}">${good ? (garbage ? "wrong" : "ok") : "blew up t=" + (+r.t).toFixed(2)}</td><td>${r.steps}</td><td>${r.step_wall.toFixed(0)}</td><td>${(r.step_wall / Math.max(r.steps, 1)).toFixed(3)}</td><td>${(+r.outer_mean).toFixed(1)}</td><td>${good ? e(r.err_cl) : "–"}</td><td>${good ? e(r.err_cd) : "–"}</td><td>${good ? e(r.err_tcl) : "–"}</td><td>${good ? e(r.err_dp) : "–"}</td></tr>`);
  }
  t.appendChild(tb);
})();
</script>
"""


def main(argv):
    out, findings_path, summaries = argv[0], argv[1], argv[2:]
    runs = [r for p in summaries for r in parse(p) if r.get("kind") == "run"
            and r.get("status") in ("ok", "diverged") and "cfl_target" in r]
    data = [{k: r.get(k) for k in KEEP} for r in runs]
    for d in data:
        d["mref"] = int(d["mref"] or 0)
        d["ext"] = int(d["ext"] or 2)
    with open(findings_path) as f:
        findings = f.read()
    # Script contents are raw text (no entity decoding): only "</" must not
    # appear, or it would end the script element early.
    page = PAGE.replace("__FINDINGS__", findings).replace(
        "__DATA__", json.dumps(data).replace("</", "<\\/"))
    with open(out, "w") as f:
        f.write(page)
    print("wrote %s (%d runs)" % (out, len(data)))


if __name__ == "__main__":
    main(sys.argv[1:])
