"""Offline self-contained HTML5 interactive knowledge graph exporter."""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from second_brain.config import BrainConfig
from second_brain.ingest.pdf_adapter import _slugify, rebuild_cross_standard_edges
from second_brain.storage.sqlite_store import SqliteStore, atomic_write_lf

_PALETTE = [
    "#58a6ff",
    "#3fb950",
    "#d29922",
    "#f78166",
    "#bc8cff",
    "#39c5cf",
    "#ffa657",
    "#79c0ff",
    "#56d364",
    "#e3b341",
    "#ff7b72",
    "#d2a8ff",
    "#a5d6ff",
    "#7ee787",
    "#f0883e",
]

_BREADCRUMB_RE = re.compile(r"^(?:<!--\s*breadcrumb:[^\n]*-->\s*\n?)+")


def _clean_preview(text: str, limit: int = 360) -> str:
    stripped = _BREADCRUMB_RE.sub("", text).strip()
    collapsed = re.sub(r"\s+", " ", stripped)
    return collapsed[:limit]


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>2ndBrain Interactive Knowledge Graph</title>
<style>
  :root {
    --bg: #0d1117;
    --panel: #161b22;
    --border: #30363d;
    --text: #e6edf3;
    --muted: #8b949e;
    --accent: #58a6ff;
    --btn: #21262d;
    --btn-hover: #30363d;
  }
  * { box-sizing: border-box; }
  html, body {
    margin: 0; padding: 0; width: 100%; height: 100%; overflow: hidden;
    background: var(--bg); color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    font-size: 13px;
  }
  #app { display: flex; flex-direction: column; width: 100%; height: 100%; }
  #topbar {
    display: flex; flex-wrap: wrap; align-items: center; gap: 8px;
    padding: 8px 12px; background: var(--panel); border-bottom: 1px solid var(--border);
    z-index: 20;
  }
  .brand { font-weight: 700; color: var(--accent); margin-right: 4px; white-space: nowrap; }
  select, input[type="text"], button {
    background: var(--btn); color: var(--text); border: 1px solid var(--border);
    border-radius: 6px; padding: 5px 9px; font-size: 12px;
  }
  button { cursor: pointer; }
  button:hover, select:hover { background: var(--btn-hover); }
  button.primary { background: #1f6feb; border-color: #388bfd; }
  button.primary:hover { background: #388bfd; }
  .search-wrap { position: relative; min-width: 240px; flex: 1; max-width: 360px; }
  .search-wrap input { width: 100%; }
  #search-results {
    position: absolute; top: 100%; left: 0; right: 0; margin-top: 4px;
    background: var(--panel); border: 1px solid var(--border); border-radius: 6px;
    max-height: 320px; overflow-y: auto; z-index: 50; display: none;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.5);
  }
  .sr-item { padding: 7px 10px; border-bottom: 1px solid var(--border); cursor: pointer; }
  .sr-item:hover { background: var(--btn-hover); }
  .sr-uri { font-family: ui-monospace, Consolas, monospace; font-size: 11px; color: var(--accent); }
  .sr-path { font-size: 11px; color: var(--muted); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  label.chk { display: inline-flex; align-items: center; gap: 4px; cursor: pointer; user-select: none; color: var(--muted); }
  label.chk input { cursor: pointer; }
  #stats-pill { margin-left: auto; font-size: 11px; color: var(--muted); white-space: nowrap; }
  #main { display: flex; flex: 1; min-height: 0; position: relative; }
  #canvas-wrap { flex: 1; position: relative; min-width: 0; }
  canvas { display: block; width: 100%; height: 100%; cursor: grab; }
  canvas.dragging { cursor: grabbing; }
  #legend {
    position: absolute; left: 12px; bottom: 12px; background: rgba(22, 27, 34, 0.9);
    border: 1px solid var(--border); border-radius: 6px; padding: 8px 10px;
    font-size: 11px; color: var(--muted); pointer-events: none; line-height: 1.6;
  }
  .dot { display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 5px; vertical-align: middle; }
  #sidebar {
    width: 350px; background: var(--panel); border-left: 1px solid var(--border);
    padding: 12px; overflow-y: auto; display: flex; flex-direction: column; gap: 10px;
  }
  .sec-title { font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em; color: var(--muted); margin-bottom: 4px; }
  .card { background: var(--bg); border: 1px solid var(--border); border-radius: 6px; padding: 9px; }
  .mono { font-family: ui-monospace, Consolas, monospace; font-size: 11px; word-break: break-all; color: var(--accent); }
  .meta-row { font-size: 11px; color: var(--muted); margin-top: 4px; }
  .preview { font-size: 12px; line-height: 1.45; color: var(--text); margin-top: 6px; }
  .nb-list { list-style: none; padding: 0; margin: 0; display: flex; flex-direction: column; gap: 4px; max-height: 200px; overflow-y: auto; }
  .nb-btn {
    width: 100%; text-align: left; padding: 5px 7px; border-radius: 5px;
    background: var(--btn); border: 1px solid var(--border); color: var(--text); font-size: 11px;
  }
  .nb-btn:hover { background: var(--btn-hover); border-color: var(--accent); }
  .badge { display: inline-block; padding: 1px 5px; border-radius: 4px; font-size: 10px; margin-right: 5px; background: #30363d; color: #e6edf3; }
  .badge-cross { background: #1f6feb; }
  .badge-cites { background: #8957e5; }
  .badge-part { background: #238636; }
  .badge-ext { background: #6e7681; }
  .action-row { display: flex; gap: 6px; flex-wrap: wrap; margin-top: 8px; }
</style>
</head>
<body>
<div id="app">
  <div id="topbar">
    <span class="brand">2ndBrain Graph</span>
    <select id="mode-select" title="Select view">
      <option value="__overview__">Overview (All Standards)</option>
    </select>
    <div class="search-wrap">
      <input id="search-input" type="text" placeholder="Search clauses, URIs, or text..." autocomplete="off">
      <div id="search-results"></div>
    </div>
    <label class="chk"><input type="checkbox" id="chk-cross" checked> Cross-Std</label>
    <label class="chk"><input type="checkbox" id="chk-intra" checked> Internal CITES</label>
    <label class="chk"><input type="checkbox" id="chk-part" checked> PART_OF</label>
    <label class="chk"><input type="checkbox" id="chk-ext" checked> External</label>
    <button id="btn-physics">Pause Physics</button>
    <button id="btn-reset">Reset View</button>
    <span id="stats-pill"></span>
  </div>
  <div id="main">
    <div id="canvas-wrap">
      <canvas id="canvas"></canvas>
      <div id="legend">
        <div><span class="dot" style="background:#58a6ff"></span>Standard / Clause Nodes</div>
        <div><span class="dot" style="background:#8b949e"></span>External Ref</div>
        <div>Scroll: Zoom | Drag: Pan or Move Node | Double-Click Standard: Drill Down</div>
      </div>
    </div>
    <aside id="sidebar">
      <div class="card">
        <div class="sec-title">Inspector</div>
        <div id="inspector-body" class="meta-row">Click any node to inspect links, metadata, and text preview.</div>
      </div>
      <div class="card">
        <div class="sec-title">Outgoing Links (<span id="out-count">0</span>)</div>
        <ul id="out-list" class="nb-list"></ul>
      </div>
      <div class="card">
        <div class="sec-title">Incoming Links (<span id="in-count">0</span>)</div>
        <ul id="in-list" class="nb-list"></ul>
      </div>
    </aside>
  </div>
</div>
<script id="graph-data" type="application/json">__GRAPH_JSON__</script>
<script>
(() => {
  const DATA = JSON.parse(document.getElementById("graph-data").textContent);
  const stdById = Object.fromEntries(DATA.standards.map(s => [s.source_id, s]));
  const docUriToStd = Object.fromEntries(DATA.standards.map(s => [s.safe_doc_uri, s.source_id]));

  const EDGE_COLORS = {
    0: "#3fb950", // PART_OF
    1: "#bc8cff", // CITES intra
    2: "#58a6ff", // CITES cross-std
    3: "#6e7681"  // CITES external
  };
  const EDGE_NAMES = { 0: "PART_OF", 1: "CITES", 2: "CROSS_STD", 3: "EXTERNAL" };

  const chunks = DATA.chunks;
  const allEdges = DATA.edges;
  const outIdx = {};
  const inIdx = {};
  for (let i = 0; i < allEdges.length; i++) {
    const [s, d, k] = allEdges[i];
    (outIdx[s] ||= []).push([d, k]);
    (inIdx[d] ||= []).push([s, k]);
  }

  const canvas = document.getElementById("canvas");
  const ctx = canvas.getContext("2d");
  const modeSel = document.getElementById("mode-select");
  const searchInp = document.getElementById("search-input");
  const searchBox = document.getElementById("search-results");
  const chkCross = document.getElementById("chk-cross");
  const chkIntra = document.getElementById("chk-intra");
  const chkPart = document.getElementById("chk-part");
  const chkExt = document.getElementById("chk-ext");
  const btnPhysics = document.getElementById("btn-physics");
  const btnReset = document.getElementById("btn-reset");
  const statsPill = document.getElementById("stats-pill");

  for (const s of DATA.standards) {
    const opt = document.createElement("option");
    opt.value = "std:" + s.source_id;
    opt.textContent = `${s.source_id} (${s.chunk_count} chunks)`;
    modeSel.appendChild(opt);
  }

  let viewState = { mode: "overview", stdId: null, egoUri: null };
  let simNodes = [];
  let nodeMap = {};
  let simEdges = [];
  let cam = { x: 0, y: 0, zoom: 1 };
  let physicsRunning = true;
  let selectedId = null;
  let hoverId = null;
  let dragNode = null;
  let panning = false;
  let dragStart = { x: 0, y: 0 };
  let didMove = false;

  function resizeCanvas() {
    const rect = canvas.parentElement.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.floor(rect.width * dpr));
    canvas.height = Math.max(1, Math.floor(rect.height * dpr));
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  window.addEventListener("resize", resizeCanvas);
  resizeCanvas();

  function edgeVisible(kind) {
    if (kind === 0 && !chkPart.checked) return false;
    if (kind === 1 && !chkIntra.checked) return false;
    if (kind === 2 && !chkCross.checked) return false;
    if (kind === 3 && !chkExt.checked) return false;
    return true;
  }

  function layoutSeed(nodes) {
    const n = nodes.length;
    const rad = Math.max(180, Math.sqrt(n) * 42);
    nodes.forEach((nd, i) => {
      const a = (2 * Math.PI * i) / Math.max(1, n);
      nd.x = Math.cos(a) * rad * (0.65 + 0.35 * ((i * 7) % 11) / 11);
      nd.y = Math.sin(a) * rad * (0.65 + 0.35 * ((i * 13) % 11) / 11);
      nd.vx = 0;
      nd.vy = 0;
    });
  }

  function loadOverview() {
    viewState = { mode: "overview", stdId: null, egoUri: null };
    modeSel.value = "__overview__";
    selectedId = null;
    const nodes = [];
    const seen = new Set();
    for (const s of DATA.standards) {
      const id = "STD:" + s.source_id;
      seen.add(id);
      nodes.push({
        id,
        kind: "std",
        sourceId: s.source_id,
        label: s.source_id,
        sub: `${s.source_version} | ${s.chunk_count} chunks`,
        r: Math.max(14, Math.min(34, 10 + Math.sqrt(s.chunk_count) * 0.55)),
        color: s.color,
        pinned: false
      });
    }
    const edges = [];
    for (const se of DATA.std_edges) {
      if (se.kind === "external" && !chkExt.checked) continue;
      if (se.kind === "ingested" && !chkCross.checked) continue;
      const srcId = "STD:" + se.src;
      const dstId = se.kind === "ingested" ? ("STD:" + se.dst) : ("EXT:" + se.dst);
      if (!seen.has(dstId)) {
        seen.add(dstId);
        nodes.push({
          id: dstId,
          kind: "ext",
          sourceId: se.dst,
          label: se.dst.toUpperCase(),
          sub: "External reference",
          r: 10,
          color: "#6e7681",
          pinned: false
        });
      }
      edges.push({
        s: srcId,
        t: dstId,
        kind: se.kind === "ingested" ? 2 : 3,
        count: se.count,
        sampleUris: se.sample_uris || []
      });
    }
    activateGraph(nodes, edges);
  }

  function loadStandardDrill(stdId) {
    viewState = { mode: "std", stdId, egoUri: null };
    modeSel.value = "std:" + stdId;
    selectedId = null;

    const candidates = [];
    for (const [uri, info] of Object.entries(chunks)) {
      if (info[0] !== stdId) continue;
      const deg = (outIdx[uri] ? outIdx[uri].length : 0) + (inIdx[uri] ? inIdx[uri].length : 0);
      const depth = (info[1].match(/>/g) || []).length;
      const score = deg * 10 - depth * 4;
      candidates.push([uri, score, deg]);
    }
    candidates.sort((a, b) => b[1] - a[1]);
    const picked = new Set(candidates.slice(0, 320).map(x => x[0]));

    const nodes = [];
    const seen = new Set();
    const stdColor = stdById[stdId] ? stdById[stdId].color : "#58a6ff";
    for (const uri of picked) {
      const info = chunks[uri];
      seen.add(uri);
      const deg = (outIdx[uri] ? outIdx[uri].length : 0) + (inIdx[uri] ? inIdx[uri].length : 0);
      nodes.push({
        id: uri,
        kind: "chunk",
        sourceId: stdId,
        label: uri.split("#")[1] || uri,
        sub: info[1],
        r: Math.max(5, Math.min(16, 5 + Math.sqrt(deg))),
        color: stdColor,
        pinned: false
      });
    }

    const edges = [];
    for (const [s, d, k] of allEdges) {
      if (!edgeVisible(k)) continue;
      if (!picked.has(s)) continue;
      if (k === 2 && !seen.has(d)) {
        const tgtStd = docUriToStd[d];
        if (tgtStd) {
          seen.add(d);
          nodes.push({
            id: d,
            kind: "std",
            sourceId: tgtStd,
            label: tgtStd,
            sub: "Ingested Standard Hub",
            r: 16,
            color: stdById[tgtStd] ? stdById[tgtStd].color : "#58a6ff",
            pinned: false
          });
        }
      } else if (k === 3 && !seen.has(d) && nodes.length < 350) {
        seen.add(d);
        nodes.push({
          id: d,
          kind: "ext",
          sourceId: d.split("#")[1] || d,
          label: (d.split("#")[1] || d).toUpperCase(),
          sub: "External reference",
          r: 7,
          color: "#6e7681",
          pinned: false
        });
      }
      if (seen.has(d)) edges.push({ s, t: d, kind: k, count: 1 });
    }
    activateGraph(nodes, edges);
  }

  function loadEgoGraph(centerUri) {
    viewState = { mode: "ego", stdId: null, egoUri: centerUri };
    selectedId = centerUri;
    const hop0 = new Set([centerUri]);
    const hop1 = new Set([centerUri]);
    for (const [d, k] of (outIdx[centerUri] || [])) if (edgeVisible(k)) hop1.add(d);
    for (const [s, k] of (inIdx[centerUri] || [])) if (edgeVisible(k)) hop1.add(s);

    const hop2 = new Set(hop1);
    for (const u of hop1) {
      if (hop2.size >= 340) break;
      for (const [d, k] of (outIdx[u] || [])) {
        if (edgeVisible(k)) hop2.add(d);
        if (hop2.size >= 340) break;
      }
      for (const [s, k] of (inIdx[u] || [])) {
        if (edgeVisible(k)) hop2.add(s);
        if (hop2.size >= 340) break;
      }
    }

    const nodes = [];
    for (const u of hop2) {
      if (chunks[u]) {
        const info = chunks[u];
        nodes.push({
          id: u,
          kind: "chunk",
          sourceId: info[0],
          label: u.split("#")[1] || u,
          sub: info[1],
          r: u === centerUri ? 15 : 8,
          color: u === centerUri ? "#ffa657" : (stdById[info[0]] ? stdById[info[0]].color : "#58a6ff"),
          pinned: false
        });
      } else if (docUriToStd[u]) {
        const sid = docUriToStd[u];
        nodes.push({
          id: u,
          kind: "std",
          sourceId: sid,
          label: sid,
          sub: "Ingested Standard Hub",
          r: 15,
          color: stdById[sid] ? stdById[sid].color : "#58a6ff",
          pinned: false
        });
      } else {
        nodes.push({
          id: u,
          kind: "ext",
          sourceId: u.split("#")[1] || u,
          label: (u.split("#")[1] || u).toUpperCase(),
          sub: "Target URI",
          r: 7,
          color: "#6e7681",
          pinned: false
        });
      }
    }

    const edges = [];
    for (const [s, d, k] of allEdges) {
      if (!edgeVisible(k)) continue;
      if (hop2.has(s) && hop2.has(d)) edges.push({ s, t: d, kind: k, count: 1 });
    }
    activateGraph(nodes, edges);
    updateInspector(centerUri);
  }

  function activateGraph(nodes, edges) {
    simNodes = nodes;
    nodeMap = Object.fromEntries(nodes.map(n => [n.id, n]));
    simEdges = edges.filter(e => nodeMap[e.s] && nodeMap[e.t]);
    layoutSeed(simNodes);
    cam = { x: 0, y: 0, zoom: 1 };
    physicsRunning = true;
    btnPhysics.textContent = "Pause Physics";
    statsPill.textContent = `Showing ${simNodes.length} nodes | ${simEdges.length} edges (DB: ${Object.keys(chunks).length} chunks, ${allEdges.length} edges)`;
    if (!selectedId) clearInspector();
  }

  function clearInspector() {
    document.getElementById("inspector-body").innerHTML = "Click any node to inspect links, metadata, and text preview.";
    document.getElementById("out-count").textContent = "0";
    document.getElementById("in-count").textContent = "0";
    document.getElementById("out-list").innerHTML = "";
    document.getElementById("in-list").innerHTML = "";
  }

  function escapeHtml(str) {
    return String(str).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }

  function badgeClass(k) {
    if (k === 0) return "badge-part";
    if (k === 1) return "badge-cites";
    if (k === 2) return "badge-cross";
    return "badge-ext";
  }

  function updateInspector(id) {
    const nd = nodeMap[id];
    const body = document.getElementById("inspector-body");
    const outList = document.getElementById("out-list");
    const inList = document.getElementById("in-list");
    outList.innerHTML = "";
    inList.innerHTML = "";

    let html = "";
    if (nd && nd.kind === "std") {
      const st = stdById[nd.sourceId];
      html += `<div><strong>${escapeHtml(nd.sourceId)}</strong> (${escapeHtml(st ? st.source_version : "")})</div>`;
      html += `<div class="meta-row">${st ? escapeHtml(st.authority_tier) + " | " + st.chunk_count + " chunks" : ""}</div>`;
      html += `<div class="action-row"><button class="primary" onclick="window.__drillStd('${escapeHtml(nd.sourceId)}')">Drill Into Standard</button></div>`;
    } else if (chunks[id]) {
      const c = chunks[id];
      html += `<div class="mono">${escapeHtml(id)}</div>`;
      html += `<div class="meta-row"><strong>${escapeHtml(c[0])}</strong> | ${escapeHtml(c[2])} | ${c[3]} tok</div>`;
      html += `<div class="meta-row">${escapeHtml(c[1])}</div>`;
      if (c[4]) html += `<div class="meta-row">File: ${escapeHtml(c[4])}:${c[5]}</div>`;
      html += `<div class="preview">${escapeHtml(c[6])}</div>`;
      html += `<div class="action-row">
        <button class="primary" onclick="window.__focusEgo('${escapeHtml(id)}')">Focus Ego-Graph (2-hop)</button>
        <button onclick="window.__drillStd('${escapeHtml(c[0])}')">View ${escapeHtml(c[0])}</button>
      </div>`;
    } else {
      html += `<div class="mono">${escapeHtml(id)}</div><div class="meta-row">${nd ? escapeHtml(nd.sub) : ""}</div>`;
    }
    body.innerHTML = html;

    const outs = [];
    const ins = [];
    for (const e of simEdges) {
      if (e.s === id) outs.push([e.t, e.kind, e.count]);
      if (e.t === id) ins.push([e.s, e.kind, e.count]);
    }
    document.getElementById("out-count").textContent = String(outs.length);
    document.getElementById("in-count").textContent = String(ins.length);

    function appendNb(ul, targetId, kind, count) {
      const li = document.createElement("li");
      const btn = document.createElement("button");
      btn.className = "nb-btn";
      const lbl = nodeMap[targetId] ? `${nodeMap[targetId].label} (${nodeMap[targetId].sourceId})` : targetId;
      const cntTxt = count > 1 ? ` x${count}` : "";
      btn.innerHTML = `<span class="badge ${badgeClass(kind)}">${EDGE_NAMES[kind]}${cntTxt}</span>${escapeHtml(lbl)}`;
      btn.onclick = () => {
        selectedId = targetId;
        updateInspector(targetId);
        if (nodeMap[targetId]) {
          cam.x = -nodeMap[targetId].x * cam.zoom;
          cam.y = -nodeMap[targetId].y * cam.zoom;
        }
      };
      li.appendChild(btn);
      ul.appendChild(li);
    }

    outs.slice(0, 80).forEach(([t, k, cnt]) => appendNb(outList, t, k, cnt));
    ins.slice(0, 80).forEach(([s, k, cnt]) => appendNb(inList, s, k, cnt));
  }

  window.__drillStd = (sid) => loadStandardDrill(sid);
  window.__focusEgo = (uri) => loadEgoGraph(uri);

  function stepPhysics() {
    if (!physicsRunning || simNodes.length === 0) return;
    const n = simNodes.length;
    const repulse = viewState.mode === "overview" ? 9500 : 2800;
    for (let i = 0; i < n; i++) {
      const a = simNodes[i];
      for (let j = i + 1; j < n; j++) {
        const b = simNodes[j];
        let dx = a.x - b.x;
        let dy = a.y - b.y;
        let d2 = dx * dx + dy * dy + 0.01;
        if (d2 > 220000) continue;
        const f = repulse / d2;
        const dist = Math.sqrt(d2);
        const fx = (dx / dist) * f;
        const fy = (dy / dist) * f;
        a.vx += fx; a.vy += fy;
        b.vx -= fx; b.vy -= fy;
      }
    }
    const targetLen = viewState.mode === "overview" ? 190 : 85;
    const kSpring = viewState.mode === "overview" ? 0.015 : 0.04;
    for (const e of simEdges) {
      const a = nodeMap[e.s], b = nodeMap[e.t];
      if (!a || !b) continue;
      const dx = b.x - a.x;
      const dy = b.y - a.y;
      const dist = Math.sqrt(dx * dx + dy * dy) || 1;
      const diff = (dist - targetLen) * kSpring;
      const fx = (dx / dist) * diff;
      const fy = (dy / dist) * diff;
      a.vx += fx; a.vy += fy;
      b.vx -= fx; b.vy -= fy;
    }
    for (const nd of simNodes) {
      nd.vx += -nd.x * 0.004;
      nd.vy += -nd.y * 0.004;
      if (nd.pinned || nd === dragNode) {
        nd.vx = 0; nd.vy = 0;
        continue;
      }
      nd.vx *= 0.84;
      nd.vy *= 0.84;
      nd.x += Math.max(-18, Math.min(18, nd.vx));
      nd.y += Math.max(-18, Math.min(18, nd.vy));
    }
  }

  function draw() {
    const rect = canvas.parentElement.getBoundingClientRect();
    const w = rect.width, h = rect.height;
    ctx.clearRect(0, 0, w, h);
    ctx.save();
    ctx.translate(w / 2 + cam.x, h / 2 + cam.y);
    ctx.scale(cam.zoom, cam.zoom);

    const active = selectedId || hoverId;
    const adj = new Set();
    if (active) {
      adj.add(active);
      for (const e of simEdges) {
        if (e.s === active) adj.add(e.t);
        if (e.t === active) adj.add(e.s);
      }
    }

    for (const e of simEdges) {
      const a = nodeMap[e.s], b = nodeMap[e.t];
      if (!a || !b) continue;
      const isFocus = !active || e.s === active || e.t === active;
      ctx.strokeStyle = EDGE_COLORS[e.kind] || "#6e7681";
      ctx.globalAlpha = isFocus ? (e.kind === 2 ? 0.85 : 0.65) : 0.08;
      ctx.lineWidth = viewState.mode === "overview" ? Math.max(1, Math.min(5, Math.log2(e.count + 1))) : 1.1;
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.stroke();

      const ang = Math.atan2(b.y - a.y, b.x - a.x);
      const ax = b.x - Math.cos(ang) * (b.r + 3);
      const ay = b.y - Math.sin(ang) * (b.r + 3);
      ctx.beginPath();
      ctx.moveTo(ax, ay);
      ctx.lineTo(ax - Math.cos(ang - 0.35) * 7, ay - Math.sin(ang - 0.35) * 7);
      ctx.lineTo(ax - Math.cos(ang + 0.35) * 7, ay - Math.sin(ang + 0.35) * 7);
      ctx.closePath();
      ctx.fillStyle = EDGE_COLORS[e.kind] || "#6e7681";
      ctx.fill();

      if (viewState.mode === "overview" && e.count > 1 && isFocus) {
        ctx.fillStyle = "#c9d1d9";
        ctx.font = "10px sans-serif";
        ctx.fillText(String(e.count), (a.x + b.x) / 2 + 4, (a.y + b.y) / 2 - 4);
      }
    }

    for (const nd of simNodes) {
      const isFocus = !active || adj.has(nd.id);
      ctx.globalAlpha = isFocus ? 1.0 : 0.16;
      ctx.beginPath();
      ctx.arc(nd.x, nd.y, nd.r, 0, Math.PI * 2);
      ctx.fillStyle = nd.color;
      ctx.fill();
      if (nd.id === selectedId || nd.id === hoverId) {
        ctx.lineWidth = 2.5;
        ctx.strokeStyle = "#ffffff";
        ctx.stroke();
      }
      if (viewState.mode === "overview" || nd.r >= 9 || cam.zoom > 1.25 || nd.id === selectedId || nd.id === hoverId) {
        ctx.fillStyle = "#e6edf3";
        ctx.font = nd.kind === "std" ? "bold 11px sans-serif" : "10px sans-serif";
        ctx.fillText(nd.label, nd.x + nd.r + 4, nd.y + 3);
      }
    }
    ctx.restore();
  }

  function screenToWorld(sx, sy) {
    const rect = canvas.getBoundingClientRect();
    return {
      x: (sx - rect.width / 2 - cam.x) / cam.zoom,
      y: (sy - rect.height / 2 - cam.y) / cam.zoom
    };
  }

  function hitTest(wx, wy) {
    for (let i = simNodes.length - 1; i >= 0; i--) {
      const nd = simNodes[i];
      const dx = wx - nd.x, dy = wy - nd.y;
      if (dx * dx + dy * dy <= (nd.r + 4) * (nd.r + 4)) return nd;
    }
    return null;
  }

  canvas.addEventListener("mousedown", ev => {
    const rect = canvas.getBoundingClientRect();
    const sx = ev.clientX - rect.left, sy = ev.clientY - rect.top;
    const wpt = screenToWorld(sx, sy);
    const hit = hitTest(wpt.x, wpt.y);
    didMove = false;
    dragStart = { x: ev.clientX, y: ev.clientY };
    if (hit) {
      dragNode = hit;
    } else {
      panning = true;
      canvas.classList.add("dragging");
    }
  });

  window.addEventListener("mousemove", ev => {
    const rect = canvas.getBoundingClientRect();
    const sx = ev.clientX - rect.left, sy = ev.clientY - rect.top;
    if (dragNode) {
      if (Math.hypot(ev.clientX - dragStart.x, ev.clientY - dragStart.y) > 3) didMove = true;
      const wpt = screenToWorld(sx, sy);
      dragNode.x = wpt.x;
      dragNode.y = wpt.y;
      dragNode.pinned = true;
    } else if (panning) {
      if (Math.hypot(ev.clientX - dragStart.x, ev.clientY - dragStart.y) > 3) didMove = true;
      cam.x += ev.clientX - dragStart.x;
      cam.y += ev.clientY - dragStart.y;
      dragStart = { x: ev.clientX, y: ev.clientY };
    } else {
      const wpt = screenToWorld(sx, sy);
      const hit = hitTest(wpt.x, wpt.y);
      hoverId = hit ? hit.id : null;
    }
  });

  window.addEventListener("mouseup", () => {
    if (dragNode && !didMove) {
      selectedId = dragNode.id;
      updateInspector(dragNode.id);
    } else if (panning && !didMove) {
      selectedId = null;
      clearInspector();
    }
    dragNode = null;
    panning = false;
    canvas.classList.remove("dragging");
  });

  canvas.addEventListener("dblclick", ev => {
    const rect = canvas.getBoundingClientRect();
    const wpt = screenToWorld(ev.clientX - rect.left, ev.clientY - rect.top);
    const hit = hitTest(wpt.x, wpt.y);
    if (hit && hit.kind === "std") loadStandardDrill(hit.sourceId);
  });

  canvas.addEventListener("wheel", ev => {
    ev.preventDefault();
    const rect = canvas.getBoundingClientRect();
    const sx = ev.clientX - rect.left - rect.width / 2;
    const sy = ev.clientY - rect.top - rect.height / 2;
    const factor = ev.deltaY < 0 ? 1.12 : 0.89;
    const newZoom = Math.max(0.15, Math.min(6, cam.zoom * factor));
    const scale = newZoom / cam.zoom;
    cam.x = sx - (sx - cam.x) * scale;
    cam.y = sy - (sy - cam.y) * scale;
    cam.zoom = newZoom;
  }, { passive: false });

  modeSel.addEventListener("change", () => {
    const val = modeSel.value;
    if (val === "__overview__") loadOverview();
    else if (val.startsWith("std:")) loadStandardDrill(val.slice(4));
  });

  function reloadCurrentView() {
    if (viewState.mode === "overview") loadOverview();
    else if (viewState.mode === "std") loadStandardDrill(viewState.stdId);
    else if (viewState.mode === "ego") loadEgoGraph(viewState.egoUri);
  }
  [chkCross, chkIntra, chkPart, chkExt].forEach(el => el.addEventListener("change", reloadCurrentView));

  btnPhysics.addEventListener("click", () => {
    physicsRunning = !physicsRunning;
    btnPhysics.textContent = physicsRunning ? "Pause Physics" : "Resume Physics";
  });
  btnReset.addEventListener("click", () => { cam = { x: 0, y: 0, zoom: 1 }; });

  searchInp.addEventListener("input", () => {
    const q = searchInp.value.trim().toLowerCase();
    if (q.length < 2) { searchBox.style.display = "none"; return; }
    const hits = [];
    for (const [uri, info] of Object.entries(chunks)) {
      if (uri.toLowerCase().includes(q) || info[1].toLowerCase().includes(q) || info[6].toLowerCase().includes(q)) {
        hits.push([uri, info]);
        if (hits.length >= 15) break;
      }
    }
    if (hits.length === 0) { searchBox.style.display = "none"; return; }
    searchBox.innerHTML = "";
    for (const [uri, info] of hits) {
      const div = document.createElement("div");
      div.className = "sr-item";
      div.innerHTML = `<div class="sr-uri">${escapeHtml(uri)}</div><div class="sr-path">${escapeHtml(info[1])}</div>`;
      div.onclick = () => {
        searchBox.style.display = "none";
        searchInp.value = uri;
        loadEgoGraph(uri);
      };
      searchBox.appendChild(div);
    }
    searchBox.style.display = "block";
  });

  document.addEventListener("click", ev => {
    if (!ev.target.closest(".search-wrap")) searchBox.style.display = "none";
  });

  loadOverview();
  function loop() {
    stepPhysics();
    draw();
    requestAnimationFrame(loop);
  }
  requestAnimationFrame(loop);
})();
</script>
</body>
</html>
"""


def export_interactive_graph_html(
    *,
    cfg: BrainConfig,
    store: SqliteStore,
    output_path: Path,
    domain_filter: str | None = None,
    rebuild_cross_refs: bool = False,
) -> dict[str, Any]:
    """Export an offline self-contained interactive HTML graph from brain.sqlite."""
    rebuilt: dict[str, int] | None = None
    if rebuild_cross_refs:
        rebuilt = rebuild_cross_standard_edges(store)

    if domain_filter:
        chunk_rows = store.conn.execute(
            "SELECT uri, domain, source_id, source_version, authority_tier, clause_path, "
            "token_count, file_relpath, start_line, text "
            "FROM chunks WHERE store = 'parsed' AND domain = ? ORDER BY source_id, uri",
            (domain_filter,),
        ).fetchall()
    else:
        chunk_rows = store.conn.execute(
            "SELECT uri, domain, source_id, source_version, authority_tier, clause_path, "
            "token_count, file_relpath, start_line, text "
            "FROM chunks WHERE store = 'parsed' ORDER BY source_id, uri"
        ).fetchall()

    chunks_compact: dict[str, list[Any]] = {}
    std_meta: dict[str, dict[str, Any]] = {}
    doc_uri_to_std: dict[str, str] = {}
    uri_to_source: dict[str, str] = {}

    for r in chunk_rows:
        uri = str(r["uri"])
        src_id = str(r["source_id"])
        src_ver = str(r["source_version"])
        chunks_compact[uri] = [
            src_id,
            str(r["clause_path"] or ""),
            str(r["authority_tier"]),
            int(r["token_count"] or 0),
            str(r["file_relpath"] or ""),
            int(r["start_line"] or 0),
            _clean_preview(str(r["text"] or ""), 360),
        ]
        uri_to_source[uri] = src_id
        if src_id not in std_meta:
            safe_doc_uri = f"vault://{_slugify(src_id, 32)}@{_slugify(src_ver, 24)}#_doc"
            doc_uri_to_std[safe_doc_uri] = src_id
            std_meta[src_id] = {
                "source_id": src_id,
                "source_version": src_ver,
                "authority_tier": str(r["authority_tier"]),
                "domain": str(r["domain"]),
                "chunk_count": 0,
                "safe_doc_uri": safe_doc_uri,
            }
        std_meta[src_id]["chunk_count"] += 1

    standards = []
    for idx, src_id in enumerate(sorted(std_meta.keys())):
        entry = std_meta[src_id]
        entry["color"] = _PALETTE[idx % len(_PALETTE)]
        standards.append(entry)

    edge_rows = store.conn.execute("SELECT src_uri, dst_uri, rel_type FROM edges").fetchall()
    edges_compact: list[list[Any]] = []
    agg_cross: dict[tuple[str, str, str], dict[str, Any]] = {}
    ext_totals: dict[str, int] = defaultdict(int)
    cross_std_count = 0
    external_count = 0

    for er in edge_rows:
        s_uri = str(er["src_uri"])
        if s_uri not in chunks_compact:
            continue
        d_uri = str(er["dst_uri"])
        rel = str(er["rel_type"])
        src_id = uri_to_source[s_uri]

        if rel == "PART_OF":
            edges_compact.append([s_uri, d_uri, 0])
            continue
        if rel == "CITES":
            if d_uri in doc_uri_to_std:
                tgt_std = doc_uri_to_std[d_uri]
                if tgt_std != src_id:
                    edges_compact.append([s_uri, d_uri, 2])
                    cross_std_count += 1
                    key = (src_id, tgt_std, "ingested")
                    rec = agg_cross.setdefault(
                        key, {"src": src_id, "dst": tgt_std, "kind": "ingested", "count": 0, "sample_uris": []}
                    )
                    rec["count"] += 1
                    if len(rec["sample_uris"]) < 6:
                        rec["sample_uris"].append(s_uri)
                continue
            if d_uri.startswith("vault://external@unresolved#"):
                ext_label = d_uri.split("#", 1)[1]
                edges_compact.append([s_uri, d_uri, 3])
                external_count += 1
                ext_totals[ext_label] += 1
                key = (src_id, ext_label, "external")
                rec = agg_cross.setdefault(
                    key, {"src": src_id, "dst": ext_label, "kind": "external", "count": 0, "sample_uris": []}
                )
                rec["count"] += 1
                if len(rec["sample_uris"]) < 6:
                    rec["sample_uris"].append(s_uri)
                continue
            edges_compact.append([s_uri, d_uri, 1])

    top_ext = {k for k, _ in sorted(ext_totals.items(), key=lambda kv: kv[1], reverse=True)[:20]}
    std_edges = [
        rec
        for rec in agg_cross.values()
        if rec["kind"] == "ingested" or rec["dst"] in top_ext
    ]
    std_edges.sort(key=lambda r: (0 if r["kind"] == "ingested" else 1, -r["count"], r["src"], r["dst"]))

    payload = {
        "standards": standards,
        "std_edges": std_edges,
        "chunks": chunks_compact,
        "edges": edges_compact,
    }
    raw_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    html_text = _HTML_TEMPLATE.replace("__GRAPH_JSON__", raw_json)
    atomic_write_lf(output_path, html_text)

    return {
        "output_path": str(output_path),
        "size_bytes": output_path.stat().st_size,
        "standards_count": len(standards),
        "chunks_count": len(chunks_compact),
        "edges_count": len(edges_compact),
        "cross_standard_edges": cross_std_count,
        "external_edges": external_count,
        "rebuilt_cross_refs": rebuilt,
    }
