/* ============================================================
   FACT — project page behavior
   ============================================================ */
(function () {
  "use strict";

  /* ---------- scroll reveal ---------- */
  const io = new IntersectionObserver((entries) => {
    entries.forEach((e) => {
      if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); }
    });
  }, { threshold: 0.12, rootMargin: "0px 0px -8% 0px" });
  document.querySelectorAll(".reveal").forEach((el) => io.observe(el));

  /* ---------- kick autoplay on montage / diagnosis videos ---------- */
  function kick() {
    document.querySelectorAll("video[data-auto]").forEach((v) => {
      const p = v.play();
      if (p && p.catch) p.catch(() => {});
    });
  }
  window.addEventListener("load", kick);
  document.addEventListener("visibilitychange", () => { if (!document.hidden) kick(); });

  /* ---------- hero video custom controls ---------- */
  (function heroControls() {
    const box = document.getElementById("heroVid");
    if (!box) return;
    const v = box.querySelector("video");
    const speedBtn = box.querySelector('[data-act="speed"]');
    const SPEEDS = [1, 1.5, 2, 0.5];
    let si = 0;

    function syncPlay() {
      box.classList.toggle("playing", !v.paused);
      box.classList.toggle("paused", v.paused);
    }
    v.addEventListener("play", syncPlay);
    v.addEventListener("pause", syncPlay);
    syncPlay();  // no play/pause event fires on load — set initial state

    function syncMute() { box.classList.toggle("muted", v.muted); }
    v.addEventListener("volumechange", syncMute);
    syncMute();  // volumechange never fires on load — set initial icon state

    function toggle() { if (v.paused) { const p = v.play(); if (p && p.catch) p.catch(() => {}); } else v.pause(); }

    box.querySelector('[data-act="play"]').addEventListener("click", toggle);
    v.addEventListener("click", toggle);

    box.querySelector('[data-act="mute"]').addEventListener("click", () => { v.muted = !v.muted; });

    /* --- scrub bar --- */
    const scrub = box.querySelector('[data-act="scrub"]');
    const fill = box.querySelector(".vfill");
    const knob = box.querySelector(".vknob");
    const timeEl = box.querySelector('[data-act="time"]');

    const fmt = (s) => {
      if (!isFinite(s) || s < 0) s = 0;
      const m = Math.floor(s / 60);
      return m + ":" + String(Math.floor(s % 60)).padStart(2, "0");
    };

    function syncTime() {
      const d = v.duration;
      const pct = isFinite(d) && d > 0 ? (v.currentTime / d) * 100 : 0;
      fill.style.width = pct + "%";
      knob.style.left = pct + "%";
      scrub.setAttribute("aria-valuenow", Math.round(pct));
      scrub.setAttribute("aria-valuetext", fmt(v.currentTime) + " of " + fmt(d));
      timeEl.textContent = fmt(v.currentTime) + " / " + fmt(d);
    }
    v.addEventListener("timeupdate", syncTime);
    v.addEventListener("loadedmetadata", syncTime);
    v.addEventListener("durationchange", syncTime);
    syncTime();

    function seekTo(clientX) {
      const r = scrub.getBoundingClientRect();
      const ratio = Math.min(1, Math.max(0, (clientX - r.left) / r.width));
      if (isFinite(v.duration)) { v.currentTime = ratio * v.duration; syncTime(); }
    }

    scrub.addEventListener("pointerdown", (e) => {
      e.preventDefault();
      scrub.classList.add("dragging");
      scrub.setPointerCapture(e.pointerId);
      seekTo(e.clientX);
    });
    scrub.addEventListener("pointermove", (e) => {
      if (scrub.classList.contains("dragging")) seekTo(e.clientX);
    });
    const endDrag = (e) => {
      scrub.classList.remove("dragging");
      if (e.pointerId != null && scrub.hasPointerCapture(e.pointerId))
        scrub.releasePointerCapture(e.pointerId);
    };
    scrub.addEventListener("pointerup", endDrag);
    scrub.addEventListener("pointercancel", endDrag);

    scrub.addEventListener("keydown", (e) => {
      const step = e.shiftKey ? 10 : 5;
      if (e.key === "ArrowRight") { v.currentTime = Math.min(v.duration, v.currentTime + step); }
      else if (e.key === "ArrowLeft") { v.currentTime = Math.max(0, v.currentTime - step); }
      else if (e.key === "Home") { v.currentTime = 0; }
      else if (e.key === "End") { v.currentTime = v.duration; }
      else if (e.key === " " || e.key === "Enter") { toggle(); }
      else return;
      e.preventDefault();
      syncTime();
    });

    speedBtn.addEventListener("click", () => {
      si = (si + 1) % SPEEDS.length;
      v.playbackRate = SPEEDS[si];
      speedBtn.textContent = SPEEDS[si] + "×";
    });

    box.querySelector('[data-act="fs"]').addEventListener("click", () => {
      // exit if already fullscreen (desktop / Android)
      if (document.fullscreenElement || document.webkitFullscreenElement) {
        (document.exitFullscreen || document.webkitExitFullscreen).call(document);
        return;
      }
      // Element Fullscreen API — works on desktop and Android for the container
      if (box.requestFullscreen) { box.requestFullscreen(); return; }
      if (box.webkitRequestFullscreen) { box.webkitRequestFullscreen(); return; }
      // iOS Safari on iPhone has no element fullscreen — only the video element
      // itself can go fullscreen, via its own webkit method.
      if (v.webkitEnterFullscreen) { v.webkitEnterFullscreen(); return; }
      if (v.webkitSupportsFullscreen && v.webkitEnterFullscreen) v.webkitEnterFullscreen();
    });

    // start paused; reflect initial state
    syncPlay();
    syncMute();
  })();

  /* ============================================================
     Diagnosis — toggle swaps each clip for its force-profile version
     (same animation rendered with a live |F| trace beneath it)
     ============================================================ */
  (function diagToggle() {
    const sw = document.getElementById("fp-switch");
    const tript = document.getElementById("tript");
    if (!sw || !tript) return;
    const videos = [...tript.querySelectorAll("video[data-force]")];
    sw.addEventListener("click", () => {
      const on = sw.getAttribute("aria-pressed") !== "true";
      sw.setAttribute("aria-pressed", String(on));
      // the profile clips are taller (chart below the plug) — switch the box
      // to contain so the chart isn't cropped away
      tript.classList.toggle("fp-on", on);
      videos.forEach((v) => {
        v.src = on ? v.dataset.force : v.dataset.plain;
        v.load();
        const p = v.play();
        if (p && p.catch) p.catch(() => {});
      });
    });
  })();

  /* ============================================================
     LN vs Beta noise-schedule chart (SVG)
     ============================================================ */
  function logit(t) { return Math.log(t / (1 - t)); }
  function lnDensity(t, m, s) {
    const z = logit(t) + m;
    return (1 / (s * Math.sqrt(2 * Math.PI))) * (1 / (t * (1 - t))) * Math.exp(-(z * z) / (2 * s * s));
  }
  /* Lanczos log-gamma — needed to normalize Beta(a,b) for arbitrary a,b. */
  const LG = [676.5203681218851, -1259.1392167224028, 771.32342877765313,
              -176.61502916214059, 12.507343278686905, -0.13857109526572012,
              9.9843695780195716e-6, 1.5056327351493116e-7];
  function logGamma(z) {
    if (z < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * z)) - logGamma(1 - z);
    z -= 1;
    let x = 0.99999999999980993;
    for (let i = 0; i < LG.length; i++) x += LG[i] / (z + i + 1);
    const t = z + LG.length - 0.5;
    return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(x);
  }
  function logBeta(a, b) { return logGamma(a) + logGamma(b) - logGamma(a + b); }
  /* Beta(a,b) density, normalized so it integrates to 1 over (0,1). */
  function betaDensity(t, a = 1.5, b = 1.0) {
    if (t <= 0 || t >= 1) return 0;
    return Math.exp((a - 1) * Math.log(t) + (b - 1) * Math.log(1 - t) - logBeta(a, b));
  }

  function buildScheduleChart(svg, opts) {
    const { m = 1.5, s = 1.0, a = 1.5, b = 1.0, tau = 0.2 } = opts || {};
    /* The SVG fills its container width but keeps a fixed viewBox, so on a phone
       the 340-wide box renders at ~230px — a 0.68x scale that shrinks every
       label. Using a NARROWER viewBox on phones (200 vs 340) flips the scale
       above 1x, so text in viewBox units renders larger than its nominal size.
       Font sizes are then chosen for the on-screen result, not the box unit. */
    const narrow = window.matchMedia("(max-width: 560px)").matches;
    const W = narrow ? 200 : 340, H = narrow ? 165 : 150;
    const padL = narrow ? 20 : 26, padR = narrow ? 10 : 14, padT = narrow ? 16 : 18, padB = narrow ? 34 : 38;
    // in a 200-wide box at ~230px display (~1.15x), 8 units ≈ 9.5px on screen
    const FS = narrow ? 8 : 6, FS2 = narrow ? 8.5 : 6.5;
    const iw = W - padL - padR, ih = H - padT - padB;
    /* Sample the open interval, then anchor both ends at the domain edges.
       Sampling only (0,1) leaves each curve hanging mid-air where its density
       is still large — LN at t→0, Beta at t→1 — which reads as a clipped plot. */
    const N = 400, EPS = 1e-4;
    const xs = [], ln = [], be = [];
    const safe = (fn) => { const v = fn(); return isFinite(v) ? v : 0; };
    // both densities are evaluated just inside the open interval at the edges,
    // so a curve that is still high at t→0 or t→1 meets the axis instead of
    // stopping short of it
    for (let i = 0; i <= N; i++) {
      const t = i === 0 ? EPS : i === N ? 1 - EPS : i / N;
      xs.push(i === 0 ? 0 : i === N ? 1 : t);
      ln.push(safe(() => lnDensity(t, m, s)));
      be.push(safe(() => betaDensity(t, a, b)));
    }
    /* Fixed y-scale. Both curves are true densities (each integrates to 1), so a
       constant shared scale keeps their areas comparable AND decouples the
       sliders — adjusting one distribution never rescales the other. Peaks past
       the ceiling clip at the top, which is the intended tradeoff. */
    const norm = 5.5;
    const X = (t) => padL + t * iw;
    // clamp so a diverging LN peak is cut off at the top rather than overflowing
    const Y = (v) => padT + ih - Math.min(v / norm, 1) * ih * 0.92;

    function path(arr, close) {
      let d = "";
      for (let i = 0; i < xs.length; i++) {
        d += (i === 0 ? "M" : "L") + X(xs[i]).toFixed(1) + " " + Y(arr[i]).toFixed(1) + " ";
      }
      if (close) d += "L" + X(xs[xs.length - 1]).toFixed(1) + " " + Y(0) + " L" + X(xs[0]).toFixed(1) + " " + Y(0) + " Z";
      return d;
    }
    const NS = "http://www.w3.org/2000/svg";
    function el(tag, attrs) { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); return e; }

    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    svg.innerHTML = "";

    // axes
    svg.appendChild(el("line", { x1: padL, y1: padT + ih, x2: W - padR, y2: padT + ih, stroke: "var(--faint)", "stroke-width": 0.6 }));
    svg.appendChild(el("line", { x1: padL, y1: padT, x2: padL, y2: padT + ih, stroke: "var(--faint)", "stroke-width": 0.6 }));

    // tau < 0.2 shaded region
    svg.appendChild(el("rect", { x: padL, y: padT, width: (X(tau) - padL), height: ih, fill: "var(--good)", opacity: 0.07 }));
    const vline = el("line", { x1: X(tau), y1: padT, x2: X(tau), y2: padT + ih, stroke: "var(--dim)", "stroke-width": 0.6, "stroke-dasharray": "2.5 2.5" });
    svg.appendChild(vline);

    // beta fill + curve (purple)
    svg.appendChild(el("path", { d: path(be, true), fill: "var(--beta)", opacity: 0.08 }));
    svg.appendChild(el("path", { d: path(be, false), fill: "none", stroke: "var(--beta)", "stroke-width": 1.3, "stroke-linejoin": "round" }));
    // ln fill + curve (yellow)
    svg.appendChild(el("path", { d: path(ln, true), fill: "var(--ln)", opacity: 0.12 }));
    svg.appendChild(el("path", { d: path(ln, false), fill: "none", stroke: "var(--ln)", "stroke-width": 1.4, "stroke-linejoin": "round" }));

    /* Legend, top-right inside the plot. Drawn after the curves so it sits on
       top of them; a translucent backing keeps it readable if a curve runs
       underneath at some parameter setting. */
    const legend = [
      { color: "var(--beta)", label: "Beta" },
      { color: "var(--ln)", label: "Logit-Normal" },
    ];
    const lgW = 58, lgRowH = 8;
    const lgX = W - padR - lgW, lgY = padT + 4;
    svg.appendChild(el("rect", {
      x: lgX - 4, y: lgY - 5, width: lgW + 8, height: legend.length * lgRowH + 4,
      fill: "#fff", opacity: 0.72, rx: 2,
    }));
    legend.forEach((it, i) => {
      const y = lgY + i * lgRowH;
      svg.appendChild(el("line", {
        x1: lgX, y1: y, x2: lgX + 9, y2: y,
        stroke: it.color, "stroke-width": 1.6, "stroke-linecap": "round",
      }));
      const t = el("text", {
        x: lgX + 12, y: y + 2, fill: "var(--dim)",
        "font-size": FS, "font-family": "var(--mono)",
      });
      t.textContent = it.label;
      svg.appendChild(t);
    });

    // x ticks
    const tickY = padT + ih + 10;
    [0, 0.5, 1].forEach((t) => {
      const tx = el("text", { x: X(t), y: tickY, "text-anchor": "middle", fill: "var(--dim)", "font-size": FS2, "font-family": "var(--mono)" });
      tx.textContent = String(t);
      svg.appendChild(tx);
    });

    /* What τ means at each end — flow matching integrates from τ=1 (noise) down
       to τ=0 (the clean action), so low τ is where fine corrections are made.
       Labelling the ends is what makes the LN mass shift legible.
       Sits on its own row well below the ticks; measured so the two labels and
       the arrow between them never collide. */
    /* Anchored to the plot edges, not centred on the ticks — centring pushes
       both labels past the viewBox and clips them. */
    const endY = padT + ih + 20;
    const endLabel = (t, anchor, txt) => {
      const e = el("text", {
        x: X(t), y: endY, "text-anchor": anchor, fill: "var(--ink)",
        "font-size": FS, "font-weight": 700, "font-family": "var(--mono)",
      });
      e.textContent = txt;
      svg.appendChild(e);
    };
    endLabel(0, "start", "clean action");
    endLabel(1, "end", "pure noise");

    // arrow showing the denoising direction, τ=1 → τ=0
    const ay = endY - 2;
    const ax0 = X(0) + 50, ax1 = X(1) - 42;
    svg.appendChild(el("line", {
      x1: ax1, y1: ay, x2: ax0, y2: ay,
      stroke: "var(--faint)", "stroke-width": 0.6,
    }));
    svg.appendChild(el("path", {
      d: `M${(ax0 + 3).toFixed(1)} ${(ay - 2).toFixed(1)} L${ax0.toFixed(1)} ${ay.toFixed(1)} L${(ax0 + 3).toFixed(1)} ${(ay + 2).toFixed(1)}`,
      fill: "none", stroke: "var(--faint)", "stroke-width": 0.6,
      "stroke-linecap": "round", "stroke-linejoin": "round",
    }));
    /* y-axis label sits above the axis, outside the plot area entirely —
       rotating it into the left margin risks clipping at the viewBox edge. */
    const ax = el("text", {
      x: padL, y: padT - 4, "text-anchor": "middle", fill: "var(--dim)",
      "font-size": FS, "font-family": "var(--mono)",
    });
    ax.textContent = "p(τ)";
    svg.appendChild(ax);
    const lbl = el("text", { x: X(tau) + 4, y: padT + 12, fill: "var(--dim)", "font-size": FS, "font-family": "var(--mono)" });
    lbl.textContent = "τ=" + tau.toFixed(2);
    svg.appendChild(lbl);
  }
  const schedSvg = document.getElementById("sched-chart");
  if (schedSvg) {
    const DEFAULTS = { m: 1.5, s: 1.0, a: 1.5, b: 1.0, tau: 0.2 };
    const mI = document.getElementById("sc-m");
    const sI = document.getElementById("sc-s");
    const aI = document.getElementById("sc-a");
    const bI = document.getElementById("sc-b");
    const tI = document.getElementById("sc-t");

    // fraction of probability mass below tau, by numeric integration
    function massBelow(fn, tau) {
      const N = 2000;
      let below = 0, total = 0;
      for (let i = 1; i < N; i++) {
        const t = i / N;
        const v = fn(t);
        if (!isFinite(v)) continue;
        total += v;
        if (t < tau) below += v;
      }
      return total > 0 ? below / total : 0;
    }

    function render() {
      const m = mI ? +mI.value : DEFAULTS.m;
      const s = sI ? +sI.value : DEFAULTS.s;
      const a = aI ? +aI.value : DEFAULTS.a;
      const b = bI ? +bI.value : DEFAULTS.b;
      const tau = tI ? +tI.value : DEFAULTS.tau;
      buildScheduleChart(schedSvg, { m, s, a, b, tau });

      const set = (id, txt) => { const e = document.getElementById(id); if (e) e.textContent = txt; };
      set("sc-m-out", m.toFixed(1));
      set("sc-s-out", s.toFixed(2));
      set("sc-a-out", a.toFixed(1));
      set("sc-b-out", b.toFixed(1));
      set("sc-t-out", tau.toFixed(2));
      set("sc-tau", "τ = " + tau.toFixed(2));

      const lnMass = massBelow((t) => lnDensity(t, m, s), tau);
      const beMass = massBelow((t) => betaDensity(t, a, b), tau);
      set("sc-ratio", beMass > 0 ? (lnMass / beMass).toFixed(1) + "×" : "—");
    }

    [mI, sI, aI, bI, tI].forEach((el) => { if (el) el.addEventListener("input", render); });
    const reset = document.getElementById("sc-reset");
    if (reset) reset.addEventListener("click", () => {
      if (mI) mI.value = DEFAULTS.m;
      if (sI) sI.value = DEFAULTS.s;
      if (aI) aI.value = DEFAULTS.a;
      if (bI) bI.value = DEFAULTS.b;
      if (tI) tI.value = DEFAULTS.tau;
      render();
    });
    render();
  }

  /* ============================================================
     Collapsed title bar — appears once the hero title scrolls away
     ============================================================ */
  (function titleBar() {
    const bar = document.getElementById("titlebar");
    const h1 = document.querySelector(".header h1");
    if (!bar || !h1) return;

    /* Watch the h1 rather than a fixed scroll offset so the handoff happens at
       the same visual moment regardless of viewport or zoom. rootMargin pulls
       the trigger line below the bar's own height, so the collapsed title takes
       over only once the real one is genuinely gone — not while it's still
       half-visible underneath. */
    const obs = new IntersectionObserver(
      ([e]) => bar.classList.toggle("show", !e.isIntersecting),
      { threshold: 0, rootMargin: "-56px 0px 0px 0px" }
    );
    obs.observe(h1);

    // only the title scrolls to top — the nav links navigate normally
    const home = bar.querySelector(".tb-home");
    if (home) home.addEventListener("click", () => {
      closeMenu();
      const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
      window.scrollTo({ top: 0, behavior: reduce ? "auto" : "smooth" });
    });

    /* Mobile menu: the links collapse behind a hamburger below 1000px. */
    const burger = bar.querySelector("#tb-burger");
    const menu = bar.querySelector("#tb-menu");
    function closeMenu() {
      if (!burger || !menu) return;
      menu.classList.remove("is-open");
      burger.setAttribute("aria-expanded", "false");
      burger.setAttribute("aria-label", "Open menu");
    }
    if (burger && menu) {
      burger.addEventListener("click", (e) => {
        e.stopPropagation();
        const open = menu.classList.toggle("is-open");
        burger.setAttribute("aria-expanded", String(open));
        burger.setAttribute("aria-label", open ? "Close menu" : "Open menu");
      });
      // dismiss on link tap, outside click, or Escape
      menu.addEventListener("click", (e) => { if (e.target.closest("a")) closeMenu(); });
      document.addEventListener("click", (e) => {
        if (!bar.contains(e.target)) closeMenu();
      });
      document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeMenu(); });
      // returning to desktop width must not leave the panel stuck open
      window.matchMedia("(min-width: 1001px)").addEventListener("change", closeMenu);
    }
  })();

  /* ============================================================
     Time-aware force injection — click a component to view it
     ============================================================ */
  (function archPicker() {
    const list = document.getElementById("arch-list");
    const img = document.getElementById("arch-img");
    if (!list || !img) return;

    const VIEWS = {
      state: {
        src: "assets/icons/contact_state.png",
        alt: "Contact state injected into AdaRMSNorm layers",
        desc: "Targeting sensitivity modulation, we inject the current force reading into the normalization layers of the action generation module to adapt the policy's responsiveness to the contact state.",
      },
      history: {
        src: "assets/icons/contact_history.png",
        alt: "Contact history encoded and prepended as tokens",
        desc: "Targeting temporal structure, the preceding 2s of F/T readings are encoded independently by a shared causal encoder. The resulting tokens are prepended to the input of the action generation module, enabling the policy to reason over the trajectory of contact.",
      },
      gating: {
        /* two states of the same gate: closed in free space, open at contact.
           Alternating them shows the mechanism better than a single still. */
        frames: [
          { src: "assets/icons/contact_gating.png", note: "No contact — gradient blocked" },
          { src: "assets/icons/contact_gating_2.png", note: "Contact detected — gradient flows" },
        ],
        src: "assets/icons/contact_gating.png",
        alt: "Gradient gate blocking contact-free steps",
        desc: "Targeting contact sparsity, a gradient gate with threshold δ blocks gradient through the force encoding components for steps where no contact is detected, preventing the force encoders from fitting to uninformative near-zero readings.",
      },
    };

    const opts = [...list.querySelectorAll(".arch-opt")];

    let cycle = null;

    function select(key) {
      const v = VIEWS[key];
      if (!v) return;
      if (cycle) { clearInterval(cycle); cycle = null; }

      img.alt = v.alt;
      const desc = document.getElementById("arch-desc");
      if (desc && v.desc) desc.innerHTML = v.desc;

      const note = document.getElementById("arch-note");
      if (v.frames) {
        // alternate the gate open/closed so the mechanism is visible
        let i = 0;
        const show = () => {
          img.src = v.frames[i].src;
          if (note) { note.textContent = v.frames[i].note; note.hidden = false; }
          i = (i + 1) % v.frames.length;
        };
        show();
        if (!window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
          cycle = setInterval(show, 1600);
        }
      } else {
        img.src = v.src;
        if (note) { note.textContent = ""; note.hidden = true; }
      }

      opts.forEach((o) => {
        const on = o.dataset.arch === key;
        o.setAttribute("aria-selected", on ? "true" : "false");
        o.classList.toggle("is-on", on);
      });
    }

    opts.forEach((o) => {
      o.addEventListener("click", () => select(o.dataset.arch));
      o.addEventListener("keydown", (e) => {
        const i = opts.indexOf(o);
        let j = null;
        if (e.key === "ArrowDown" || e.key === "ArrowRight") j = (i + 1) % opts.length;
        else if (e.key === "ArrowUp" || e.key === "ArrowLeft") j = (i - 1 + opts.length) % opts.length;
        if (j === null) return;
        e.preventDefault();
        opts[j].focus();
        select(opts[j].dataset.arch);
      });
    });

    // if an image is missing, say so rather than showing a broken icon
    img.addEventListener("error", () => { document.getElementById("archfig").classList.add("img-missing"); });
    img.addEventListener("load", () => { document.getElementById("archfig").classList.remove("img-missing"); });

    select("state");
  })();

  /* ============================================================
     Results bar chart — animate heights when in view
     ============================================================ */
  const chart = document.getElementById("barchart");
  if (chart) {
    const cap = 74;
    const cobs = new IntersectionObserver((entries) => {
      entries.forEach((e) => {
        if (!e.isIntersecting) return;
        chart.querySelectorAll(".col").forEach((c, i) => {
          const v = +c.dataset.val;
          setTimeout(() => { c.style.height = (v / cap * 100) + "%"; }, 120 + i * 110);
        });
        cobs.disconnect();
      });
    }, { threshold: 0.4 });
    cobs.observe(chart);
  }

  /* ============================================================
     Interactive rollout gallery
     ============================================================ */
  const G = document.getElementById("gallery");
  if (G) {
    const folder = { FACT: "FACT", ForceVLA: "ForceVLA", TAVLA: "TAVLA" };
    const suffix = { FACT: "fact", ForceVLA: "forcevla", TAVLA: "tavla" };
    const taskName = { plug: "Plug insertion", usb: "USB insertion", button: "Button push", key: "Key insertion", board: "Board erasing" };
    const methodName = { FACT: "FACT (ours)", ForceVLA: "ForceVLA", TAVLA: "TA-VLA" };

    const state = { task: "plug", method: "FACT", outcome: "success" };

    const stage = G.querySelector(".stage");
    const videos = [...G.querySelectorAll(".stage video")];
    const missingMsg = G.querySelector(".missing .msg");
    const replayBtn = G.querySelector("#replay");

    function src(view) {
      return `assets/videos/${state.task}/${folder[state.method]}/${state.task}_${view}_${state.outcome}_${suffix[state.method]}.mp4`;
    }
    // clips no longer loop — the pair ends together and waits for a replay
    function play() {
      stage.classList.remove("is-ended", "is-paused");
      videos.forEach((v) => {
        v.currentTime = 0;
        const p = v.play();
        if (p && p.catch) p.catch(() => {});
      });
    }
    function load() {
      stage.classList.remove("is-missing", "is-ended");
      // both views share one clip pair, so a failure on either means the
      // rollout doesn't exist — track them together.
      let failed = 0;
      videos.forEach((v) => {
        v.onerror = () => { if (++failed >= videos.length) stage.classList.add("is-missing"); };
        v.src = src(v.dataset.view);
        v.load();
        const p = v.play();
        if (p && p.catch) p.catch(() => {});
      });
      missingMsg.textContent = `No ${state.outcome} clip available for ${methodName[state.method]} on ${taskName[state.task].toLowerCase()}.`;
    }

    // the two clips can differ slightly in length: only offer replay once
    // every view has finished, so the button never interrupts the longer one.
    videos.forEach((v) => {
      v.addEventListener("ended", () => {
        if (videos.every((o) => o.ended)) stage.classList.add("is-ended");
      });
    });

    replayBtn.addEventListener("click", (e) => { e.stopPropagation(); play(); });

    // pause / play on click — both views move together
    stage.addEventListener("click", () => {
      if (stage.classList.contains("is-missing")) return;
      if (stage.classList.contains("is-ended")) { play(); return; }
      const paused = videos[0].paused;
      videos.forEach((v) => { if (paused) v.play(); else v.pause(); });
      stage.classList.toggle("is-paused", !paused);
    });

    G.querySelectorAll(".seg").forEach((seg) => {
      seg.addEventListener("click", (e) => {
        const btn = e.target.closest("button"); if (!btn) return;
        const key = seg.dataset.key;
        state[key] = btn.dataset.val;
        seg.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", String(b === btn)));
        stage.classList.remove("is-paused");
        loaded = true;      // a picker tap is explicit intent — load regardless
        load();
      });
    });

    /* Don't fetch or autoplay the clips on page load — wait until the gallery
       actually scrolls into view. A picker click before that also counts. */
    let loaded = false;
    const gobs = new IntersectionObserver((entries) => {
      entries.forEach((e) => {
        if (e.isIntersecting && !loaded) {
          loaded = true;
          load();
          gobs.disconnect();
        }
      });
    }, { threshold: 0.25 });
    gobs.observe(stage);
  }

  /* ============================================================
     BibTeX copy
     ============================================================ */
  const copyBtn = document.querySelector(".bibtex .copy");
  if (copyBtn) {
    copyBtn.addEventListener("click", () => {
      const txt = document.getElementById("bib-src").textContent.trim();
      navigator.clipboard.writeText(txt).then(() => {
        const old = copyBtn.textContent;
        copyBtn.textContent = "copied ✓";
        setTimeout(() => { copyBtn.textContent = old; }, 1600);
      }).catch(() => {});
    });
  }

  /* ============================================================
     Results table — hover crosshair (row via CSS, column via JS)
     ============================================================ */
  document.querySelectorAll("table.res").forEach((tbl) => {
    let lit = [];
    const clear = () => { lit.forEach((c) => c.classList.remove("col-hl")); lit = []; };
    const light = (cell) => {
      clear();
      if (!cell) return;
      const i = cell.cellIndex;
      if (i < 1) return;   // skip the method-name column
      tbl.querySelectorAll("tr").forEach((row) => {
        const c = row.cells[i];
        if (c) { c.classList.add("col-hl"); lit.push(c); }
      });
    };
    tbl.addEventListener("mouseover", (e) => {
      const cell = e.target.closest("td, th");
      light(cell && tbl.contains(cell) ? cell : null);
    });
    tbl.addEventListener("mouseleave", clear);
  });

  /* ============================================================
     Back to top
     ============================================================ */
  const toTop = document.getElementById("toTop");
  if (toTop) {
    const sync = () => toTop.classList.toggle("show", window.scrollY > 600);
    window.addEventListener("scroll", sync, { passive: true });
    sync();
    toTop.addEventListener("click", () => {
      const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
      window.scrollTo({ top: 0, behavior: reduce ? "auto" : "smooth" });
    });
  }
})();
