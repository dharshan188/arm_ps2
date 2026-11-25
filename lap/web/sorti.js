/* Sorti v4 frontend — talks only to the local server (lap/server.py): GET /api/state (polled), MJPEG /video,
   POST /api/ask/{start,stop,text}, /api/lang, /api/welcome (start over), /api/video/{ready,done,skip}.
   The server owns the camera, mic and speaker; this page only shows state. Vanilla JS, no network beyond the Pi. */
(function () {
  "use strict";
  var $ = function (id) { return document.getElementById(id); };
  var LANGS = ["en", "hi", "ta"];
  var T = {
    status: {
      loading: { en: "Loading…", hi: "लोड हो रहा है…", ta: "லோட் ஆகுது…" },
      watching: { en: "Watching", hi: "देख रहा हूँ", ta: "பார்க்கிறேன்" },
      listening: { en: "Listening…", hi: "सुन रहा हूँ…", ta: "கேட்கிறேன்…" },
      thinking: { en: "Thinking…", hi: "सोच रहा हूँ…", ta: "யோசிக்கிறேன்…" },
      speaking: { en: "Speaking", hi: "बोल रहा हूँ", ta: "பேசுறேன்" },
      video: { en: "Playing video", hi: "वीडियो चल रहा है", ta: "வீடியோ ஓடுது" }
    },
    asst: { watching: "Active", listening: "Listening", thinking: "Thinking", speaking: "Speaking", video: "Video", loading: "Loading" },
    emptyTitle: { en: "Show your waste to the camera", hi: "अपना कचरा कैमरा को दिखाओ", ta: "உங்க குப்பைய கேமராவுக்கு காட்டுங்க" },
    emptySub: { en: "Sorti finds every item and tells you its bin.", hi: "Sorti हर चीज़ पहचानकर उसका बिन बताएगा।", ta: "Sorti ஒவ்வொரு பொருளுக்கும் பின் சொல்லும்." },
    checking: { en: "Checking…", hi: "जाँच रहा हूँ…", ta: "பார்க்கிறேன்…" },
    notSure: { en: "Item — not sure", hi: "पक्का नहीं", ta: "உறுதியா தெரியல" },
    items: { en: " items", hi: " चीज़ें", ta: " பொருட்கள்" },
    item: { en: " item", hi: " चीज़", ta: " பொருள்" },
    placeholder: { en: "Ask Sorti about disposal, materials or rules…", hi: "Sorti से पूछो — कौन सा बिन, रीसायकल…", ta: "Sorti கிட்ட கேளுங்க — எந்த பின், ரீசைக்கிள்…" },
    hint: { en: "Press the button on the mic and ask — English, हिंदी or தமிழ்", hi: "माइक का बटन दबाकर पूछो — English, हिंदी, தமிழ்", ta: "மைக் பட்டன அழுத்தி கேளுங்க — English, हिंदी, தமிழ்" },
    hintAlways: { en: "Always listening — switch the mic on and just speak", hi: "हमेशा सुन रहा हूँ — माइक ऑन करो और बोलो", ta: "எப்பவும் கேட்டுட்டு இருக்கேன் — மைக் ஆன் பண்ணி பேசுங்க" },
    hintWait: { en: "Wait for Sorti to finish, then speak", hi: "Sorti के खत्म होने तक रुको, फिर बोलो", ta: "Sorti முடிக்கிற வரை காத்திருங்க, அப்புறம் பேசுங்க" },
    hintRec: { en: "Listening… I'll answer when you pause", hi: "सुन रहा हूँ… रुकोगे तो जवाब दूँगा", ta: "கேட்கிறேன்… நிறுத்தினதும் பதில் சொல்றேன்" },
    thinking: { en: "Thinking", hi: "सोच रहा हूँ", ta: "யோசிக்கிறேன்" },
    learn: { en: "Learn more", hi: "और जानो", ta: "மேலும் தெரிஞ்சுக்கோங்க" },
    skip: { en: "Skip", hi: "छोड़ो", ta: "ஸ்கிப்" },
    playing: { en: "Video: ", hi: "वीडियो: ", ta: "வீடியோ: " },
    camOn: { en: "Camera connected", hi: "कैमरा चालू", ta: "கேமரா ஆன்" },
    camOff: { en: "Camera connecting", hi: "कैमरा जुड़ रहा है", ta: "கேமரா இணைகிறது" },
    busy: { en: "Just a moment — still answering", hi: "एक सेकंड — अभी जवाब दे रहा हूँ", ta: "ஒரு நிமிஷம் — பதில் சொல்லிட்டு இருக்கேன்" },
    offline: { en: "Reconnecting to Sorti…", hi: "Sorti से फिर जुड़ रहा है…", ta: "Sorti கூட மறுபடி இணைகிறது…" },
    pills: {
      en: ["Which bin is this?", "Is this recyclable?", "How can I reduce plastic at home?", "Where do I give e-waste?"],
      hi: ["ये किस बिन में जाएगा?", "क्या ये रीसायकल होता है?", "घर पर प्लास्टिक कैसे कम करें?"],
      ta: ["இது எந்த பின்ல போடணும்?", "இது ரீசைக்கிள் ஆகுமா?", "வீட்டுல பிளாஸ்டிக் எப்படி குறைக்கலாம்?"]
    }
  };
  var ICON = { biodegradable: "i-leaf", recyclable: "i-recycle", hazardous: "i-warn", e_waste: "i-chip", general: "i-bin" };
  var BIN_WORD = { green: { en: "Green bin", hi: "ग्रीन बिन", ta: "கிரீன் பின்" }, yellow: { en: "Yellow bin", hi: "येलो बिन", ta: "யெல்லோ பின்" }, blue: { en: "Blue bin", hi: "ब्लू बिन", ta: "ப்ளூ பின்" },
                   red: { en: "Red bin", hi: "रेड बिन", ta: "ரெட் பின்" }, grey: { en: "E-waste centre", hi: "ई-वेस्ट सेंटर", ta: "இ-வேஸ்ட் சென்டர்" } };

  var clockOff = null, clockRtt = 1e9;   // seconds to add to Date.now()/1000 to get the Pi's time
  var S = null, lang = "en", lastId = 0, pollN = 0, failures = 0, msgEls = {}, staticLang = null;
  var el = { body: document.body, msgs: $("msgs"), feed: $("feed"), q: $("q"), mic: $("mic") };

  function tr(o, l) { return o ? (o[l] || o.en || "") : ""; }
  function setText(n, t) { if (n && n.textContent !== t) n.textContent = t; }
  function icon(id) { var s = document.createElementNS("http://www.w3.org/2000/svg", "svg"); var u = document.createElementNS("http://www.w3.org/2000/svg", "use"); u.setAttribute("href", "#" + id); s.appendChild(u); return s; }
  function mk(tag, cls, text) { var n = document.createElement(tag); if (cls) n.className = cls; if (text != null) n.textContent = text; return n; }
  function pct(x) { return Math.round(Math.max(0, Math.min(1, +x || 0)) * 100) + "%"; }
  function clock(t) { var d = t ? new Date(t * 1000) : new Date(); return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }); }

  var toastT = null;
  function toast(msg, sticky) {
    var n = $("toast"); n.textContent = msg; n.hidden = false; clearTimeout(toastT);
    if (!sticky) toastT = setTimeout(function () { n.hidden = true; }, 2600);
  }
  function post(path, body) {
    return fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body || {}) })
      .then(function (r) { return r.json().catch(function () { return {}; }).then(function (j) { j._ok = r.ok; return j; }); })
      .catch(function () { toast(tr(T.offline, lang)); return { _ok: false }; });
  }

  // ------------------------------------------------------------------ polling
  function poll() {
    var full = (pollN++ % 8 === 0), sent = Date.now();
    fetch("/api/state?since=" + (full ? 0 : lastId), { cache: "no-store" })
      .then(function (r) { if (!r.ok) throw new Error(r.status); return r.json(); })
      .then(function (st) {
        var got = Date.now(), rtt = got - sent;
        // Pi clock - this computer's clock (the Pi has no RTC; they can differ by seconds). Prefer low-RTT samples.
        if (typeof st.now === "number") {
          var off = st.now - (sent + got) / 2000;
          if (clockOff === null || rtt < clockRtt + 15) { clockOff = clockOff === null ? off : clockOff * 0.7 + off * 0.3; clockRtt = Math.min(rtt, clockRtt * 1.02 + 1); }
        }
        if (failures) $("toast").hidden = true; failures = 0; apply(st, full); setTimeout(poll, 250);
      })
      .catch(function () { failures++; pollN = 0; if (failures > 2) toast(tr(T.offline, lang), true); setTimeout(poll, Math.min(250 * Math.pow(2, failures), 4000)); });
  }

  function apply(st, full) {
    S = st;
    if (LANGS.indexOf(st.lang) >= 0) lang = st.lang;
    if (staticLang !== lang) localise();
    var status = st.status || "loading";
    el.body.setAttribute("data-status", status);
    setText($("status-text"), tr(T.status[status], lang));
    setText($("asst-state").lastChild, T.asst[status] || "Active");
    // the server's recorder is the truth (rec_s > 0 while arecord runs); a click is trusted for 1.5 s until it shows up
    var always = !!st.always_listening;
    var rec = always ? status === "listening" : (Date.now() < micUntil ? micWant : st.rec_s > 0);
    el.body.classList.toggle("rec", rec);
    el.body.classList.toggle("always", always);
    // always-listening: the level bar is live, so you can see whether the mic switch is on
    el.mic.style.setProperty("--lv", (rec || always) ? (+st.mic_level || 0).toFixed(3) : 0);
    var deaf = always && ["speaking", "video", "thinking", "loading"].indexOf(status) >= 0;
    setText($("hint"), tr(rec ? T.hintRec : always ? (deaf ? T.hintWait : T.hintAlways) : T.hint, lang));
    var m = st.models || {};
    $("sys").textContent = (st.ram_mb ? "RAM " + (st.ram_mb / 1024).toFixed(1) + " GB" : "") +
      (m.llama === "ready" ? " · LLM ready" : m.llama === "loading" ? " · LLM loading" : "");
    renderCamera(st);
    renderDetections(st);
    renderMessages(st.messages || [], full);
    renderThinking(status);
    renderVideo(st);
  }

  function localise() {
    staticLang = lang;
    document.documentElement.lang = lang;
    Array.prototype.forEach.call($("langs").children, function (b) { b.setAttribute("aria-pressed", String(b.dataset.lang === lang)); });
    el.q.placeholder = tr(T.placeholder, lang);
    setText($("det-empty-title"), tr(T.emptyTitle, lang));
    setText($("det-empty-sub"), tr(T.emptySub, lang));
    setText($("edu-kicker"), tr(T.learn, lang));
    setText($("edu-skip"), tr(T.skip, lang));
    var p = $("pills"); p.textContent = "";
    (T.pills[lang] || T.pills.en).forEach(function (q) {
      var b = mk("button", "", q); b.type = "button"; b.onclick = function () { askText(q); }; p.appendChild(b);
    });
  }

  // ------------------------------------------------------------------ camera + detections
  var camOk = false;
  el.feed.addEventListener("load", function () { camOk = true; $("cam-off").hidden = true; });
  el.feed.addEventListener("error", function () { camOk = false; $("cam-off").hidden = false; setTimeout(function () { el.feed.src = "/video?r=" + Date.now(); }, 2000); });

  function confirmedObjects(st) {
    return (st.objects || []).filter(function (o) { return o && o.confirmed && o.cls !== "background"; });
  }
  function renderCamera(st) {
    var on = st.models && st.models.vision === "ready";
    $("cam-dot").classList.toggle("on", !!on);
    setText($("cam-state"), tr(on ? T.camOn : T.camOff, lang));
    var so = st.sorter || {}, live = $("cam-live");
    var soTxt = so.state === "sorting" ? "SORTER · SORTING" : so.state === "ready" ? "SORTER · READY" : "LIVE STREAM";
    setText(live, soTxt);
    live.style.color = so.state === "sorting" ? "var(--amber)" : so.state === "ready" ? "var(--teal)" : "";
    var fps = st.pred && st.pred.fps;
    var n = confirmedObjects(st).length;
    setText($("cam-meta"), (n ? n + (n === 1 ? tr(T.item, lang) : tr(T.items, lang)) + " • " : "") + (fps ? fps.toFixed(1) : "—") + " FPS");
  }

  var detKey = "";
  function renderDetections(st) {
    var objs = confirmedObjects(st);
    var unsure = (st.objects || []).filter(function (o) { return o && o.unsure; });
    // group identical classes: "2 × Plastic"
    var groups = [], by = {};
    objs.forEach(function (o) {
      if (!by[o.cls]) { by[o.cls] = { o: o, n: 0, conf: 0 }; groups.push(by[o.cls]); }
      by[o.cls].n++; by[o.cls].conf = Math.max(by[o.cls].conf, +o.conf || 0);
    });
    var key = lang + "|" + groups.map(function (g) { return g.o.cls + g.n + Math.round(g.conf * 20); }).join(",") + "|" + unsure.length;
    if (key === detKey) return;
    detKey = key;
    var list = $("det-list"); list.textContent = "";
    $("det-empty").hidden = groups.length > 0 || unsure.length > 0;
    groups.forEach(function (g) {
      var o = g.o, li = mk("li"); li.setAttribute("data-bin", o.color || "");
      var l = mk("div", "det-l"), ic = mk("div", "det-ic"); ic.appendChild(icon(ICON[o.stream] || "i-bin")); l.appendChild(ic);
      var nm = mk("div", "det-name");
      nm.appendChild(mk("span", "det-title", (g.n > 1 ? g.n + " × " : "") + (o.title || o.cls)));
      var ok = mk("span", "ok"); ok.appendChild(icon("i-check")); nm.appendChild(ok);
      l.appendChild(nm); li.appendChild(l);
      var badge = mk("span", "binbadge"); badge.appendChild(icon("i-bin"));
      badge.appendChild(mk("span", "", o.bin || tr(BIN_WORD[o.color], lang))); li.appendChild(badge);
      list.appendChild(li);
    });
    if (!groups.length && unsure.length) {
      var li = mk("li", "unsure"); li.setAttribute("data-bin", "general");
      var l = mk("div", "det-l"), ic = mk("div", "det-ic"); ic.appendChild(icon("i-recycle")); l.appendChild(ic);
      l.appendChild(mk("span", "det-title", tr(T.checking, lang))); li.appendChild(l); list.appendChild(li);
    }
  }

  // ------------------------------------------------------------------ chat
  function renderMessages(list, full) {
    if (full) {
      var keep = {}; list.forEach(function (m) { keep[m.id] = 1; });
      Object.keys(msgEls).forEach(function (id) { if (!keep[id]) { msgEls[id].remove(); delete msgEls[id]; } });
    }
    var added = false, nearBottom = el.msgs.scrollHeight - el.msgs.scrollTop - el.msgs.clientHeight < 80;
    list.forEach(function (m) {
      if (!m || m.id == null) return;
      var li = msgEls[m.id];
      var sig = m.text + "|" + (m.streaming ? 1 : 0) + "|" + (S.speaking_id === m.id ? 1 : 0) + "|" + (m.stt_s || "") + "|" + (m.llm_s || "");
      if (li && li._sig === sig) return;
      if (!li) {
        li = mk("li"); msgEls[m.id] = li; added = true;
        var after = null;
        Array.prototype.some.call(el.msgs.children, function (o) { if (+o.dataset.id > m.id) { after = o; return true; } return false; });
        el.msgs.insertBefore(li, after);
        li.dataset.id = m.id;
      }
      li._sig = sig;
      fill(li, m);
      if (m.id > lastId) lastId = m.id;
    });
    if (S && S.last_id > lastId) lastId = S.last_id;
    while (el.msgs.querySelectorAll("li[data-id]").length > 60) { var f = el.msgs.querySelector("li[data-id]"); delete msgEls[f.dataset.id]; f.remove(); }
    if (added || nearBottom) el.msgs.scrollTop = el.msgs.scrollHeight;
  }

  function fill(li, m) {
    var user = m.role === "user";
    li.className = "msg " + (user ? "user" : (m.kind || "answer")) + (m.streaming ? " streaming" : "") + (S && S.speaking_id === m.id ? " speaking" : "");
    li.lang = m.lang || lang;
    li.textContent = "";
    if (!user) {
      var av = mk("div", "av"); av.appendChild(icon(m.kind === "prompt" ? "i-leaf" : m.kind === "video" ? "i-play" : "i-bot")); li.appendChild(av);
    }
    var body = mk("div", "body"), b = mk("div", "bubble");
    if (m.kind === "video") {
      b.appendChild(icon("i-play"));
      var title = S && S.video && S.video.title;
      b.appendChild(mk("span", "", tr(T.playing, m.lang || lang) + (title || (m.text || "").split(/[.!?।]/)[0])));
    } else {
      b.appendChild(document.createTextNode(m.text || ""));
    }
    // bin card for a single explained item
    if (!user && m.stream && m.kind === "explain") {
      var col = { biodegradable: "green", recyclable: "yellow", hazardous: "red", e_waste: "grey" }[m.stream] || "general";
      var card = mk("div", "bincard"); card.setAttribute("data-bin", col);
      var bi = mk("div", "bi"); bi.appendChild(icon(ICON[m.stream] || "i-bin")); card.appendChild(bi);
      var tx = mk("div"); tx.appendChild(mk("div", "bn", tr(BIN_WORD[col], m.lang || lang))); card.appendChild(tx);
      b.appendChild(card);
    }
    body.appendChild(b);
    var meta = mk("div", "meta"); meta.appendChild(mk("span", "", clock(m.t)));
    if (m.stt_s != null) { var s1 = mk("span", "", "STT "); s1.appendChild(mk("b", "", m.stt_s + "s")); meta.appendChild(s1); }
    if (m.llm_s != null) { var s2 = mk("span", "", "LLM "); s2.appendChild(mk("b", "", m.llm_s + "s")); meta.appendChild(s2); }
    if (m.translation && !user) meta.appendChild(mk("span", "", "EN: " + m.translation));
    body.appendChild(meta);
    li.appendChild(body);
  }

  var thinkEl = null;
  function renderThinking(status) {
    var streaming = Object.keys(msgEls).some(function (k) { return msgEls[k].classList.contains("streaming"); });
    var on = status === "thinking" && !streaming;
    if (on && !thinkEl) {
      thinkEl = mk("li", "msg thinking-row");
      var av = mk("div", "av"); av.appendChild(icon("i-bot")); thinkEl.appendChild(av);
      var b = mk("div", "bubble"); b.appendChild(document.createTextNode(tr(T.thinking, lang) + " "));
      var d = mk("span", "dots"); d.appendChild(mk("i")); d.appendChild(mk("i")); d.appendChild(mk("i")); b.appendChild(d);
      thinkEl.appendChild(b); el.msgs.appendChild(thinkEl); el.msgs.scrollTop = el.msgs.scrollHeight;
    } else if (on && thinkEl.nextSibling) {
      el.msgs.appendChild(thinkEl);
    } else if (!on && thinkEl) { thinkEl.remove(); thinkEl = null; }
  }

  // ------------------------------------------------------------------ asking
  function askStart() {
    post("/api/ask/start").then(function (j) {
      if (!j._ok) { micSet(false); toast(j.error === "busy" ? tr(T.busy, lang) : (j.error || "Mic not ready")); }
    });
  }
  function askStop() { post("/api/ask/stop"); }
  var micWant = false, micUntil = 0;
  function micSet(on) { micWant = on; micUntil = Date.now() + 1500; el.body.classList.toggle("rec", on); }
  function recording() { return Date.now() < micUntil ? micWant : !!(S && S.rec_s > 0); }
  // the USB mic's own button starts/stops recording (server side); this icon only shows the state
  var spaceDown = false;      // hold Space to talk (when not typing)
  document.addEventListener("keydown", function (e) {
    if (e.code !== "Space" || e.repeat || document.activeElement === el.q || (S && S.always_listening)) return;
    e.preventDefault(); if (!spaceDown) { spaceDown = true; micSet(true); askStart(); }
  });
  document.addEventListener("keyup", function (e) {
    if (e.code !== "Space" || !spaceDown) return;
    e.preventDefault(); spaceDown = false; micSet(false); askStop();
  });
  function askText(q) {
    q = (q || "").trim(); if (!q) return;
    post("/api/ask/text", { text: q });
  }
  $("ask-form").addEventListener("submit", function (e) { e.preventDefault(); askText(el.q.value); el.q.value = ""; });
  $("langs").addEventListener("click", function (e) {
    var b = e.target.closest("button[data-lang]"); if (!b || b.dataset.lang === lang) return;
    lang = b.dataset.lang; localise(); post("/api/lang", { lang: lang });
  });
  $("restart").addEventListener("click", function () {
    el.msgs.textContent = ""; msgEls = {}; lastId = 0; pollN = 0; post("/api/welcome");
  });
  $("edu-skip").addEventListener("click", function () { post("/api/video/skip"); });

  // ------------------------------------------------------------------ educational video (A/V sync with the speaker)
  var V = { root: $("edu"), v: $("edu-video"), prog: $("edu-prog") }, videoId = null, videoT = null;
  function videoStop() {
    var v = V.v; v.onended = v.onerror = null;
    try { v.pause(); } catch (_) {}
    if (v.hasAttribute("src")) { v.removeAttribute("src"); try { v.load(); } catch (_) {} }
    v.removeAttribute("poster");
  }
  function renderVideo(st) {
    var vd = st.video && typeof st.video === "object" ? st.video : null;
    V.root.hidden = !vd;
    if (!vd) { if (videoId !== null) { videoStop(); videoId = null; } videoT = null; V.prog.style.setProperty("--p", 0); return; }
    var id = String(vd.id) + "|" + vd.url, silent = !!st.no_speaker, v = V.v;
    if (id !== videoId) {
      videoId = id; videoStop();
      v.muted = true; v.loop = false; v.playsInline = true;
      if (vd.poster) v.poster = vd.poster;
      if (silent) {                         // no speaker: play muted with captions, report the end
        v.onended = function () { if (videoId === id) post("/api/video/done"); };
        v.onerror = function () { if (videoId === id) post("/api/video/skip"); };
        v.autoplay = true; v.src = vd.url;
      } else {                              // speaker: buffer, sit at 0, tell the server; it starts the voice-over
        v.autoplay = false; v.preload = "auto"; v.src = vd.url;
        var sent = false, ready = function () {
          if (sent || videoId !== id) return; sent = true;
          try { v.pause(); v.currentTime = 0; } catch (_) {}
          post("/api/video/ready");
        };
        v.addEventListener("canplaythrough", ready, { once: true });
        v.addEventListener("canplay", function () { setTimeout(ready, 600); }, { once: true });
        try { v.load(); } catch (_) {}
      }
    }
    videoT = typeof vd.t === "number" && vd.t > 1e9 ? vd.t : null;
    setText($("edu-title"), vd.title || "");
    setText($("edu-credit"), vd.credit ? "Photo: " + vd.credit : "");
    var cap = $("edu-cap"); cap.hidden = !silent; if (silent) setText(cap, vd.text || "");
  }
  // keep currentTime = now - t (voice-over clock from the server); measured drift ~0.03 s
  function frame() {
    var v = V.v;
    if (videoId !== null && v.src) {
      if (videoT !== null) {
        // picture time = Pi now - voice-over start - speaker latency (Bluetooth buffers ~0.3 s)
        var want = Date.now() / 1000 + (clockOff || 0) - videoT - ((S && S.av_delay) || 0), dur = v.duration || 0;
        if (want < 0 || (dur && want >= dur)) { if (!v.paused) v.pause(); }
        else if (v.paused) { try { v.currentTime = want; } catch (_) {} var pr = v.play(); if (pr && pr.catch) pr.catch(function () {}); }
        else {
          var drift = v.currentTime - want;
          if (Math.abs(drift) > 0.4) { try { v.currentTime = want; } catch (_) {} v.playbackRate = 1; }
          else if (Math.abs(drift) > 0.04) v.playbackRate = Math.max(0.92, Math.min(1.08, 1 - drift * 0.8));
          else if (v.playbackRate !== 1) v.playbackRate = 1;
        }
      }
      if (v.duration) V.prog.style.setProperty("--p", Math.min(1, v.currentTime / v.duration).toFixed(4));
    }
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);

  localise();
  poll();
})();
