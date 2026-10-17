"use strict";

(() => {
  const $ = (id) => document.getElementById(id);
  const state = { file: null, config: null, job: null, results: null, jobs: [], timer: null, xhr: null, generation: 0, page: 0, video: null, seekVersion: 0, frames: [], frameIndex: -1, frameLimit: 24, resultView: "events" };
  const ACTIVE = new Set(["queued", "running"]);
  const STATUS = { queued: "В очереди", running: "Обрабатывается", completed: "Готово", partial: "Готово частично", failed: "Ошибка обработки", cancelled: "Остановлено", interrupted: "Прервано", uploading: "Загрузка" };
  const MODES = { trained: "Объекты кассы", desk: "Предметы на столе", people: "Люди и перемещения", source: "Исходное видео" };
  const STAGES = { trained: "Контекст кассы", desk: "Проверка рабочего места", people: "Присутствие людей", visual_states: "Состояния двери и сейфа", metadata: "Проверка видео", prepare: "Подготовка", export: "Сохранение результатов" };
  const PAGE_SIZE = 100;

  function element(tag, text, className) {
    const el = document.createElement(tag);
    if (text !== undefined && text !== null) el.textContent = String(text);
    if (className) el.className = className;
    return el;
  }
  function visible(el, show) { el.hidden = !show; }
  function savedJob(id) { try { if (id) localStorage.setItem("cash-video-job", id); else localStorage.removeItem("cash-video-job"); } catch (_) {} }
  function readSavedJob() { try { return localStorage.getItem("cash-video-job"); } catch (_) { return null; } }
  function safeURL(value) {
    try {
      const url = new URL(value, location.origin);
      return url.origin === location.origin && url.pathname.startsWith("/api/") ? url.href : null;
    } catch (_) { return null; }
  }
  function timeLabel(value) {
    const total = Math.max(0, Number(value) || 0);
    const hours = Math.floor(total / 3600);
    const minutes = Math.floor(total / 60) % 60;
    const seconds = Math.floor(total % 60);
    const base = hours ? `${hours}:${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}` : `${minutes}:${String(seconds).padStart(2, "0")}`;
    const tenth = Math.floor((total % 1) * 10 + 0.01);
    return tenth ? `${base},${tenth}` : base;
  }
  function sizeLabel(bytes) { return `${(Number(bytes) / 1024 / 1024).toLocaleString("ru-RU", { maximumFractionDigits: 1 })} МБ`; }
  function dateLabel(value) {
    const date = new Date(value);
    return Number.isNaN(date.getTime()) ? "" : date.toLocaleString("ru-RU", { day: "2-digit", month: "2-digit", hour: "2-digit", minute: "2-digit" });
  }
  function connectionError(message) { $("connection-alert").textContent = message || ""; visible($("connection-alert"), Boolean(message)); }
  function fileError(message) { $("file-error").textContent = message || ""; visible($("file-error"), Boolean(message)); }
  function userError(error) { return error && error.message ? error.message : "Не удалось выполнить действие. Попробуйте ещё раз."; }
  async function request(path, options = {}) {
    let response;
    try { response = await fetch(path, options); }
    catch (_) { throw new Error("Нет связи с локальным приложением. Проверьте, что оно запущено, и повторите действие."); }
    let data;
    try { data = await response.json(); } catch (_) { data = null; }
    if (!response.ok) {
      const detail = data && (data.detail || data.error);
      const error = new Error(typeof detail === "string" ? detail : `Не удалось выполнить действие (код ${response.status}).`);
      error.status = response.status;
      throw error;
    }
    return data;
  }
  function setControls() {
    const busy = Boolean(state.xhr) || Boolean(state.job && ACTIVE.has(state.job.status));
    const unavailable = ["trained", "desk", "people"].filter(key => state.config && state.config.capabilities && state.config.capabilities[key] === false);
    $("process-button").disabled = !state.config || !state.file || busy || unavailable.length > 0;
    $("video-file").disabled = busy;
    $("clear-file").disabled = busy;
    $("capability-warning").textContent = unavailable.length ? `Проверка недоступна: не найдены файлы моделей «${unavailable.map(key => MODES[key]).join("», «") }». Восстановите их в папке проекта и обновите страницу.` : "";
    visible($("capability-warning"), unavailable.length > 0);
    $("dropzone").classList.toggle("disabled", busy);
  }
  function selectFile(file) {
    if (state.xhr || (state.job && ACTIVE.has(state.job.status))) return;
    fileError("");
    if (!file) { state.file = null; visible($("selected-file"), false); setControls(); return; }
    const extension = file.name.split(".").pop().toLowerCase();
    if (!["mov", "mp4", "avi", "mkv", "m4v", "webm"].includes(extension)) {
      state.file = null; $("video-file").value = ""; visible($("selected-file"), false);
      fileError("Выберите видео в формате MP4, MOV, AVI, MKV, M4V или WebM."); setControls(); return;
    }
    const maxMB = Number(state.config && state.config.max_upload_mb) || 512;
    if (!file.size || file.size > maxMB * 1024 * 1024) {
      state.file = null; $("video-file").value = ""; visible($("selected-file"), false);
      fileError(file.size ? `Файл слишком большой. Максимальный размер — ${maxMB} МБ.` : "Этот файл пустой. Выберите другое видео."); setControls(); return;
    }
    state.file = file;
    $("selected-name").textContent = file.name;
    $("selected-size").textContent = sizeLabel(file.size);
    visible($("selected-file"), true);
    setControls();
  }
  function resetResults() {
    state.results = null; state.video = null; state.page = 0;
    visible($("results"), false); visible($("result-loading"), false);
    visible($("evidence-context"), false); $("evidence-context").textContent = "";
    $("result-video").pause(); $("result-video").removeAttribute("src"); $("result-video").load();
    $("filter-class").value = ""; $("filter-status").value = ""; $("filter-mode").value = "";
  }
  function beginUpload() {
    const file = state.file;
    if (!file || $("process-button").disabled) return;
    fileError(""); connectionError(""); clearTimeout(state.timer);
    const generation = ++state.generation;
    state.job = null; resetResults();
    visible($("empty-state"), false); visible($("job-panel"), true); visible($("processing-panel"), true);
    visible($("job-error"), false); $("stage-list").replaceChildren(); visible($("stage-errors"), false);
    $("job-name").textContent = file.name; $("video-meta").textContent = sizeLabel(file.size);
    $("job-status").textContent = "Загрузка"; $("job-status").className = "status-pill running";
    $("progress-label").textContent = "Загрузка видео"; $("progress-percent").textContent = "0%";
    $("job-progress").value = 0; $("progress-detail").textContent = "Передаём файл локальному приложению";
    $("cancel-button").disabled = false; $("cancel-button").textContent = "Остановить";
    const form = new FormData(); form.append("file", file); form.append("options", "{}");
    const xhr = new XMLHttpRequest(); state.xhr = xhr; setControls();
    xhr.open("POST", "/api/jobs");
    xhr.upload.addEventListener("progress", event => {
      if (generation !== state.generation) return;
      if (event.lengthComputable) {
        const progress = event.loaded / event.total; $("job-progress").value = progress;
        $("progress-percent").textContent = `${Math.round(progress * 100)}%`;
        if (progress >= 1) $("progress-detail").textContent = "Проверяем видео перед обработкой";
      }
    });
    xhr.addEventListener("load", () => {
      state.xhr = null;
      if (generation !== state.generation) { setControls(); return; }
      let data; try { data = JSON.parse(xhr.responseText); } catch (_) {}
      if (xhr.status >= 200 && xhr.status < 300 && data && data.id) {
        state.job = data; savedJob(data.id); renderJob(data); refreshHistory();
        if (ACTIVE.has(data.status)) schedulePoll(generation); else loadResults(data, generation);
      } else {
        const detail = data && (data.detail || data.error);
        uploadFailed(typeof detail === "string" ? detail : "Видео не удалось загрузить. Проверьте формат и попробуйте ещё раз.");
      }
      setControls();
    });
    xhr.addEventListener("error", () => { state.xhr = null; if (generation === state.generation) uploadFailed("Нет связи с приложением. Проверьте, что локальный сервер запущен, и повторите загрузку."); setControls(); });
    xhr.addEventListener("abort", () => { state.xhr = null; if (generation === state.generation) uploadFailed("Загрузка остановлена. Можно выбрать другое видео или запустить её снова.", true); setControls(); });
    xhr.send(form);
    $("workspace").focus({ preventScroll: true });
  }
  function uploadFailed(message, cancelled = false) {
    visible($("processing-panel"), false); visible($("job-error"), true);
    $("job-error").textContent = message;
    $("job-status").textContent = cancelled ? "Загрузка остановлена" : "Ошибка загрузки";
    $("job-status").className = "status-pill " + (cancelled ? "cancelled" : "failed");
  }
  function renderJob(job) {
    visible($("empty-state"), false); visible($("job-panel"), true);
    $("job-name").textContent = job.name || "Видео";
    $("job-status").textContent = STATUS[job.status] || job.status;
    $("job-status").className = "status-pill " + (Object.hasOwn(STATUS, job.status) ? job.status : "");
    const meta = job.video || {};
    const details = [];
    if (meta.duration != null) details.push(timeLabel(meta.duration));
    if (meta.width && meta.height) details.push(`${meta.width} × ${meta.height}`);
    if (meta.fps) details.push(`${Number(meta.fps).toLocaleString("ru-RU", { maximumFractionDigits: 1 })} кадр/с`);
    $("video-meta").textContent = details.join("   •   ") || dateLabel(job.created_at);
    const active = ACTIVE.has(job.status); visible($("processing-panel"), active);
    if (active) {
      const progress = Math.max(0, Math.min(1, Number(job.progress) || 0));
      $("progress-label").textContent = job.stage || STATUS[job.status];
      $("progress-percent").textContent = `${Math.round(progress * 100)}%`;
      $("job-progress").value = progress;
      $("progress-detail").textContent = "Страницу можно обновить — обработка продолжится";
      $("cancel-button").disabled = Boolean(job.cancel_requested);
      $("cancel-button").textContent = job.cancel_requested ? "Останавливаем…" : "Остановить";
    }
    visible($("job-error"), Boolean(job.error)); $("job-error").textContent = job.error || "";
    $("stage-list").replaceChildren();
    $("stage-error-details").replaceChildren();
    visible($("stage-errors"), false);
    (Array.isArray(job.stages) ? job.stages : []).forEach(stage => {
      const li = element("li"); li.append(element("span", "", "stage-dot " + stage.status));
      const label = stage.label || STAGES[stage.key] || stage.key;
      const suffix = stage.warning ? "с замечанием" : { completed: "готово", failed: "не выполнено", skipped: "пропущено", cancelled: "остановлено", pending: "ожидает", running: "в работе", queued: "ожидает" }[stage.status];
      li.append(element("span", `${label}${suffix ? ": " + suffix : ""}`));
      if (stage.error) {
        li.title = String(stage.error);
        const detail = element("div"); detail.append(element("strong", label), element("p", stage.error));
        $("stage-error-details").append(detail); visible($("stage-errors"), true);
      }
      if (stage.warning) {
        const detail = element("div", null, "stage-warning");
        detail.append(element("strong", label), element("p", stage.warning));
        $("stage-error-details").append(detail); visible($("stage-errors"), true);
        $("stage-errors").open = true;
      }
      $("stage-list").append(li);
    });
    setControls(); renderHistory();
  }
  function schedulePoll(generation, delay = 1500) {
    clearTimeout(state.timer);
    state.timer = setTimeout(() => pollJob(generation), delay);
  }
  async function pollJob(generation) {
    if (generation !== state.generation || !state.job) return;
    const id = state.job.id;
    try {
      const job = await request(`/api/jobs/${encodeURIComponent(id)}`);
      if (generation !== state.generation) return;
      connectionError(""); state.job = job; renderJob(job);
      if (ACTIVE.has(job.status)) schedulePoll(generation);
      else { await loadResults(job, generation); refreshHistory(); }
    } catch (error) {
      if (generation !== state.generation) return;
      connectionError(userError(error));
      if (error.status !== 404) schedulePoll(generation, 4000);
      else { savedJob(null); setControls(); }
    }
  }
  async function openJob(id) {
    if (state.xhr) return;
    const generation = ++state.generation; clearTimeout(state.timer); resetResults();
    try {
      const job = await request(`/api/jobs/${encodeURIComponent(id)}`);
      if (generation !== state.generation) return;
      connectionError(""); state.job = job; savedJob(id); renderJob(job);
      if (ACTIVE.has(job.status)) schedulePoll(generation); else await loadResults(job, generation);
    } catch (error) {
      if (generation !== state.generation) return;
      connectionError(userError(error)); if (error.status === 404) savedJob(null);
    }
  }
  async function cancelJob() {
    if (state.xhr) { state.xhr.abort(); return; }
    if (!state.job || !ACTIVE.has(state.job.status)) return;
    const id = state.job.id; const generation = state.generation;
    $("cancel-button").disabled = true; $("cancel-button").textContent = "Останавливаем…";
    try {
      const job = await request(`/api/jobs/${encodeURIComponent(id)}/cancel`, { method: "POST" });
      if (generation !== state.generation) return;
      state.job = job; renderJob(job);
      if (ACTIVE.has(job.status)) schedulePoll(generation); else { await loadResults(job, generation); refreshHistory(); }
    } catch (error) { connectionError(userError(error)); if (generation === state.generation) $("cancel-button").disabled = false; }
  }
  async function loadResults(job, generation) {
    if (generation !== state.generation) return;
    visible($("result-loading"), true);
    try {
      const results = await request(`/api/jobs/${encodeURIComponent(job.id)}/results`);
      if (generation !== state.generation) return;
      state.results = results; renderResults(results);
    } catch (error) {
      if (generation !== state.generation) return;
      if (!(error.status === 404 && ["failed", "cancelled", "interrupted"].includes(job.status))) {
        $("job-error").textContent = `Не удалось открыть результаты. ${userError(error)}`; visible($("job-error"), true);
      }
    } finally { if (generation === state.generation) visible($("result-loading"), false); }
  }
  function playbackURL(video) {
    if (!video) return null;
    if ($("result-video").canPlayType('video/webm; codecs="vp9"')) {
      const webm = safeURL(video.webm_url);
      if (webm) return webm;
    }
    return safeURL(video.mp4_url) || safeURL(video.url);
  }
  function setVideo(video, seekTime, scrollToPlayer = true) {
    const seekVersion = ++state.seekVersion;
    const player = $("result-video"); const url = playbackURL(video);
    if (!url) return;
    const change = !state.video || state.video.key !== video.key || player.getAttribute("src") !== url;
    state.video = video;
    $("current-video-label").textContent = video.label || MODES[video.key] || "";
    if (change) { player.pause(); player.src = url; player.load(); }
    visible(player, true); visible($("no-video"), false);
    $("video-tabs").querySelectorAll("button").forEach(button => {
      const active = button.dataset.key === String(video.key); button.classList.toggle("active", active); button.setAttribute("aria-pressed", String(active));
    });
    const interval = Number(state.job && state.job.options && state.job.options.interval) || 1;
    $("preview-note").textContent = video.key === "people"
      ? "Непрерывное видео с временными номерами траекторий. Эти номера не определяют личность человека."
      : video.key === "source" ? "Исходная запись. Отметки объектов доступны в других режимах."
      : `Один кадр каждые ${interval.toLocaleString("ru-RU")} с. Это выборочное превью; время наблюдений указано по исходной записи.`;
    if (Number.isFinite(seekTime)) {
      const seek = () => { if (seekVersion !== state.seekVersion) return; player.currentTime = Math.max(0, Math.min(seekTime, Number.isFinite(player.duration) ? Math.max(0, player.duration - 0.01) : seekTime)); };
      if (!change && player.readyState >= 1) seek(); else player.addEventListener("loadedmetadata", seek, { once: true });
      if (scrollToPlayer) player.scrollIntoView({ block: "center", behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "instant" : "smooth" });
    }
  }
  function renderResults(results) {
    visible($("results"), true);
    const videos = Array.isArray(results.videos) ? results.videos : [];
    $("video-tabs").replaceChildren();
    videos.forEach(video => {
      if (!playbackURL(video)) return;
      const button = element("button", video.label || MODES[video.key] || video.key, "video-tab");
      button.type = "button"; button.dataset.key = video.key; button.setAttribute("aria-pressed", "false");
      button.addEventListener("click", () => setVideo(video)); $("video-tabs").append(button);
    });
    const firstVideo = (!Array.isArray(results.events) || results.events.length === 0
      ? videos.find(video => video.key === "people" && playbackURL(video)) : null) || videos.find(video => playbackURL(video));
    if (firstVideo) setVideo(firstVideo);
    else { visible($("result-video"), false); visible($("no-video"), true); $("preview-note").textContent = ""; $("current-video-label").textContent = ""; }
    renderEvents(results); renderCoverage(results);
    const samples = results.analyzed_samples || {};
    const closeup = results.closeup_review;
    visible($("closeup-summary"), Boolean(closeup));
    $("closeup-summary").replaceChildren();
    if (closeup) {
      $("closeup-summary").append(element("strong", closeup.title_ru || (closeup.status === "completed" ? `Уточнение мелких предметов: ${closeup.analyzed_samples} кадров` : "Уточнение мелких предметов не выполнено")));
      $("closeup-summary").append(element("p", closeup.note_ru));
    }
    $("sample-summary").textContent = Object.entries(samples).map(([key, count]) => `${MODES[key] || key}: ${count} кадров`).join("; ");
    $("class-counts").replaceChildren();
    (results.class_counts || []).forEach(item => {
      const li = element("li"); li.append(element("span", item.label_ru || item.label), element("strong", item.frames ?? item.boxes ?? 0));
      if (item.boxes != null) li.title = `${item.boxes} отметок на обработанных кадрах; это не число уникальных объектов`;
      $("class-counts").append(li);
    });
    if (!$("class-counts").childElementCount) $("class-counts").append(element("li", "Объекты не найдены"));
    visible($("people-summary"), Boolean(results.people)); $("people-summary").replaceChildren();
    if (results.people) {
      const tracks = results.people.track_ids;
      const count = Array.isArray(tracks) ? tracks.length : (Number.isFinite(Number(tracks)) ? Number(tracks) : null);
      $("people-summary").append(element("strong", count == null ? "Траектории людей" : `Временных траекторий: ${count}`));
      $("people-summary").append(element("p", results.people.note || "Номера траекторий могут меняться. Их количество не равно числу клиентов."));
      const service = results.service || {};
      const role = service.role_summary || {};
      const reflections = results.reflections || {};
      if (Array.isArray(reflections.excluded_track_ids) && reflections.excluded_track_ids.length) {
        $("people-summary").append(element("p", `Отражения исключены из проверок людей: ${reflections.excluded_track_ids.map(id => "№" + id).join(", ")}. Серые рамки в видео отмечают эти кандидаты.`));
      }
      if (role.status === "candidate" && role.cashier_track_id != null) {
        const position = role.role_method === "lone_stable_workstation_occupant" ? "единственный устойчивый человек у рабочего места с ноутбуком" : "со стороны сейфа относительно стола";
        $("people-summary").append(element("p", `Предполагаемый кассир — траектория №${role.cashier_track_id}: ${position}. Положение экрана и поза пока не подтверждаются.`));
        const sessions = Array.isArray(service.sessions) ? service.sessions : [];
        if (sessions.length) {
          $("people-summary").append(element("p", "Устойчиво видимые посещения; начало и конец банковской операции не установлены:"));
          const visits = element("ul");
          sessions.forEach(visit => visits.append(element("li", `Посетитель, траектория №${visit.track_id}: ${timeLabel(visit.start_sec)}–${timeLabel(visit.end_sec)}`)));
          $("people-summary").append(visits);
        }
      } else if (role.detail) {
        $("people-summary").append(element("p", `Роль кассира не установлена. ${role.detail}`));
      }
    }
    renderThumbnails(results.thumbnails || []); renderDownloads(results);
    $("limitations-list").replaceChildren();
    const limits = Array.isArray(results.limitations) && results.limitations.length ? results.limitations : ["Модель может пропустить объект или ошибиться с его классом. Проверьте отмеченные кадры вручную."];
    const baseLimit = "Эпизоды показывают возможные нарушения и требуют проверки по записи. Наличие пакета не подтверждает, что внутри деньги.";
    [...new Set([baseLimit, ...limits])].forEach(text => $("limitations-list").append(element("li", text)));
    const objects = Array.isArray(results.objects) ? results.objects : [];
    fillSelect($("filter-class"), "Все объекты", [...new Map(objects.map(row => [row.label, row.label_ru || row.label])).entries()]);
    fillSelect($("filter-mode"), "Все режимы", [...new Set(objects.map(row => row.mode))].filter(Boolean).map(mode => [mode, MODES[mode] || mode]));
    state.page = 0; renderTable();
  }
  function eventTime(event) {
    const start = timeLabel(event.start_sec);
    return Number(event.end_sec) > Number(event.start_sec) ? `${start}–${timeLabel(event.end_sec)}` : start;
  }
  function eventVideo(event) {
    const videos = state.results && state.results.videos || [];
    const target = ReviewHelpers.eventTarget(event, videos, playbackURL);
    return target && target.video;
  }
  function openFrame(url, caption) {
    const safe = safeURL(url); if (!safe) return;
    $("frame-caption").textContent = caption; $("frame-image").src = safe;
    state.frameIndex = state.frames.findIndex(frame => safeURL(frame.url) === safe);
    $("frame-position").textContent = state.frameIndex >= 0 ? `${state.frameIndex + 1} / ${state.frames.length}` : "Кадр эпизода";
    $("previous-frame").disabled = state.frameIndex <= 0;
    $("next-frame").disabled = state.frameIndex < 0 || state.frameIndex >= state.frames.length - 1;
    if (!$("frame-dialog").open) $("frame-dialog").showModal();
  }
  function selectEvent(event, item, scrollToPlayer = true) {
    [...$("event-list").children].forEach(row => row.classList.toggle("selected", row === item));
    $("evidence-context").textContent = `${eventTime(event)} — ${event.title || "Эпизод для проверки"}`;
    visible($("evidence-context"), true);
    const target = ReviewHelpers.eventTarget(event, state.results.videos || [], playbackURL);
    if (target) {
      setVideo(target.video, target.time, scrollToPlayer);
    } else if (scrollToPlayer && event.thumbnail_url) {
      openFrame(event.thumbnail_url, `${eventTime(event)} — ${event.title || "Кадр эпизода"}`);
    }
  }
  function renderEvents(results) {
    const hasAnalysis = Array.isArray(results.events);
    const events = hasAnalysis ? [...results.events].sort((a, b) => (Number(a.start_sec) || 0) - (Number(b.start_sec) || 0)) : [];
    const count = events.length;
    const word = count % 10 === 1 && count % 100 !== 11 ? "эпизод" : [2, 3, 4].includes(count % 10) && ![12, 13, 14].includes(count % 100) ? "эпизода" : "эпизодов";
    $("event-count").textContent = hasAnalysis ? `${count} ${word}` : "Нужна повторная проверка";
    $("review-summary-text").textContent = results.summary || (count ? "Выделены моменты, которые могут указывать на нарушение правил. Откройте эпизод, чтобы проверить его по видео." : hasAnalysis ? "Эпизоды для проверки не выделены. Это не подтверждает соблюдение всех правил: проверьте охват анализа ниже." : "Эта запись обработана без анализа эпизодов. Доступны исходные наблюдения; для проверки правил загрузите видео повторно.");
    $("event-list").replaceChildren(); visible($("no-events"), count === 0);
    $("no-events").textContent = hasAnalysis ? "Модель не выделила эпизоды для просмотра. Непроверенные правила остаются без оценки." : "В сохранённых результатах есть только исходные наблюдения модели.";
    $("raw-observations").open = !hasAnalysis;
    events.forEach((event, index) => {
      const item = element("li", null, "event-item"); item.dataset.eventId = String(event.id || index);
      const heading = element("div", null, "event-heading");
      heading.append(element("strong", eventTime(event), "event-time"), element("span", event.confidence_label || "Требует проверки", "event-review-label"));
      const title = element("h4"); const titleButton = element("button", event.title || "Возможное нарушение", "event-title-button");
      titleButton.type = "button"; titleButton.addEventListener("click", () => selectEvent(event, item)); title.append(titleButton);
      item.append(heading, title);
      item.addEventListener("click", click => { if (!click.target.closest("button,a,summary,details,input")) selectEvent(event, item); });
      if (event.description) item.append(element("p", event.description, "event-description"));
      if (Array.isArray(event.steps) && event.steps.length) {
        const section = element("div", null, "episode-steps");
        section.append(element("strong", "Что произошло по шагам"));
        const list = element("ol");
        event.steps.forEach(step => {
          const line = element("li");
          const stamp = typeof step.timestamp_sec === "number" && Number.isFinite(step.timestamp_sec) ? `${timeLabel(step.timestamp_sec)} · ` : "";
          line.append(element("strong", stamp + (step.title || "Наблюдение")));
          if (step.detail) line.append(element("p", step.detail));
          list.append(line);
        });
        section.append(list); item.append(section);
      }
      const evidence = element("div", null, "event-evidence");
      const thumbnail = safeURL(event.thumbnail_url);
      if (thumbnail) {
        const fps = state.job && state.job.video && Number(state.job.video.fps);
        const frameTime = fps > 0 && Number.isFinite(Number(event.source_frame)) ? Number(event.source_frame) / fps : Number(event.start_sec);
        const picture = element("button", null, "event-thumbnail"); picture.type = "button"; picture.setAttribute("aria-label", `Открыть кадр эпизода ${eventTime(event)}`);
        const image = element("img"); image.src = thumbnail; image.alt = `Кадр ${timeLabel(frameTime)}`; image.loading = "lazy";
        picture.append(image); picture.addEventListener("click", () => openFrame(thumbnail, `Кадр ${timeLabel(frameTime)} — ${event.title || "Кадр эпизода"}`)); evidence.append(picture);
      }
      if (event.evidence_text) { const text = element("div"); text.append(element("strong", "Основание"), element("p", event.evidence_text)); evidence.append(text); }
      if (evidence.childElementCount) item.append(evidence);
      const canSeek = Boolean(eventVideo(event));
      const action = element("button", canSeek ? "Показать в видео" : thumbnail ? "Открыть кадр" : "Кадр недоступен", "button button-small event-action");
      action.type = "button"; action.disabled = !canSeek && !thumbnail; action.addEventListener("click", () => selectEvent(event, item)); item.append(action);
      if (Array.isArray(event.limitations) && event.limitations.length) {
        const details = element("details", null, "event-limitations"); details.append(element("summary", "Что нужно уточнить"));
        const list = element("ul"); event.limitations.forEach(text => list.append(element("li", text))); details.append(list); item.append(details);
      }
      $("event-list").append(item);
    });
    if (events.length && eventVideo(events[0])) selectEvent(events[0], $("event-list").firstElementChild, false);
  }
  function renderCoverage(results) {
    const coverage = Array.isArray(results.rule_coverage) ? results.rule_coverage : [];
    $("rule-coverage").replaceChildren(); visible($("coverage-empty"), coverage.length === 0);
    const available = coverage.filter(rule => rule.status === "checked" || rule.status === "partial").length;
    $("coverage-count").textContent = coverage.length ? `Визуальные проверки доступны для ${available} из ${coverage.length} правил` : "";
    const labels = { checked: "Проверено", partial: "Частично", not_checked: "Не проверено" };
    coverage.forEach(rule => {
      const status = Object.hasOwn(labels, rule.status) ? rule.status : "not_checked";
      const item = element("li", null, "coverage-item");
      const heading = element("div", null, "coverage-heading"); heading.append(element("strong", rule.title || rule.rule_id || "Правило"), element("span", rule.status_label || labels[status], "coverage-status " + status)); item.append(heading);
      if (rule.detail) item.append(element("p", rule.detail));
      if (Array.isArray(rule.checks) && rule.checks.length) {
        const checks = element("ul", null, "concrete-checks");
        rule.checks.forEach(check => {
          const label = check.available ? (check.observation_count > 0 ? `Наблюдений: ${check.observation_count}` : "Проверено, признаков не найдено") : "Недостаточно данных";
          const line = element("li");
          line.append(element("strong", check.title), element("span", ` — ${label}`));
          checks.append(line);
        });
        item.append(checks);
      }
      if (Array.isArray(rule.procedure_steps) && rule.procedure_steps.length) {
        const procedure = element("details", null, "rule-procedure");
        procedure.append(element("summary", rule.procedure_status === "planned" ? "Как должна работать проверка — ещё не реализовано" : "Как проверяем, шаг за шагом"));
        if (rule.procedure_note) procedure.append(element("p", rule.procedure_note));
        const list = element("ol");
        rule.procedure_steps.forEach(step => {
          const line = element("li");
          if (typeof step === "string") line.textContent = step;
          else { line.append(element("strong", step.title || "Шаг")); if (step.detail) line.append(element("p", step.detail)); }
          list.append(line);
        });
        procedure.append(list); item.append(procedure);
      }
      if (Number(rule.event_count) > 0) item.append(element("span", `Эпизодов для просмотра: ${Number(rule.event_count)}`, "coverage-events"));
      $("rule-coverage").append(item);
    });
  }
  function fillSelect(select, placeholder, entries) {
    const current = select.value; select.replaceChildren();
    const first = element("option", placeholder); first.value = ""; select.append(first);
    entries.forEach(([value, label]) => { const option = element("option", label); option.value = value; select.append(option); });
    if ([...select.options].some(option => option.value === current)) select.value = current;
  }
  function renderThumbnails(thumbnails) {
    $("thumbnail-strip").replaceChildren();
    const fps = Number(state.job && state.job.video && state.job.video.fps);
    const eventFrames = (state.results.events || []).filter(e => e.thumbnail_url).map(e => ({url:e.thumbnail_url,timestamp_sec:fps > 0 && e.source_frame != null ? Number(e.source_frame)/fps : e.start_sec,mode:e.mode,caption:e.title}));
    state.frames = [...new Map([...eventFrames, ...thumbnails].filter(item => safeURL(item.url)).map(item => [safeURL(item.url),item])).values()].sort((a,b) => Number(a.timestamp_sec)-Number(b.timestamp_sec));
    const picked = state.frames.slice(0,state.frameLimit);
    $("frame-total").textContent = String(state.frames.length);
    visible($("frames-empty"), !state.frames.length); visible($("more-frames"), state.frameLimit < state.frames.length);
    picked.forEach(item => {
      const button = element("button", null, "thumbnail-button"); button.type = "button";
      const img = element("img"); img.src = safeURL(item.url); img.alt = `Кадр ${timeLabel(item.timestamp_sec)}, ${MODES[item.mode] || "объекты"}`; img.loading = "lazy";
      button.append(img, element("span", timeLabel(item.timestamp_sec)));
      button.addEventListener("click", () => {
        openFrame(item.url,`${timeLabel(item.timestamp_sec)} — ${item.caption || MODES[item.mode] || "Кадр с отметками"}`);
      });
      $("thumbnail-strip").append(button);
    });
    visible($("frames-section"), state.resultView === "frames");
  }
  function renderDownloads(results) {
    $("download-links").replaceChildren();
    $("log-download-links").replaceChildren();
    visible($("diagnostic-downloads"), false);
    const seen = new Set();
    (results.videos || []).forEach(item => {
      const primary = safeURL(item.mp4_url) || safeURL(item.url);
      const formats = [
        { url: primary, label: primary && new URL(primary).pathname.toLowerCase().endsWith(".webm") ? "WebM" : "MP4" },
        { url: safeURL(item.webm_url), label: "WebM" }
      ];
      formats.forEach(format => {
        if (!format.url || seen.has(format.url)) return; seen.add(format.url);
        const a = element("a", `${item.label || MODES[item.key] || "Видео"} — ${format.label}`, "button");
        a.href = format.url; a.download = ""; $("log-download-links").append(a); visible($("diagnostic-downloads"), true);
      });
    });
    const links = Array.isArray(results.downloads) ? results.downloads : [];
    links.forEach(item => {
      const url = safeURL(item.url); if (!url || seen.has(url)) return; seen.add(url);
      const a = element("a", item.label || "Скачать файл", "button"); a.href = url; a.download = "";
      if (/\/report\/(?:review\.html|violations\.(?:csv|json))$/i.test(new URL(url).pathname)) {
        $("log-download-links").append(a); visible($("diagnostic-downloads"), true);
      } else { $("log-download-links").append(a); visible($("diagnostic-downloads"), true); }
    });
    if (state.job) {
      const all = element("a", "Все результаты ZIP", "button download-all");
      all.href = `/api/jobs/${encodeURIComponent(state.job.id)}/archive`; all.download = ""; $("log-download-links").append(all); visible($("diagnostic-downloads"), true);
    }
  }
  function renderTable() {
    const all = state.results && Array.isArray(state.results.objects) ? state.results.objects : [];
    const classFilter = $("filter-class").value; const statusFilter = $("filter-status").value; const modeFilter = $("filter-mode").value;
    const rows = all.filter(row => (!classFilter || row.label === classFilter) && (!statusFilter || row.status === statusFilter) && (!modeFilter || row.mode === modeFilter));
    const pages = Math.max(1, Math.ceil(rows.length / PAGE_SIZE)); state.page = Math.min(state.page, pages - 1);
    const start = state.page * PAGE_SIZE; const end = Math.min(rows.length, start + PAGE_SIZE);
    $("observation-total").textContent = `${all.length.toLocaleString("ru-RU")} отметок`;
    $("observation-rows").replaceChildren();
    rows.slice(start, end).forEach(row => {
      const tr = element("tr"); const time = element("td");
      const video = (state.results.videos || []).find(item => item.key === row.mode && playbackURL(item));
      if (video && Number.isFinite(Number(row.playback_sec)) && row.playback_sec != null) {
        const button = element("button", timeLabel(row.timestamp_sec), "time-button"); button.type = "button"; button.title = "Показать этот кадр в видео с отметками";
        button.addEventListener("click", () => setVideo(video, Number(row.playback_sec))); time.append(button);
      } else { time.textContent = timeLabel(row.timestamp_sec); }
      const score = Number(row.score); const reviewedReference = row.source === "human_confirmed_reference_match";
      const scoreCell = element("td", Number.isFinite(score) ? `${reviewedReference ? "Сходство " : ""}${Math.round(score * 100)}%` : "—", "score-value");
      if (reviewedReference) scoreCell.title = "Сходство с исправленным пользователем образцом в этом же видео; это не уверенность модели.";
      const status = element("td"); const review = row.status === "review";
      status.append(element("span", review ? "Проверить" : reviewedReference ? "По уточнению" : "Найден", "observation-tag" + (review ? " review" : "")));
      tr.append(time, element("td", row.label_ru || row.label), scoreCell, status, element("td", MODES[row.mode] || row.mode || "—"));
      $("observation-rows").append(tr);
    });
    visible($("no-observations"), rows.length === 0);
    $("page-info").textContent = rows.length ? `${start + 1}–${end} из ${rows.length.toLocaleString("ru-RU")}` : "0 наблюдений";
    $("previous-page").disabled = state.page <= 0; $("next-page").disabled = state.page >= pages - 1;
  }
  async function refreshHistory() {
    try {
      const data = await request("/api/jobs"); state.jobs = Array.isArray(data.jobs) ? data.jobs : []; renderHistory();
    } catch (_) { if (!state.jobs.length) $("history-empty").textContent = "Не удалось загрузить список. Нажмите «Обновить»."; }
  }
  function renderHistory() {
    $("history-list").replaceChildren(); visible($("history-empty"), state.jobs.length === 0);
    state.jobs.slice(0, 12).forEach(job => {
      const li = element("li"); const button = element("button", null, "history-item"); button.type = "button";
      const current = state.job && state.job.id === job.id; button.classList.toggle("current", Boolean(current));
      if (current) button.setAttribute("aria-current", "true");
      button.append(element("strong", job.name || "Видео"), element("span", `${STATUS[job.status] || job.status}   ${dateLabel(job.created_at)}`));
      button.addEventListener("click", () => openJob(job.id)); button.disabled = Boolean(state.xhr); li.append(button); $("history-list").append(li);
    });
  }
  async function init() {
    try {
      state.config = await request("/api/config");
      const max = Number(state.config.max_upload_mb) || 512;
      $("file-hint").textContent = `MP4, MOV, AVI, MKV, M4V, WebM · до ${max} МБ`;
      setControls();
      await refreshHistory();
      const id = readSavedJob(); if (id) await openJob(id);
    } catch (error) { connectionError(userError(error)); }
  }

  $("upload-form").addEventListener("submit", event => { event.preventDefault(); beginUpload(); });
  $("video-file").addEventListener("change", event => selectFile(event.target.files[0]));
  $("clear-file").addEventListener("click", () => { $("video-file").value = ""; selectFile(null); });
  $("dropzone").addEventListener("dragover", event => { event.preventDefault(); if (!$("video-file").disabled) $("dropzone").classList.add("dragover"); });
  $("dropzone").addEventListener("dragleave", event => { if (!$("dropzone").contains(event.relatedTarget)) $("dropzone").classList.remove("dragover"); });
  $("dropzone").addEventListener("drop", event => {
    event.preventDefault(); $("dropzone").classList.remove("dragover");
    if (event.dataTransfer.files.length > 1) { fileError("Выберите одно видео. Следующее можно загрузить после обработки."); return; }
    selectFile(event.dataTransfer.files[0]);
  });
  document.addEventListener("dragover", event => { if (event.dataTransfer && [...event.dataTransfer.types].includes("Files")) event.preventDefault(); });
  document.addEventListener("drop", event => event.preventDefault());
  $("cancel-button").addEventListener("click", cancelJob);
  $("refresh-history").addEventListener("click", refreshHistory);
  ["filter-class", "filter-status", "filter-mode"].forEach(id => $(id).addEventListener("change", () => { state.page = 0; renderTable(); }));
  $("reset-filters").addEventListener("click", () => { ["filter-class", "filter-status", "filter-mode"].forEach(id => { $(id).value = ""; }); state.page = 0; renderTable(); });
  $("previous-page").addEventListener("click", () => { state.page--; renderTable(); });
  $("next-page").addEventListener("click", () => { state.page++; renderTable(); });
  function switchResultView(view) {
    state.resultView = view;
    visible($("event-view"), view === "events"); visible($("frames-section"), view === "frames");
    $("show-events").setAttribute("aria-pressed", String(view === "events"));
    $("show-frames").setAttribute("aria-pressed", String(view === "frames"));
  }
  function advanceFrame(delta) {
    const frame = state.frames[state.frameIndex + delta]; if (!frame) return;
    openFrame(frame.url,`${timeLabel(frame.timestamp_sec)} — ${frame.caption || MODES[frame.mode] || "Кадр записи"}`);
  }
  $("show-events").addEventListener("click", () => switchResultView("events"));
  $("show-frames").addEventListener("click", () => switchResultView("frames"));
  $("more-frames").addEventListener("click", () => { state.frameLimit += 24; renderThumbnails(state.results.thumbnails || []); });
  $("previous-frame").addEventListener("click", () => advanceFrame(-1));
  $("next-frame").addEventListener("click", () => advanceFrame(1));
  $("frame-dialog").addEventListener("keydown", event => {
    if (event.key === "ArrowLeft" || event.key === "ArrowRight") { event.preventDefault(); advanceFrame(event.key === "ArrowLeft" ? -1 : 1); }
  });
  $("save-results").addEventListener("click", () => {
    if (!state.job || !state.results) return;
    const payload = ReviewHelpers.exportResult(state.job,state.results);
    const blob = new Blob([JSON.stringify(payload,null,2)],{type:"application/json;charset=utf-8"});
    const url = URL.createObjectURL(blob); const link = document.createElement("a");
    link.href = url; link.download = `results-${state.job.id.slice(0,8)}.json`;
    document.body.append(link); link.click(); link.remove();
    setTimeout(() => URL.revokeObjectURL(url),60000);
    $("save-status").textContent = "JSON подготовлен и передан браузеру для сохранения.";
  });
  $("close-frame").addEventListener("click", () => $("frame-dialog").close());
  $("frame-dialog").addEventListener("click", event => { if (event.target === $("frame-dialog")) $("frame-dialog").close(); });
  $("result-video").addEventListener("error", () => { if ($("result-video").hasAttribute("src")) { $("preview-note").textContent = "Браузер не смог воспроизвести превью. Скачайте MP4 или откройте отдельные кадры ниже."; } });
  init();
})();
