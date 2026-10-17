"use strict";

(() => {
  const byId = (id) => document.getElementById(id);
  const ui = Object.fromEntries([
    "api-key", "key-visibility", "upload-form", "video-file", "drop-zone", "selected-file", "file-name", "file-size", "remove-file", "analyze-button", "admission-note", "form-error", "service-dot", "service-label", "status-title", "status-badge", "status-description", "progress-section", "progress-track", "progress-fill", "stage-label", "elapsed-label", "job-controls", "job-id", "poll-button", "poll-error", "source-video", "video-empty", "video-caption", "video-name", "video-error", "results", "verdict-card", "verdict-icon", "verdict-title", "verdict-description", "coverage-section", "coverage-metrics", "review-notes-section", "review-notes", "findings-heading", "finding-count", "findings", "review-queue-section", "review-queue-count", "review-queue", "catalog-section", "catalog-version", "catalog-reload", "catalog-status", "catalog-table-wrap", "catalog-items", "assessments-section", "assessments-version", "assessment-items", "open-job-form", "job-input", "video-stage", "detection-overlay", "detection-controls", "show-detections", "overlay-status", "detections-section", "detection-count", "detections", "case-decision-section", "case-decision-count", "case-decisions", "details-warning", "jury-verdict"
  ].map((id) => [id, byId(id)]));
  let file = null;
  let ready = false;
  let busy = false;
  let activeJobId = null;
  let activeJobTimer = null;
  let activeJobController = null;
  let activeJobGeneration = 0;
  let submitting = false;
  let currentJob = null;
  let sourceUrl = null;
  let pollTimer = null;
  let elapsedTimer = null;
  let pollController = null;
  let pollGeneration = 0;
  let polling = false;
  let acceptedAt = null;
  let lastDetails = {};
  let lastResult = null;
  let loadedJobVideo = null;
  let catalogTimer = null;
  let catalogController = null;
  let catalogGeneration = 0;
  let showAllAssessments = false;
  let detections = [];
  let detectionGeneration = 0;
  let detectionController = null;
  const detectionImageUrls = new Map();

  class ApiError extends Error {
    constructor(message, status, retryAfter) { super(message); this.status = status; this.retryAfter = retryAfter; }
  }
  function display(element, visible) { element.hidden = !visible; }
  function message(element, value) { element.textContent = value || ""; display(element, Boolean(value)); }
  function node(tag, className, content) {
    const result = document.createElement(tag);
    if (className) result.className = className;
    if (content !== undefined) result.textContent = String(content);
    return result;
  }
  function finite(value) { return value !== null && value !== "" && Number.isFinite(Number(value)); }
  function formatTime(value) {
    if (!finite(value)) return "—";
    const seconds = Math.max(0, Number(value));
    const minutes = Math.floor(seconds / 60);
    const remainder = Math.floor(seconds % 60);
    return `${String(minutes).padStart(2, "0")}:${String(remainder).padStart(2, "0")}`;
  }
  function formatSize(value) { return `${(value / 1024 / 1024).toFixed(1)} MB`; }
  function sampleTime(value) {
    const milliseconds = Math.max(0, Math.round(Number(value) * 1000));
    return `${String(Math.floor(milliseconds / 60000)).padStart(2, "0")}:${String(Math.floor(milliseconds / 1000) % 60).padStart(2, "0")}.${String(milliseconds % 1000).padStart(3, "0")}`;
  }
  function friendly(value) {
    const labels = {personal_food_at_workstation:"Personal belongings at workplace (food)",cashier_eating_at_workstation:"Eating at cashier workstation"};
    return labels[value] || String(value || "").replaceAll("_", " ").replace(/^./, (c) => c.toUpperCase());
  }
  function syncSubmit() {
    ui["analyze-button"].disabled = !ready || busy || !file || !ui["api-key"].value.trim() || submitting || Boolean(currentJob && polling);
    ui["analyze-button"].firstChild.textContent = submitting ? "Uploading video… " : "Analyze video ";
    ui["admission-note"].textContent = !ready ? "Waiting for the analysis worker." : submitting ? "Uploading to the local service." : currentJob && polling ? "Your review is in progress." : busy ? "An analysis is already running. Wait for it to finish before uploading another video." : "The service is ready for a new review.";
    if (busy && activeJobId && activeJobId !== currentJob && !submitting) {
      const open = node("button", "text-button", "Open running analysis"); open.type = "button"; open.id = "open-active-job";
      open.addEventListener("click", () => { message(ui["form-error"], ""); beginJob(activeJobId); });
      ui["admission-note"].append(document.createTextNode(" "), open);
    }
  }
  async function request(path, options = {}, authenticated = true) {
    // Keep both video data and credentials on this service, including future callers.
    if (typeof path !== "string" || !path.startsWith("/api/") || path.includes("\\")) {
      throw new ApiError("Only this service’s local API routes may be requested.", 0);
    }
    const headers = new Headers(options.headers || {});
    if (authenticated) {
      const key = ui["api-key"].value.trim();
      if (!key) throw new ApiError("Enter your service API key to continue.", 401);
      headers.set("X-API-Key", key);
    }
    let response;
    try { response = await fetch(path, { ...options, headers, cache: "no-store", credentials: "same-origin", redirect: "error" }); }
    catch (error) { if (error.name === "AbortError") throw error; throw new ApiError("The local service could not be reached. Check that it is running, then retry.", 0); }
    if (!response.ok) {
      let detail = "";
      try {
        const data = await response.json();
        detail = typeof data.detail === "string" ? data.detail : typeof data.error === "string" ? data.error : typeof data.message === "string" ? data.message : "";
      } catch (_) { /* A plain error response still gets a clear status-specific message. */ }
      const errors = {401:"The API key was not accepted. Check the key and retry.",403:"This API key does not have access to this review.",404:"This review was not found. Check the job ID.",413:"This video exceeds the service upload limit.",429:"An analysis is already running. Open its progress or wait for it to finish.",503:"The analysis worker is unavailable or still loading. Retry when it is ready."};
      const retryAfter = response.headers.get("Retry-After");
      throw new ApiError(errors[response.status] || detail || `The request failed (HTTP ${response.status}).`, response.status, retryAfter);
    }
    return response;
  }
  async function json(path, options, authenticated) { return (await request(path, options, authenticated)).json(); }
  function catalogEntries(value) {
    return Array.isArray(value) ? value.filter((item) => item && typeof item === "object" && !Array.isArray(item)) : [];
  }
  function scoreText(value) {
    return typeof value === "number" && Number.isFinite(value) && value >= 0 && value <= 100 ? `${Number(value.toFixed(2))}%` : null;
  }
  function scoringKind(value) {
    return ({event_occurrence:"Event occurrence",required_procedure:"Required procedure"})[value] || "Unknown";
  }
  function confidenceText(value) {
    return typeof value === "number" && Number.isFinite(value) && value >= 0 && value <= 1 ? scoreText(value * 100) : null;
  }
  function automationLabel(value) {
    return ({automatic_when_supported:"Automatic when supported",review_only:"Review only",manual:"Manual",excluded:"Excluded"})[value] || null;
  }
  function assessmentLabel(value) {
    return ({unassessed:"Unassessed",observed:"Event evidence",completion_evidence:"Correct procedure evidence",completed:"Procedure completed",unresolved:"Unresolved"})[value] || "Unassessed";
  }
  function episodeLabel(item) {
    const summary = item.scope === "per_episode_summary";
    const id = summary && typeof item.latest_episode_id === "string" ? item.latest_episode_id : typeof item.episode_id === "string" ? item.episode_id : typeof item.latest_episode_id === "string" ? item.latest_episode_id : null;
    if (!id) return null;
    const startTime = summary ? item.latest_episode_start_sec : item.start_sec;
    const endTime = summary ? item.latest_episode_end_sec : item.end_sec;
    const start = finite(startTime) ? ` · ${formatTime(startTime)}${finite(endTime) ? `–${formatTime(endTime)}` : ""}` : "";
    return "Episode" + start;
  }
  function scoreRow(item, initialOnly, episodes = []) {
    const row = node("tr");
    const id = typeof item.item_id === "string" ? item.item_id : typeof item.id === "string" ? item.id : "";
    const label = node("th", "", typeof item.label === "string" && item.label ? item.label : friendly(id) || "Untitled check");
    label.setAttribute("scope", "row");
    const applicability = initialOnly ? "unknown" : item.applicability;
    const assessment = initialOnly ? "unassessed" : item.assessment;
    const excluded = applicability === "not_applicable";
    const checked = !initialOnly && !excluded && item.automation !== "manual"
      && !["unassessed", "unresolved"].includes(assessment)
      && scoreText(item.score) !== null;
    const untouched = initialOnly || item.automation === "manual" || assessment === "unassessed";
    const result = excluded ? "Not applicable" : untouched ? "Not checked" : item.confirmed_finding === true ? "Violation"
      : assessment === "completed" && applicability === "applicable" ? "Completed" : "Needs review";
    const confidence = confidenceText(item.model_confidence);
    const estimate = checked && confidence ? confidence + (item.scoring_kind === "required_procedure" ? " completion support" : " event support") : "—";
    row.append(label, node("td", "score-result", result), node("td", "score-value", estimate));
    if (excluded) row.classList.add("score-excluded");
    const detail = node("details", "score-details"), body = node("div", "score-detail-body");
    detail.append(node("summary", "", "Details"));
    body.append(node("p", "score-state", scoringKind(item.scoring_kind) + " · " + assessmentLabel(assessment)),
      node("p", "score-state", "Applicability: " + (({unknown:"Unknown",applicable:"Applicable",not_applicable:"Not applicable"})[applicability] || "Unknown")));
    const automation = automationLabel(item.automation);
    if (automation) body.append(node("p", "score-state", "Review route: " + automation));
    const source = safeRuleUrl(item.source_url);
    if (source) {
      const policy = item.basis === "workplace_policy";
      if (policy) body.append(node("p", "score-state", "Basis: User-provided workplace policy"));
      const link = node("a", "", policy ? "Read workplace policy ↗" : "Read official source ↗");
      link.href = source; link.target = "_blank"; link.rel = "noopener noreferrer"; body.append(link);
    }
    if (!initialOnly) {
      const refs = Array.isArray(item.evidence_refs) ? item.evidence_refs : [];
      body.append(node("p", "score-state", refs.length + " evidence " + (refs.length === 1 ? "reference" : "references")));
      if (confidence) body.append(node("p", "score-state", (item.scoring_kind === "required_procedure" ? "Correct-procedure confidence: " : "Event confidence: ") + confidence + " · uncalibrated"));
      if (checked && scoreText(item.score)) body.append(node("p", "score-state", (item.scoring_kind === "required_procedure" ? "Remaining procedure estimate: " : "Event estimate: ") + scoreText(item.score) + " · uncalibrated; estimates do not determine the verdict"));
      const scope = ({clip_event_summary:"Strongest event evidence in this clip",per_episode_summary:"Highest remaining estimate across assessed transactions",episode_assessment:"One transaction or event episode",per_episode_verification_required:"Verify each applicable transaction",verified_finding_episode:"One verified event episode"})[item.scope];
      if (scope) body.append(node("p", "score-state", "Scope: " + scope));
      if (item.unassessed_episode_count > 0) body.append(node("p", "score-state", "Incomplete coverage: " + item.unassessed_episode_count + " known episodes have no assessment for this check; applicability may be unknown"));
      if (Number.isSafeInteger(item.unscoped_unknown_observation_count) && item.unscoped_unknown_observation_count > 0) body.append(node("p", "score-state", "Incomplete coverage: " + item.unscoped_unknown_observation_count + " uncertain windows have no established procedure trigger; existing completion evidence is retained"));
      if (item.scope_certainty === "window_only") body.append(node("p", "score-state", "Window-scoped evidence · transaction boundary unconfirmed"));
      const latest = episodeLabel(item);
      if (latest) body.append(node("p", "score-state", "Latest assessed " + latest.toLowerCase()));
      const moments = node("div", "check-evidence");
      for (const ref of refs.slice(0, 3)) {
        if (typeof ref?.timestamp_sec !== "number" || !Number.isFinite(ref.timestamp_sec) || ref.timestamp_sec < 0) continue;
        const time = node("button", "text-button", sampleTime(ref.timestamp_sec)); time.type = "button";
        time.setAttribute("aria-label", "Seek to check evidence at " + sampleTime(ref.timestamp_sec));
        time.addEventListener("click", () => seek(ref.timestamp_sec)); moments.append(time);
      }
      if (moments.childElementCount) body.append(moments);
      if (episodes.length) {
        const unresolved = Number.isSafeInteger(item.pending_episode_count) && item.pending_episode_count >= 0 ? item.pending_episode_count : episodes.filter(episode => episode.applicability !== "not_applicable" && (episode.assessment !== "completed" || episode.applicability === "unknown")).length;
        const count = Number.isSafeInteger(item.episode_count) && item.episode_count >= episodes.length ? item.episode_count : episodes.length;
        body.append(node("p", "score-state", count + " episodes · " + unresolved + " pending or unknown"));
        const history = node("details", "score-episodes"), list = node("ul");
        history.append(node("summary", "", "Episode scores"));
        for (const [index, episode] of episodes.entries()) {
          const assessed = episode.applicability !== "not_applicable" && episode.automation !== "manual" && !["unassessed", "unresolved"].includes(episode.assessment);
          const estimate = episode.applicability === "not_applicable" ? "Not applicable" : assessed ? scoreText(episode.score) || "Not checked" : "Not checked";
          const certainty = confidenceText(episode.model_confidence);
          const interval = finite(episode.start_sec) ? " · " + formatTime(episode.start_sec) + (finite(episode.end_sec) ? "–" + formatTime(episode.end_sec) : "") : "";
          list.append(node("li", "", "Episode " + (index + 1) + interval + ": " + estimate + " · " + assessmentLabel(episode.assessment) + (certainty ? " · " + (episode.scoring_kind === "required_procedure" ? "correct-procedure" : "event") + " confidence " + certainty : "") + (episode.applicability === "unknown" ? " · applicability unknown" : "")));
        }
        history.append(list); body.append(history);
      }
    }
    detail.append(body); label.append(detail);
    return row;
  }
  function renderAssessments(metadata) {
    const entries = catalogEntries(metadata.details?.assessments);
    const episodes = catalogEntries(metadata.details?.assessment_episodes);
    const relevant = entries.filter(item => item.confirmed_finding === true || item.automation !== "manual" && (Array.isArray(item.evidence_refs) && item.evidence_refs.length > 0 || item.unassessed_episode_count > 0 || item.unscoped_unknown_observation_count > 0 || typeof item.model_confidence === "number"));
    const shown = showAllAssessments ? entries : relevant;
    ui["assessment-items"].replaceChildren(...shown.map((item) => scoreRow(item, false, episodes.filter((episode) => (episode.item_id || episode.id) === (item.item_id || item.id)))));
    const version = metadata.details?.catalog_version;
    ui["assessments-version"].textContent = shown.length + " of " + entries.length + " checks" + (typeof version === "string" || typeof version === "number" ? " · Catalog version " + version : "");
    const toggle = byId("assessment-toggle");
    if (toggle) { toggle.textContent = showAllAssessments ? "Show relevant checks" : "Show all checks"; display(toggle, relevant.length < entries.length); }
    display(ui["assessments-section"], entries.length > 0);
  }
  async function loadCatalog() {
    clearTimeout(catalogTimer);
    if (catalogController) catalogController.abort();
    const generation = ++catalogGeneration;
    ui["catalog-items"].replaceChildren();
    ui["catalog-version"].textContent = "";
    display(ui["catalog-table-wrap"], false);
    if (!ui["api-key"].value.trim()) { display(ui["catalog-section"], false); return; }
    display(ui["catalog-section"], true);
    ui["catalog-status"].textContent = "Loading the authenticated initial catalog…";
    catalogController = new AbortController();
    try {
      const catalog = await json("/api/ui/catalog", {signal:catalogController.signal});
      if (generation !== catalogGeneration) return;
      if (!catalog || !Array.isArray(catalog.items)) throw new ApiError("The service returned an invalid review catalog.", 0);
      const entries = catalogEntries(catalog.items);
      ui["catalog-items"].replaceChildren(...entries.map((item) => scoreRow(item, true)));
      ui["catalog-version"].textContent = typeof catalog.version === "string" || typeof catalog.version === "number" ? `Catalog version ${catalog.version}` : "";
      const automaticEvents = entries.filter((item) => item.scoring_kind === "event_occurrence" && item.automation === "automatic_when_supported").length;
      ui["catalog-status"].textContent = entries.length ? `${entries.length} catalog items · ${automaticEvents} event checks eligible for automatic findings` : "The catalog has no items.";
      display(ui["catalog-table-wrap"], entries.length > 0);
    } catch (error) {
      if (error.name !== "AbortError" && generation === catalogGeneration) ui["catalog-status"].textContent = `Catalog unavailable. ${error.message}`;
    }
  }
  function scheduleCatalog() {
    clearTimeout(catalogTimer);
    if (catalogController) catalogController.abort();
    catalogGeneration += 1;
    ui["catalog-items"].replaceChildren();
    ui["catalog-version"].textContent = "";
    display(ui["catalog-table-wrap"], false);
    display(ui["catalog-section"], Boolean(ui["api-key"].value.trim()));
    ui["catalog-status"].textContent = "Waiting to load the initial catalog…";
    if (ui["api-key"].value.trim()) catalogTimer = setTimeout(loadCatalog, 400);
  }
  async function checkHealth() {
    try {
      const health = await json("/api/v1/health", {}, false);
      ready = health.ready === true || health.worker?.ready === true || health.worker_ready === true || health.status === "ready";
      busy = health.busy === true;
      if (!busy) activeJobId = null;
      ui["service-dot"].className = `dot ${ready ? "ready" : "failed"}`;
      ui["service-label"].textContent = ready ? busy ? "Analysis running" : "Worker ready" : "Worker loading / unavailable";
    } catch (_) { ready = false; ui["service-dot"].className = "dot failed"; ui["service-label"].textContent = "Service unavailable"; }
    syncSubmit();
    if (busy && ui["api-key"].value.trim()) recoverActiveJob();
  }
  async function recoverActiveJob() {
    const key = ui["api-key"].value.trim();
    if (!key || submitting) return;
    if (activeJobController) activeJobController.abort();
    const generation = ++activeJobGeneration;
    activeJobController = new AbortController();
    try {
      const active = await json("/api/ui/active-job", {signal:activeJobController.signal});
      if (generation !== activeJobGeneration || ui["api-key"].value.trim() !== key) return;
      activeJobId = typeof active.job_id === "string" && active.job_id.length > 0 ? active.job_id : null;
      if (activeJobId) {
        busy = true;
        if (!currentJob && !file && !submitting && !polling) { message(ui["form-error"], ""); beginJob(activeJobId); }
      }
      syncSubmit();
    } catch (_) { /* Health still blocks duplicate uploads if recovery is temporarily unavailable. */ }
  }
  function scheduleActiveJobRecovery() {
    clearTimeout(activeJobTimer);
    if (activeJobController) activeJobController.abort();
    activeJobGeneration += 1; activeJobId = null;
    if (ui["api-key"].value.trim()) activeJobTimer = setTimeout(recoverActiveJob, 400);
  }
  function setSource(url, name) {
    if (sourceUrl) URL.revokeObjectURL(sourceUrl);
    sourceUrl = url;
    ui["source-video"].src = url;
    ui["source-video"].load();
    ui["video-name"].textContent = name || "Source video";
    display(ui["source-video"], true); display(ui["video-empty"], false); display(ui["video-caption"], true);
    message(ui["video-error"], "");
  }
  function clearSource() {
    if (sourceUrl) URL.revokeObjectURL(sourceUrl);
    sourceUrl = null;
    ui["source-video"].removeAttribute("src"); ui["source-video"].load();
    display(ui["source-video"], false); display(ui["video-empty"], true); display(ui["video-caption"], false); message(ui["video-error"], "");
    ui["detection-overlay"].replaceChildren(); display(ui["detection-overlay"], false);
  }
  function selectFile(selected) {
    if (!selected) return;
    if (!/\.mov$/i.test(selected.name)) { message(ui["form-error"], "Choose a MOV video (.mov) to match the analysis API."); return; }
    if (selected.size === 0) { message(ui["form-error"], "The selected video is empty. Choose another file."); return; }
    if (submitting || polling) { message(ui["form-error"], "Wait for the current review, or pause its updates before selecting another video."); return; }
    stopPolling(); resetDetections(); currentJob = null; lastDetails = {}; lastResult = null; loadedJobVideo = null;
    file = selected; ui["file-name"].textContent = file.name; ui["file-size"].textContent = formatSize(file.size);
    display(ui["selected-file"], true); display(ui["drop-zone"], false);
    display(ui["results"], false); display(ui["job-controls"], false); display(ui["progress-section"], false);
    ui["status-title"].textContent = "Video selected"; ui["status-description"].textContent = "Start an analysis to review the visible sequence against the supported rules.";
    setBadge("Ready to analyze", ""); message(ui["form-error"], ""); message(ui["poll-error"], "");
    setSource(URL.createObjectURL(file), file.name); syncSubmit();
  }
  function setBadge(text, className) { ui["status-badge"].textContent = text; ui["status-badge"].className = `status-badge ${className}`; }
  function stopPolling() {
    polling = false; pollGeneration += 1;
    clearTimeout(pollTimer); clearInterval(elapsedTimer);
    if (pollController) pollController.abort();
    pollController = null; ui["poll-button"].textContent = "Resume updates";
    syncSubmit();
  }
  function updateElapsed() {
    if (!acceptedAt) { ui["elapsed-label"].textContent = ""; return; }
    ui["elapsed-label"].textContent = `${formatTime((Date.now() - acceptedAt) / 1000)} elapsed`;
  }
  function beginJob(id, localUpload = false) {
    if (localUpload) { busy = true; activeJobId = id; }
    stopPolling(); resetDetections(); showAllAssessments = false; currentJob = id; lastDetails = {}; lastResult = null;
    if (!localUpload) { file = null; display(ui["selected-file"], false); display(ui["drop-zone"], true); ui["video-file"].value = ""; clearSource(); loadedJobVideo = null; }
    acceptedAt = localUpload ? Date.now() : null;
    ui["job-id"].textContent = id; display(ui["job-controls"], true); display(ui["poll-button"], true); display(ui["results"], false); display(ui["progress-section"], true);
    ui["status-title"].textContent = "Review in progress"; ui["status-description"].textContent = "The worker is checking the visual sequence and preparing evidence.";
    setBadge("Processing", "processing"); message(ui["poll-error"], ""); resumePolling();
  }
  function resumePolling() {
    if (!currentJob) return;
    stopPolling(); polling = true; ui["poll-button"].textContent = "Pause updates"; message(ui["poll-error"], "");
    elapsedTimer = setInterval(updateElapsed, 1000); updateElapsed(); syncSubmit();
    poll(pollGeneration);
  }
  function updateProgress(metadata) {
    const progress = metadata.progress;
    const known = finite(progress);
    const percent = known ? Math.max(0, Math.min(100, Number(progress) <= 1 ? Number(progress) * 100 : Number(progress))) : 0;
    ui["progress-track"].classList.toggle("indeterminate", !known);
    if (known) { ui["progress-fill"].style.width = `${percent}%`; ui["progress-track"].setAttribute("aria-valuenow", String(Math.round(percent))); }
    else ui["progress-track"].removeAttribute("aria-valuenow");
    ui["stage-label"].textContent = friendly(metadata.stage || "Processing video");
    const began = metadata.accepted_at || metadata.created_at;
    const parsed = typeof began === "number" ? began * (began < 100000000000 ? 1000 : 1) : Date.parse(began);
    if (!acceptedAt && began && Number.isFinite(parsed)) acceptedAt = parsed;
  }
  async function poll(generation) {
    if (!polling || generation !== pollGeneration) return;
    pollController = new AbortController();
    try {
      const result = await json(`/api/v1/jobs/${encodeURIComponent(currentJob)}`, { signal: pollController.signal });
      let metadata = {};
      try { metadata = await json(`/api/ui/jobs/${encodeURIComponent(currentJob)}`, { signal: pollController.signal }); }
      catch (error) { if (error.name === "AbortError" || error.status === 401 || error.status === 403) throw error; metadata = {details_unavailable:true}; }
      if (generation !== pollGeneration) return;
      lastDetails = metadata;
      if (metadata.original_name) ui["video-name"].textContent = String(metadata.original_name);
      updateProgress(metadata);
      if (result.status === "done") {
        if (!["violation", "no_violation"].includes(result.verdict)) throw new ApiError("The completed response has no valid verdict. The review cannot be displayed as clear.", 0);
        stopPolling(); lastResult = result;
        display(ui["progress-section"], false); display(ui["poll-button"], metadata.details_unavailable === true);
        if (metadata.details_unavailable) ui["poll-button"].textContent = "Retry review details";
        renderResult(result, metadata);
        checkHealth();
        if (!sourceUrl) await loadJobVideo(currentJob, metadata.original_name);
        return;
      }
      if (result.status === "failed") {
        stopPolling(); display(ui["progress-section"], false); display(ui["poll-button"], false); display(ui["results"], false);
        setBadge("Failed", "failed"); ui["status-title"].textContent = "Analysis could not finish";
        ui["status-description"].textContent = "No verdict was produced. Review the error and submit the video again when the service is ready.";
        const error = metadata.error?.message || metadata.error || metadata.details?.error || "The worker reported a processing failure.";
        message(ui["poll-error"], typeof error === "string" ? error : "The worker reported a processing failure.");
        checkHealth();
        if (!sourceUrl) await loadJobVideo(currentJob, metadata.original_name);
        return;
      }
      if (result.status !== "processing") throw new ApiError("The service returned an unrecognized job status. Resume updates to retry.", 0);
      busy = true; activeJobId = currentJob; syncSubmit();
      display(ui["poll-button"], true);
      if (!sourceUrl && loadedJobVideo !== currentJob) await loadJobVideo(currentJob, metadata.original_name);
      if (generation === pollGeneration && polling) pollTimer = setTimeout(() => poll(generation), 1800);
    } catch (error) {
      if (error.name === "AbortError" || generation !== pollGeneration) return;
      stopPolling(); display(ui["poll-button"], true);
      ui["status-title"].textContent = "Updates paused";
      ui["status-description"].textContent = "The server job may still be running. Resume updates to retrieve its latest status.";
      message(ui["poll-error"], error.message); if (error.status === 401) ui["api-key"].focus();
    }
  }
  async function loadJobVideo(id, name) {
    loadedJobVideo = id;
    try {
      const response = await request(`/api/ui/jobs/${encodeURIComponent(id)}/video`);
      const blob = await response.blob();
      if (id === currentJob && !sourceUrl) setSource(URL.createObjectURL(blob), name || "Source video");
    } catch (_) { if (id === currentJob) message(ui["video-error"], "The source video could not be loaded. The evidence frames below remain available."); }
  }
  function seek(value) {
    if (!sourceUrl || !finite(value)) return;
    const video = ui["source-video"];
    const apply = () => { video.currentTime = Math.max(0, Math.min(Number(value), Number.isFinite(video.duration) ? video.duration : Number(value))); video.focus({ preventScroll: true }); video.scrollIntoView({ behavior: "smooth", block: "center" }); };
    if (video.readyState >= 1) apply(); else video.addEventListener("loadedmetadata", apply, { once: true });
  }
  function safeRuleUrl(value) {
    if (["/policies/workplace-belongings", "/policies/workplace-eating"].includes(value)) return value;
    try {
      const url = new URL(value);
      if (url.protocol === "https:" && !url.username && !url.password && (!url.port || url.port === "443") && ["adilet.zan.kz", "old.adilet.zan.kz", "nationalbank.kz", "www.nationalbank.kz"].includes(url.hostname)) return url.href;
    } catch (_) { /* Model-provided or malformed URLs are never made clickable. */ }
    return null;
  }
  function notesFor(metadata) {
    const details = metadata.details || {};
    return [details.review_notes, details.limitations, details.analysis_limitations, metadata.review_notes]
      .flatMap((value) => Array.isArray(value) ? value : typeof value === "string" ? [value] : [])
      .map((value) => {
        if (typeof value === "string") return value;
        if (value && typeof value.message === "string") {
          const context = [finite(value.timecode_sec) ? formatTime(value.timecode_sec) : "", value.type_id ? friendly(value.type_id) : ""].filter(Boolean).join(" · ");
          return context ? `${context}: ${value.message}` : value.message;
        }
        return "";
      }).filter((value) => value.trim());
  }
  function renderCoverage(metadata) {
    const coverage = metadata.details?.coverage;
    const metrics = [];
    const number = (value) => typeof value === "number" && Number.isFinite(value) && value >= 0;
    const add = (label, value) => {
      const metric = node("div", "coverage-metric");
      metric.append(node("dt", "", label), node("dd", "", value));
      metrics.push(metric);
    };
    if (coverage && typeof coverage === "object" && !Array.isArray(coverage)) {
      if (number(coverage.duration_sec)) add("Video duration", formatTime(coverage.duration_sec));
      if (number(coverage.windows_completed)) {
        const planned = number(coverage.windows_planned) ? ` / ${Math.floor(coverage.windows_planned)}` : "";
        add("Screening windows checked", `${Math.floor(coverage.windows_completed)}${planned}`);
      }
      if (number(coverage.max_sample_gap_sec)) add("Largest sample gap", `${coverage.max_sample_gap_sec.toFixed(2)} s`);
      if (number(coverage.source_frames_total)) add("Source frames", Math.floor(coverage.source_frames_total).toLocaleString("en-US"));
      const context = Array.isArray(coverage.context_windows_used) ? coverage.context_windows_used.length : coverage.context_windows_used;
      if (number(context)) add("Context windows checked", Math.floor(context).toLocaleString("en-US"));
      if (Array.isArray(coverage.visibility_gaps)) add("Intervals with limited view", coverage.visibility_gaps.length.toLocaleString("en-US"));
      if (Number.isSafeInteger(coverage.unresolved_verification_count) && coverage.unresolved_verification_count > 0) add("Checks not established", coverage.unresolved_verification_count.toLocaleString("en-US"));
      if (Number.isSafeInteger(coverage.unverified_candidate_count) && coverage.unverified_candidate_count > 0) add("Candidate checks not completed", coverage.unverified_candidate_count.toLocaleString("en-US"));
      if (coverage.verification_complete === false) add("Candidate verification", "Incomplete");
      if (number(coverage.perception?.frames_analyzed)) add("Motion and person samples", coverage.perception.frames_analyzed.toLocaleString("en-US"));
      if (number(coverage.perception?.max_gap_sec)) add("Perception sample gap", coverage.perception.max_gap_sec.toFixed(2) + " s");
      if (Array.isArray(metadata.details?.transactions?.episodes)) add("Tentative transaction episodes", metadata.details.transactions.episodes.length.toLocaleString("en-US"));
      const unknownZones = metadata.details?.scene?.zones?.filter(zone => zone.status !== "supported");
      if (Array.isArray(unknownZones)) add("Unresolved scene regions", unknownZones.length.toLocaleString("en-US"));
      const capabilities = metadata.details?.perception?.capabilities;
      for (const [key, label] of [["pose", "Hand position analysis"], ["open_vocabulary_objects", "Object analysis"]]) {
        const capability = capabilities?.[key];
        if (capability && typeof capability.status === "string") {
          const count = number(capability.frames_inspected) ? ` · ${Math.floor(capability.frames_inspected)} frames` : "";
          add(label, (capability.status === "enabled" ? "Available" : "Unavailable") + count);
        }
      }
      if (Array.isArray(metadata.details?.event_proposals?.proposals)) add("Activity proposals", metadata.details.event_proposals.proposals.length.toLocaleString("en-US"));
      const scenarios = metadata.details?.scenario_coverage;
      if (scenarios && typeof scenarios === "object" && !Array.isArray(scenarios)) {
        const rows = Object.values(scenarios).filter(row => row && typeof row === "object");
        add("Scenario families screened", rows.length.toLocaleString("en-US"));
        add("Visible scenarios verified", rows.reduce((count, row) => count + (Number.isSafeInteger(row.supported_visual_episodes) ? row.supported_visual_episodes : 0), 0).toLocaleString("en-US"));
      }
      if (typeof metadata.details?.calibration?.status === "string") add("Decision thresholds", metadata.details.calibration.status === "empirical_thresholds" ? "Reviewed calibration available" : "Awaiting reviewed calibration");
    }
    ui["coverage-metrics"].replaceChildren(...metrics);
    display(ui["coverage-section"], metrics.length > 0);
  }
  function needsCaseReview(item) {
    if (!item || item.status === "rejected" || item.status === "confirmed" || item.requires_review === false) return false;
    return item.requires_review === true || item.status === "review_required";
  }
  function sameReviewCase(left, right) {
    if ((left.type_id || left.type) !== (right.type_id || right.type)) return false;
    if (left.episode_id && right.episode_id && left.episode_id !== right.episode_id) return false;
    return !finite(left.timecode_sec) || !finite(right.timecode_sec) || Math.abs(Number(left.timecode_sec) - Number(right.timecode_sec)) < .01;
  }
  function activeReviewQueue(metadata) {
    const cases = metadata.details?.case_decision?.cases;
    return catalogEntries(metadata.details?.review_queue).filter(entry => {
      if (entry.requires_review === false || ["rejected", "confirmed", "diagnostic"].includes(entry.status)) return false;
      if (Array.isArray(entry.reason_codes) && entry.reason_codes.some(code => ["action_rejected", "action_contradiction", "legal_context_rejected", "context_contradiction"].includes(code))) return false;
      // New host decisions are authoritative. Older saved jobs have only a queue.
      return !Array.isArray(cases) || cases.some(item => item && sameReviewCase(item, entry) && needsCaseReview(item));
    });
  }
  function reviewExplanation(item) {
    const labels = {
      visible_scenario_supported:"The physical scenario is visible. Check whether the rule applies.",
      action_uncertain:"The visible action could not be established.", action_visibility_limited:"The action is obscured or outside the sampled view.",
      action_checks_unresolved:"The visible action could not be established.", context_not_verified:"The transaction context has not been verified.",
      legal_context_uncertain:"The transaction context remains unclear.", context_checks_unresolved:"The rule's applicability remains unclear.",
      limited_visibility:"The sampled view is incomplete.", insufficient_context:"More transaction context is needed.",
      source_timestamp_gap:"A gap in the source video limits this check.", review_only_rule:"This rule requires a human policy check.",
      rule_not_mapped:"The visible observation needs a human policy check."
    };
    const code = Array.isArray(item.reason_codes) ? item.reason_codes.find(value => labels[value]) : null;
    return labels[code] || (item.visual_scenario_supported === true ? labels.visible_scenario_supported : "The available evidence does not establish a violation. Check the source video.");
  }
  function renderReviewQueue(metadata) {
    const entries = activeReviewQueue(metadata);
    const cards = entries.map((entry, index) => {
      const card = node("article", "review-item");
      const heading = node("div", "review-item-heading");
      const title = node("div");
      title.append(node("p", "review-item-label", `${entry.visual_scenario_supported === true ? "VISIBLE SCENARIO · REVIEW POLICY" : "UNCONFIRMED"} ${String(index + 1).padStart(2, "0")}`), node("h4", "", typeof entry.type_id === "string" && entry.type_id ? friendly(entry.type_id) : "Visible action to review"));
      heading.append(title);
      if (typeof entry.timecode_sec === "number" && Number.isFinite(entry.timecode_sec) && entry.timecode_sec >= 0) {
        const time = node("button", "time-button", `▶ ${formatTime(entry.timecode_sec)}`);
        time.type = "button";
        time.setAttribute("aria-label", `Seek to unconfirmed observation at ${formatTime(entry.timecode_sec)}`);
        time.addEventListener("click", () => seek(entry.timecode_sec));
        heading.append(time);
      }
      card.append(heading);
      card.append(node("p", "review-item-message", reviewExplanation(entry)));
      if (entry.visual_scenario_supported === true && typeof entry.observed_action === "string" && entry.observed_action.trim()) {
        const observation = node("p", "review-observation");
        observation.append(node("strong", "", "Local observation: "), document.createTextNode(entry.observed_action));
        card.append(observation);
      }
      return card;
    });
    ui["review-queue"].replaceChildren(...cards);
    ui["review-queue-count"].textContent = `${entries.length} ${entries.length === 1 ? "item" : "items"}`;
    display(ui["review-queue-section"], entries.length > 0);
  }
  function resetDetections() {
    detectionGeneration += 1;
    if (detectionController) detectionController.abort();
    detectionController = null;
    for (const url of detectionImageUrls.values()) URL.revokeObjectURL(url);
    detectionImageUrls.clear(); detections = [];
    ui["detections"].replaceChildren(); ui["detection-overlay"].replaceChildren();
    display(ui["detection-overlay"], false); display(ui["detection-controls"], false); display(ui["detections-section"], false);
  }
  function validDetection(item) {
    return item && typeof item === "object" && typeof item.label === "string" && item.label.trim()
      && typeof item.confidence === "number" && Number.isFinite(item.confidence) && item.confidence >= 0 && item.confidence <= 1
      && typeof item.timestamp_sec === "number" && Number.isFinite(item.timestamp_sec) && item.timestamp_sec >= 0
      && Array.isArray(item.bbox) && item.bbox.length === 4 && item.bbox.every(value => typeof value === "number" && Number.isFinite(value) && value >= 0 && value <= 1)
      && item.bbox[0] < item.bbox[2] && item.bbox[1] < item.bbox[3];
  }
  function detectionBox(item) {
    const box = node("div", "detection-box"), [x1, y1, x2, y2] = item.bbox;
    box.style.left = x1 * 100 + "%"; box.style.top = y1 * 100 + "%";
    box.style.width = (x2 - x1) * 100 + "%"; box.style.height = (y2 - y1) * 100 + "%";
    box.append(node("span", "box-label", item.label + " · " + confidenceText(item.confidence)));
    return box;
  }
  function refreshDetectionOverlay(time = ui["source-video"].currentTime) {
    const video = ui["source-video"], overlay = ui["detection-overlay"];
    // Exact source-frame matches only. A sampled box must never become a track.
    const matches = detections.filter(item => Math.abs(item.timestamp_sec - time) <= .001);
    const shown = sourceUrl && ui["show-detections"].checked && matches.length && video.videoWidth > 0 && video.videoHeight > 0;
    ui["overlay-status"].textContent = !ui["show-detections"].checked ? "Boxes hidden" : matches.length ? matches.length + " localized items at " + sampleTime(time) : "No boxes at this unsampled moment";
    overlay.replaceChildren(); display(overlay, Boolean(shown));
    if (!shown) return;
    const scale = Math.min(video.clientWidth / video.videoWidth, video.clientHeight / video.videoHeight);
    const width = video.videoWidth * scale, height = video.videoHeight * scale;
    overlay.style.left = video.offsetLeft + (video.clientWidth - width) / 2 + "px";
    overlay.style.top = video.offsetTop + (video.clientHeight - height) / 2 + "px";
    overlay.style.width = width + "px"; overlay.style.height = height + "px";
    overlay.append(...matches.map(detectionBox));
  }
  function renderDetections(metadata) {
    resetDetections();
    const supplied = metadata.details?.detections;
    detections = Array.isArray(supplied) ? supplied.filter(validDetection) : [];
    const generation = detectionGeneration, job = currentJob;
    detectionController = new AbortController();
    const signal = detectionController.signal, pendingImages = new Map();
    const imageFor = filename => {
      if (!pendingImages.has(filename)) pendingImages.set(filename, (async () => {
        const response = await request("/api/ui/jobs/" + encodeURIComponent(job) + "/evidence/" + encodeURIComponent(filename), {signal});
        const blob = await response.blob();
        if (generation !== detectionGeneration || job !== currentJob) return null;
        if (blob.type !== "image/jpeg") throw new ApiError("Source frame was not a JPEG image.", 0);
        const url = URL.createObjectURL(blob); detectionImageUrls.set(filename, url); return url;
      })());
      return pendingImages.get(filename);
    };
    let shown = 0;
    const more = node("button", "text-button"); more.type = "button";
    const appendPage = () => {
    for (const item of detections.slice(shown, shown + 24)) {
      const card = node("article", "detection-card"), top = node("div", "detection-card-heading");
      top.append(node("h4", "", item.label), node("span", "detection-confidence", confidenceText(item.confidence)));
      const time = node("button", "time-button", "▶ " + sampleTime(item.timestamp_sec)); time.type = "button";
      time.setAttribute("aria-label", "Seek to " + item.label + " at " + sampleTime(item.timestamp_sec));
      time.addEventListener("click", () => seek(item.timestamp_sec));
      card.append(top, node("p", "detection-kind", friendly(item.kind || "visible item") + " · model estimate"), time);
      const filename = item.evidence_frame?.filename;
      if (typeof filename === "string" && filename && !/[\\/]/.test(filename) && filename !== "." && filename !== "..") {
        const frame = node("div", "detection-frame"), image = node("img"), state = node("p", "detection-frame-state", "Loading source frame…");
        image.alt = item.label + " in sampled source frame at " + sampleTime(item.timestamp_sec);
        display(image, false); frame.append(image); card.append(frame, state);
        image.addEventListener("load", () => { display(image, true); frame.append(detectionBox(item)); display(state, false); });
        image.addEventListener("error", () => { state.textContent = "Source frame could not be displayed. Seek the video to inspect this item."; });
        imageFor(filename).then(url => { if (url && generation === detectionGeneration) image.src = url; }).catch(error => {
          if (error.name !== "AbortError" && generation === detectionGeneration) state.textContent = "Source frame unavailable. Seek the video to inspect this item.";
        });
      } else card.append(node("p", "detection-frame-state", "Source-frame preview unavailable. Use the timestamp to inspect the video."));
      ui["detections"].append(card);
    }
    shown = Math.min(detections.length, shown + 24);
    more.textContent = "Show more sampled items (" + shown + " of " + detections.length + ")";
    display(more, shown < detections.length); ui["detections"].append(more);
    };
    more.addEventListener("click", appendPage); appendPage();
    ui["detection-count"].textContent = detections.length + " localized " + (detections.length === 1 ? "item" : "items");
    if (Array.isArray(supplied) && !detections.length) ui["detections"].append(node("p", "detection-frame-state", "No items were localized in the sampled frames. This does not establish that an object is absent."));
    display(ui["detections-section"], Array.isArray(supplied)); display(ui["detection-controls"], detections.length > 0);
    refreshDetectionOverlay();
  }
  function caseTitle(type) {
    return ({active_intake_departure:"Leaving an unfinished cash intake",cash_intake_without_client_presence:"Cash intake without the client",previous_transaction_cash_exposed:"Earlier client's cash left exposed",loose_cash_issued_without_observed_count:"Payout counting",unmapped_visual_event:"Other visible observation"})[type] || friendly(type) || "Visible observation";
  }
  function reviewCases(metadata, result) {
    const cases = metadata.details?.case_decision?.cases;
    if (Array.isArray(cases)) return catalogEntries(cases);
    const events = Array.isArray(result?.violations) ? result.violations : [];
    const queue = activeReviewQueue(metadata);
    return [
      ...events.map(event => ({type_id:event.type,status:"confirmed",timecode_sec:event.timecode_sec,observed_action:event.description,reasons:["The source evidence supports a finding."],checks:[]})),
      ...queue.map(item => ({...item,status:"review_required",requires_review:true,reasons:[reviewExplanation(item)],checks:[]}))
    ];
  }
  function renderCaseDecisions(metadata, result) {
    const cases = reviewCases(metadata, result);
    ui["case-decisions"].replaceChildren(...cases.flatMap((item, index) => {
      const card = node("tr", "case-card"), name = node("th", "check-name", caseTitle(item.type_id));
      name.setAttribute("scope", "row");
      const reasons = Array.isArray(item.reasons) ? item.reasons.filter(reason => typeof reason === "string" && reason.trim()) : [];
      const short = reasons[0] || reviewExplanation(item);
      if (short) name.append(node("p", "case-reason-short", short.length > 110 ? short.slice(0, 107) + "…" : short));
      const labels = {confirmed:"Violation",review_required:"Needs review",rejected:"Not supported"};
      const outcome = node("td"), probability = node("td", "case-probability"), moment = node("td", "case-time"), detailCell = node("td");
      const active = needsCaseReview(item), supported = item.status === "confirmed";
      outcome.append(node("span", "case-status " + (supported ? "confirmed" : active ? "pending" : "rejected"), supported ? labels.confirmed : active ? labels.review_required : labels.rejected));
      const estimate = confidenceText(item.probability);
      probability.append(node("span", "case-probability-value", estimate === null ? "Not assessed" : estimate));
      if (estimate !== null) {
        probability.title = "Uncalibrated model estimate that the stated check is true; not a measured probability.";
        const scope = ({joint_checks:"All rule checks",action_only:"Visible action only",procedure_completion:"Procedure completed",event_occurrence:"Event occurs"})[item.probability_scope];
        if (scope) probability.append(node("span", "case-probability-scope", scope));
      }
      if (typeof item.timecode_sec === "number" && Number.isFinite(item.timecode_sec) && item.timecode_sec >= 0) {
        const time = node("button", "time-button", "▶ " + formatTime(item.timecode_sec)); time.type = "button";
        time.setAttribute("aria-label", "Seek to " + caseTitle(item.type_id) + " at " + sampleTime(item.timecode_sec));
        time.addEventListener("click", () => seek(item.timecode_sec)); moment.append(time);
      } else moment.textContent = "—";
      const detail = node("details", "case-details");
      const summary = node("summary", "", "View"), detailRow = node("tr", "case-detail-row"), detailBody = node("td", "case-detail-body");
      detailBody.colSpan = 5; detailRow.id = "case-detail-" + index;
      summary.setAttribute("aria-controls", detailRow.id); detail.append(summary);
      display(detailRow, false);
      detail.addEventListener("toggle", () => display(detailRow, detail.open));
      if ((supported || item.visual_scenario_supported === true || item.action_decision === "supported") && typeof item.observed_action === "string" && item.observed_action.trim()) detailBody.append(node("p", "case-observation", item.observed_action));
      const checks = Array.isArray(item.checks) ? item.checks.filter(check => check && typeof check.name === "string") : [];
      if (checks.length) {
        const list = node("ul", "case-checks");
        for (const check of checks) {
          const status = ["yes", "no"].includes(check.status) ? check.status : "unknown", row = node("li");
          const label = node("span", "check-label", (check.stage === "action" ? "Visible action" : check.stage === "exception_exclusion" ? "Exception excluded" : "Rule context") + " · " + friendly(check.name));
          const refs = Array.isArray(check.evidence_refs) ? check.evidence_refs.filter(ref => typeof ref?.timestamp_sec === "number" && Number.isFinite(ref.timestamp_sec) && ref.timestamp_sec >= 0).slice(0, 3) : [];
          if (refs.length) {
            const moments = node("span", "check-evidence");
            for (const ref of refs) {
              const time = node("button", "text-button", sampleTime(ref.timestamp_sec)); time.type = "button";
              time.setAttribute("aria-label", "Seek to evidence for " + friendly(check.name) + " at " + sampleTime(ref.timestamp_sec));
              time.addEventListener("click", () => seek(ref.timestamp_sec)); moments.append(time);
            }
            label.append(moments);
          }
          row.append(label, node("strong", "check-status " + status, ({yes:"Yes",no:"No",unknown:"Unknown"})[status])); list.append(row);
        }
        detailBody.append(list);
      }
      if (reasons.length) { const list = node("ul", "case-reasons"); list.append(...reasons.map(reason => node("li", "", reason))); detailBody.append(list); }
      if (!checks.length && !reasons.length && !item.observed_action) detailBody.append(node("p", "case-observation", "No additional check details were saved."));
      detailCell.append(detail); card.append(name, outcome, probability, moment, detailCell);
      detailRow.append(detailBody);
      return [card, detailRow];
    }));
    ui["case-decision-count"].textContent = cases.length + (cases.length === 1 ? " check" : " checks");
    display(ui["case-decision-section"], cases.length > 0);
  }
  function hasIncompleteVerification(metadata) {
    const details = metadata.details || {}, coverage = details.coverage || {};
    return details.processing_status === "review_incomplete" || metadata.processing_status === "review_incomplete"
      || coverage.verification_complete === false
      || Number.isSafeInteger(coverage.unverified_candidate_count) && coverage.unverified_candidate_count > 0;
  }
  function hasPendingReview(metadata) {
    const details = metadata.details || {}, coverage = details.coverage || {};
    const cases = details.case_decision?.cases;
    if (hasIncompleteVerification(metadata) || metadata.details_unavailable || (Array.isArray(cases) ? cases.some(needsCaseReview) : details.case_decision?.status === "review_required")
      || activeReviewQueue(metadata).length > 0
      || Array.isArray(coverage.visibility_gaps) && coverage.visibility_gaps.length > 0
      || Number.isSafeInteger(coverage.unresolved_verification_count) && coverage.unresolved_verification_count > 0
      || typeof coverage.windows_completed === "number" && typeof coverage.windows_planned === "number" && coverage.windows_completed < coverage.windows_planned) return true;
    return catalogEntries(details.assessments).some(item => {
      if (item.automation === "manual" || item.applicability === "not_applicable" || item.confirmed_finding) return false;
      if (item.unassessed_episode_count > 0 || item.unscoped_unknown_observation_count > 0 || ["unresolved", "partial_support"].includes(item.assessment)) return true;
      const evidenced = Array.isArray(item.evidence_refs) && item.evidence_refs.length > 0;
      return evidenced && typeof item.score === "number" && item.score > 0
        && (item.scoring_kind === "event_occurrence" || item.scoring_kind === "required_procedure" && item.assessment !== "unassessed" && item.assessment !== "completed");
    });
  }
  function renderResult(result, metadata) {
    const events = Array.isArray(result.violations) ? result.violations : [];
    const flagged = events.length > 0;
    const verificationIncomplete = hasIncompleteVerification(metadata);
    display(ui["results"], true);
    ui["status-title"].textContent = verificationIncomplete ? "Review incomplete" : "Review complete"; setBadge(verificationIncomplete ? "Incomplete" : "Complete", verificationIncomplete ? "queued" : "done");
    ui["status-description"].textContent = verificationIncomplete ? "Some candidate checks were not completed within the processing budget." : "Your results are ready.";
    const reviewRequired = !flagged && hasPendingReview(metadata);
    ui["verdict-card"].classList.toggle("flagged", flagged); ui["verdict-card"].classList.toggle("review-required", reviewRequired);
    ui["verdict-icon"].textContent = flagged ? "!" : reviewRequired ? "?" : "✓";
    ui["verdict-title"].textContent = flagged ? "Violation detected" : reviewRequired ? "Needs review" : "No violation detected";
    const count = reviewCases(metadata, result).filter(needsCaseReview).length;
    ui["verdict-description"].textContent = flagged ? events.length + (events.length === 1 ? " finding needs human review." : " findings need human review.")
      : verificationIncomplete ? "A violation is not confirmed. Some candidate checks were not completed within the processing budget."
      : reviewRequired ? "A violation is not confirmed. " + (count ? count + (count === 1 ? " case needs a check." : " cases need a check.") : "Review details need a check.")
      : "No supported violation was found in the sampled video.";
    ui["jury-verdict"].textContent = `Jury API verdict: ${flagged ? "violation" : "no_violation"} · ${events.length} confirmed ${events.length === 1 ? "finding" : "findings"}. Unconfirmed cases and item boxes do not change this API verdict.`;
    message(ui["details-warning"], metadata.details_unavailable ? "Review details could not be loaded. Case decisions, detected items, coverage and the unconfirmed queue may be missing. Use ‘Retry review details’ above before treating this review as complete." : "");
    renderCaseDecisions(metadata, result); renderDetections(metadata);
    const notes = [...new Set(notesFor(metadata))];
    ui["review-notes"].replaceChildren(...notes.map((note) => node("li", "", note))); display(ui["review-notes-section"], notes.length > 0);
    renderCoverage(metadata);
    renderAssessments(metadata);
    renderReviewQueue(metadata);
    display(ui["findings-heading"], events.length > 0); ui["finding-count"].textContent = `${events.length} ${events.length === 1 ? "event" : "events"}`;
    const findings = metadata.details?.findings || metadata.findings || [];
    ui["findings"].replaceChildren(...events.map((event, index) => {
      const match = Array.isArray(findings) ? findings.find((finding) => (finding.type_id || finding.type) === event.type && Math.abs(Number(finding.timecode_sec ?? finding.timestamp_sec ?? finding.start_sec) - Number(event.timecode_sec)) < 3) || findings[index] || {} : {};
      return renderFinding(event, match, index);
    }));
  }
  function renderFinding(event, detail, index) {
    const card = node("article", "finding-card card");
    const policy = detail.basis === "workplace_policy" || detail.rule?.basis === "workplace_policy";
    const top = node("div", "finding-top"), heading = node("div");
    heading.append(node("p", "finding-index", `EVENT ${String(index + 1).padStart(2, "0")}`), node("h3", "", detail.title || detail.type_name || friendly(event.type)));
    if (policy) heading.append(node("p", "score-state", "Workplace policy violation"));
    const time = node("button", "time-button", `▶ ${formatTime(event.timecode_sec)}`); time.type = "button"; time.setAttribute("aria-label", `Seek video to ${formatTime(event.timecode_sec)}`); time.addEventListener("click", () => seek(event.timecode_sec));
    top.append(heading, time); card.append(top, node("p", "finding-description", event.description || "Review the evidence and applicable rule."));
    const frames = Array.isArray(event.evidence_frames) ? event.evidence_frames.slice(0, 3) : [];
    const grid = node("div", "evidence-grid");
    for (const [frameIndex, frame] of frames.entries()) {
      const raw = String(frame.image_base64 || "").replace(/^data:image\/jpe?g;base64,/i, "").replace(/\s/g, "");
      if (!raw || raw.length > 24000000 || !/^[A-Za-z0-9+/]+=*$/.test(raw)) continue;
      const button = node("button", "evidence-button"); button.type = "button";
      const image = node("img"); image.src = `data:image/jpeg;base64,${raw}`; image.alt = `Evidence ${frameIndex + 1} for event ${index + 1} at ${formatTime(frame.timestamp_sec)}`; image.loading = "lazy";
      button.append(image, node("span", "", `${formatTime(frame.timestamp_sec)} · Evidence ${frameIndex + 1}`)); button.addEventListener("click", () => seek(frame.timestamp_sec));
      grid.append(button);
    }
    if (grid.childElementCount) card.append(grid);
    const rule = detail.rule || {};
    const ruleName = detail.rule_name || rule.name || rule.title || detail.rule_id;
    const paragraph = detail.paragraph || rule.paragraph;
    const url = safeRuleUrl(detail.rule_url || detail.source_url || rule.url || rule.source_url);
    if (ruleName || paragraph || url) {
      const box = node("div", "rule-box"), content = node("div");
      content.append(node("strong", "", ruleName || (policy ? "User-provided workplace policy" : "Applicable regulatory reference")));
      if (paragraph) content.append(node("p", "", `${policy ? "Section" : "Paragraph"} ${paragraph}`));
      if (detail.rule_explanation || rule.requirement) content.append(node("p", "", detail.rule_explanation || rule.requirement));
      if (url) { const link = node("a", "", policy ? "Read workplace policy ↗" : "Read official source ↗"); link.href = url; link.target = "_blank"; link.rel = "noopener noreferrer"; content.append(link); }
      box.append(node("span", "rule-symbol", "§"), content); card.append(box);
    }
    const uncertainty = detail.uncertainty || detail.visibility_note;
    if (typeof uncertainty === "string" && uncertainty.trim()) card.append(node("p", "uncertainty", `Review note: ${uncertainty}`));
    return card;
  }

  ui["api-key"].addEventListener("input", () => { scheduleActiveJobRecovery(); syncSubmit(); scheduleCatalog(); });
  ui["catalog-reload"].addEventListener("click", loadCatalog);
  byId("assessment-toggle")?.addEventListener("click", () => { showAllAssessments = !showAllAssessments; renderAssessments(lastDetails); });
  ui["key-visibility"].addEventListener("click", () => { const shown = ui["api-key"].type === "password"; ui["api-key"].type = shown ? "text" : "password"; ui["key-visibility"].textContent = shown ? "Hide" : "Show"; ui["key-visibility"].setAttribute("aria-label", shown ? "Hide API key" : "Show API key"); ui["key-visibility"].setAttribute("aria-pressed", String(shown)); });
  ui["video-file"].addEventListener("change", () => selectFile(ui["video-file"].files[0]));
  ui["remove-file"].addEventListener("click", () => { if (submitting || polling) return; file = null; ui["video-file"].value = ""; display(ui["selected-file"], false); display(ui["drop-zone"], true); clearSource(); syncSubmit(); });
  for (const eventName of ["dragenter", "dragover"]) ui["drop-zone"].addEventListener(eventName, (event) => { event.preventDefault(); ui["drop-zone"].classList.add("drag-over"); });
  for (const eventName of ["dragleave", "drop"]) ui["drop-zone"].addEventListener(eventName, (event) => { event.preventDefault(); ui["drop-zone"].classList.remove("drag-over"); });
  ui["drop-zone"].addEventListener("drop", (event) => selectFile(event.dataTransfer.files[0]));
  ui["upload-form"].addEventListener("submit", async (event) => {
    event.preventDefault();
    if (!file || !ready || busy || submitting || polling) return;
    submitting = true; syncSubmit(); message(ui["form-error"], "");
    const data = new FormData(); data.append("file", file);
    try {
      const result = await json("/api/v1/analyze", { method: "POST", body: data });
      if (typeof result.job_id !== "string" || !result.job_id) throw new ApiError("The service did not return a job ID. Check the service logs before retrying.", 0);
      beginJob(result.job_id, true);
    } catch (error) {
      message(ui["form-error"], error.message + (error.retryAfter ? ` Retry after ${error.retryAfter} seconds.` : ""));
      if (error.status === 401) ui["api-key"].focus();
      if (error.status === 503) checkHealth();
      if (error.status === 429) { busy = true; recoverActiveJob(); }
    } finally { submitting = false; syncSubmit(); if (busy && !activeJobId) recoverActiveJob(); }
  });
  ui["open-job-form"].addEventListener("submit", (event) => {
    event.preventDefault(); const id = ui["job-input"].value.trim();
    if (!ui["api-key"].value.trim()) { message(ui["form-error"], "Enter your service API key before opening a review."); ui["api-key"].focus(); return; }
    if (submitting) return;
    if (id) { display(ui["poll-button"], true); beginJob(id); }
  });
  ui["poll-button"].addEventListener("click", () => {
    if (polling) { stopPolling(); ui["status-title"].textContent = "Updates paused"; ui["status-description"].textContent = "Only browser updates are paused. The server analysis continues; resume to retrieve its result."; }
    else { ui["status-title"].textContent = "Checking review status"; resumePolling(); }
  });
  ui["source-video"].addEventListener("error", () => { if (sourceUrl) message(ui["video-error"], "Video preview unavailable. Evidence frames are still available."); });
  ui["show-detections"].addEventListener("change", () => refreshDetectionOverlay());
  for (const name of ["loadedmetadata", "seeked", "timeupdate", "pause", "resize"]) ui["source-video"].addEventListener(name, () => refreshDetectionOverlay());
  ui["source-video"].addEventListener("seeking", () => { ui["detection-overlay"].replaceChildren(); display(ui["detection-overlay"], false); });
  window.addEventListener("resize", () => refreshDetectionOverlay());
  if (typeof ui["source-video"].requestVideoFrameCallback === "function") {
    const presented = (_, metadata) => { refreshDetectionOverlay(metadata.mediaTime); ui["source-video"].requestVideoFrameCallback(presented); };
    ui["source-video"].requestVideoFrameCallback(presented);
  }
  window.addEventListener("beforeunload", () => { stopPolling(); resetDetections(); clearTimeout(catalogTimer); clearTimeout(activeJobTimer); if (catalogController) catalogController.abort(); if (activeJobController) activeJobController.abort(); if (sourceUrl) URL.revokeObjectURL(sourceUrl); ui["api-key"].value = ""; });
  checkHealth(); setInterval(checkHealth, 10000);
})();
