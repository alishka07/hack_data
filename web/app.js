"use strict";

(() => {
  const byId = (id) => document.getElementById(id);
  const ui = Object.fromEntries([
    "api-key", "key-visibility", "upload-form", "video-file", "drop-zone", "selected-file", "file-name", "file-size", "remove-file", "analyze-button", "admission-note", "form-error", "service-dot", "service-label", "status-title", "status-badge", "status-description", "progress-section", "progress-track", "progress-fill", "stage-label", "elapsed-label", "job-controls", "job-id", "poll-button", "poll-error", "source-video", "video-empty", "video-caption", "video-name", "video-error", "results", "verdict-card", "verdict-icon", "verdict-title", "verdict-description", "coverage-section", "coverage-metrics", "review-notes-section", "review-notes", "findings-heading", "finding-count", "findings", "review-queue-section", "review-queue-count", "review-queue", "catalog-section", "catalog-version", "catalog-reload", "catalog-status", "catalog-table-wrap", "catalog-items", "assessments-section", "assessments-version", "assessment-items", "open-job-form", "job-input", "video-stage", "detection-overlay", "detection-controls", "show-detections", "overlay-status", "detections-section", "detection-count", "detections", "case-decision-section", "case-decision-count", "case-decisions", "details-warning", "jury-verdict", "reviewer-section", "download-report", "reviewer-name", "review-camera", "review-session", "review-source-group", "review-coverage", "save-review-meta", "review-save-status", "reload-review", "review-error", "case-filters", "case-search", "case-empty", "missed-event-section", "missed-event-form", "missed-type", "missed-start", "missed-end", "missed-comment", "missed-events", "cancel-missed-event", "history-section", "history-search", "history-reload", "history-status", "history-items", "history-more", "quality-section", "quality-reload", "quality-status", "quality-summary", "quality-items", "quality-limitations", "event-navigation", "selected-case-label", "next-case", "event-timeline", "play-context", "play-context-status"
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

  let reviewState = null;
  let reviewDirty = false;
  let reviewSaving = false;
  let selectedCaseKey = null;
  let visibleCaseGroups = [];
  let allCaseGroups = [];
  let caseFilter = "all";
  let historyGeneration = 0;
  let historyTimer = null;
  let historyOffset = 0;
  let historyTotal = 0;
  let qualityGeneration = 0;
  let contextEnd = null;
  let pendingManualId = null;
  let manualFormDirty = false;
  const reviewDrafts = new Map();

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
    const labels = {"personal_food_at_workstation": "Личные вещи на рабочем месте: продукты питания", "workstation_departure_screen_left_on": "Уход с рабочего места с включённым экраном", "cashier_eating_at_workstation": "Приём пищи на рабочем месте", "active_intake_departure": "Уход во время незавершённого приёма наличных", "cash_intake_without_client_presence": "Приём наличных без присутствия клиента", "previous_transaction_cash_exposed": "Наличные предыдущего клиента вне места хранения", "intake_banknotes_counted": "Пересчёт принимаемых банкнот по одной", "intake_coins_counted": "Пересчёт принимаемых монет поштучно", "intake_client_presence_required": "Приём и пересчёт наличных при клиенте", "previous_cash_stored": "Хранение ранее принятых наличных", "loose_cash_issued_without_observed_count": "Пересчёт выдаваемых наличных при клиенте", "damaged_incomplete_strap_count": "Пересчёт банкнот из неполных или повреждённых пачек", "incomplete_coin_sack_count": "Пересчёт монет из неполных мешков", "payout_bank_packaging": "Выдача наличных в разрешённой упаковке", "client_recount_supervision": "Присутствие сотрудника при пересчёте клиентом", "backoffice_processing_control": "Контроль обработки наличных", "intake_authenticity_validity_checks": "Проверка подлинности и платёжности наличных", "machine_security_features": "Проверка защитных признаков оборудованием", "intake_client_identification": "Идентификация клиента при приёме наличных", "payout_client_identification": "Идентификация клиента при выдаче наличных", "intake_confirmation_document": "Документ о приёме наличных", "payout_confirmation_document": "Документ о выдаче наличных", "personal_belongings_in_cash_unit": "Личные вещи в кассовом узле", "personal_cash_valuables_in_cash_unit": "Личные деньги и ценности в кассовом узле", "personal_device_exception_eligibility": "Основание для использования личного устройства", "vault_entry_authorization": "Право доступа в хранилище", "vault_other_person_accompaniment": "Сопровождение посетителей хранилища", "keys_stamps_secure_handling": "Сохранность ключей, кодов, штампов и ценностей", "cash_unit_isolation": "Изоляция кассового узла и закрытие входов", "cashier_room_lockable_entrance": "Отдельный запираемый вход в кассу", "cash_transfer_device_serviceable": "Исправность устройства передачи наличных", "cash_transfer_controls_inside": "Управление передаточным устройством из кассы", "counting_room_lockable_entrance": "Запираемый вход в помещение пересчёта", "collection_loading_access_restriction": "Ограничение доступа при погрузке инкассации", "collection_loading_view_restriction": "Ограничение обзора погрузки инкассации", "vault_cash_segregation_storage": "Разделение и хранение наличных в хранилище", "unmapped_visual_event": "Другое наблюдаемое действие", "person": "Человек", "hand": "Рука", "cash": "Наличные", "computer": "Компьютер", "phone": "Телефон", "bag": "Сумка", "counter": "Рабочая поверхность", "drawer": "Ящик", "safe": "Сейф", "window": "Окно", "queued": "В очереди", "ingest": "Подготовка видео", "perception": "Поиск людей и предметов", "scene_calibration": "Определение зон", "screening": "Просмотр сцен", "verification": "Проверка событий", "finalizing": "Подготовка результата", "processing": "Обработка", "done": "Завершено", "workstation_departure": "Уход с рабочего места", "actual_workstation_departure": "Наблюдаемый уход с рабочего места", "workstation_departure_observed": "Наблюдаемый уход с рабочего места", "computer_at_workstation_identified": "Компьютер на рабочем месте определён", "workstation_unattended_after_departure": "Рабочее место без оператора после ухода", "same_screen_visibly_on_after_departure": "Тот же экран включён после ухода", "departure_screen_sequence_visible": "Последовательность ухода и состояния экрана видна", "non_cash_object_visually_supported": "Виден предмет, не относящийся к наличным", "object_cash_area_relationship_visible": "Расположение предмета в кассовой зоне видно", "food_item_visually_identified": "Продукты питания определены по кадру", "food_on_cashier_work_surface": "Продукты на рабочей поверхности кассира", "personal_food_on_work_table": "Продукты на рабочем столе", "unfinished_intake": "Приём наличных не завершён", "cash_exposed": "Наличные вне хранения", "cash_acceptance_or_count_observed": "Видимый приём или пересчёт наличных", "cash_form_visible": "Видимость формы и упаковки наличных", "cash_handling_observed": "Обращение с наличными", "cash_handover_observed": "Передача наличных", "cash_intake_established": "Установленный приём наличных", "cash_on_work_surface_observed": "Наличные вне контейнера", "cash_outside_designated_storage": "Наличные вне места хранения", "cash_payout_established": "Установленная выдача наличных", "cash_persistence_visually_supported": "Наличные в последовательных кадрах", "cash_processing_observed": "Обработка наличных", "cash_storage_relationship_observed": "Расположение наличных у хранилища", "client_absence_visually_supported": "Отсутствие клиента в обозреваемой зоне", "client_area_visible": "Видимость клиентской зоны", "client_cash_lineage_established": "Связь наличных с операцией клиента", "client_demonstrably_absent": "Подтверждение отсутствия клиента", "client_service_active": "Продолжение обслуживания клиента", "client_service_transition_observed": "Переход к другой операции", "complete_preparation_to_handover_visible": "Видимость подготовки и передачи целиком", "food_consumption_visibly_observed": "Видимый приём пищи", "handling_or_placement_observed": "Перемещение или размещение предмета", "key_or_stamp_visually_supported": "Опознание ключа или штампа", "loose_cash_established": "Неупакованные наличные", "next_distinct_client_transaction_active": "Начало следующей операции другого клиента", "no_earlier_or_offscreen_count_possible": "Исключение более раннего или скрытого пересчёта", "no_manual_or_machine_count_observed": "Отсутствие видимого ручного или машинного пересчёта", "object_continuity_established": "Непрерывность наблюдения одного предмета", "packaging_exception_excluded": "Исключение допустимой выдачи в упаковке", "person_operating_cashier_workstation": "Работа человека на кассовом месте", "previous_client_cash_identified": "Идентификация наличных предыдущего клиента", "previous_transaction_completed": "Завершение предыдущей операции", "processing_area_visible": "Видимость зоны обработки наличных", "relevant_context_visible": "Видимость необходимого контекста", "service_not_completed": "Незавершённое обслуживание", "storage_container_visible": "Видимость контейнера или хранилища", "storage_relationship_visible": "Расположение наличных относительно поверхности или хранилища", "workstation_work_context_visible": "Рабочий контекст кассового места", "verifying": "Проверка событий"};
    return labels[value] || String(value || "").replaceAll("_", " ").replace(/^./, (c) => c.toUpperCase());
  }
  function syncSubmit() {
    ui["analyze-button"].disabled = !ready || busy || !file || !ui["api-key"].value.trim() || submitting || reviewDirty || reviewSaving || Boolean(currentJob && polling);
    ui["analyze-button"].firstChild.textContent = submitting ? "Загружаем видео… " : "Анализировать видео ";
    ui["admission-note"].textContent = !ready ? "Ожидание обработчика анализа." : submitting ? "Загрузка в локальный сервис." : currentJob && polling ? "Идёт анализ вашего видео." : busy ? "Сейчас выполняется анализ. Следующее видео можно загрузить после его завершения." : "Сервис готов к новому анализу.";
    if (busy && activeJobId && activeJobId !== currentJob && !submitting) {
      const open = node("button", "text-button", "Открыть текущий анализ"); open.type = "button"; open.id = "open-active-job";
      open.addEventListener("click", () => { message(ui["form-error"], ""); beginJob(activeJobId); });
      ui["admission-note"].append(document.createTextNode(" "), open);
    }
  }
  async function request(path, options = {}, authenticated = true) {
    // Keep both video data and credentials on this service, including future callers.
    if (typeof path !== "string" || !path.startsWith("/api/") || path.includes("\\")) {
      throw new ApiError("Разрешены только локальные API-маршруты этого сервиса.", 0);
    }
    const headers = new Headers(options.headers || {});
    if (authenticated) {
      const key = ui["api-key"].value.trim();
      if (!key) throw new ApiError("Введите ключ API сервиса.", 401);
      headers.set("X-API-Key", key);
    }
    let response;
    try { response = await fetch(path, { ...options, headers, cache: "no-store", credentials: "same-origin", redirect: "error" }); }
    catch (error) { if (error.name === "AbortError") throw error; throw new ApiError("Не удалось связаться с локальным сервисом. Проверьте, что он запущен, и повторите.", 0); }
    if (!response.ok) {
      let detail = "";
      try {
        const data = await response.json();
        detail = typeof data.detail === "string" ? data.detail : typeof data.error === "string" ? data.error : typeof data.message === "string" ? data.message : "";
      } catch (_) { /* A plain error response still gets a clear status-specific message. */ }
      const errors = {400:"Не удалось выполнить запрос. Проверьте введённые данные.",401:"Ключ API не принят. Проверьте ключ и повторите.",403:"У этого ключа API нет доступа к анализу.",404:"Анализ не найден. Проверьте ID.",409:"Сохранённая версия изменилась. Загрузите актуальную версию.",413:"Видео превышает допустимый размер загрузки.",415:"Выберите видео в формате MOV.",422:"Проверьте имя, решение и границы интервалов. Интервал должен находиться внутри видео.",429:"Уже выполняется анализ. Откройте его или дождитесь завершения.",503:"Обработчик недоступен или загружается. Повторите после его запуска."};
      const retryAfter = response.headers.get("Retry-After");
      throw new ApiError(errors[response.status] || detail || `Запрос завершился ошибкой (HTTP ${response.status}).`, response.status, retryAfter);
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
    return ({event_occurrence:"Наличие события",required_procedure:"Обязательная процедура"})[value] || "Неизвестно";
  }
  function confidenceText(value) {
    return typeof value === "number" && Number.isFinite(value) && value >= 0 && value <= 1 ? scoreText(value * 100) : null;
  }
  function automationLabel(value) {
    return ({automatic_when_supported:"Автоматически при наличии доказательств",review_only:"Проверка человеком",manual:"Ручная проверка",excluded:"Исключено"})[value] || null;
  }
  function assessmentLabel(value) {
    return ({unassessed:"Не оценено",observed:"Данные о событии",completion_evidence:"Данные о выполнении процедуры",completed:"Процедура выполнена",unresolved:"Не установлено"})[value] || "Не оценено";
  }
  function episodeLabel(item) {
    const summary = item.scope === "per_episode_summary";
    const id = summary && typeof item.latest_episode_id === "string" ? item.latest_episode_id : typeof item.episode_id === "string" ? item.episode_id : typeof item.latest_episode_id === "string" ? item.latest_episode_id : null;
    if (!id) return null;
    const startTime = summary ? item.latest_episode_start_sec : item.start_sec;
    const endTime = summary ? item.latest_episode_end_sec : item.end_sec;
    const start = finite(startTime) ? ` · ${formatTime(startTime)}${finite(endTime) ? `–${formatTime(endTime)}` : ""}` : "";
    return "Эпизод" + start;
  }
  function scoreRow(item, initialOnly, episodes = []) {
    const row = node("tr");
    const id = typeof item.item_id === "string" ? item.item_id : typeof item.id === "string" ? item.id : "";
    const label = node("th", "", /[А-Яа-яЁё]/.test(friendly(id)) ? friendly(id) : typeof item.label === "string" && item.label ? item.label : friendly(id) || "Проверка без названия");
    label.setAttribute("scope", "row");
    const applicability = initialOnly ? "unknown" : item.applicability;
    const assessment = initialOnly ? "unassessed" : item.assessment;
    const excluded = applicability === "not_applicable";
    const checked = !initialOnly && !excluded && item.automation !== "manual"
      && !["unassessed", "unresolved"].includes(assessment)
      && scoreText(item.score) !== null;
    const untouched = initialOnly || item.automation === "manual" || assessment === "unassessed";
    const result = excluded ? "Не применимо" : untouched ? "Не проверено" : item.confirmed_finding === true ? "Подтверждено моделью"
      : assessment === "completed" && applicability === "applicable" ? "Выполнено" : "Требует проверки";
    const confidence = confidenceText(item.model_confidence);
    const estimate = checked && confidence ? confidence + (item.scoring_kind === "required_procedure" ? " · выполнение процедуры" : " · наличие события") : "—";
    row.append(label, node("td", "score-result", result), node("td", "score-value", estimate));
    if (excluded) row.classList.add("score-excluded");
    const detail = node("details", "score-details"), body = node("div", "score-detail-body");
    detail.append(node("summary", "", "Подробности"));
    body.append(node("p", "score-state", scoringKind(item.scoring_kind) + " · " + assessmentLabel(assessment)),
      node("p", "score-state", "Применимость: " + (({unknown:"Неизвестно",applicable:"Применимо",not_applicable:"Не применимо"})[applicability] || "Неизвестно")));
    const automation = automationLabel(item.automation);
    if (automation) body.append(node("p", "score-state", "Способ проверки: " + automation));
    const source = safeRuleUrl(item.source_url);
    if (source) {
      const policy = item.basis === "workplace_policy";
      if (policy) body.append(node("p", "score-state", "Основание: правило рабочего места от пользователя"));
      const link = node("a", "", policy ? "Открыть правило рабочего места ↗" : "Открыть официальный источник ↗");
      link.href = source; link.target = "_blank"; link.rel = "noopener noreferrer"; body.append(link);
    }
    if (!initialOnly) {
      const refs = Array.isArray(item.evidence_refs) ? item.evidence_refs : [];
      body.append(node("p", "score-state", refs.length + " ссылок на кадры"));
      if (confidence) body.append(node("p", "score-state", (item.scoring_kind === "required_procedure" ? "Оценка выполнения процедуры: " : "Оценка события: ") + confidence + " · без калибровки"));
      if (checked && scoreText(item.score)) body.append(node("p", "score-state", (item.scoring_kind === "required_procedure" ? "Остаточная оценка процедуры: " : "Оценка события: ") + scoreText(item.score) + " · без калибровки; оценка не определяет итог"));
      const scope = ({clip_event_summary:"Самое сильное подтверждение события в ролике",per_episode_summary:"Максимальная остаточная оценка среди проверенных операций",episode_assessment:"Один эпизод операции или события",per_episode_verification_required:"Проверяйте каждую применимую операцию",verified_finding_episode:"Один проверенный эпизод события"})[item.scope];
      if (scope) body.append(node("p", "score-state", "Область оценки: " + scope));
      if (item.unassessed_episode_count > 0) body.append(node("p", "score-state", "Неполный охват: " + item.unassessed_episode_count + " известных эпизодов без оценки этой проверки; применимость может быть неизвестна"));
      if (Number.isSafeInteger(item.unscoped_unknown_observation_count) && item.unscoped_unknown_observation_count > 0) body.append(node("p", "score-state", "Неполный охват: " + item.unscoped_unknown_observation_count + " окон с неопределённым началом процедуры; данные о выполнении сохранены"));
      if (item.scope_certainty === "window_only") body.append(node("p", "score-state", "Данные ограничены фрагментом · границы операции не подтверждены"));
      const latest = episodeLabel(item);
      if (latest) body.append(node("p", "score-state", "Последний оценённый " + latest.toLowerCase()));
      const moments = node("div", "check-evidence");
      for (const ref of refs.slice(0, 3)) {
        if (typeof ref?.timestamp_sec !== "number" || !Number.isFinite(ref.timestamp_sec) || ref.timestamp_sec < 0) continue;
        const time = node("button", "text-button", sampleTime(ref.timestamp_sec)); time.type = "button";
        time.setAttribute("aria-label", "Перейти к доказательству: " + sampleTime(ref.timestamp_sec));
        time.addEventListener("click", () => seek(ref.timestamp_sec)); moments.append(time);
      }
      if (moments.childElementCount) body.append(moments);
      if (episodes.length) {
        const unresolved = Number.isSafeInteger(item.pending_episode_count) && item.pending_episode_count >= 0 ? item.pending_episode_count : episodes.filter(episode => episode.applicability !== "not_applicable" && (episode.assessment !== "completed" || episode.applicability === "unknown")).length;
        const count = Number.isSafeInteger(item.episode_count) && item.episode_count >= episodes.length ? item.episode_count : episodes.length;
        body.append(node("p", "score-state", count + " эпизодов · " + unresolved + " требуют проверки или неизвестны"));
        const history = node("details", "score-episodes"), list = node("ul");
        history.append(node("summary", "", "Оценки эпизодов"));
        for (const [index, episode] of episodes.entries()) {
          const assessed = episode.applicability !== "not_applicable" && episode.automation !== "manual" && !["unassessed", "unresolved"].includes(episode.assessment);
          const estimate = episode.applicability === "not_applicable" ? "Не применимо" : assessed ? scoreText(episode.score) || "Не проверено" : "Не проверено";
          const certainty = confidenceText(episode.model_confidence);
          const interval = finite(episode.start_sec) ? " · " + formatTime(episode.start_sec) + (finite(episode.end_sec) ? "–" + formatTime(episode.end_sec) : "") : "";
          list.append(node("li", "", "Эпизод " + (index + 1) + interval + ": " + estimate + " · " + assessmentLabel(episode.assessment) + (certainty ? " · " + (episode.scoring_kind === "required_procedure" ? "выполнение процедуры" : "event") + " оценка " + certainty : "") + (episode.applicability === "unknown" ? " · применимость неизвестна" : "")));
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
    ui["assessments-version"].textContent = shown.length + " из " + entries.length + " проверок" + (typeof version === "string" || typeof version === "number" ? " · Версия каталога " + version : "");
    const toggle = byId("assessment-toggle");
    if (toggle) { toggle.textContent = showAllAssessments ? "Значимые проверки" : "Все проверки"; display(toggle, relevant.length < entries.length); }
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
    ui["catalog-status"].textContent = "Загружаем каталог проверок…";
    catalogController = new AbortController();
    try {
      const catalog = await json("/api/ui/catalog", {signal:catalogController.signal});
      if (generation !== catalogGeneration) return;
      if (!catalog || !Array.isArray(catalog.items)) throw new ApiError("Сервис вернул некорректный каталог проверок.", 0);
      const entries = catalogEntries(catalog.items);
      ui["missed-type"].replaceChildren(...entries.map(item => { const option = node("option", "", caseTitle(item.id || item.item_id)); option.value = item.id || item.item_id; return option; }));
      ui["catalog-items"].replaceChildren(...entries.map((item) => scoreRow(item, true)));
      ui["catalog-version"].textContent = typeof catalog.version === "string" || typeof catalog.version === "number" ? `Версия каталога ${catalog.version}` : "";
      const automaticEvents = entries.filter((item) => item.scoring_kind === "event_occurrence" && item.automation === "automatic_when_supported").length;
      ui["catalog-status"].textContent = entries.length ? `${entries.length} проверок в каталоге · ${automaticEvents} проверок событий с автоматическим подтверждением` : "В каталоге нет проверок.";
      display(ui["catalog-table-wrap"], entries.length > 0);
    } catch (error) {
      if (error.name !== "AbortError" && generation === catalogGeneration) ui["catalog-status"].textContent = `Каталог недоступен. ${error.message}`;
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
    ui["catalog-status"].textContent = "Ожидание загрузки каталога…";
    if (ui["api-key"].value.trim()) catalogTimer = setTimeout(loadCatalog, 400);
  }
  async function checkHealth() {
    try {
      const health = await json("/api/v1/health", {}, false);
      ready = health.ready === true || health.worker?.ready === true || health.worker_ready === true || health.status === "ready";
      busy = health.busy === true;
      if (!busy) activeJobId = null;
      ui["service-dot"].className = `dot ${ready ? "ready" : "failed"}`;
      ui["service-label"].textContent = ready ? busy ? "Идёт анализ" : "Обработчик готов" : "Обработчик загружается / недоступен";
    } catch (_) { ready = false; ui["service-dot"].className = "dot failed"; ui["service-label"].textContent = "Сервис недоступен"; }
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
    ui["video-name"].textContent = name || "Исходное видео";
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
    if (reviewDirty || reviewSaving) { message(ui["review-error"], "Сохраните изменения проверки перед выбором другого видео."); ui["reviewer-section"].scrollIntoView({block:"center"}); return; }
    if (!/\.mov$/i.test(selected.name)) { message(ui["form-error"], "Выберите видео в формате MOV (.mov)."); return; }
    if (selected.size === 0) { message(ui["form-error"], "Выбран пустой файл. Выберите другое видео."); return; }
    if (submitting || polling) { message(ui["form-error"], "Дождитесь текущего анализа или приостановите обновления перед выбором другого видео."); return; }
    stopPolling(); resetDetections(); currentJob = null; lastDetails = {}; lastResult = null; loadedJobVideo = null;
    file = selected; ui["file-name"].textContent = file.name; ui["file-size"].textContent = formatSize(file.size);
    display(ui["selected-file"], true); display(ui["drop-zone"], false);
    display(ui["results"], false); display(ui["job-controls"], false); display(ui["progress-section"], false);
    ui["status-title"].textContent = "Видео выбрано"; ui["status-description"].textContent = "Начните анализ для проверки наблюдаемых действий.";
    setBadge("Готово к анализу", ""); message(ui["form-error"], ""); message(ui["poll-error"], "");
    setSource(URL.createObjectURL(file), file.name); syncSubmit();
  }
  function setBadge(text, className) { ui["status-badge"].textContent = text; ui["status-badge"].className = `status-badge ${className}`; }
  function stopPolling() {
    polling = false; pollGeneration += 1;
    clearTimeout(pollTimer); clearInterval(elapsedTimer);
    if (pollController) pollController.abort();
    pollController = null; ui["poll-button"].textContent = "Возобновить обновления";
    syncSubmit();
  }
  function updateElapsed() {
    if (!acceptedAt) { ui["elapsed-label"].textContent = ""; return; }
    ui["elapsed-label"].textContent = `${formatTime((Date.now() - acceptedAt) / 1000)} прошло`;
  }
  function beginJob(id, localUpload = false) {
    if (localUpload) { busy = true; activeJobId = id; }
    if ((reviewDirty || reviewSaving) && currentJob) { message(ui["review-error"], "Есть несохранённые изменения. Сохраните их перед открытием анализа."); ui["reviewer-section"].scrollIntoView({block:"center"}); return; }
    reviewState = null; reviewDirty = false; reviewDrafts.clear(); selectedCaseKey = null; caseFilter = "all"; ui["case-search"].value = "";
    stopPolling(); resetDetections(); showAllAssessments = false; currentJob = id; lastDetails = {}; lastResult = null;
    if (!localUpload) { file = null; display(ui["selected-file"], false); display(ui["drop-zone"], true); ui["video-file"].value = ""; clearSource(); loadedJobVideo = null; }
    acceptedAt = localUpload ? Date.now() : null;
    ui["job-id"].textContent = id; display(ui["job-controls"], true); display(ui["poll-button"], true); display(ui["results"], false); display(ui["progress-section"], true);
    ui["status-title"].textContent = "Идёт анализ"; ui["status-description"].textContent = "Обработчик проверяет видео и подготавливает доказательства.";
    setBadge("Обработка", "processing"); message(ui["poll-error"], ""); resumePolling();
  }
  function resumePolling() {
    if (!currentJob) return;
    stopPolling(); polling = true; ui["poll-button"].textContent = "Приостановить обновления"; message(ui["poll-error"], "");
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
    ui["stage-label"].textContent = friendly(metadata.stage || "Обработка видео");
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
        if (!["violation", "no_violation"].includes(result.verdict)) throw new ApiError("Ответ завершённого анализа не содержит корректного итога. Результат нельзя считать отсутствием нарушений.", 0);
        stopPolling(); lastResult = result;
        display(ui["progress-section"], false); display(ui["poll-button"], metadata.details_unavailable === true);
        if (metadata.details_unavailable) ui["poll-button"].textContent = "Повторить загрузку подробностей";
        renderResult(result, metadata);
        checkHealth();
        if (!sourceUrl) await loadJobVideo(currentJob, metadata.original_name);
        return;
      }
      if (result.status === "failed") {
        stopPolling(); display(ui["progress-section"], false); display(ui["poll-button"], false); display(ui["results"], false);
        setBadge("Ошибка", "failed"); ui["status-title"].textContent = "Не удалось завершить анализ";
        ui["status-description"].textContent = "Итог не сформирован. Проверьте ошибку и загрузите видео снова, когда сервис будет готов.";
        const error = metadata.error?.message || metadata.error || metadata.details?.error || "Обработчик сообщил об ошибке анализа.";
        message(ui["poll-error"], typeof error === "string" ? error : "Обработчик сообщил об ошибке анализа.");
        checkHealth();
        if (!sourceUrl) await loadJobVideo(currentJob, metadata.original_name);
        return;
      }
      if (result.status !== "processing") throw new ApiError("Сервис вернул неизвестный статус. Возобновите обновления.", 0);
      busy = true; activeJobId = currentJob; syncSubmit();
      display(ui["poll-button"], true);
      if (!sourceUrl && loadedJobVideo !== currentJob) await loadJobVideo(currentJob, metadata.original_name);
      if (generation === pollGeneration && polling) pollTimer = setTimeout(() => poll(generation), 1800);
    } catch (error) {
      if (error.name === "AbortError" || generation !== pollGeneration) return;
      stopPolling(); display(ui["poll-button"], true);
      ui["status-title"].textContent = "Обновления приостановлены";
      ui["status-description"].textContent = "Анализ на сервере может продолжаться. Возобновите обновления, чтобы получить результат.";
      message(ui["poll-error"], error.message); if (error.status === 401) ui["api-key"].focus();
    }
  }
  async function loadJobVideo(id, name) {
    loadedJobVideo = id;
    try {
      const response = await request(`/api/ui/jobs/${encodeURIComponent(id)}/video`);
      const blob = await response.blob();
      if (id === currentJob && !sourceUrl) setSource(URL.createObjectURL(blob), name || "Исходное видео");
    } catch (_) { if (id === currentJob) message(ui["video-error"], "Исходное видео не загрузилось. Сохранённые кадры доказательств остаются доступны."); }
  }
  function seek(value) {
    if (!sourceUrl || !finite(value)) return;
    const video = ui["source-video"];
    contextEnd = null;
    const apply = () => { video.currentTime = Math.max(0, Math.min(Number(value), Number.isFinite(video.duration) ? video.duration : Number(value))); video.focus({ preventScroll: true }); if (video.getBoundingClientRect().top < 0) video.scrollIntoView({ behavior: "smooth", block: "start" }); };
    if (video.readyState >= 1) apply(); else video.addEventListener("loadedmetadata", apply, { once: true });
  }
  function safeRuleUrl(value) {
    if (["/policies/workplace-belongings", "/policies/workplace-eating", "/policies/workplace-screen"].includes(value)) return value;
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
      if (number(coverage.duration_sec)) add("Длительность видео", formatTime(coverage.duration_sec));
      if (number(coverage.windows_completed)) {
        const planned = number(coverage.windows_planned) ? ` / ${Math.floor(coverage.windows_planned)}` : "";
        add("Просмотрено фрагментов", `${Math.floor(coverage.windows_completed)}${planned}`);
      }
      if (number(coverage.max_sample_gap_sec)) add("Наибольший промежуток между кадрами", `${coverage.max_sample_gap_sec.toFixed(2)} s`);
      if (number(coverage.source_frames_total)) add("Кадров в исходном видео", Math.floor(coverage.source_frames_total).toLocaleString("en-US"));
      const context = Array.isArray(coverage.context_windows_used) ? coverage.context_windows_used.length : coverage.context_windows_used;
      if (number(context)) add("Проверено контекстных фрагментов", Math.floor(context).toLocaleString("en-US"));
      if (Array.isArray(coverage.visibility_gaps)) add("Интервалов с ограниченным обзором", coverage.visibility_gaps.length.toLocaleString("en-US"));
      if (Number.isSafeInteger(coverage.unresolved_verification_count) && coverage.unresolved_verification_count > 0) add("Проверок без установленного результата", coverage.unresolved_verification_count.toLocaleString("en-US"));
      if (Number.isSafeInteger(coverage.unverified_candidate_count) && coverage.unverified_candidate_count > 0) add("Незавершённых проверок кандидатов", coverage.unverified_candidate_count.toLocaleString("en-US"));
      if (coverage.verification_complete === false) add("Проверка кандидатов", "Не завершено");
      if (number(coverage.perception?.frames_analyzed)) add("Кадров анализа людей и движения", coverage.perception.frames_analyzed.toLocaleString("en-US"));
      if (number(coverage.perception?.max_gap_sec)) add("Промежуток анализа движения", coverage.perception.max_gap_sec.toFixed(2) + " s");
      if (Array.isArray(metadata.details?.transactions?.episodes)) add("Предварительных эпизодов операций", metadata.details.transactions.episodes.length.toLocaleString("en-US"));
      const unknownZones = metadata.details?.scene?.zones?.filter(zone => zone.status !== "supported");
      if (Array.isArray(unknownZones)) add("Неопределённых зон сцены", unknownZones.length.toLocaleString("en-US"));
      const capabilities = metadata.details?.perception?.capabilities;
      for (const [key, label] of [["pose", "Анализ положения рук"], ["open_vocabulary_objects", "Анализ предметов"]]) {
        const capability = capabilities?.[key];
        if (capability && typeof capability.status === "string") {
          const count = number(capability.frames_inspected) ? ` · ${Math.floor(capability.frames_inspected)} frames` : "";
          add(label, (capability.status === "enabled" ? "Доступен" : "Недоступен") + count);
        }
      }
      if (Array.isArray(metadata.details?.event_proposals?.proposals)) add("Предложено событий", metadata.details.event_proposals.proposals.length.toLocaleString("en-US"));
      const scenarios = metadata.details?.scenario_coverage;
      if (scenarios && typeof scenarios === "object" && !Array.isArray(scenarios)) {
        const rows = Object.values(scenarios).filter(row => row && typeof row === "object");
        add("Просмотрено типов сценариев", rows.length.toLocaleString("en-US"));
        add("Проверено видимых сценариев", rows.reduce((count, row) => count + (Number.isSafeInteger(row.supported_visual_episodes) ? row.supported_visual_episodes : 0), 0).toLocaleString("en-US"));
      }
      if (typeof metadata.details?.calibration?.status === "string") add("Пороги решения", metadata.details.calibration.status === "empirical_thresholds" ? "Калибровка по проверенной разметке доступна" : "Ожидается калибровка по разметке");
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
      visible_scenario_supported:"Сценарий виден. Проверьте применимость правила.",
      action_uncertain:"По доступным кадрам действие не установлено.", action_visibility_limited:"Действие скрыто или вне проанализированной области.",
      action_checks_unresolved:"По доступным кадрам действие не установлено.", context_not_verified:"Контекст операции не проверен.",
      legal_context_uncertain:"Контекст операции неясен.", context_checks_unresolved:"Применимость правила не установлена.",
      limited_visibility:"Проанализированный обзор неполон.", insufficient_context:"Нужен дополнительный контекст операции.",
      source_timestamp_gap:"Разрыв исходного видео ограничивает проверку.", review_only_rule:"Применимость правила должен проверить человек.",
      rule_not_mapped:"Наблюдение требует проверки применимости правила человеком."
    };
    const code = Array.isArray(item.reason_codes) ? item.reason_codes.find(value => labels[value]) : null;
    return labels[code] || (item.visual_scenario_supported === true ? labels.visible_scenario_supported : "Доказательств недостаточно для подтверждения нарушения. Проверьте исходное видео.");
  }
  function renderReviewQueue(metadata) {
    const entries = activeReviewQueue(metadata);
    const cards = entries.map((entry, index) => {
      const card = node("article", "review-item");
      const heading = node("div", "review-item-heading");
      const title = node("div");
      title.append(node("p", "review-item-label", `${entry.visual_scenario_supported === true ? "СЦЕНАРИЙ ВИДЕН · ПРОВЕРЬТЕ ПРАВИЛО" : "НЕ ПОДТВЕРЖДЕНО"} ${String(index + 1).padStart(2, "0")}`), node("h4", "", typeof entry.type_id === "string" && entry.type_id ? friendly(entry.type_id) : "Действие для проверки"));
      heading.append(title);
      if (typeof entry.timecode_sec === "number" && Number.isFinite(entry.timecode_sec) && entry.timecode_sec >= 0) {
        const time = node("button", "time-button", `▶ ${formatTime(entry.timecode_sec)}`);
        time.type = "button";
        time.setAttribute("aria-label", `Перейти к неподтверждённому наблюдению: ${formatTime(entry.timecode_sec)}`);
        time.addEventListener("click", () => seek(entry.timecode_sec));
        heading.append(time);
      }
      card.append(heading);
      card.append(node("p", "review-item-message", reviewExplanation(entry)));
      if (entry.visual_scenario_supported === true && typeof entry.observed_action === "string" && entry.observed_action.trim()) {
        const observation = node("p", "review-observation");
        observation.append(node("strong", "", "Исходное наблюдение модели: "), document.createTextNode(entry.observed_action));
        card.append(observation);
      }
      return card;
    });
    ui["review-queue"].replaceChildren(...cards);
    ui["review-queue-count"].textContent = `${entries.length} ${entries.length === 1 ? "объект" : "объектов"}`;
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
    ui["overlay-status"].textContent = !ui["show-detections"].checked ? "Рамки скрыты" : matches.length ? matches.length + " найденных объектов в " + sampleTime(time) : "Этот момент не анализировался: рамок нет";
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
        if (blob.type !== "image/jpeg") throw new ApiError("Исходный кадр не является изображением JPEG.", 0);
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
      time.setAttribute("aria-label", "Перейти к " + item.label + " в " + sampleTime(item.timestamp_sec));
      time.addEventListener("click", () => seek(item.timestamp_sec));
      card.append(top, node("p", "detection-kind", friendly(item.kind || "видимый предмет") + " · оценка модели"), time);
      const filename = item.evidence_frame?.filename;
      if (typeof filename === "string" && filename && !/[\\/]/.test(filename) && filename !== "." && filename !== "..") {
        const frame = node("div", "detection-frame"), image = node("img"), state = node("p", "detection-frame-state", "Загружаем исходный кадр…");
        image.alt = item.label + " в исходном кадре: " + sampleTime(item.timestamp_sec);
        display(image, false); frame.append(image); card.append(frame, state);
        image.addEventListener("load", () => { display(image, true); frame.append(detectionBox(item)); display(state, false); });
        image.addEventListener("error", () => { state.textContent = "Не удалось показать кадр. Откройте соответствующий момент видео."; });
        imageFor(filename).then(url => { if (url && generation === detectionGeneration) image.src = url; }).catch(error => {
          if (error.name !== "AbortError" && generation === detectionGeneration) state.textContent = "Исходный кадр недоступен. Откройте соответствующий момент видео.";
        });
      } else card.append(node("p", "detection-frame-state", "Предпросмотр кадра недоступен. Откройте соответствующий момент видео."));
      ui["detections"].append(card);
    }
    shown = Math.min(detections.length, shown + 24);
    more.textContent = "Ещё найденные объекты (" + shown + " из " + detections.length + ")";
    display(more, shown < detections.length); ui["detections"].append(more);
    };
    more.addEventListener("click", appendPage); appendPage();
    ui["detection-count"].textContent = detections.length + " найденных " + (detections.length === 1 ? "объект" : "объектов");
    if (Array.isArray(supplied) && !detections.length) ui["detections"].append(node("p", "detection-frame-state", "В проанализированных кадрах объекты не локализованы. Это не доказывает их отсутствие."));
    display(ui["detections-section"], Array.isArray(supplied)); display(ui["detection-controls"], detections.length > 0);
    refreshDetectionOverlay();
  }
  function caseTitle(type) {
    return friendly(type) || "Наблюдаемое действие";
  }
  function reviewCases(metadata, result) {
    const cases = Array.isArray(metadata.review_cases) ? metadata.review_cases : metadata.details?.case_decision?.cases;
    if (Array.isArray(cases)) return catalogEntries(cases);
    const events = Array.isArray(result?.violations) ? result.violations : [];
    const queue = activeReviewQueue(metadata);
    return [
      ...events.map(event => ({type_id:event.type,status:"confirmed",timecode_sec:event.timecode_sec,observed_action:event.description,reasons:["Исходные кадры подтверждают событие."],checks:[]})),
      ...queue.map(item => ({...item,status:"review_required",requires_review:true,reasons:[reviewExplanation(item)],checks:[]}))
    ];
  }
  function caseResultStatus(item) {
    return item.status === "confirmed" ? "confirmed" : needsCaseReview(item) ? "review_required" : "rejected";
  }
  function validCaseTime(value) {
    return typeof value === "number" && Number.isFinite(value) && value >= 0;
  }
  function newEventId() {
    if (typeof crypto.randomUUID === "function") return crypto.randomUUID();
    const bytes = crypto.getRandomValues(new Uint8Array(16)); bytes[6] = (bytes[6] & 15) | 64; bytes[8] = (bytes[8] & 63) | 128;
    const hex = [...bytes].map(value => value.toString(16).padStart(2,"0")).join("");
    return `${hex.slice(0,8)}-${hex.slice(8,12)}-${hex.slice(12,16)}-${hex.slice(16,20)}-${hex.slice(20)}`;
  }
  function displayCaseGroups(cases) {
    // Presence checks can repeat across sampling windows. Event-specific actions
    // remain separate, and the underlying decisions and evidence are untouched.
    const presenceTypes = new Set(["personal_belongings_in_cash_unit", "personal_food_at_workstation"]);
    const groups = [], byKey = new Map();
    for (const item of cases) {
      const status = caseResultStatus(item);
      const key = presenceTypes.has(item.type_id) && validCaseTime(item.timecode_sec)
        ? JSON.stringify([item.type_id, status, status === "rejected" ? rejectedLabel(item) : null, item.probability_scope || null]) : null;
      let group = key === null ? null : byKey.get(key);
      if (!group) {
        group = {items:[], status, order:groups.length}; groups.push(group);
        if (key !== null) byKey.set(key, group);
      }
      group.items.push(item);
    }
    for (const group of groups) {
      group.items.sort((a, b) => (validCaseTime(a.timecode_sec) ? a.timecode_sec : Infinity) - (validCaseTime(b.timecode_sec) ? b.timecode_sec : Infinity));
      group.start_sec = validCaseTime(group.items[0].timecode_sec) ? group.items[0].timecode_sec : null;
      group.end_sec = validCaseTime(group.items.at(-1).timecode_sec) ? group.items.at(-1).timecode_sec : null;
    }
    const priority = {review_required:0, confirmed:1, rejected:2};
    return groups.sort((a, b) => priority[a.status] - priority[b.status]
      || (a.start_sec ?? Infinity) - (b.start_sec ?? Infinity) || a.order - b.order);
  }
  function caseProbabilityScope(item) {
    return ({joint_checks:"Все условия правила",action_only:"Только видимое действие",procedure_completion:"Процедура выполнена",event_occurrence:"Наличие события"})[item.probability_scope] || "";
  }
  function caseObservationFacts(item) {
    const checks = new Map();
    for (const check of Array.isArray(item.checks) ? item.checks : []) {
      if (!check || !["action", "context"].includes(check.stage) || typeof check.name !== "string") continue;
      if (!checks.has(check.name)) checks.set(check.name, []);
      checks.get(check.name).push(check);
    }
    const sources = name => {
      const rows = checks.get(name);
      if (!rows || rows.some(check => check.status !== "yes")) return new Set();
      const perCheck = rows.map(check => new Set((Array.isArray(check.evidence_refs) ? check.evidence_refs : [])
        .filter(ref => Number.isSafeInteger(ref?.source_index) && ref.source_index >= 0 && validCaseTime(ref.timestamp_sec))
        .map(ref => ref.source_index + ":" + ref.timestamp_sec)));
      return perCheck.some(refs => !refs.size) ? new Set() : new Set(perCheck.flatMap(refs => [...refs]));
    };
    const timestamp = ref => Number(ref.slice(ref.indexOf(":") + 1));
    const temporalBlocked = Array.isArray(item.reason_codes) && item.reason_codes.some(code => ["source_timestamp_gap", "scene_change_context", "insufficient_before_after"].includes(code));
    const sequence = refs => !temporalBlocked && new Set([...refs].map(ref => ref.split(":")[0])).size >= 2 && new Set([...refs].map(timestamp)).size >= 2;
    const departures = ["actual_workstation_departure", "workstation_departure_observed", "workstation_departure"].map(sources).filter(sequence);
    const afterDeparture = refs => refs.size && departures.some(before => Math.min(...[...refs].map(timestamp)) >= Math.max(...[...before].map(timestamp)));
    // Curated descriptions report only source-linked positive checks. Labels,
    // reason strings and unrestricted model prose cannot become observed facts.
    const phrases = [
      {family:"food",names:["food_item_visually_identified", "food_on_cashier_work_surface"],shared:true,text:"На рабочем столе видны продукты питания."},
      {family:"food",names:["personal_food_on_work_table"],text:"На рабочем столе видны продукты питания."},
      {family:"food",names:["food_item_visually_identified"],text:"В кадре видны продукты питания."},
      {family:"object",names:["non_cash_object_visually_supported", "object_cash_area_relationship_visible"],shared:true,text:"В кассовой зоне виден предмет, не относящийся к наличным."},
      {family:"object",names:["non_cash_object_visually_supported"],text:"В кадре виден предмет, не относящийся к наличным."},
      {family:"departure",names:["actual_workstation_departure"],sequence:true,text:"Человек покидает рабочее место."},
      {family:"departure",names:["workstation_departure_observed"],sequence:true,text:"Человек покидает рабочее место."},
      {family:"departure",names:["workstation_departure"],sequence:true,text:"Человек покидает рабочее место."},
      {family:"screen",names:["same_screen_visibly_on_after_departure"],text:afterDeparture(sources("same_screen_visibly_on_after_departure")) ? "После ухода человека экран виден включённым." : "Экран компьютера виден включённым."},
      {family:"computer",names:["computer_at_workstation_identified"],text:"На рабочем месте виден компьютер."},
      {family:"unattended",names:["workstation_unattended_after_departure"],text:afterDeparture(sources("workstation_unattended_after_departure")) ? "После ухода человека рабочее место видно без оператора." : "Рабочее место видно без оператора."},
      {family:"cash",names:["cash_on_work_surface_observed"],text:"Вне контейнера видны предметы, похожие на банкноты."},
      {family:"cash",names:["cash_outside_designated_storage"],text:"Наличные видны вне места хранения."},
      {family:"handover",names:["cash_handover_observed"],text:"Видна передача предметов, похожих на банкноты."},
      {family:"cash",names:["cash_acceptance_or_count_observed"],text:"Видна работа с предметами, похожими на банкноты."},
      {family:"cash",names:["cash_handling_observed"],text:"Видна работа с предметами, похожими на банкноты."},
      {family:"cash",names:["cash_processing_observed"],text:"Видна обработка предметов, похожих на банкноты."},
      {family:"cash",names:["loose_cash_established"],text:"В кадре видны неупакованные наличные."},
      {family:"client",names:["client_area_visible", "client_absence_visually_supported"],shared:true,text:"В обозреваемой клиентской зоне клиент не виден."},
      {family:"client",names:["client_area_visible", "client_demonstrably_absent"],shared:true,text:"В обозреваемой клиентской зоне клиент не виден."},
      {family:"key",names:["key_or_stamp_visually_supported"],text:"В кадре виден предмет, похожий на ключ или штамп."},
      {family:"storage",names:["cash_storage_relationship_observed"],text:"У места хранения видны предметы, похожие на банкноты."},
      {family:"storage",names:["storage_relationship_visible"],text:"Видно расположение предметов, похожих на банкноты."},
      {family:"storage",names:["storage_container_visible"],text:"В кадре виден контейнер или место хранения."},
      {family:"eating",names:["food_consumption_visibly_observed"],text:"Человек принимает пищу."}
    ];
    return phrases.filter(phrase => {
      const refs = phrase.names.map(sources);
      return refs.every(values => values.size) && (!phrase.sequence || refs.every(sequence))
        && (!phrase.shared || [...refs[0]].some(ref => refs.every(values => values.has(ref))));
    });
  }
  function caseJustification(items) {
    const facts = items.map(caseObservationFacts);
    const shared = facts[0].filter(fact => facts.every(rows => rows.some(other => other.text === fact.text)));
    const families = new Set(), sentences = [];
    for (const fact of shared) {
      if (families.has(fact.family)) continue;
      families.add(fact.family); sentences.push(fact.text);
      if (sentences.length === 2) break;
    }
    return sentences.join(" ") || (items.length > 1 && facts.some(rows => rows.length)
      ? "Наблюдения в этом диапазоне различаются; отдельные кадры — в подробностях."
      : items.every(item => item.status === "confirmed") ? "Для описания действия недостаточно сохранённых данных; проверьте кадры."
      : "По доступным кадрам действие не подтверждено.");
  }
  function justificationNode(items) {
    const description = node("p", "case-justification");
    description.append(node("strong", "", "Обоснование — "), document.createTextNode(caseJustification(items)));
    return description;
  }
  function appendCaseDetails(target, item) {
    const reasons = Array.isArray(item.reasons) ? item.reasons.filter(reason => typeof reason === "string" && reason.trim()) : [];
    target.append(justificationNode([item]));
    const checks = Array.isArray(item.checks) ? item.checks.filter(check => check && typeof check.name === "string") : [];
    if (checks.length) {
      const list = node("ul", "case-checks");
      for (const check of checks) {
        const status = ["yes", "no"].includes(check.status) ? check.status : "unknown", row = node("li");
        const label = node("span", "check-label", (check.stage === "action" ? "Видимое действие" : check.stage === "exception_exclusion" ? "Исключение проверено" : "Контекст правила") + " · " + friendly(check.name));
        const refs = Array.isArray(check.evidence_refs) ? check.evidence_refs.filter(ref => validCaseTime(ref?.timestamp_sec)).slice(0, 3) : [];
        if (refs.length) {
          const moments = node("span", "check-evidence");
          for (const ref of refs) {
            const time = node("button", "text-button", sampleTime(ref.timestamp_sec)); time.type = "button";
            time.setAttribute("aria-label", "Перейти к доказательству: " + friendly(check.name) + " в " + sampleTime(ref.timestamp_sec));
            time.addEventListener("click", () => seek(ref.timestamp_sec)); moments.append(time);
          }
          label.append(moments);
        }
        row.append(label, node("strong", "check-status " + status, ({yes:"Да",no:"Нет",unknown:"Неизвестно"})[status])); list.append(row);
      }
      target.append(list);
    }
    if (reasons.length) { const list = node("ul", "case-reasons"); list.append(...reasons.map(reason => node("li", "", /[А-Яа-яЁё]/.test(reason) ? reason : reviewExplanation(item)))); target.append(list); }
    if (!checks.length && !reasons.length) target.append(node("p", "case-observation", "Дополнительные сведения о проверке не сохранены."));
  }
  function renderCaseDecisions(metadata, result) {
    allCaseGroups = displayCaseGroups(reviewCases(metadata, result));
    for (const group of allCaseGroups) group.key = group.items.map(item => item.case_id || String(reviewCases(metadata, result).indexOf(item))).join("|");
    for (const button of ui["case-filters"].querySelectorAll("button")) {
      const count = button.dataset.filter === "all" ? allCaseGroups.length : allCaseGroups.filter(group => group.status === button.dataset.filter).length;
      button.querySelector("span").textContent = count;
      button.setAttribute("aria-pressed", String(caseFilter === button.dataset.filter));
    }
    const query = ui["case-search"].value.trim().toLocaleLowerCase("ru");
    const cases = allCaseGroups.filter(group => (caseFilter === "all" || group.status === caseFilter) && (!query || (caseTitle(group.items[0].type_id) + " " + caseJustification(group.items)).toLocaleLowerCase("ru").includes(query)));
    visibleCaseGroups = cases;
    display(ui["case-empty"], !cases.length);
    ui["case-decisions"].replaceChildren(...cases.flatMap((group, index) => {
      const item = group.items[0], repeated = group.items.length > 1;
      const card = node("tr", "case-card"), name = node("th", "check-name", caseTitle(item.type_id));
      card.dataset.caseKey = group.key; card.tabIndex = 0;
      card.addEventListener("click", event => { if (!event.target.closest("button,input,textarea,select,details,summary,a,label")) selectCase(group.key); });
      card.addEventListener("keydown", event => { if (event.target === card && ["Enter", " "].includes(event.key)) { event.preventDefault(); selectCase(group.key); } });
      name.setAttribute("scope", "row");
      name.append(justificationNode(group.items));
      if (repeated) name.append(node("p", "case-observation-count", group.items.length + " наблюдений"));
      const labels = {confirmed:"Подтверждено моделью",review_required:"Требует проверки",rejected:"Недостаточно данных"};
      const outcome = node("td"), probability = node("td", "case-probability"), moment = node("td", "case-time"), detailCell = node("td");
      const active = needsCaseReview(item), supported = item.status === "confirmed";
      outcome.append(node("span", "case-status " + (supported ? "confirmed" : active ? "pending" : "rejected"), supported ? labels.confirmed : active ? labels.review_required : rejectedLabel(item)));
      appendHumanStatus(outcome, group);
      const estimates = group.items.filter(member => confidenceText(member.probability) !== null).map(member => member.probability);
      const low = estimates.length ? confidenceText(Math.min(...estimates)) : null;
      const high = estimates.length ? confidenceText(Math.max(...estimates)) : null;
      const estimate = low === high ? low : low + "–" + high;
      probability.append(node("span", "case-probability-value", estimate === null ? "Нет оценки" : estimate));
      if (estimate !== null) {
        probability.title = repeated ? "Диапазон отдельных некалиброванных оценок; не общая вероятность за интервал." : "Некалиброванная оценка модели для указанной проверки, не измеренная вероятность.";
        const scope = caseProbabilityScope(item);
        if (scope) probability.append(node("span", "case-probability-scope", scope));
        if (repeated) probability.append(node("span", "case-probability-scope", estimates.length < group.items.length ? estimates.length + " из " + group.items.length + " оценено" : "По отдельным наблюдениям"));
      }
      if (group.start_sec !== null) {
        const range = group.end_sec > group.start_sec;
        const time = node("button", "time-button" + (range ? " case-time-range" : "")); time.type = "button";
        time.append(node("span", "", "▶ " + formatTime(group.start_sec)));
        if (range) time.append(node("span", "", "–" + formatTime(group.end_sec)));
        time.setAttribute("aria-label", "Перейти к " + caseTitle(item.type_id) + " в " + sampleTime(group.start_sec) + (range ? "; наблюдения до " + sampleTime(group.end_sec) : ""));
        if (range) time.title = "Диапазон наблюдений. Переход к первому кадру; отдельные моменты — в подробностях.";
        time.addEventListener("click", () => selectCase(group.key)); moment.append(time);
      } else moment.textContent = "—";
      const corrected = groupDecisions(group);
      if (corrected.length && corrected.every(value => value && validCaseTime(value.start_sec) && validCaseTime(value.end_sec) && value.start_sec === corrected[0].start_sec && value.end_sec === corrected[0].end_sec)) moment.append(node("span", "human-interval", "Человек: " + formatTime(corrected[0].start_sec) + "–" + formatTime(corrected[0].end_sec)));
      const detail = node("details", "case-details");
      const summary = node("summary", "", "Подробнее"), detailRow = node("tr", "case-detail-row"), detailBody = node("td", "case-detail-body");
      detailBody.colSpan = 5; detailRow.id = "case-detail-" + index;
      summary.setAttribute("aria-controls", detailRow.id); detail.append(summary);
      display(detailRow, false);
      detail.addEventListener("toggle", () => display(detailRow, detail.open));
      if (repeated) {
        detailBody.append(node("p", "case-group-note", "Наблюдения относятся к отдельным кадрам внутри диапазона. Непрерывное присутствие между кадрами не установлено."));
        for (const member of group.items) {
          const occurrence = node("section", "case-occurrence"), heading = node("h4", "case-occurrence-heading");
          const time = node("button", "time-button", "▶ " + sampleTime(member.timecode_sec)); time.type = "button";
          time.setAttribute("aria-label", "Перейти к наблюдению: " + caseTitle(member.type_id) + " в " + sampleTime(member.timecode_sec));
          time.addEventListener("click", () => seek(member.timecode_sec)); heading.append(time);
          const estimate = confidenceText(member.probability), scope = caseProbabilityScope(member);
          heading.append(node("span", "case-occurrence-probability", estimate === null ? "Нет оценки" : estimate + (scope ? " · " + scope : "")));
          occurrence.append(heading); appendCaseDetails(occurrence, member); detailBody.append(occurrence);
        }
      } else appendCaseDetails(detailBody, item);
      appendReviewEditor(detailBody, group);
      detailCell.append(detail); card.append(name, outcome, probability, moment, detailCell);
      detailRow.append(detailBody);
      return [card, detailRow];
    }));
    ui["case-decision-count"].textContent = cases.length + (cases.length === 1 ? " проверка" : " проверок");
    display(ui["case-decision-section"], allCaseGroups.length > 0);
    renderTimeline();
    if (selectedCaseKey && !cases.some(group => group.key === selectedCaseKey)) selectedCaseKey = null;
    selectCase(selectedCaseKey || cases[0]?.key, false);
  }
  function rejectedLabel(item) {
    const contradiction = ["action_rejected", "action_contradiction", "legal_context_rejected", "context_contradiction"];
    const negative = item.action_decision === "rejected" || item.context_decision === "rejected" || (Array.isArray(item.reason_codes) && item.reason_codes.some(code => contradiction.includes(code))) || (Array.isArray(item.checks) && item.checks.some(check => check?.status === "no"));
    return negative ? "Не подтвердилось" : "Недостаточно данных";
  }
  function markReviewDirty() {
    reviewDirty = true;
    ui["review-save-status"].textContent = "Есть несохранённые изменения";
    ui["reviewer-section"].classList.add("has-unsaved");
    syncSubmit();
  }
  function reviewMetadata() {
    return {reviewer:ui["reviewer-name"].value.trim(),camera_id:ui["review-camera"].value.trim(),session_id:ui["review-session"].value.trim(),source_group:ui["review-source-group"].value.trim(),coverage_complete:ui["review-coverage"].checked};
  }
  function initializeReview(metadata) {
    const available = metadata.review_state && typeof metadata.review_state === "object" && Array.isArray(metadata.review_cases);
    display(ui["reviewer-section"], true); display(ui["missed-event-section"], Boolean(available));
    ui["download-report"].disabled = Boolean(metadata.details_unavailable);
    for (const id of ["reviewer-name","review-camera","review-session","review-source-group","review-coverage","save-review-meta"]) ui[id].disabled = !available;
    if (!available) {
      reviewState = null; ui["review-save-status"].textContent = "Сохранение решений недоступно для этой версии сервера. Обновите сервер и повторно откройте анализ."; return;
    }
    reviewState = structuredClone(metadata.review_state);
    ui["reviewer-name"].value = reviewState.reviewer || ""; ui["review-camera"].value = reviewState.camera_id || "";
    ui["review-session"].value = reviewState.session_id || ""; ui["review-source-group"].value = reviewState.source_group || ""; ui["review-coverage"].checked = reviewState.coverage_complete === true;
    reviewDirty = false; reviewDrafts.clear(); pendingManualId = null; manualFormDirty = false; ui["missed-event-form"].reset(); ui["reviewer-section"].classList.remove("has-unsaved"); syncSubmit();
    ui["review-save-status"].textContent = reviewState.revision ? "Сохранено · версия " + reviewState.revision : "Решения проверяющего ещё не сохранены";
    display(ui["reload-review"], false); message(ui["review-error"], ""); renderMissedEvents();
  }
  function groupDecisions(group) {
    return group.items.map(item => (reviewState?.decisions || []).find(decision => decision.case_id === item.case_id) || null);
  }
  function appendHumanStatus(target, group) {
    const decisions = groupDecisions(group), values = new Set(decisions.map(decision => decision?.decision || "unreviewed"));
    const status = values.size === 1 ? [...values][0] : "mixed";
    const labels = {confirmed:"Человек: подтверждено",dismissed:"Человек: отклонено",uncertain:"Человек: недостаточно данных",unreviewed:"Человек: не проверено",mixed:"Человек: разные решения"};
    target.append(node("span", "human-status " + status, labels[status]));
  }
  function appendReviewEditor(target, group) {
    if (!reviewState || group.items.some(item => typeof item.case_id !== "string" || !item.case_id)) return;
    const editor = node("form", "review-editor"), title = node("h4", "", "Решение проверяющего");
    editor.append(title, node("p", "coverage-note", (group.items.length > 1 ? "Решение применяется ко всем " + group.items.length + " наблюдениям строки. " : "Оценка модели сохраняется отдельно. ") + "Проверьте границы события по видео; указанные границы сохраняются как разметка человека."));
    const existing = groupDecisions(group), first = existing[0];
    const common = key => first && existing.every(value => value && value[key] === first[key]) ? first[key] : "";
    const saved = reviewDrafts.get(group.key) || {decision:common("decision"),comment:common("comment"),start_sec:common("start_sec") === "" ? group.start_sec ?? "" : common("start_sec"),end_sec:common("end_sec") === "" ? group.end_sec ?? "" : common("end_sec")};
    const choices = node("div", "review-decision-choices");
    for (const [value,label] of [["confirmed","Подтвердить"],["dismissed","Отклонить"],["uncertain","Недостаточно данных"]]) {
      const choice = node("label", "review-decision-choice"), input = node("input"); input.type = "radio"; input.name = "human-decision-" + group.order; input.value = value; input.checked = saved.decision === value; input.required = true;
      choice.append(input, document.createTextNode(label)); choices.append(choice);
    }
    const times = node("div", "review-interval");
    const timeInput = (label,value,name) => { const wrap = node("label", "", label), input = node("input"); input.type = "number"; input.min = "0"; input.step = "0.01"; input.name = name; input.value = value ?? ""; wrap.append(input); times.append(wrap); return input; };
    const start = timeInput("Начало, сек.",saved.start_sec,"start_sec"), end = timeInput("Конец, сек.",saved.end_sec,"end_sec");
    const commentLabel = node("label", "", "Комментарий проверяющего"), comment = node("textarea"); comment.name = "comment"; comment.maxLength = 2000; comment.rows = 2; comment.placeholder = "Кратко опишите наблюдаемое действие и основание решения"; comment.value = saved.comment || ""; commentLabel.append(comment);
    const save = node("button", "button button-secondary", "Сохранить решение"), next = node("button", "text-button", "Следующий случай →"); save.type = "submit"; next.type = "button"; next.addEventListener("click", nextCase);
    const status = node("span", "editor-state", reviewDrafts.has(group.key) ? "Не сохранено" : "");
    const values = () => ({decision:editor.querySelector("input[type=radio]:checked")?.value || "",comment:comment.value.trim(),start_sec:start.value === "" ? null : Number(start.value),end_sec:end.value === "" ? null : Number(end.value)});
    editor.addEventListener("input", () => { reviewDrafts.set(group.key,values()); markReviewDirty(); status.textContent = "Не сохранено"; });
    editor.addEventListener("submit", async event => {
      event.preventDefault(); const draft = values();
      if (!draft.decision) return;
      if ((draft.start_sec === null) !== (draft.end_sec === null) || draft.start_sec !== null && (!validCaseTime(draft.start_sec) || !validCaseTime(draft.end_sec) || draft.end_sec < draft.start_sec)) { message(ui["review-error"], "Укажите обе границы интервала: конец должен быть не раньше начала."); return; }
      reviewDrafts.set(group.key,draft); markReviewDirty(); save.disabled = true;
      try { await saveReviewState(); } finally { save.disabled = false; }
    });
    editor.append(choices,times,commentLabel,save,next,status); target.append(editor);
  }
  async function saveReviewState(extra = {}) {
    if (!reviewState || reviewSaving) return false;
    if (!ui["reviewer-name"].value.trim()) { message(ui["review-error"], "Укажите имя проверяющего перед сохранением."); ui["reviewer-section"].open = true; ui["reviewer-name"].focus(); return false; }
    const next = {...structuredClone(reviewState),...reviewMetadata(),...extra};
    if (manualFormDirty) {
      const start = ui["missed-start"].value === "" ? null : Number(ui["missed-start"].value), end = ui["missed-end"].value === "" ? null : Number(ui["missed-end"].value), comment = ui["missed-comment"].value.trim(), type = ui["missed-type"].value;
      if (!validCaseTime(start) || !validCaseTime(end) || end < start || !comment || !type) { message(ui["review-error"], "Завершите форму пропущенного события: тип, интервал и описание. Либо очистите форму."); return false; }
      pendingManualId ||= newEventId();
      next.missed_events = [...(next.missed_events || []).filter(value => value.event_id !== pendingManualId),{event_id:pendingManualId,type_id:type,start_sec:start,end_sec:end,comment}];
    }
    const decisions = new Map((next.decisions || []).map(decision => [decision.case_id,decision]));
    for (const [key,draft] of reviewDrafts) {
      const group = allCaseGroups.find(value => value.key === key); if (!group) continue;
      if (!draft.decision) { message(ui["review-error"], "Выберите решение для изменённых строк перед сохранением."); return false; }
      if ((draft.start_sec === null) !== (draft.end_sec === null) || draft.start_sec !== null && (!validCaseTime(draft.start_sec) || !validCaseTime(draft.end_sec) || draft.end_sec < draft.start_sec)) { message(ui["review-error"], "Проверьте границы интервалов в изменённых строках."); return false; }
      for (const item of group.items) decisions.set(item.case_id,{case_id:item.case_id,...draft});
    }
    next.decisions = [...decisions.values()];
    const job = currentJob; reviewSaving = true;
    const locked = [...document.querySelectorAll("#api-key,#key-visibility,#reviewer-section input,#reviewer-section button,#case-filters button,#case-search,.review-editor input,.review-editor textarea,.review-editor button,#missed-event-form input,#missed-event-form select,#missed-event-form textarea,#missed-event-form button,#open-job-form button,#video-file,#remove-file,#history-items button")];
    locked.forEach(element => { element.disabled = true; }); ui["review-save-status"].textContent = "Сохраняем…"; message(ui["review-error"], "");
    try {
      const saved = await json(`/api/ui/jobs/${encodeURIComponent(job)}/review`,{method:"PUT",headers:{"Content-Type":"application/json"},body:JSON.stringify(next)});
      if (job !== currentJob) return false;
      if (!Number.isSafeInteger(saved.revision) || !Array.isArray(saved.decisions)) throw new ApiError("Сервер вернул некорректные данные сохранения.",0);
      lastDetails.review_state = saved; initializeReview(lastDetails); renderCaseDecisions(lastDetails,lastResult); loadHistory(); return true;
    } catch (error) {
      if (job !== currentJob) return false;
      ui["review-save-status"].textContent = "Изменения не сохранены";
      message(ui["review-error"], error.status === 409 ? "Сохранённая версия изменилась в другой вкладке. Ваши изменения остаются на экране. Загрузите сохранённую версию перед повторной проверкой." : "Не удалось сохранить. " + error.message);
      display(ui["reload-review"], error.status === 409); return false;
    } finally { reviewSaving = false; locked.forEach(element => { element.disabled = false; }); ui["save-review-meta"].disabled = !reviewState; syncSubmit(); }
  }
  function renderMissedEvents() {
    ui["missed-events"].replaceChildren(...(reviewState?.missed_events || []).map(event => {
      const row = node("li", "missed-event"), jump = node("button", "text-button", formatTime(event.start_sec) + "–" + formatTime(event.end_sec)); jump.type = "button"; jump.addEventListener("click", () => seek(event.start_sec));
      row.append(node("strong", "", caseTitle(event.type_id)), document.createTextNode(" · "),jump,node("p", "", event.comment)); return row;
    }));
  }
  function selectCase(key, move = true) {
    if (reviewSaving) return;
    if (key && !visibleCaseGroups.some(group => group.key === key) && allCaseGroups.some(group => group.key === key)) {
      caseFilter = "all"; ui["case-search"].value = ""; selectedCaseKey = key; renderCaseDecisions(lastDetails,lastResult);
    }
    const group = allCaseGroups.find(value => value.key === key); selectedCaseKey = group?.key || null;
    for (const row of ui["case-decisions"].querySelectorAll(".case-card")) { const selected = row.dataset.caseKey === selectedCaseKey; row.classList.toggle("is-selected",selected); row.setAttribute("aria-selected",String(selected)); }
    for (const marker of ui["event-timeline"].querySelectorAll("button")) marker.setAttribute("aria-pressed",String(marker.dataset.caseKey === selectedCaseKey));
    ui["selected-case-label"].textContent = group ? caseTitle(group.items[0].type_id) : "Выберите событие"; ui["play-context"].disabled = !group || group.start_sec === null;
    ui["next-case"].disabled = visibleCaseGroups.length < 2;
    if (move && group?.start_sec !== null && group) seek(group.start_sec);
  }
  function nextCase() {
    if (!visibleCaseGroups.length) return;
    const index = visibleCaseGroups.findIndex(group => group.key === selectedCaseKey), next = visibleCaseGroups[(index+1)%visibleCaseGroups.length]; selectCase(next.key);
    const row = [...ui["case-decisions"].querySelectorAll(".case-card")].find(value => value.dataset.caseKey === next.key); row?.scrollIntoView({block:"nearest",behavior:"smooth"});
  }
  function renderTimeline() {
    const duration = lastDetails.details?.coverage?.duration_sec;
    const max = validCaseTime(duration) && duration > 0 ? duration : Math.max(1,...allCaseGroups.map(group => group.end_sec || 0));
    ui["event-timeline"].replaceChildren(...allCaseGroups.filter(group => group.start_sec !== null).map((group,index) => {
      const marker = node("button", "event-marker " + group.status); marker.type = "button"; marker.dataset.caseKey = group.key;
      marker.style.left = Math.min(98,group.start_sec/max*100) + "%"; marker.style.width = Math.max(1.4,Math.min(100-group.start_sec/max*100,(group.end_sec-group.start_sec)/max*100)) + "%";
      marker.style.top = (index%3)*12+6 + "px"; marker.title = caseTitle(group.items[0].type_id) + " · " + formatTime(group.start_sec) + "–" + formatTime(group.end_sec); marker.setAttribute("aria-label",marker.title); marker.setAttribute("aria-pressed",String(group.key === selectedCaseKey)); marker.addEventListener("click",() => selectCase(group.key)); return marker;
    }));
    display(ui["event-navigation"],allCaseGroups.length > 0);
  }
  function playContext() {
    const group = allCaseGroups.find(value => value.key === selectedCaseKey); if (!group || group.start_sec === null || !sourceUrl) return;
    const video = ui["source-video"], start = Math.max(0,group.start_sec-3), end = group.end_sec+3;
    seek(start); contextEnd = Math.min(Number.isFinite(video.duration) ? video.duration : end,end); ui["play-context-status"].textContent = formatTime(start) + "–" + formatTime(contextEnd);
    const playback = video.play(); if (playback?.catch) playback.catch(() => { ui["play-context-status"].textContent = "Нажмите воспроизведение в плеере"; });
  }
  async function downloadReport() {
    if (!currentJob || reviewDirty) { message(ui["review-error"], "Сохраните изменения перед скачиванием отчёта."); return; }
    const job = currentJob; ui["download-report"].disabled = true; message(ui["review-error"], "");
    try {
      const response = await request(`/api/ui/jobs/${encodeURIComponent(job)}/report.html`), blob = await response.blob();
      if (!blob.type.startsWith("text/html")) throw new ApiError("Сервер вернул файл неизвестного формата.",0);
      const url = URL.createObjectURL(blob), link = node("a"); link.href = url; link.download = "forte-review-" + job + ".html"; link.hidden = true; document.body.append(link); link.click(); setTimeout(() => { link.remove(); URL.revokeObjectURL(url); },60000);
    } catch(error) { if (currentJob === job) message(ui["review-error"], "Не удалось скачать отчёт. " + error.message); }
    finally { ui["download-report"].disabled = false; }
  }
  function scheduleHistory() {
    clearTimeout(historyTimer); historyGeneration += 1;
    const available = Boolean(ui["api-key"].value.trim()); display(ui["history-section"],available);
    ui["history-items"].replaceChildren(); ui["history-status"].textContent = ""; display(ui["history-more"],false);
    if (available) historyTimer = setTimeout(() => loadHistory(),400);
  }
  async function loadHistory(append = false) {
    if (!ui["api-key"].value.trim()) return;
    const generation = ++historyGeneration, offset = append ? historyOffset : 0; ui["history-status"].textContent = "Загружаем…";
    try {
      const history = await json(`/api/ui/history?query=${encodeURIComponent(ui["history-search"].value.trim())}&limit=20&offset=${offset}`);
      if (generation !== historyGeneration) return;
      const items = catalogEntries(history.items); if (!append) ui["history-items"].replaceChildren();
      for (const item of items) {
        if (typeof item.job_id !== "string") continue;
        const open = node("button", "history-item"); open.type = "button";
        const states = {done:"Готово",processing:"Обработка",failed:"Ошибка"};
        open.append(node("strong", "",item.original_name || "Видео без названия"),node("span", "",(states[item.status] || "Статус неизвестен") + (item.review_revision ? " · проверка сохранена" : " · без решения человека")));
        open.addEventListener("click", () => beginJob(item.job_id)); ui["history-items"].append(open);
      }
      historyOffset = offset+items.length; historyTotal = Number.isSafeInteger(history.total) ? history.total : historyOffset;
      ui["history-status"].textContent = historyTotal ? "Найдено: " + historyTotal : "Сохранённых анализов нет"; display(ui["history-more"],historyOffset < historyTotal);
    } catch(error) { if (generation === historyGeneration) ui["history-status"].textContent = "История недоступна. " + error.message; }
  }
  async function loadQuality() {
    const generation = ++qualityGeneration; ui["quality-status"].textContent = "Загружаем проверенную разметку…";
    try {
      const data = await json("/api/ui/quality"); if (generation !== qualityGeneration) return;
      ui["quality-status"].textContent = data.status === "no_reviewed_labels" ? "Проверенной разметки пока недостаточно для оценки качества." : "Предварительные метрики по сохранённым решениям человека";
      const records = data.records || {}, grouping = data.grouping || {};
      ui["quality-summary"].replaceChildren(node("p", "", "Анализов с разметкой: " + (records.with_review || 0) + ". Полностью проверено: " + (records.eligible_complete_coverage || 0) + "."),node("p", "", "Исходных групп: " + (grouping.source_families || 0) + "; без группы: " + (grouping.unknown_group_records || 0) + ". Варианты одного исходного видео должны оставаться в одной выборке."));
      const rate = value => typeof value === "number" && Number.isFinite(value) ? (value*100).toFixed(1) + "%" : "Недостаточно данных";
      ui["quality-items"].replaceChildren(...catalogEntries(data.per_type).map(item => {
        const section = node("details", "quality-type"), automatic = item.automatic_decision || {}, coverage = item.event_coverage || {}, temporal = item.temporal_localization || {}, calibration = item.probability_evaluation || {};
        section.append(node("summary", "",caseTitle(item.type_id)),node("p", "", "Точность срабатываний автоматики, проверенная часть: " + rate(automatic.precision)),node("p", "", "Ложные тревоги автоматики: " + (automatic.false_positive ?? "—") + "; подтверждённые человеком пропуски: " + (item.known_misses?.total ?? "—")),node("p", "", "Полнота событий: " + rate(coverage.recall) + (coverage.denominator_status !== "complete_reviewed_subset" ? " · нет полного знаменателя" : " · полностью проверенная часть")),node("p", "", "Совпадение интервалов (IoU): " + rate(temporal.mean_iou) + " · пар: " + (temporal.pairs || 0)),node("p", "", "Калибровка оценки: " + (calibration.samples || 0) + " примеров; Brier: " + (typeof calibration.brier_score === "number" ? calibration.brier_score.toFixed(3) : "—")));
        return section;
      }));
      const limitations = [...(Array.isArray(data.limitations) ? data.limitations : []),...(Array.isArray(grouping.leakage_warnings) ? grouping.leakage_warnings : [])].filter(value => typeof value === "string");
      ui["quality-limitations"].replaceChildren(...limitations.map(value => node("li", "",value)));
    } catch(error) { if (generation === qualityGeneration) ui["quality-status"].textContent = "Метрики недоступны. " + error.message; }
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
    ui["status-title"].textContent = verificationIncomplete ? "Анализ не завершён полностью" : "Анализ завершён"; setBadge(verificationIncomplete ? "Не завершено" : "Завершено", verificationIncomplete ? "queued" : "done");
    ui["status-description"].textContent = verificationIncomplete ? "Часть кандидатов не проверена за отведённое время." : "Результаты готовы.";
    const reviewRequired = !flagged && hasPendingReview(metadata);
    ui["verdict-card"].classList.toggle("flagged", flagged); ui["verdict-card"].classList.toggle("review-required", reviewRequired);
    ui["verdict-icon"].textContent = flagged ? "!" : reviewRequired ? "?" : "✓";
    ui["verdict-title"].textContent = flagged ? "Модель подтвердила нарушение" : reviewRequired ? "Требует проверки" : "Модель не подтвердила нарушений";
    const count = displayCaseGroups(reviewCases(metadata, result)).filter(group => group.status === "review_required").length;
    ui["verdict-description"].textContent = flagged ? events.length + (events.length === 1 ? " событие требует проверки человеком." : " событий требуют проверки человеком.")
      : verificationIncomplete ? "Нарушение не подтверждено. Часть кандидатов не проверена за отведённое время."
      : reviewRequired ? "Нарушение не подтверждено. " + (count ? count + (count === 1 ? " случай требует проверки." : " случаев требуют проверки.") : "Подробности анализа требуют проверки.")
      : "В проанализированных кадрах подтверждённое нарушение не найдено.";
    ui["jury-verdict"].textContent = `Исходный итог API: ${flagged ? "violation" : "no_violation"} · подтверждено моделью: ${events.length}. Решения человека и рамки объектов сохраняются отдельно.`;
    message(ui["details-warning"], metadata.details_unavailable ? "Подробности анализа не загрузились. Список проверок, найденные предметы, охват и наблюдения могут быть неполными. Повторите загрузку подробностей." : "");
    initializeReview(metadata); renderCaseDecisions(metadata, result); renderDetections(metadata);
    const notes = [...new Set(notesFor(metadata))];
    ui["review-notes"].replaceChildren(...notes.map((note) => node("li", "", note))); display(ui["review-notes-section"], notes.length > 0);
    renderCoverage(metadata);
    renderAssessments(metadata);
    renderReviewQueue(metadata);
    display(ui["findings-heading"], events.length > 0); ui["finding-count"].textContent = `${events.length} ${events.length === 1 ? "событие" : "событий"}`;
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
    heading.append(node("p", "finding-index", `СОБЫТИЕ ${String(index + 1).padStart(2, "0")}`), node("h3", "", caseTitle(event.type)));
    if (policy) heading.append(node("p", "score-state", "Нарушение правила рабочего места"));
    const time = node("button", "time-button", `▶ ${formatTime(event.timecode_sec)}`); time.type = "button"; time.setAttribute("aria-label", `Перейти в видео: ${formatTime(event.timecode_sec)}`); time.addEventListener("click", () => seek(event.timecode_sec));
    top.append(heading, time);
    const matched = reviewCases(lastDetails, lastResult).filter(item => item.type_id === event.type && validCaseTime(item.timecode_sec) && Math.abs(item.timecode_sec - event.timecode_sec) < 3);
    card.append(top, node("p", "finding-description", caseJustification(matched.length ? [matched[0]] : [{status:"confirmed",checks:[]}])));
    const frames = Array.isArray(event.evidence_frames) ? event.evidence_frames.slice(0, 3) : [];
    const grid = node("div", "evidence-grid");
    for (const [frameIndex, frame] of frames.entries()) {
      const raw = String(frame.image_base64 || "").replace(/^data:image\/jpe?g;base64,/i, "").replace(/\s/g, "");
      if (!raw || raw.length > 24000000 || !/^[A-Za-z0-9+/]+=*$/.test(raw)) continue;
      const button = node("button", "evidence-button"); button.type = "button";
      const image = node("img"); image.src = `data:image/jpeg;base64,${raw}`; image.alt = `Кадр ${frameIndex + 1} события ${index + 1} в ${formatTime(frame.timestamp_sec)}`; image.loading = "lazy";
      button.append(image, node("span", "", `${formatTime(frame.timestamp_sec)} · Кадр ${frameIndex + 1}`)); button.addEventListener("click", () => seek(frame.timestamp_sec));
      grid.append(button);
    }
    if (grid.childElementCount) card.append(grid);
    const rule = detail.rule || {};
    const ruleName = detail.rule_name || rule.name || rule.title || detail.rule_id;
    const paragraph = detail.paragraph || rule.paragraph;
    const url = safeRuleUrl(detail.rule_url || detail.source_url || rule.url || rule.source_url);
    if (ruleName || paragraph || url) {
      const box = node("div", "rule-box"), content = node("div");
      content.append(node("strong", "", ruleName || (policy ? "Правило рабочего места от пользователя" : "Нормативное основание")));
      if (paragraph) content.append(node("p", "", `${policy ? "Раздел" : "Пункт"} ${paragraph}`));
      if (detail.rule_explanation || rule.requirement) content.append(node("p", "", detail.rule_explanation || rule.requirement));
      if (url) { const link = node("a", "", policy ? "Открыть правило рабочего места ↗" : "Открыть официальный источник ↗"); link.href = url; link.target = "_blank"; link.rel = "noopener noreferrer"; content.append(link); }
      box.append(node("span", "rule-symbol", "§"), content); card.append(box);
    }
    const uncertainty = detail.uncertainty || detail.visibility_note;
    if (typeof uncertainty === "string" && uncertainty.trim()) card.append(node("p", "uncertainty", `Примечание: ${uncertainty}`));
    return card;
  }

  for (const button of ui["case-filters"].querySelectorAll("button")) button.addEventListener("click", () => { if (reviewSaving) return; caseFilter = button.dataset.filter; renderCaseDecisions(lastDetails,lastResult); });
  ui["case-search"].addEventListener("input", () => { if (!reviewSaving) renderCaseDecisions(lastDetails,lastResult); });
  for (const id of ["reviewer-name","review-camera","review-session","review-source-group","review-coverage"]) ui[id].addEventListener("input",markReviewDirty);
  ui["save-review-meta"].addEventListener("click",() => saveReviewState());
  ui["download-report"].addEventListener("click",downloadReport);
  ui["next-case"].addEventListener("click",nextCase); ui["play-context"].addEventListener("click",playContext);
  ui["reload-review"].addEventListener("click",async () => {
    if (reviewSaving) return;
    const job = currentJob, key = ui["api-key"].value.trim(); reviewSaving = true;
    const locked = [...document.querySelectorAll("#api-key,#key-visibility,#reviewer-section input,#reviewer-section button,#case-filters button,#case-search,.review-editor input,.review-editor textarea,.review-editor button,#missed-event-form input,#missed-event-form select,#missed-event-form textarea,#missed-event-form button,#open-job-form button,#video-file,#remove-file,#history-items button")];
    locked.forEach(element => { element.disabled = true; });
    try { const metadata = await json(`/api/ui/jobs/${encodeURIComponent(job)}`); if (job === currentJob && key === ui["api-key"].value.trim()) { lastDetails = metadata; initializeReview(metadata); renderCaseDecisions(metadata,lastResult); } }
    catch(error) { message(ui["review-error"],"Не удалось загрузить сохранённую версию. " + error.message); }
    finally { reviewSaving = false; locked.forEach(element => { element.disabled = false; }); syncSubmit(); }
  });
  ui["missed-event-form"].addEventListener("submit",async event => {
    event.preventDefault(); if (!reviewState || reviewSaving) return; manualFormDirty = true; markReviewDirty(); await saveReviewState();
  });
  ui["missed-event-form"].addEventListener("input",() => { if (!reviewSaving) { manualFormDirty = true; markReviewDirty(); } });
  ui["cancel-missed-event"].addEventListener("click",() => {
    if (reviewSaving) return; ui["missed-event-form"].reset(); manualFormDirty = false; pendingManualId = null;
    const meta = reviewMetadata(), changed = Object.keys(meta).some(key => meta[key] !== (reviewState?.[key] ?? (key === "coverage_complete" ? false : "")));
    reviewDirty = reviewDrafts.size > 0 || changed; ui["reviewer-section"].classList.toggle("has-unsaved",reviewDirty); ui["review-save-status"].textContent = reviewDirty ? "Есть несохранённые изменения" : "Изменения отменены"; syncSubmit();
  });
  ui["history-reload"].addEventListener("click",() => loadHistory()); ui["history-more"].addEventListener("click",() => loadHistory(true));
  ui["history-search"].addEventListener("input",() => { clearTimeout(historyTimer); historyGeneration += 1; historyTimer = setTimeout(() => loadHistory(),250); });
  ui["history-section"].addEventListener("toggle",() => { if (ui["history-section"].open) loadHistory(); });
  ui["quality-reload"].addEventListener("click",loadQuality); ui["quality-section"].addEventListener("toggle",() => { if (ui["quality-section"].open) loadQuality(); });
  ui["source-video"].addEventListener("timeupdate",() => { if (contextEnd !== null && ui["source-video"].currentTime >= contextEnd) { ui["source-video"].pause(); contextEnd = null; } });
  window.addEventListener("beforeunload",event => { if (reviewDirty) { event.preventDefault(); event.returnValue = ""; } });
  ui["api-key"].addEventListener("input", () => { scheduleActiveJobRecovery(); syncSubmit(); scheduleCatalog(); scheduleHistory(); display(ui["quality-section"], Boolean(ui["api-key"].value.trim())); });
  ui["catalog-reload"].addEventListener("click", loadCatalog);
  byId("assessment-toggle")?.addEventListener("click", () => { showAllAssessments = !showAllAssessments; renderAssessments(lastDetails); });
  ui["key-visibility"].addEventListener("click", () => { const shown = ui["api-key"].type === "password"; ui["api-key"].type = shown ? "text" : "password"; ui["key-visibility"].textContent = shown ? "Скрыть" : "Показать"; ui["key-visibility"].setAttribute("aria-label", shown ? "Скрыть ключ API" : "Показать ключ API"); ui["key-visibility"].setAttribute("aria-pressed", String(shown)); });
  ui["video-file"].addEventListener("change", () => selectFile(ui["video-file"].files[0]));
  ui["remove-file"].addEventListener("click", () => { if (submitting || polling) return; if (reviewDirty || reviewSaving) { message(ui["review-error"], "Сохраните изменения проверки перед удалением видео."); return; } file = null; ui["video-file"].value = ""; display(ui["selected-file"], false); display(ui["drop-zone"], true); clearSource(); syncSubmit(); });
  for (const eventName of ["dragenter", "dragover"]) ui["drop-zone"].addEventListener(eventName, (event) => { event.preventDefault(); ui["drop-zone"].classList.add("drag-over"); });
  for (const eventName of ["dragleave", "drop"]) ui["drop-zone"].addEventListener(eventName, (event) => { event.preventDefault(); ui["drop-zone"].classList.remove("drag-over"); });
  ui["drop-zone"].addEventListener("drop", (event) => selectFile(event.dataTransfer.files[0]));
  ui["upload-form"].addEventListener("submit", async (event) => {
    event.preventDefault();
    if (!file || !ready || busy || submitting || polling || reviewDirty || reviewSaving) return;
    submitting = true; syncSubmit(); message(ui["form-error"], "");
    const data = new FormData(); data.append("file", file);
    try {
      const result = await json("/api/v1/analyze", { method: "POST", body: data });
      if (typeof result.job_id !== "string" || !result.job_id) throw new ApiError("Сервис не вернул ID анализа. Проверьте журнал сервиса перед повтором.", 0);
      beginJob(result.job_id, true);
    } catch (error) {
      message(ui["form-error"], error.message + (error.retryAfter ? ` Повторите через ${error.retryAfter} сек.` : ""));
      if (error.status === 401) ui["api-key"].focus();
      if (error.status === 503) checkHealth();
      if (error.status === 429) { busy = true; recoverActiveJob(); }
    } finally { submitting = false; syncSubmit(); if (busy && !activeJobId) recoverActiveJob(); }
  });
  ui["open-job-form"].addEventListener("submit", (event) => {
    event.preventDefault(); const id = ui["job-input"].value.trim();
    if (!ui["api-key"].value.trim()) { message(ui["form-error"], "Введите ключ API перед открытием анализа."); ui["api-key"].focus(); return; }
    if (submitting) return;
    if (id) { display(ui["poll-button"], true); beginJob(id); }
  });
  ui["poll-button"].addEventListener("click", () => {
    if (polling) { stopPolling(); ui["status-title"].textContent = "Обновления приостановлены"; ui["status-description"].textContent = "Приостановлены обновления страницы. Анализ на сервере продолжается; возобновите обновления для получения результата."; }
    else { ui["status-title"].textContent = "Проверяем статус анализа"; resumePolling(); }
  });
  ui["source-video"].addEventListener("error", () => { if (sourceUrl) message(ui["video-error"], "Предпросмотр видео недоступен. Сохранённые кадры остаются доступны."); });
  ui["show-detections"].addEventListener("change", () => refreshDetectionOverlay());
  for (const name of ["loadedmetadata", "seeked", "timeupdate", "pause", "resize"]) ui["source-video"].addEventListener(name, () => refreshDetectionOverlay());
  ui["source-video"].addEventListener("seeking", () => { ui["detection-overlay"].replaceChildren(); display(ui["detection-overlay"], false); });
  window.addEventListener("resize", () => refreshDetectionOverlay());
  if (typeof ui["source-video"].requestVideoFrameCallback === "function") {
    const presented = (_, metadata) => { refreshDetectionOverlay(metadata.mediaTime); ui["source-video"].requestVideoFrameCallback(presented); };
    ui["source-video"].requestVideoFrameCallback(presented);
  }
  window.addEventListener("pagehide", () => { stopPolling(); resetDetections(); clearTimeout(catalogTimer); clearTimeout(activeJobTimer); if (catalogController) catalogController.abort(); if (activeJobController) activeJobController.abort(); if (sourceUrl) URL.revokeObjectURL(sourceUrl); ui["api-key"].value = ""; });
  checkHealth(); setInterval(checkHealth, 10000);
})();
