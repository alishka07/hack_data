"use strict";

// Only interface-owned source text belongs here. Uploaded names, model prose,
// reviewer names and comments must be rendered through the raw-text bindings.
(() => {
  const exact = Object.fromEntries(`
Forte · Проверка кассы|Forte · Cash review
Forte — проверка кассы|Forte — cash review
Проверка кассы|Cash review
Проверяем сервис|Checking service
Проверка работы кассы|Cash operations review
Видео, доказательства и решения проверяющего в одном рабочем пространстве.|Video, evidence and reviewer decisions in one workspace.
Новый анализ|New analysis
Ключ API|API key
Введите ключ API|Enter API key
Показать ключ API|Show API key
Скрыть ключ API|Hide API key
Показать|Show
Скрыть|Hide
Хранится только в памяти этой вкладки.|Stored only in this tab's memory.
Перетащите видео MOV|Drop a MOV video here
Выбрать видео|Choose video
Обрабатывается локально|Processed locally
Убрать выбранное видео|Remove selected video
Анализировать видео|Analyze video
Ожидание обработчика.|Waiting for worker.
Открыть анализ по ID|Open analysis by ID
ID анализа|Analysis ID
Введите ID анализа|Enter analysis ID
Открыть анализ|Open analysis
История анализов|Analysis history
Поиск в истории|Search history
Название видео или ID|Video name or ID
Обновить|Refresh
Показать ещё|Show more
Качество по проверенной разметке|Quality from reviewed labels
Обновить метрики|Refresh metrics
Доступные проверки|Available checks
Список доступных проверок|List of available checks
Проверка|Check
Модель|Model
Оценка|Estimate
Просмотр видео|Video viewer
Выберите видео|Choose a video
Готово|Ready
Здесь появятся результаты.|Results will appear here.
Ход анализа|Analysis progress
В очереди|Queued
Приостановить обновления|Pause updates
Возобновить обновления|Resume updates
Выбрать видео с локального диска|Choose video from local drive
Предпросмотр видео|Video preview
Нажмите, чтобы выбрать MOV с компьютера|Click to choose a MOV file from your computer
Исходное загруженное видео|Original uploaded video
Исходное видео|Source video
Оригинал|Original
Показывать рамки|Show boxes
События на видео|Video events
Выберите событие|Select an event
Следующий случай →|Next case →
События на временной шкале|Events on timeline
▶ Фрагмент ±3 сек.|▶ Play ±3 sec.
Результат модели|Model result
Решение проверяющего|Reviewer decision
Проверяющий|Reviewer
Имя проверяющего|Reviewer name
Сохранить|Save
Отменить изменения и загрузить сохранённую версию|Discard changes and load saved version
Данные для оценки качества|Quality evaluation data
Решение человека хранится отдельно. Отчёт — один HTML-файл с кадрами.|Human decisions are stored separately. The report is one HTML file with frames.
Камера|Camera
Камера / точка|Camera / location
Сессия|Session
Смена / дата|Shift / date
Исходная группа|Source group
Один исходный ролик и все его варианты|One source video and all its variants
Я просмотрел(а) весь ролик и добавил(а) пропущенные события|I reviewed the entire video and added missed events
Результаты|Results
Проценты — некалиброванная оценка модели. Оценка видимого действия не означает, что все условия нарушения подтверждены. Для объединённых наблюдений показан диапазон отдельных оценок.|Percentages are uncalibrated model estimates. An estimate of a visible action does not confirm every condition of a violation. Grouped observations show the range of individual estimates.
Фильтр результатов|Filter results
Требует проверки|Needs review
Подтверждено моделью|Model confirmed
Остальные|Other results
Другие результаты|Other results
Все|All
Поиск по результатам|Search results
Поиск по проверке или обоснованию|Search checks or justification
По этому фильтру событий нет.|No events match this filter.
Проверки, оценки модели, время и доказательства|Checks, model estimates, time and evidence
Оценка модели|Model estimate
Время|Time
Подробности|Details
Добавить пропущенное событие|Add missed event
Отметьте событие, которого нет в результатах модели. Оно сохраняется как разметка проверяющего.|Add an event missing from the model results. It will be saved as a reviewer annotation.
Тип проверки|Check type
Начало, сек.|Start, sec.
Конец, сек.|End, sec.
Наблюдаемое действие|Observed action
Что видно в кадрах|What is visible in the frames
Добавить и сохранить|Add and save
Очистить форму|Clear form
Кадры подтверждённых моделью событий|Frames of model-confirmed events
Найденные предметы и люди|Detected objects and people
Рамки определены только в отдельных проанализированных кадрах.|Boxes are located only in individual analyzed frames.
Технические сведения анализа|Analysis technical details
Оценки проверок|Check estimates
Все проверки|All checks
Оценки модели помогают проверке и сами по себе не подтверждают нарушение.|Model estimates support review and do not confirm a violation on their own.
Оценённые проверки и данные модели|Assessed checks and model data
Охват видео|Video coverage
Наблюдения для проверки|Observations for review
Ограничения|Limitations
Завершение проверки|Complete review
Отчёт по анализу|Analysis report
Результаты, решения проверяющего и кадры — в одном HTML-файле.|Results, reviewer decisions and frames in one HTML file.
Сохранить изменения|Save changes
Скачать отчёт · HTML|Download report · HTML
Локальный анализ видео|Local video analysis
Личные вещи на рабочем месте: продукты питания|Personal belongings at workstation: food
Уход с рабочего места с включённым экраном|Leaving workstation with screen on
Приём пищи на рабочем месте|Eating at workstation
Уход во время незавершённого приёма наличных|Leaving during unfinished cash intake
Приём наличных без присутствия клиента|Cash intake without client present
Наличные предыдущего клиента вне места хранения|Previous client's cash outside storage
Пересчёт принимаемых банкнот по одной|Counting incoming banknotes individually
Пересчёт принимаемых монет поштучно|Counting incoming coins individually
Приём и пересчёт наличных при клиенте|Receiving and counting cash with client present
Хранение ранее принятых наличных|Storing previously received cash
Пересчёт выдаваемых наличных при клиенте|Counting outgoing cash with client present
Пересчёт банкнот из неполных или повреждённых пачек|Counting banknotes from incomplete or damaged bundles
Пересчёт монет из неполных мешков|Counting coins from incomplete bags
Выдача наличных в разрешённой упаковке|Issuing cash in permitted packaging
Присутствие сотрудника при пересчёте клиентом|Employee presence during client recount
Контроль обработки наличных|Cash processing supervision
Проверка подлинности и платёжности наличных|Cash authenticity and fitness checks
Проверка защитных признаков оборудованием|Equipment checks for security features
Идентификация клиента при приёме наличных|Client identification for cash intake
Идентификация клиента при выдаче наличных|Client identification for cash payout
Документ о приёме наличных|Cash intake document
Документ о выдаче наличных|Cash payout document
Личные вещи в кассовом узле|Personal belongings in cash area
Личные деньги и ценности в кассовом узле|Personal money and valuables in cash area
Основание для использования личного устройства|Eligibility to use a personal device
Право доступа в хранилище|Vault access authorization
Сопровождение посетителей хранилища|Accompanying vault visitors
Сохранность ключей, кодов, штампов и ценностей|Safekeeping of keys, codes, stamps and valuables
Изоляция кассового узла и закрытие входов|Cash area isolation and closed entrances
Отдельный запираемый вход в кассу|Separate lockable entrance to cash office
Исправность устройства передачи наличных|Cash transfer device serviceability
Управление передаточным устройством из кассы|Cash transfer controls inside cash office
Запираемый вход в помещение пересчёта|Lockable entrance to counting room
Ограничение доступа при погрузке инкассации|Restricted access during cash collection loading
Ограничение обзора погрузки инкассации|Restricted view of cash collection loading
Разделение и хранение наличных в хранилище|Cash segregation and storage in vault
Другое наблюдаемое действие|Other observed action
Человек|Person
Рука|Hand
Наличные|Cash
Компьютер|Computer
Телефон|Phone
Сумка|Bag
Рабочая поверхность|Work surface
Ящик|Drawer
Сейф|Safe
Окно|Window
Подготовка видео|Preparing video
Поиск людей и предметов|Detecting people and objects
Определение зон|Identifying zones
Просмотр сцен|Screening scenes
Проверка событий|Verifying events
Подготовка результата|Preparing results
Обработка|Processing
Обработка видео|Processing video
Завершено|Completed
Уход с рабочего места|Workstation departure
Наблюдаемый уход с рабочего места|Observed workstation departure
Компьютер на рабочем месте определён|Workstation computer identified
Рабочее место без оператора после ухода|Workstation unattended after departure
Тот же экран включён после ухода|Same screen on after departure
Последовательность ухода и состояния экрана видна|Departure and screen-state sequence visible
Виден предмет, не относящийся к наличным|Non-cash object visible
Расположение предмета в кассовой зоне видно|Object's position in cash area visible
Продукты питания определены по кадру|Food identified in frame
Продукты на рабочей поверхности кассира|Food on cashier's work surface
Продукты на рабочем столе|Food on work table
Приём наличных не завершён|Cash intake unfinished
Наличные вне хранения|Cash outside storage
Видимый приём или пересчёт наличных|Visible cash intake or counting
Видимость формы и упаковки наличных|Cash form and packaging visible
Обращение с наличными|Cash handling
Передача наличных|Cash handover
Установленный приём наличных|Cash intake established
Наличные вне контейнера|Cash outside container
Наличные вне места хранения|Cash outside designated storage
Установленная выдача наличных|Cash payout established
Наличные в последовательных кадрах|Cash in consecutive frames
Обработка наличных|Cash processing
Расположение наличных у хранилища|Cash position near storage
Отсутствие клиента в обозреваемой зоне|Client absent from visible area
Видимость клиентской зоны|Client area visible
Связь наличных с операцией клиента|Cash linked to client's transaction
Подтверждение отсутствия клиента|Client absence confirmed
Продолжение обслуживания клиента|Client service ongoing
Переход к другой операции|Transition to another transaction
Видимость подготовки и передачи целиком|Full preparation and handover visible
Видимый приём пищи|Visible eating
Перемещение или размещение предмета|Object handling or placement
Опознание ключа или штампа|Key or stamp identified
Неупакованные наличные|Loose cash
Начало следующей операции другого клиента|Next transaction for a different client started
Исключение более раннего или скрытого пересчёта|Earlier or off-screen counting excluded
Отсутствие видимого ручного или машинного пересчёта|No visible manual or machine count
Непрерывность наблюдения одного предмета|Continuous observation of same object
Исключение допустимой выдачи в упаковке|Permitted packaged payout excluded
Работа человека на кассовом месте|Person operating cashier workstation
Идентификация наличных предыдущего клиента|Previous client's cash identified
Завершение предыдущей операции|Previous transaction completed
Видимость зоны обработки наличных|Cash processing area visible
Видимость необходимого контекста|Relevant context visible
Незавершённое обслуживание|Service unfinished
Видимость контейнера или хранилища|Container or storage visible
Расположение наличных относительно поверхности или хранилища|Cash position relative to surface or storage
Рабочий контекст кассового места|Cashier workstation context visible
Загружаем видео…|Uploading video…
Ожидание обработчика анализа.|Waiting for analysis worker.
Загрузка в локальный сервис.|Uploading to local service.
Идёт анализ вашего видео.|Your video is being analyzed.
Сейчас выполняется анализ. Следующее видео можно загрузить после его завершения.|An analysis is running. You can upload another video once it finishes.
Сервис готов к новому анализу.|Service ready for a new analysis.
Открыть текущий анализ|Open current analysis
Сохраняем решения проверяющего…|Saving reviewer decisions…
Есть несохранённые правки. Сохраните их перед скачиванием отчёта.|You have unsaved changes. Save them before downloading the report.
Проверьте результаты и сохраните решения, чтобы включить их в отчёт.|Review the results and save your decisions to include them in the report.
Разрешены только локальные API-маршруты этого сервиса.|Only local API routes of this service are allowed.
Введите ключ API сервиса.|Enter the service API key.
Не удалось связаться с локальным сервисом. Проверьте, что он запущен, и повторите.|Could not reach the local service. Check that it is running and try again.
Не удалось выполнить запрос. Проверьте введённые данные.|Request failed. Check the entered data.
Ключ API не принят. Проверьте ключ и повторите.|API key rejected. Check the key and try again.
У этого ключа API нет доступа к анализу.|This API key cannot access the analysis.
Анализ не найден. Проверьте ID.|Analysis not found. Check the ID.
Сохранённая версия изменилась. Загрузите актуальную версию.|The saved version has changed. Load the current version.
Видео превышает допустимый размер загрузки.|Video exceeds the upload size limit.
Выберите видео в формате MOV (.mov).|Choose a MOV video (.mov).
Выберите видео в формате MOV.|Choose a MOV video.
Проверьте имя, решение и границы интервалов. Интервал должен находиться внутри видео.|Check the name, decision and interval boundaries. The interval must be within the video.
Уже выполняется анализ. Откройте его или дождитесь завершения.|An analysis is already running. Open it or wait for it to finish.
Обработчик недоступен или загружается. Повторите после его запуска.|Worker unavailable or loading. Retry once it starts.
Язык / Language|Language
Наличие события|Event occurrence
Обязательная процедура|Required procedure
Неизвестно|Unknown
Автоматически при наличии доказательств|Automatic when supported by evidence
Проверка человеком|Human review
Ручная проверка|Manual review
Исключено|Excluded
Не оценено|Not assessed
Данные о событии|Event evidence
Данные о выполнении процедуры|Procedure completion evidence
Процедура выполнена|Procedure completed
Не установлено|Unresolved
Эпизод|Episode
Проверка без названия|Unnamed check
Не применимо|Not applicable
Не проверено|Not checked
Выполнено|Completed
Применимо|Applicable
Основание: правило рабочего места от пользователя|Basis: user-defined workplace rule
Открыть правило рабочего места ↗|Open workplace rule ↗
Открыть официальный источник ↗|Open official source ↗
Самое сильное подтверждение события в ролике|Strongest event evidence in video
Максимальная остаточная оценка среди проверенных операций|Maximum residual estimate across assessed transactions
Один эпизод операции или события|One transaction or event episode
Проверяйте каждую применимую операцию|Review each applicable transaction
Один проверенный эпизод события|One verified event episode
Данные ограничены фрагментом · границы операции не подтверждены|Evidence limited to clip · transaction boundaries unconfirmed
Оценки эпизодов|Episode estimates
Значимые проверки|Relevant checks
Загружаем каталог проверок…|Loading check catalog…
Сервис вернул некорректный каталог проверок.|Service returned an invalid check catalog.
В каталоге нет проверок.|No checks in catalog.
Ожидание загрузки каталога…|Waiting to load catalog…
Идёт анализ|Analysis running
Обработчик готов|Worker ready
Обработчик загружается / недоступен|Worker loading / unavailable
Сервис недоступен|Service unavailable
Сохраните изменения проверки перед выбором другого видео.|Save review changes before choosing another video.
Выбран пустой файл. Выберите другое видео.|The selected file is empty. Choose another video.
Дождитесь текущего анализа или приостановите обновления перед выбором другого видео.|Wait for the current analysis or pause updates before choosing another video.
Видео выбрано|Video selected
Начните анализ для проверки наблюдаемых действий.|Start analysis to check visible actions.
Готово к анализу|Ready to analyze
Есть несохранённые изменения. Сохраните их перед открытием анализа.|You have unsaved changes. Save them before opening an analysis.
Обработчик проверяет видео и подготавливает доказательства.|The worker is analyzing the video and preparing evidence.
Ответ завершённого анализа не содержит корректного итога. Результат нельзя считать отсутствием нарушений.|The completed analysis has no valid verdict. This result cannot establish that no violation occurred.
Повторить загрузку подробностей|Retry loading details
Ошибка|Error
Не удалось завершить анализ|Analysis could not be completed
Итог не сформирован. Проверьте ошибку и загрузите видео снова, когда сервис будет готов.|No verdict was produced. Check the error and upload the video again when the service is ready.
Обработчик сообщил об ошибке анализа.|The worker reported an analysis error.
Сервис вернул неизвестный статус. Возобновите обновления.|Service returned an unknown status. Resume updates.
Обновления приостановлены|Updates paused
Анализ на сервере может продолжаться. Возобновите обновления, чтобы получить результат.|Analysis may still be running on the server. Resume updates to get the result.
Исходное видео не загрузилось. Сохранённые кадры доказательств остаются доступны.|Source video did not load. Saved evidence frames remain available.
Длительность видео|Video duration
Просмотрено фрагментов|Clips reviewed
Наибольший промежуток между кадрами|Maximum gap between frames
Кадров в исходном видео|Source video frames
Проверено контекстных фрагментов|Context clips checked
Интервалов с ограниченным обзором|Limited-visibility intervals
Проверок без установленного результата|Checks with unresolved results
Незавершённых проверок кандидатов|Unfinished candidate checks
Проверка кандидатов|Candidate verification
Не завершено|Incomplete
Кадров анализа людей и движения|People and motion analysis frames
Промежуток анализа движения|Motion analysis gap
Предварительных эпизодов операций|Proposed transaction episodes
Неопределённых зон сцены|Unknown scene zones
Анализ положения рук|Hand pose analysis
Анализ предметов|Object analysis
Доступен|Available
Недоступен|Unavailable
Предложено событий|Proposed events
Просмотрено типов сценариев|Scenario types screened
Проверено видимых сценариев|Visible scenarios verified
Пороги решения|Decision thresholds
Калибровка по проверенной разметке доступна|Calibration from reviewed labels available
Ожидается калибровка по разметке|Awaiting label-based calibration
Сценарий виден. Проверьте применимость правила.|The scenario is visible. Check whether the rule applies.
По доступным кадрам действие не установлено.|The available frames do not establish the action.
Действие скрыто или вне проанализированной области.|The action is obscured or outside the analyzed area.
Контекст операции не проверен.|Transaction context was not verified.
Контекст операции неясен.|Transaction context is unclear.
Применимость правила не установлена.|Rule applicability is unresolved.
Проанализированный обзор неполон.|Analyzed visibility is incomplete.
Нужен дополнительный контекст операции.|More transaction context is needed.
Разрыв исходного видео ограничивает проверку.|A gap in the source video limits verification.
Применимость правила должен проверить человек.|A reviewer must check whether the rule applies.
Наблюдение требует проверки применимости правила человеком.|A reviewer must assess rule applicability for this observation.
Доказательств недостаточно для подтверждения нарушения. Проверьте исходное видео.|Evidence is insufficient to confirm a violation. Review the source video.
Действие для проверки|Action for review
Исходное наблюдение модели:|Original model observation:
Рамки скрыты|Boxes hidden
Этот момент не анализировался: рамок нет|This moment was not analyzed: no boxes
Исходный кадр не является изображением JPEG.|Source frame is not a JPEG image.
Видимый предмет|Visible object
видимый предмет|visible object
Загружаем исходный кадр…|Loading source frame…
Не удалось показать кадр. Откройте соответствующий момент видео.|Could not display frame. Open the corresponding moment in the video.
Исходный кадр недоступен. Откройте соответствующий момент видео.|Source frame unavailable. Open the corresponding moment in the video.
Предпросмотр кадра недоступен. Откройте соответствующий момент видео.|Frame preview unavailable. Open the corresponding moment in the video.
В проанализированных кадрах объекты не локализованы. Это не доказывает их отсутствие.|No objects were localized in the analyzed frames. This does not establish their absence.
Исходные кадры подтверждают событие.|Source frames support the event.
Все условия правила|All rule conditions
Только видимое действие|Visible action only
На рабочем столе видны продукты питания.|Food is visible on the work table.
В кадре видны продукты питания.|Food is visible in the frame.
В кассовой зоне виден предмет, не относящийся к наличным.|A non-cash object is visible in the cash area.
В кадре виден предмет, не относящийся к наличным.|A non-cash object is visible in the frame.
Человек покидает рабочее место.|A person leaves the workstation.
После ухода человека экран виден включённым.|The screen is visibly on after the person leaves.
Экран компьютера виден включённым.|The computer screen is visibly on.
На рабочем месте виден компьютер.|A computer is visible at the workstation.
После ухода человека рабочее место видно без оператора.|The workstation is visibly unattended after the person leaves.
Рабочее место видно без оператора.|The workstation is visibly unattended.
Вне контейнера видны предметы, похожие на банкноты.|Objects resembling banknotes are visible outside the container.
Наличные видны вне места хранения.|Cash is visible outside storage.
Видна передача предметов, похожих на банкноты.|A handover of objects resembling banknotes is visible.
Видна работа с предметами, похожими на банкноты.|Handling of objects resembling banknotes is visible.
Видна обработка предметов, похожих на банкноты.|Processing of objects resembling banknotes is visible.
В кадре видны неупакованные наличные.|Loose cash is visible in the frame.
В обозреваемой клиентской зоне клиент не виден.|No client is visible in the observable client area.
В кадре виден предмет, похожий на ключ или штамп.|An object resembling a key or stamp is visible in the frame.
У места хранения видны предметы, похожие на банкноты.|Objects resembling banknotes are visible near storage.
Видно расположение предметов, похожих на банкноты.|The position of objects resembling banknotes is visible.
В кадре виден контейнер или место хранения.|A container or storage location is visible in the frame.
Человек принимает пищу.|A person is eating.
Наблюдения в этом диапазоне различаются; отдельные кадры — в подробностях.|Observations differ within this range; individual frames are in the details.
Для описания действия недостаточно сохранённых данных; проверьте кадры.|Saved data is insufficient to describe the action; review the frames.
По доступным кадрам действие не подтверждено.|The available frames do not confirm the action.
Обоснование —|Justification —
Видимое действие|Visible action
Исключение проверено|Exception checked
Контекст правила|Rule context
Да|Yes
Нет|No
Дополнительные сведения о проверке не сохранены.|No additional check details were saved.
Недостаточно данных|Insufficient evidence
Нет оценки|Not assessed
Диапазон отдельных некалиброванных оценок; не общая вероятность за интервал.|Range of individual uncalibrated estimates; not an overall probability for the interval.
Некалиброванная оценка модели для указанной проверки, не измеренная вероятность.|Uncalibrated model estimate for this check, not a measured probability.
По отдельным наблюдениям|Individual observations
Диапазон наблюдений. Переход к первому кадру; отдельные моменты — в подробностях.|Observation range. Jump to the first frame; individual moments are in the details.
Подробнее|View details
Наблюдения относятся к отдельным кадрам внутри диапазона. Непрерывное присутствие между кадрами не установлено.|Observations refer to individual frames within the range. Continuous presence between frames is not established.
Не подтвердилось|Not supported
Есть несохранённые изменения|You have unsaved changes
Сохранение решений недоступно для этой версии сервера. Обновите сервер и повторно откройте анализ.|Saving decisions is unavailable in this server version. Update the server and reopen the analysis.
Решения проверяющего ещё не сохранены|Reviewer decisions have not been saved yet
Человек: подтверждено|Reviewer: confirmed
Человек: отклонено|Reviewer: dismissed
Человек: недостаточно данных|Reviewer: insufficient evidence
Человек: не проверено|Reviewer: unreviewed
Человек: разные решения|Reviewer: mixed decisions
Подтвердить|Confirm
Отклонить|Dismiss
Комментарий проверяющего|Reviewer comment
Кратко опишите наблюдаемое действие и основание решения|Briefly describe the observed action and reason for your decision
Сохранить решение|Save decision
Не сохранено|Unsaved
Укажите обе границы интервала: конец должен быть не раньше начала.|Enter both interval boundaries: the end must not precede the start.
Укажите имя проверяющего перед сохранением.|Enter the reviewer name before saving.
Завершите форму пропущенного события: тип, интервал и описание. Либо очистите форму.|Complete the missed event form: type, interval and description. Or clear the form.
Выберите решение для изменённых строк перед сохранением.|Choose a decision for edited rows before saving.
Проверьте границы интервалов в изменённых строках.|Check interval boundaries in edited rows.
Сохраняем…|Saving…
Сервер вернул некорректные данные сохранения.|Server returned invalid save data.
Изменения не сохранены|Changes were not saved
Сохранённая версия изменилась в другой вкладке. Ваши изменения остаются на экране. Загрузите сохранённую версию перед повторной проверкой.|The saved version changed in another tab. Your edits remain on screen. Load the saved version before reviewing again.
Время от начала видео|Time from start of video
Нажмите воспроизведение в плеере|Press play in the video player
Сохраните изменения перед скачиванием отчёта.|Save changes before downloading the report.
Сервер вернул файл неизвестного формата.|Server returned a file in an unknown format.
Проверка изменилась во время подготовки отчёта. Сохраните правки и скачайте отчёт снова.|Review changed while the report was being prepared. Save changes and download the report again.
Загружаем…|Loading…
Видео без названия|Untitled video
Статус неизвестен|Unknown status
Сохранённых анализов нет|No saved analyses
Загружаем проверенную разметку…|Loading reviewed labels…
Проверенной разметки пока недостаточно для оценки качества.|There are not enough reviewed labels to estimate quality yet.
Предварительные метрики по сохранённым решениям человека|Preliminary metrics from saved reviewer decisions
Анализ не завершён полностью|Analysis partly completed
Анализ завершён|Analysis completed
Часть кандидатов не проверена за отведённое время.|Some candidates were not verified within the time limit.
Результаты готовы.|Results are ready.
Модель подтвердила нарушение|Model confirmed a violation
Модель не подтвердила нарушений|Model did not confirm any violations
Нарушение не подтверждено. Часть кандидатов не проверена за отведённое время.|No violation confirmed. Some candidates were not verified within the time limit.
Нарушение не подтверждено. Подробности анализа требуют проверки.|No violation confirmed. Analysis details require review.
В проанализированных кадрах подтверждённое нарушение не найдено.|No confirmed violation was found in the analyzed frames.
Подробности анализа не загрузились. Список проверок, найденные предметы, охват и наблюдения могут быть неполными. Повторите загрузку подробностей.|Analysis details did not load. Checks, detected objects, coverage and observations may be incomplete. Retry loading details.
Нарушение правила рабочего места|Workplace rule violation
Правило рабочего места от пользователя|User-defined workplace rule
Нормативное основание|Regulatory basis
Примечание:|Note:
Не удалось сохранить изменения. Проверьте введённые данные.|Could not save changes. Check the entered data.
Изменения отменены|Changes discarded
Сохраните изменения проверки перед удалением видео.|Save review changes before removing the video.
Сервис не вернул ID анализа. Проверьте журнал сервиса перед повтором.|Service returned no analysis ID. Check service logs before retrying.
Введите ключ API перед открытием анализа.|Enter the API key before opening an analysis.
Приостановлены обновления страницы. Анализ на сервере продолжается; возобновите обновления для получения результата.|Page updates are paused. Analysis continues on the server; resume updates to get the result.
Проверяем статус анализа|Checking analysis status
Предпросмотр видео недоступен. Сохранённые кадры остаются доступны.|Video preview unavailable. Saved frames remain available.
Не у всех видео указаны исходная группа, камера и сессия; независимость не установлена.|Some videos have no source group, camera or session; independence is not established.
Меньше трёх независимых групп: отдельные обучение, валидация и тест пока невозможны.|Fewer than three independent groups: separate training, validation and test sets are not yet possible.
Это исследовательская оценка по решениям проверяющих, а не независимая тестовая точность.|This is an exploratory evaluation based on reviewer decisions, not independent test accuracy.
Непроверенные случаи и «Недостаточно данных» не считаются отсутствием нарушения.|Unreviewed cases and “Insufficient evidence” are not treated as absence of a violation.
Точность и полнота по кандидатам относятся только к проверенным строкам; найденное человеком действие в «Требует проверки» не является автоматическим подтверждением модели.|Candidate precision and recall cover reviewed rows only; a human-confirmed action in “Needs review” is not an automatic model confirmation.
Полнота событий вычисляется только для полностью просмотренных видео с окончательными решениями, проверенными интервалами и указанными группой источника, камерой и сессией. Неоднозначные пересекающиеся интервалы исключены.|Event recall is calculated only for fully reviewed videos with final decisions, verified intervals and a source group, camera and session. Ambiguous overlapping intervals are excluded.
Нулевой знаменатель показан как отсутствие данных; временная точка не подменяет интервал для IoU.|A zero denominator is shown as unavailable data; a timestamp does not replace an interval for IoU.
Brier и диаграмма надёжности используют только оценки всего нарушения и решения человека. Оценки видимого действия и завершения процедуры исключены.|Brier score and reliability plots use only whole-violation estimates and human decisions. Visible-action and procedure-completion estimates are excluded.
Все версии одного исходника и общей камеры остаются в одной группе. Разделение и обучение автоматически не выполняются.|All variants of one source and shared camera remain in one group. Splitting and training are not performed automatically.
`.trim().split("\n").map(line => { const at = line.indexOf("|"); return [line.slice(0, at), line.slice(at + 1)]; }));

  const patterns = [];
  function translate(value) {
    const text = String(value), trimmed = text.trim();
    let translated = Object.hasOwn(exact, trimmed) ? exact[trimmed] : undefined;
    if (translated === undefined) {
      const entry = patterns.find(([pattern]) => pattern.test(trimmed));
      translated = entry ? trimmed.replace(entry[0], entry[1]) : trimmed;
    }
    return text.slice(0, text.indexOf(trimmed)) + translated + text.slice(text.indexOf(trimmed) + trimmed.length);
  }
  const t = translate;
  const count = (value, singular, plural) => `${value} ${Number(value) === 1 ? singular : plural}`;
  patterns.push(
    [/^В отчёт войдут сохранённые решения · версия (\d+)\.$/u, "The report will include saved decisions · version $1."],
    [/^Сохранено · версия (\d+)$/u, "Saved · version $1"],
    [/^Запрос завершился ошибкой \(HTTP (\d+)\)\.$/u, "Request failed (HTTP $1)."],
    [/^Применимость: (.+)$/u, (_, value) => `Applicability: ${t(value)}`],
    [/^Способ проверки: (.+)$/u, (_, value) => `Review method: ${t(value)}`],
    [/^(Наличие события|Обязательная процедура) · (.+)$/u, (_, first, second) => `${t(first)} · ${t(second)}`],
    [/^(\d+(?:\.\d+)?%) · (выполнение процедуры|наличие события)$/u, (_, value, scope) => `${value} · ${scope === "выполнение процедуры" ? "procedure completion" : "event occurrence"}`],
    [/^(\d+) ссылок на кадры$/u, (_, value) => count(value, "frame reference", "frame references")],
    [/^(Оценка выполнения процедуры|Остаточная оценка процедуры|Оценка события): (\d+(?:\.\d+)?%) · без калибровки(; оценка не определяет итог)?$/u, (_, label, value, suffix) => `${({"Оценка выполнения процедуры":"Procedure completion estimate", "Остаточная оценка процедуры":"Residual procedure estimate", "Оценка события":"Event estimate"})[label]}: ${value} · uncalibrated${suffix ? "; estimate does not determine the verdict" : ""}`],
    [/^Область оценки: (.+)$/u, (_, value) => `Estimate scope: ${t(value)}`],
    [/^Неполный охват: (\d+) известных эпизодов без оценки этой проверки; применимость может быть неизвестна$/u, "Incomplete coverage: $1 known episodes without this check's estimate; applicability may be unknown"],
    [/^Неполный охват: (\d+) окон с неопределённым началом процедуры; данные о выполнении сохранены$/u, "Incomplete coverage: $1 windows with unclear procedure start; completion evidence retained"],
    [/^Последний оценённый эпизод( · [\d:–]+)?$/u, (_, interval = "") => `Latest assessed episode${interval}`],
    [/^Эпизод( · [\d:–]+)$/u, "Episode$1"],
    [/^(\d+) эпизодов · (\d+) требуют проверки или неизвестны$/u, "$1 episodes · $2 need review or are unknown"],
    [/^Эпизод (\d+)( · [\d:–]+)?: (Не применимо|Не проверено|\d+(?:\.\d+)?%) · (Не оценено|Данные о событии|Данные о выполнении процедуры|Процедура выполнена|Не установлено)(?: · (выполнение процедуры|event) оценка (\d+(?:\.\d+)?%))?( · применимость неизвестна)?$/u, (_, index, interval = "", estimate, assessment, scope, confidence, unknown) => `Episode ${index}${interval}: ${t(estimate)} · ${t(assessment)}${confidence ? ` · ${scope === "event" ? "event" : "procedure completion"} estimate ${confidence}` : ""}${unknown ? " · applicability unknown" : ""}`],
    [/^(\d+) из (\d+) проверок(?: · Версия каталога (.+))?$/u, (_, shown, total, version) => `${shown} of ${total} checks${version ? ` · Catalog version ${version}` : ""}`],
    [/^Версия каталога (.+)$/u, "Catalog version $1"],
    [/^(\d+) проверок в каталоге · (\d+) проверок событий с автоматическим подтверждением$/u, "$1 catalog checks · $2 event checks with automatic confirmation"],
    [/^([\d:]+) прошло$/u, "$1 elapsed"],
    [/^(Доступен|Недоступен) · (\d+) frames$/u, (_, label, value) => `${t(label)} · ${count(value, "frame", "frames")}`],
    [/^(СЦЕНАРИЙ ВИДЕН · ПРОВЕРЬТЕ ПРАВИЛО|НЕ ПОДТВЕРЖДЕНО) (\d+)$/u, (_, label, index) => `${label.startsWith("СЦЕНАРИЙ") ? "SCENARIO VISIBLE · CHECK RULE" : "UNCONFIRMED"} ${index}`],
    [/^(\d+) (?:объект|объектов)$/u, (_, value) => count(value, "item", "items")],
    [/^(\d+) найденных объектов в ([\d:.]+)$/u, "$1 detected objects at $2"],
    [/^(.+) · оценка модели$/u, (_, kind) => `${t(kind)} · model estimate`],
    [/^Ещё найденные объекты \((\d+) из (\d+)\)$/u, "More detected objects ($1 of $2)"],
    [/^(\d+) найденных (?:объект|объектов)$/u, (_, value) => `${count(value, "object", "objects")} detected`],
    [/^(Видимое действие|Исключение проверено|Контекст правила) · (.+)$/u, (_, stage, label) => `${t(stage)} · ${t(label)}`],
    [/^(\d+) наблюдений$/u, (_, value) => count(value, "observation", "observations")],
    [/^(\d+) из (\d+) оценено$/u, "$1 of $2 assessed"],
    [/^Человек: ([\d:–]+)$/u, "Reviewer: $1"],
    [/^(\d+(?:\.\d+)?%) · (Все условия правила|Только видимое действие|Процедура выполнена|Наличие события)$/u, (_, value, scope) => `${value} · ${t(scope)}`],
    [/^(\d+) провер(?:ка|ок)$/u, (_, value) => count(value, "check", "checks")],
    [/^(Требует проверки|Подтверждено моделью|Другие результаты) · (\d+)$/u, (_, label, value) => `${t(label)} · ${value}`],
    [/^Оценка модели сохраняется отдельно\. Проверьте границы события по видео; указанные границы сохраняются как разметка человека\.$/u, "The model estimate is stored separately. Check the event boundaries against the video; these boundaries are saved as human annotations."],
    [/^Решение применяется ко всем (\d+) наблюдениям строки\. Проверьте границы события по видео; указанные границы сохраняются как разметка человека\.$/u, "The decision applies to all $1 observations in this row. Check the event boundaries against the video; these boundaries are saved as human annotations."],
    [/^([\d:–]+) · (диапазон наблюдений|выбранное событие)$/u, (_, interval, label) => `${interval} · ${label === "диапазон наблюдений" ? "observation range" : "selected event"}`],
    [/^(.+) · ([\d:–]+) · (Требует проверки|Подтверждено моделью|Другие результаты)( · диапазон наблюдений)?$/u, (_, label, interval, status, range) => `${t(label)} · ${interval} · ${t(status)}${range ? " · observation range" : ""}`],
    [/^(Готово|Обработка|Ошибка|Статус неизвестен) · (проверка сохранена|без решения человека)$/u, (_, status, review) => `${t(status)} · ${review === "проверка сохранена" ? "review saved" : "no human decision"}`],
    [/^Найдено: (\d+)$/u, "Found: $1"],
    [/^Анализов с разметкой: (\d+)\. Полностью проверено: (\d+)\.$/u, "Annotated analyses: $1. Fully reviewed: $2."],
    [/^Исходных групп: (\d+); без группы: (\d+)\. Варианты одного исходного видео должны оставаться в одной выборке\.$/u, "Source groups: $1; ungrouped: $2. Variants of one source video must remain in the same dataset split."],
    [/^Точность срабатываний автоматики, проверенная часть: (.+)$/u, (_, rate) => `Automatic detection precision, reviewed subset: ${t(rate)}`],
    [/^Ложные тревоги автоматики: (\d+|—); подтверждённые человеком пропуски: (\d+|—)$/u, "Automatic false alarms: $1; human-confirmed misses: $2"],
    [/^Полнота событий: (.+) · (нет полного знаменателя|полностью проверенная часть)$/u, (_, rate, state) => `Event recall: ${t(rate)} · ${state === "нет полного знаменателя" ? "complete denominator unavailable" : "fully reviewed subset"}`],
    [/^Совпадение интервалов \(IoU\): (.+) · пар: (\d+)$/u, (_, rate, pairs) => `Interval overlap (IoU): ${t(rate)} · pairs: ${pairs}`],
    [/^Калибровка оценки: (\d+) примеров; Brier: ([\d.]+|—)$/u, "Estimate calibration: $1 samples; Brier: $2"],
    [/^(\d+) событи(?:е требует|й требуют) проверки человеком\.$/u, (_, value) => `${count(value, "event requires", "events require")} human review.`],
    [/^Нарушение не подтверждено\. (\d+) случа(?:й требует|ев требуют) проверки\.$/u, (_, value) => `No violation confirmed. ${count(value, "case requires", "cases require")} review.`],
    [/^Исходный итог API: (violation|no_violation) · подтверждено моделью: (\d+)\. Решения человека и рамки объектов сохраняются отдельно\.$/u, "Original API verdict: $1 · model-confirmed: $2. Human decisions and object boxes are stored separately."],
    [/^(\d+) событи(?:е|й)$/u, (_, value) => count(value, "event", "events")],
    [/^СОБЫТИЕ (\d+)$/u, "EVENT $1"],
    [/^Перейти в видео: ([\d:]+)$/u, "Jump to video: $1"],
    [/^Кадр (\d+) события (\d+) в ([\d:]+)$/u, "Frame $1 of event $2 at $3"],
    [/^([\d:]+) · Кадр (\d+)$/u, "$1 · Frame $2"],
    [/^(Раздел|Пункт) (.+)$/u, (_, label, value) => `${label === "Раздел" ? "Section" : "Clause"} ${value}`],
    [/^Перейти к доказательству: (.+) в ([\d:.]+)$/u, (_, label, time) => `Jump to evidence: ${t(label)} at ${time}`],
    [/^Перейти к доказательству: ([\d:.]+)$/u, "Jump to evidence: $1"],
    [/^Перейти к неподтверждённому наблюдению: ([\d:]+)$/u, "Jump to unconfirmed observation: $1"],
    [/^Перейти к наблюдению: (.+) в ([\d:.]+)$/u, (_, label, time) => `Jump to observation: ${t(label)} at ${time}`],
    [/^Перейти к (.+) в ([\d:.]+)(?:; наблюдения до ([\d:.]+))?$/u, (_, label, time, end) => `Jump to ${t(label)} at ${time}${end ? `; observations through ${end}` : ""}`],
    [/^(.+) в исходном кадре: ([\d:.]+)$/u, (_, label, time) => `${t(label)} in source frame: ${time}`],
    [/^(Каталог недоступен|Не удалось сохранить|Не удалось скачать отчёт|История недоступна|Метрики недоступны|Не удалось загрузить сохранённую версию)\. (.+)$/u, (_, label, error) => `${({"Каталог недоступен":"Catalog unavailable", "Не удалось сохранить":"Could not save", "Не удалось скачать отчёт":"Could not download report", "История недоступна":"History unavailable", "Метрики недоступны":"Metrics unavailable", "Не удалось загрузить сохранённую версию":"Could not load saved version"})[label]}. ${t(error)}`],
    [/^(.+) Повторите через (\d+) сек\.$/u, (_, error, seconds) => `${t(error)} Retry in ${seconds} sec.`]
  );

  // The only concatenated factual descriptions are this reviewed sentence set.
  // Exact pairs avoid accidental translation of arbitrary observations or comments.
  const facts = Object.keys(exact).filter(value => /^(?:На рабочем столе видны|В кадре видны|В кассовой зоне виден|В кадре виден|Человек покидает|После ухода человека|Экран компьютера виден|На рабочем месте виден|Рабочее место видно|Вне контейнера видны|Наличные видны|Видна передача|Видна работа|Видна обработка|В обозреваемой клиентской зоне|У места хранения видны|Видно расположение|Человек принимает пищу)/u.test(value) && value.endsWith("."));
  for (const first of facts) for (const second of facts) exact[first + " " + second] = exact[first] + " " + exact[second];
  window.ForteTranslations = {exact, patterns};
})();
