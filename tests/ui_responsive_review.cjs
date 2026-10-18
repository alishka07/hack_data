/* Responsive review workflow against synthetic endpoints. No private media or live service. */
"use strict";
const assert = require("node:assert/strict"), fs = require("node:fs"), path = require("node:path");
const {chromium} = require(process.env.FORTE_PLAYWRIGHT_MODULE || "playwright");
const root = path.resolve(__dirname,".."), origin = "http://127.0.0.1:48124";
const job = "12345678-1234-4234-8234-123456789012";
const filename = "Касса_" + "длинноеимяисходноговидеофайла".repeat(5) + ".mov";
const longType = "неизвестнаяпроверкабезпробелов".repeat(6);
const description = "На рабочей поверхности рядом с кассиром виден предмет. По доступным кадрам нельзя определить его принадлежность; требуется решение проверяющего.";
const evidence = (timestamp_sec) => ({source_index:Math.round(timestamp_sec*30),timestamp_sec});
const cases = [
 {case_id:"presence-0",type_id:"personal_belongings_in_cash_unit",status:"review_required",requires_review:true,timecode_sec:0,probability:.6,probability_scope:"action_only",checks:[{stage:"action",name:"non_cash_object_visually_supported",status:"yes",evidence_refs:[evidence(0)]},{stage:"context",name:"personal_belonging_at_cash_unit",status:"unknown",evidence_refs:[evidence(1.5)]}],reasons:[description]},
 {case_id:"presence-8",type_id:"personal_belongings_in_cash_unit",status:"review_required",requires_review:true,timecode_sec:8,probability:.8,probability_scope:"action_only",checks:[{stage:"action",name:"non_cash_object_visually_supported",status:"yes",evidence_refs:[evidence(8)]}],reasons:[description]},
 {case_id:"screen-20",type_id:"workstation_departure_screen_left_on",status:"confirmed",timecode_sec:20,probability:.91,probability_scope:"joint_checks",checks:[{stage:"action",name:"same_screen_visibly_on_after_departure",status:"yes",evidence_refs:[evidence(20),evidence(23)]}]},
 {case_id:"cash-30",type_id:"previous_transaction_cash_exposed",status:"rejected",timecode_sec:30,action_decision:"rejected",checks:[{stage:"action",name:"cash_exposed",status:"no"}]},
 {case_id:"intake-40",type_id:"active_intake_departure",status:"rejected",timecode_sec:40,action_decision:"uncertain",checks:[]},
 {case_id:"intake-40-close-a",type_id:"active_intake_departure",status:"rejected",timecode_sec:40.05,action_decision:"uncertain",checks:[]},
 {case_id:"intake-40-close-b",type_id:"active_intake_departure",status:"rejected",timecode_sec:40.1,action_decision:"uncertain",checks:[]},
 {case_id:"food-end-60",type_id:"personal_food_at_workstation",status:"rejected",timecode_sec:60,action_decision:"uncertain",checks:[]}
];
const image = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+iRWQAAAAASUVORK5CYII=";
const widths = [320,375,390,650,768,1024,1280,1366];
const sizes = [...widths.map(width=>({width,height:900})),{width:844,height:390}];
let state = {revision:0,reviewer:"",camera_id:"",session_id:"",source_group:"",coverage_complete:false,decisions:[],missed_events:[]};
const writes = [], errors = [];
let analyzeRequests = 0, conflictNext = false, reportRequests = 0, downloadCount = 0;
let holdReport = false, releaseReport = null, signalReportStart = null;
const metadata = () => ({job_id:job,status:"done",original_name:filename,review_cases:cases,review_state:structuredClone(state),details:{coverage:{duration_sec:60},catalog_version:"responsive-test",assessments:[{item_id:longType,scoring_kind:"event_occurrence",score:61,model_confidence:.61,applicability:"applicable",assessment:"observed",automation:"review_only",evidence_refs:[evidence(0)]}],case_decision:{cases},detections:[],findings:[{type_id:"workstation_departure_screen_left_on",timecode_sec:20,basis:"workplace_policy",rule:{name:"Правило рабочего места: завершение работы за компьютером",requirement:description}}]}});

async function assertLayout(page,label) {
 const geometry = await page.evaluate(()=>{
  const within = element => { const box=element.getBoundingClientRect(); return box.width>0&&box.height>0; };
  const cards=[...document.querySelectorAll(".topbar,.upload-panel,.status-card,.video-card,#verdict-card,#reviewer-section,#case-decision-section,#missed-event-section,.finding-card,#technical-details,#report-section")].filter(within);
  const overflowing=cards.map(element=>({name:element.id||element.className,left:element.getBoundingClientRect().left,right:element.getBoundingClientRect().right,width:element.getBoundingClientRect().width})).filter(box=>box.left<-.75||box.right>innerWidth+.75);
  const offenders=[...document.querySelectorAll("body *")].filter(within).map(element=>({name:element.id||element.className||element.tagName,right:element.getBoundingClientRect().right})).filter(box=>box.right>innerWidth+1).slice(0,12);
  return {width:innerWidth,documentWidth:document.documentElement.scrollWidth,bodyWidth:document.body.scrollWidth,overflowing,offenders,htmlOverflow:getComputedStyle(document.documentElement).overflowX,bodyOverflow:getComputedStyle(document.body).overflowX};
 });
 assert(geometry.documentWidth<=geometry.width+1,`${label}: document overflow ${JSON.stringify(geometry)}`);
 assert(geometry.bodyWidth<=geometry.width+1,`${label}: body overflow ${JSON.stringify(geometry)}`);
 assert.deepEqual(geometry.overflowing,[],`${label}: major cards must fit the viewport`);
 assert.notEqual(geometry.htmlOverflow,"hidden",`${label}: do not hide overflow on the document`);
 assert.notEqual(geometry.bodyOverflow,"hidden",`${label}: do not hide overflow on the body`);
}

async function assertFormControls(page,label) {
 const problems=await page.evaluate(()=>{
  const selectors=".review-editor,.review-metadata,#missed-event-form,.reviewer-identity,.report-actions";
  const issues=[];
  for(const form of document.querySelectorAll(selectors)) {
   const parent=form.getBoundingClientRect();if(!parent.width||!parent.height)continue;
   const controls=[...form.querySelectorAll("input:not([type=radio]),textarea,select,button")].filter(element=>element.getBoundingClientRect().width>0);
   for(const element of controls) {
    const box=element.getBoundingClientRect();
    if(box.left<parent.left-1||box.right>parent.right+1)issues.push({name:element.id||element.name||element.textContent,parent:{left:parent.left,right:parent.right},control:{left:box.left,right:box.right}});
   }
   for(let a=0;a<controls.length;a++)for(let b=a+1;b<controls.length;b++){
    const first=controls[a].getBoundingClientRect(),second=controls[b].getBoundingClientRect();
    if(Math.min(first.right,second.right)-Math.max(first.left,second.left)>1&&Math.min(first.bottom,second.bottom)-Math.max(first.top,second.top)>1)issues.push({overlap:[controls[a].id||controls[a].name,controls[b].id||controls[b].name]});
   }
  }
  return issues;
 });
 assert.deepEqual(problems,[],`${label}: editor controls fit their forms without overlapping`);
}

(async()=>{
 fs.mkdirSync(path.join(root,"tmp"),{recursive:true});
 const browser=await chromium.launch({headless:true,...(process.env.FORTE_BROWSER_EXECUTABLE?{executablePath:process.env.FORTE_BROWSER_EXECUTABLE}:{})});
 try {
  const context=await browser.newContext({viewport:{width:375,height:900},colorScheme:"light"});
  const page=await context.newPage();page.on("pageerror",error=>errors.push(error.message));page.on("download",()=>{downloadCount++;});
  await page.addInitScript(()=>{
   HTMLMediaElement.prototype.load=function(){};
   HTMLMediaElement.prototype.play=function(){this.dataset.playing="true";return Promise.resolve();};
   HTMLMediaElement.prototype.pause=function(){this.dataset.playing="false";};
   Object.defineProperty(HTMLMediaElement.prototype,"readyState",{get:()=>1,configurable:true});
   Object.defineProperty(HTMLMediaElement.prototype,"duration",{get:()=>60,configurable:true});
   Object.defineProperty(HTMLMediaElement.prototype,"currentTime",{get(){return Number(this.dataset.time||0);},set(value){this.dataset.time=String(value);this.dispatchEvent(new Event("timeupdate"));},configurable:true});
  });
  await page.route("**/*",async route=>{
   const request=route.request(),url=new URL(request.url());assert.equal(url.origin,origin,"Synthetic requests stay on the test origin");
   if(["/","/app.js","/style.css","/static/i18n.js","/static/i18n-catalog.js"].includes(url.pathname))return route.fulfill({contentType:url.pathname.endsWith("js")?"text/javascript":url.pathname.endsWith("css")?"text/css":"text/html",body:fs.readFileSync(path.join(root,"web",url.pathname==="/"?"index.html":path.basename(url.pathname)))});
   if(url.pathname==="/api/v1/health")return route.fulfill({json:{ready:true,busy:false}});
   assert.equal(request.headers()["x-api-key"],"responsive-test-key","Protected requests retain their authenticated header");
   if(url.pathname==="/api/v1/analyze"){analyzeRequests++;return route.fulfill({status:500,json:{detail:"Unexpected implicit upload"}});}
   if(url.pathname==="/api/ui/active-job")return route.fulfill({json:{job_id:null}});
   if(url.pathname==="/api/ui/catalog")return route.fulfill({json:{version:"responsive-test",items:[...new Set([...cases.map(item=>item.type_id),longType])].map(id=>({id,scoring_kind:"event_occurrence",automation:"review_only"}))}});
   if(url.pathname==="/api/ui/history")return route.fulfill({json:{items:[{job_id:job,original_name:filename,status:"done",review_revision:state.revision}],total:1,limit:20,offset:0}});
   if(url.pathname==="/api/ui/quality")return route.fulfill({json:{status:"no_reviewed_labels",records:{total:1,with_review:0,eligible_complete_coverage:0},grouping:{status:"missing_metadata",source_families:0,unknown_group_records:1},per_type:[],limitations:["Без полной разметки полнота неизвестна."]}});
   if(url.pathname.endsWith("/report.html")){reportRequests++;if(holdReport){holdReport=false;signalReportStart();await new Promise(resolve=>{releaseReport=resolve;});}return route.fulfill({contentType:"text/html; charset=utf-8",body:"<!doctype html><html lang=ru><body>Единый отчёт по проверке</body></html>"});}
   if(url.pathname.endsWith("/review")){
    assert.equal(request.method(),"PUT");const payload=request.postDataJSON();assert.equal(payload.revision,state.revision);writes.push(payload);
    if(conflictNext){conflictNext=false;return route.fulfill({status:409,json:{detail:"Review revision conflict"}});}
    state={...payload,revision:state.revision+1};return route.fulfill({json:state});
   }
   if(url.pathname.endsWith("/video"))return route.fulfill({contentType:"video/quicktime",body:"synthetic video"});
   if(url.pathname.startsWith("/api/v1/jobs/"))return route.fulfill({json:{job_id:job,status:"done",verdict:"violation",violations:[{type:"workstation_departure_screen_left_on",timecode_sec:20,description,evidence_frames:[{timestamp_sec:20,image_base64:image},{timestamp_sec:23,image_base64:image}]}]}});
   if(url.pathname.startsWith("/api/ui/jobs/"))return route.fulfill({json:metadata()});
   return route.fulfill({status:404,body:"Synthetic test endpoint only"});
  });
  await page.goto(origin);await page.fill("#api-key","responsive-test-key");
  for(const viewport of sizes) {await page.setViewportSize(viewport);await assertLayout(page,`Upload ${viewport.width}×${viewport.height}`);}
  await page.setViewportSize({width:375,height:900});
  const chooserEvent=page.waitForEvent("filechooser");await page.locator("#video-empty").click();const chooser=await chooserEvent;
  assert.equal(chooser.isMultiple(),false);await chooser.setFiles({name:filename,mimeType:"video/quicktime",buffer:Buffer.from("synthetic local video")});
  assert.equal(await page.locator("#file-name").textContent(),filename);assert.equal(await page.locator("#source-video").isVisible(),true);assert.equal(analyzeRequests,0,"Local preview selection does not start analysis");
  await assertLayout(page,"Long filename local preview 375px");await page.locator("#remove-file").click();
  await page.locator(".existing-job summary").click();await page.fill("#job-input",job);await page.locator("#open-job-form button").click();
  await page.waitForFunction(()=>!document.getElementById("results").hidden&&document.getElementById("source-video").src.startsWith("blob:"));
  assert.equal(await page.locator(".case-card").count(),7);assert.equal(await page.locator(".case-time .time-button").first().textContent(),"▶ 00:00–00:08");
  assert.equal(await page.locator("#video-name").textContent(),filename);
  assert.equal(await page.locator("#download-report").evaluate(button=>Boolean(button.closest("#results"))&&!button.closest("#reviewer-section")&&document.querySelector("#results").lastElementChild===button.closest("#report-section")&&Boolean(document.querySelector("#technical-details").compareDocumentPosition(button)&Node.DOCUMENT_POSITION_FOLLOWING)),true,"The single-format report action is the final result section instead of occupying the reviewer header");
  assert.deepEqual(await page.locator("#event-timeline .event-time-axis span").allTextContents(),["00:00","00:30","01:00"],"Timeline shows the start, middle and end of the loaded video");
  assert.equal(await page.locator("#selected-case-label .event-selection-time").textContent(),"00:00–00:08 · диапазон наблюдений");
  for(const [status,label] of [["review_required","Требует проверки"],["confirmed","Подтверждено моделью"],["rejected","Другие результаты"]]){
   assert.equal(await page.locator(`#event-timeline .event-lane.${status} .event-lane-track`).count(),1);assert.match(await page.locator(`#event-timeline .event-legend-item:has(.event-legend-dot.${status})`).textContent(),new RegExp(label));
  }
  assert.equal(await page.locator("#event-timeline .event-marker").count(),7);
  for(const viewport of sizes) {
   await page.setViewportSize(viewport);await assertLayout(page,`Saved results ${viewport.width}×${viewport.height}`);
   const timelineOverflow=await page.locator("#event-timeline").evaluate(element=>{const bounds=element.getBoundingClientRect();return [...element.querySelectorAll(".event-time-axis span,.event-legend-item,.event-lane-track,.event-marker")].map(child=>({text:child.textContent||child.getAttribute("aria-label"),rect:child.getBoundingClientRect()})).filter(child=>child.rect.left<bounds.left-1||child.rect.right>bounds.right+1).map(child=>child.text);});assert.deepEqual(timelineOverflow,[],`${viewport.width}px timeline labels, endpoint markers and tracks fit inside the video card`);
   const markerOverlaps=await page.locator(".event-marker").evaluateAll(elements=>{const problems=[];for(let a=0;a<elements.length;a++)for(let b=a+1;b<elements.length;b++){const first=elements[a].getBoundingClientRect(),second=elements[b].getBoundingClientRect();if(Math.min(first.right,second.right)-Math.max(first.left,second.left)>1&&Math.min(first.bottom,second.bottom)-Math.max(first.top,second.top)>1)problems.push([elements[a].getAttribute("aria-label"),elements[b].getAttribute("aria-label")]);}return problems;});assert.deepEqual(markerOverlaps,[],`${viewport.width}px closely timed observations remain distinct clickable markers`);
   const first=page.locator(".case-card").first();
   assert.equal(await first.locator("th,td").count(),5,"The same five semantic cells survive resizing");
   assert.deepEqual(await first.locator("td").evaluateAll(elements=>elements.map(element=>element.dataset.label)),["Модель","Оценка модели","Время","Подробности"]);
   if(viewport.width<=650){
    const display=await first.evaluate(element=>getComputedStyle(element).display);assert(["grid","block","flex"].includes(display),`${viewport.width}px results use readable labelled cards`);
    const row=await first.boundingBox(),name=await first.locator(".check-name").boundingBox();assert(name.width>=row.width*.8,"Mobile check name and justification get a full-width reading area");
   }
   if(viewport.width>650){
    const clippedHeaders=await page.locator(".case-results-table thead th").evaluateAll(elements=>elements.flatMap(element=>{const bounds=element.getBoundingClientRect(),range=document.createRange();range.selectNodeContents(element);return [...range.getClientRects()].filter(rect=>rect.left<bounds.left-1||rect.right>bounds.right+1).map(rect=>({text:element.textContent,left:rect.left,right:rect.right,cell:{left:bounds.left,right:bounds.right}}));}));
    assert.deepEqual(clippedHeaders,[],`${viewport.width}px table headings wrap inside their columns`);
   }
   if(viewport.width>=1280){
    assert.equal(await first.evaluate(element=>getComputedStyle(element).display),"table-row");assert.equal(await first.locator("td").first().evaluate(element=>getComputedStyle(element).display),"table-cell");
    const bounds=await page.evaluate(()=>({video:document.querySelector(".video-column").getBoundingClientRect().right,results:document.querySelector(".decision-column").getBoundingClientRect().left}));assert(bounds.video<bounds.results,"Wide desktops retain source beside results");
   }
  }
  await page.locator('.event-marker[aria-label*="01:00"]').click();assert.equal(await page.locator("#source-video").evaluate(video=>video.currentTime),60,"The final-frame marker remains clickable at the edge of the time axis");await page.locator(".event-marker").first().click();assert.equal(await page.locator("#source-video").evaluate(video=>video.currentTime),0);
  await page.setViewportSize({width:375,height:900});
  await page.locator(".case-details summary").first().click();await page.locator("#review-quality-metadata summary").click();await page.locator("#missed-event-section summary").click();
  await page.fill("#reviewer-name","Проверяющий адаптивного интерфейса");await page.fill("#review-camera","Камера 1");await page.fill("#review-session","Утренняя смена");await page.fill("#review-source-group","original-source-A");
  let editor=page.locator(".case-detail-body").first().locator(".review-editor");
  await editor.locator('input[value="confirmed"]').check();await editor.locator("textarea").fill(description);await editor.locator('input[name="start_sec"]').fill("1");await editor.locator('input[name="end_sec"]').fill("9");
  await page.fill("#missed-start","45");await page.fill("#missed-end","48");await page.fill("#missed-comment","На столе виден ещё один предмет; требуется дополнительная проверка по видео.");
  const sourceBefore=await page.locator("#source-video").getAttribute("src");
  for(const viewport of sizes) {
   await page.setViewportSize(viewport);await assertLayout(page,`Expanded review ${viewport.width}×${viewport.height}`);await assertFormControls(page,`Expanded review ${viewport.width}px`);
   assert.equal(await editor.locator("textarea").inputValue(),description,"The review draft survives viewport changes");assert.equal(await page.locator("#missed-comment").inputValue(),"На столе виден ещё один предмет; требуется дополнительная проверка по видео.");
   assert.equal(await page.locator("#source-video").getAttribute("src"),sourceBefore,"Resize preserves the loaded video source");
   if(viewport.width<=650){
    for(const selector of [".review-metadata","#missed-event-form",...(viewport.width<360?[".review-interval"]:[])]){const columns=await page.locator(selector).first().evaluate(element=>getComputedStyle(element).gridTemplateColumns.split(" ").length);assert.equal(columns,1,`${selector} has one input column on a phone`);}
    const controls=await page.locator("#case-filters button,.case-details summary,.review-editor button,.review-decision-choice,#play-context,#next-case,#missed-event-form button").evaluateAll(elements=>elements.filter(element=>element.getBoundingClientRect().height>0).map(element=>({text:element.textContent,height:element.getBoundingClientRect().height})));assert(controls.every(control=>control.height>=40),`Mobile touch controls are at least 40px tall: ${JSON.stringify(controls.filter(control=>control.height<40))}`);
   }
   if([375,768,1024].includes(viewport.width)) {await page.locator("#case-decision-section").evaluate(element=>element.scrollIntoView({block:"start"}));await page.screenshot({path:path.join(root,"tmp",`ui-responsive-${viewport.width}.png`),fullPage:false});}
  }
  await page.setViewportSize({width:390,height:900});await page.emulateMedia({colorScheme:"dark"});await assertLayout(page,"Dark expanded review 390px");await assertFormControls(page,"Dark expanded review 390px");await page.locator("#case-decision-section").evaluate(element=>element.scrollIntoView({block:"start"}));await page.screenshot({path:path.join(root,"tmp","ui-responsive-390-dark.png"),fullPage:false});
  await page.locator(".case-occurrence .time-button").nth(1).click();assert.equal(await page.locator("#source-video").evaluate(video=>video.currentTime),8);
  await page.locator(".check-evidence button").nth(1).click();assert.equal(await page.locator("#source-video").evaluate(video=>video.currentTime),1.5);
  await page.locator(".evidence-button").last().click();assert.equal(await page.locator("#source-video").evaluate(video=>video.currentTime),23);
  await page.locator("#case-filters button[data-filter=confirmed]").click();assert.equal(await page.locator(".case-card").count(),1);assert.match(await page.locator("#selected-case-label").textContent(),/экран/);
  assert.equal(await page.locator("#selected-case-label .event-selection-time").textContent(),"00:20 · выбранное событие");assert.equal(await page.locator(".event-marker[aria-pressed=true]").count(),1);
  await page.locator("#case-filters button[data-filter=all]").click();await page.locator(".case-details summary").first().click();editor=page.locator(".case-detail-body").first().locator(".review-editor");assert.equal(await editor.locator("textarea").inputValue(),description,"Filters keep the unsaved draft too");
  assert.equal(await page.locator("#download-report").isDisabled(),true,"Unsaved drafts must not be omitted from a downloaded report");assert.match(await page.locator("#report-review-status").textContent(),/несохранённые/);await page.locator("#save-review-final").click();await page.waitForFunction(()=>document.getElementById("review-save-status").textContent.includes("версия 1"));assert.match(await page.locator("#report-review-status").textContent(),/версия 1/);assert.equal(await page.locator("#download-report").isEnabled(),true);
  assert.equal(writes.length,1,"Review and manual-event drafts are saved in one atomic request");assert.deepEqual(writes[0].decisions.map(value=>value.case_id),["presence-0","presence-8"]);assert(writes[0].decisions.every(value=>value.start_sec===1&&value.end_sec===9&&value.comment===description));assert.equal(writes[0].missed_events.length,1);assert.equal(writes[0].camera_id,"Камера 1");
  await assertLayout(page,"Saved mobile review");assert.equal(await page.locator(".case-status").first().textContent(),"Требует проверки","Human decisions remain separate from model status");
  const reportDownload=page.waitForEvent("download");await page.locator("#download-report").click();const downloaded=await reportDownload;assert.match(downloaded.suggestedFilename(),/\.html$/);assert.match(fs.readFileSync(await downloaded.path(),"utf8"),/Единый отчёт по проверке/);assert.equal(reportRequests,1,"Bottom report action downloads the authenticated single HTML file");
  for(const [width,colorScheme,suffix] of [[390,"dark","390-dark"],[1366,"light","1366"]]){
   await page.setViewportSize({width,height:900});await page.emulateMedia({colorScheme});await assertLayout(page,`Final timeline and report ${suffix}`);await assertFormControls(page,`Report ${suffix}`);await page.locator(".event-navigation").screenshot({path:path.join(root,"tmp",`ui-responsive-timeline-${suffix}.png`)});await page.locator("#report-section").screenshot({path:path.join(root,"tmp",`ui-responsive-report-${suffix}.png`)});
  }
  await page.locator(".case-details summary").first().click();editor=page.locator(".case-detail-body").first().locator(".review-editor");
  holdReport=true;const reportStarted=new Promise(resolve=>{signalReportStart=resolve;});await page.locator("#download-report").click();await reportStarted;assert.equal(await page.locator("#download-report").isDisabled(),true);
  await editor.locator("textarea").fill("Правка во время подготовки отчёта должна остаться в черновике.");releaseReport();await page.waitForFunction(()=>document.getElementById("report-error").textContent.includes("изменилась во время подготовки"));
  assert.equal(downloadCount,1,"An in-flight report must not download after the reviewer edits its underlying decisions");assert.equal(await page.locator("#download-report").isDisabled(),true);assert.equal(await editor.locator("textarea").inputValue(),"Правка во время подготовки отчёта должна остаться в черновике.");await assertLayout(page,"Report preparation race keeps local error inside footer");
  await page.locator("#technical-details > summary").click();await page.locator("#catalog-section > summary").click();await page.locator("#history-section > summary").click();await page.locator("#quality-section > summary").click();
  await page.waitForFunction(()=>document.querySelectorAll("#catalog-items tr").length===6&&document.querySelectorAll("#history-items button").length===1);
  if(!await page.locator(".case-details").first().evaluate(element=>element.open))await page.locator(".case-details summary").first().click();editor=page.locator(".case-detail-body").first().locator(".review-editor");await editor.locator("textarea").fill("Черновик должен сохраниться при конфликте версий.");assert.equal(await page.locator("#download-report").isDisabled(),true,"Unsaved review edits disable downloading the saved report");conflictNext=true;await page.locator("#save-review-final").click();await page.waitForFunction(()=>!document.getElementById("reload-review").hidden);assert.match(await page.locator("#review-error").textContent(),/другой вкладке/);
  for(const viewport of [{width:320,height:900},{width:650,height:900},{width:1024,height:900}]){await page.setViewportSize(viewport);await assertLayout(page,`Expanded technical panels and conflict ${viewport.width}px`);await assertFormControls(page,`Conflict ${viewport.width}px`);assert.equal(await editor.locator("textarea").inputValue(),"Черновик должен сохраниться при конфликте версий.");}
  assert.equal(analyzeRequests,0);assert.deepEqual(errors,[]);
  console.log("Responsive review passed: upload and saved results fit 320–1366px plus landscape and dark mode; native local picker, labelled cards, desktop table, evidence seeking, filters, touch targets, editor geometry, persistent drafts and atomic grouped saves.");
 } finally {if(releaseReport)releaseReport();await browser.close();}
})().catch(error=>{console.error(error);process.exit(1);});
