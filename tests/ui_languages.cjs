/* Russian/English UI against synthetic endpoints. No private media or live API. */
"use strict";
const assert = require("node:assert/strict"), fs = require("node:fs"), path = require("node:path");
const {chromium} = require(process.env.FORTE_PLAYWRIGHT_MODULE || "playwright");
const root = path.resolve(__dirname,".."), origin = "http://127.0.0.1:48125";
const job = "12345678-1234-4234-8234-123456789012", key = "language-test-key";
const filename = "Касса_Проверка_утро.mov", comment = "Требует проверки", missedComment = "Недостаточно данных";
const evidence = timestamp_sec => ({source_index:timestamp_sec*30,timestamp_sec});
const cases = [
 {case_id:"presence-0",type_id:"personal_belongings_in_cash_unit",status:"review_required",requires_review:true,timecode_sec:0,probability:.6,probability_scope:"action_only",checks:[{stage:"action",name:"non_cash_object_visually_supported",status:"yes",evidence_refs:[evidence(0)]}]},
 {case_id:"presence-8",type_id:"personal_belongings_in_cash_unit",status:"review_required",requires_review:true,timecode_sec:8,probability:.8,probability_scope:"action_only",checks:[{stage:"action",name:"non_cash_object_visually_supported",status:"yes",evidence_refs:[evidence(8)]}]},
 {case_id:"screen-20",type_id:"workstation_departure_screen_left_on",status:"confirmed",timecode_sec:20,probability:.91,probability_scope:"joint_checks",checks:[{stage:"action",name:"same_screen_visibly_on_after_departure",status:"yes",evidence_refs:[evidence(20),evidence(23)]}]},
 {case_id:"cash-30",type_id:"previous_transaction_cash_exposed",status:"rejected",timecode_sec:30,action_decision:"rejected",checks:[{stage:"action",name:"cash_exposed",status:"no"}]},
 {case_id:"intake-40",type_id:"active_intake_departure",status:"rejected",timecode_sec:40,action_decision:"uncertain",checks:[]}
];
let state = {revision:0,reviewer:"",camera_id:"",session_id:"",source_group:"",coverage_complete:false,decisions:[],missed_events:[]};
const errors = [], untranslated = [], requests = [], writes = [];
const metadata = () => ({job_id:job,status:"done",original_name:filename,review_cases:cases,review_state:structuredClone(state),details:{coverage:{duration_sec:60},catalog_version:"languages-test",case_decision:{cases},detections:[]}});
const english = async (locator,label) => {
 const text = (await locator.allTextContents()).join(" ").trim();
 assert(text,`${label} has visible text`);if(/[А-Яа-яЁё]/.test(text))untranslated.push({label,text});
};
async function layout(page,label) {
 const geometry = await page.evaluate(()=>{
  const buttons=[...document.querySelectorAll("#language-switcher button")].map(element=>({text:element.textContent,box:element.getBoundingClientRect().toJSON()}));
  return {viewport:innerWidth,document:document.documentElement.scrollWidth,body:document.body.scrollWidth,buttons};
 });
 assert(geometry.document<=geometry.viewport+1&&geometry.body<=geometry.viewport+1,`${label}: horizontal overflow ${JSON.stringify(geometry)}`);
 for(const {text,box} of geometry.buttons)assert(box.width>0&&box.height>=32&&box.left>=0&&box.right<=geometry.viewport+1,`${label}: ${text} must be available inside the header`);
}
async function switchLanguage(page,language) {
 const before = requests.length;
 await page.locator(`#language-switcher button[data-language=${language}]`).click();
 await page.waitForFunction(expected=>document.documentElement.lang===expected,language);
 await page.waitForFunction(expected=>document.querySelector("h1").textContent.includes(expected),language==="en"?"Cash":"Проверка");
 assert.equal(await page.locator(`#language-switcher button[data-language=${language}]`).getAttribute("aria-pressed"),"true");
 assert.equal(await page.locator(`#language-switcher button[data-language=${language==="en"?"ru":"en"}]`).getAttribute("aria-pressed"),"false");
 assert.equal(requests.length,before,"Switching UI language must not fetch, save or restart an analysis");
}
(async()=>{
 fs.mkdirSync(path.join(root,"tmp"),{recursive:true});
 const browser=await chromium.launch({headless:true,...(process.env.FORTE_BROWSER_EXECUTABLE?{executablePath:process.env.FORTE_BROWSER_EXECUTABLE}:{})});
 try {
  const page=await browser.newPage({viewport:{width:1366,height:900},colorScheme:"light"});
  page.on("pageerror",error=>errors.push(error.message));
  await page.addInitScript(()=>{
   HTMLMediaElement.prototype.load=function(){};
   HTMLMediaElement.prototype.play=function(){this.dataset.playing="true";return Promise.resolve();};
   HTMLMediaElement.prototype.pause=function(){this.dataset.playing="false";};
   Object.defineProperty(HTMLMediaElement.prototype,"readyState",{get:()=>1,configurable:true});
   Object.defineProperty(HTMLMediaElement.prototype,"duration",{get:()=>60,configurable:true});
   Object.defineProperty(HTMLMediaElement.prototype,"currentTime",{get(){return Number(this.dataset.time||0);},set(value){this.dataset.time=String(value);this.dispatchEvent(new Event("timeupdate"));},configurable:true});
  });
  await page.route("**/*",async route=>{
   const request=route.request(),url=new URL(request.url());assert.equal(url.origin,origin);
   if(["/","/app.js","/style.css","/static/i18n.js","/static/i18n-catalog.js"].includes(url.pathname))return route.fulfill({contentType:url.pathname.endsWith("js")?"text/javascript":url.pathname.endsWith("css")?"text/css":"text/html",body:fs.readFileSync(path.join(root,"web",url.pathname==="/"?"index.html":path.basename(url.pathname)))});
   if(url.pathname==="/api/v1/health")return route.fulfill({json:{ready:true,busy:false}});
   requests.push({method:request.method(),path:url.pathname});assert.equal(request.headers()["x-api-key"],key);
   if(url.pathname==="/api/ui/active-job")return route.fulfill({json:{job_id:null}});
   if(url.pathname==="/api/ui/catalog")return route.fulfill({json:{version:"languages-test",items:[...new Set(cases.map(item=>item.type_id))].map(id=>({id,scoring_kind:"event_occurrence",automation:"review_only"}))}});
   if(url.pathname==="/api/ui/history")return route.fulfill({json:{items:[{job_id:job,original_name:filename,status:"done",review_revision:state.revision}],total:1,limit:20,offset:0}});
   if(url.pathname==="/api/ui/quality")return route.fulfill({json:{status:"no_reviewed_labels",records:{total:1,with_review:0,eligible_complete_coverage:0},grouping:{status:"missing_metadata",source_families:0,unknown_group_records:1},per_type:[{type_id:"personal_belongings_in_cash_unit",automatic_decision:{precision:null,false_positive:0,false_negative:0},known_misses:{total:0},event_coverage:{recall:null,denominator_status:"incomplete"},temporal_localization:{pairs:0,mean_iou:null},probability_evaluation:{samples:0,brier_score:null}}],limitations:[]}});
   if(url.pathname.endsWith("/review")){
    assert.equal(request.method(),"PUT");const payload=request.postDataJSON();assert.equal(payload.revision,state.revision);writes.push(payload);state={...payload,revision:state.revision+1};return route.fulfill({json:state});
   }
   if(url.pathname.endsWith("/video"))return route.fulfill({contentType:"video/quicktime",body:"synthetic video"});
   if(url.pathname.startsWith("/api/v1/jobs/"))return route.fulfill({json:{job_id:job,status:"done",verdict:"no_violation",violations:[]}});
   if(url.pathname.startsWith("/api/ui/jobs/"))return route.fulfill({json:metadata()});
   return route.fulfill({status:404,body:"Synthetic endpoint only"});
  });
  await page.goto(origin);
  assert.equal(await page.locator("html").getAttribute("lang"),"ru");
  assert.deepEqual(await page.locator("#language-switcher button").allTextContents(),["Русский","English"]);
  await switchLanguage(page,"en");
  for(const selector of ["h1",".intro-copy",".upload-panel h2","#video-empty","#status-title","#status-description","#service-label","footer"])await english(page.locator(selector),selector);
  for(const selector of ["#api-key","#job-input"])assert.doesNotMatch(await page.locator(selector).getAttribute("placeholder"),/[А-Яа-яЁё]/);
  assert.doesNotMatch(await page.locator("#video-empty").getAttribute("aria-label"),/[А-Яа-яЁё]/);
  for(const width of [320,390,768,1366]){await page.setViewportSize({width,height:900});await layout(page,`English upload ${width}px`);}
  await page.fill("#api-key",key);
  const filePicker=page.waitForEvent("filechooser");await page.locator("#video-empty").click();await (await filePicker).setFiles({name:filename,mimeType:"video/quicktime",buffer:Buffer.from("synthetic")});
  await page.locator("#source-video").evaluate(video=>{video.currentTime=11;});
  const localSource=await page.locator("#source-video").getAttribute("src");assert.match(localSource,/^blob:/);
  await switchLanguage(page,"ru");await switchLanguage(page,"en");
  assert.equal(await page.locator("#file-name").textContent(),filename);assert.equal(await page.locator("#source-video").getAttribute("src"),localSource);assert.equal(await page.locator("#source-video").evaluate(video=>video.currentTime),11);
  assert(!requests.some(request=>request.path==="/api/v1/analyze"));
  await page.locator("#remove-file").click();await page.locator(".existing-job summary").click();await page.fill("#job-input",job);await page.locator("#open-job-form button").click();
  await page.waitForFunction(()=>!document.getElementById("results").hidden&&document.getElementById("source-video").src.startsWith("blob:"));
  await page.waitForFunction(()=>document.querySelector(".case-status")&&!/[А-Яа-яЁё]/.test(document.querySelector(".case-status").textContent));
  assert.equal(await page.locator(".case-card").count(),4);assert.equal(await page.locator(".case-probability-value").first().textContent(),"60%–80%");
  assert.equal(await page.locator("#video-name").textContent(),filename);
  for(const selector of ["#case-decision-title",".case-status",".check-name",".case-probability-scope","#case-probability-note","#case-filters","#selected-case-label","#event-timeline","#report-section"])await english(page.locator(selector),selector);
  assert.deepEqual(await page.locator(".case-card").first().locator("td").evaluateAll(elements=>elements.map(element=>element.dataset.label)),["Model","Model estimate","Time","Details"]);
  await page.locator("#case-filters button[data-filter=review_required]").click();await page.fill("#case-search","personal");assert.equal(await page.locator(".case-card").count(),1,"Search uses English check names");
  await page.locator(".case-details summary").first().click();await page.locator("#review-quality-metadata summary").click();await page.locator("#missed-event-section summary").click();
  let editor=page.locator(".case-detail-body").first().locator(".review-editor");
  await page.fill("#reviewer-name","Проверяющий");await page.fill("#review-camera","Камера");await page.fill("#review-session","Сессия");await page.fill("#review-source-group","Исходная группа");
  await editor.locator('input[value="confirmed"]').check();await editor.locator("textarea").fill(comment);await editor.locator('input[name="start_sec"]').fill("1");await editor.locator('input[name="end_sec"]').fill("9");
  await page.fill("#missed-start","45");await page.fill("#missed-end","48");await page.fill("#missed-comment",missedComment);await page.locator("#source-video").evaluate(video=>{video.currentTime=7;});
  const source=await page.locator("#source-video").getAttribute("src");
  for(const language of ["ru","en","ru","en"]){
   await switchLanguage(page,language);
   assert.equal(await page.locator("#api-key").inputValue(),key);assert.equal(await page.locator("#job-input").inputValue(),job);assert.equal(await page.locator("#case-search").inputValue(),"personal");assert.equal(await page.locator("#case-filters button[data-filter=review_required]").getAttribute("aria-pressed"),"true");
   assert.equal(await page.locator(".case-card").count(),1);assert.equal(await page.locator(".case-details").first().evaluate(element=>element.open),true);assert.equal(await page.locator("#review-quality-metadata").evaluate(element=>element.open),true);assert.equal(await page.locator("#missed-event-section").evaluate(element=>element.open),true);
   assert.equal(await page.locator("#reviewer-name").inputValue(),"Проверяющий");assert.equal(await page.locator("#review-camera").inputValue(),"Камера");assert.equal(await page.locator("#review-session").inputValue(),"Сессия");assert.equal(await page.locator("#review-source-group").inputValue(),"Исходная группа");
   assert.equal(await editor.locator("textarea").inputValue(),comment);assert.equal(await editor.locator('input[name="start_sec"]').inputValue(),"1");assert.equal(await editor.locator('input[name="end_sec"]').inputValue(),"9");assert.equal(await editor.locator('input[value="confirmed"]').isChecked(),true);assert.equal(await page.locator("#missed-comment").inputValue(),missedComment);
   assert.equal(await page.locator("#source-video").getAttribute("src"),source);assert.equal(await page.locator("#source-video").evaluate(video=>video.currentTime),7);assert.equal(await page.locator("#download-report").isDisabled(),true);
  }
  for(const selector of [".case-detail-body .case-justification",".case-checks",".review-editor h4","#review-quality-metadata summary","#missed-event-section summary","#review-save-status","#report-review-status"])await english(page.locator(selector),selector);
  for(const width of [320,390,768,1366]){await page.setViewportSize({width,height:900});await layout(page,`English expanded review ${width}px`);}
  await page.locator("#save-review-final").click();await page.waitForFunction(()=>document.getElementById("review-save-status").textContent.includes("1")&&!document.getElementById("download-report").disabled);
  assert.equal(writes.length,1);assert.equal(writes[0].reviewer,"Проверяющий");assert.equal(writes[0].camera_id,"Камера");assert.equal(writes[0].session_id,"Сессия");assert.equal(writes[0].source_group,"Исходная группа");assert(writes[0].decisions.every(item=>item.comment===comment));assert.equal(writes[0].missed_events[0].comment,missedComment);
  await english(page.locator(".human-status"),"Saved human status");await english(page.locator("#review-save-status"),"Save result");assert.match(await page.locator("#missed-events").textContent(),new RegExp(missedComment),"User-authored text is not translated even when it matches a UI label");
  await page.locator("#history-section summary").click();await page.waitForFunction(()=>document.querySelectorAll("#history-items button").length===1);assert.match(await page.locator("#history-items").textContent(),new RegExp(filename));await english(page.locator("#history-status"),"History count");
  await page.locator("#quality-section > summary").click();await page.waitForFunction(()=>document.querySelector("#quality-items").textContent.trim().length>0);await english(page.locator("#quality-status,#quality-summary,#quality-items"),"Quality panels");
  await page.locator("#catalog-section > summary").click();await page.waitForFunction(()=>document.querySelectorAll("#catalog-items tr").length===4);await english(page.locator("#catalog-items"),"Check catalog");
  await page.setViewportSize({width:390,height:900});await page.emulateMedia({colorScheme:"dark"});await layout(page,"English mobile dark review");await page.evaluate(()=>scrollTo(0,0));await page.screenshot({path:path.join(root,"tmp","ui-languages-390-en.png"),fullPage:false});
  assert.equal(await page.evaluate(()=>localStorage.getItem("forte-ui-language")),"en");assert(!await page.evaluate(secret=>JSON.stringify({...localStorage,...sessionStorage}).includes(secret),key));
  await page.reload();await page.waitForFunction(()=>document.documentElement.lang==="en");await english(page.locator("h1"),"Persisted English preference");assert.equal(await page.locator("#api-key").inputValue(),"","Only language persists, never credentials");
  await switchLanguage(page,"ru");assert.equal(await page.locator("#video-empty .video-empty-title").textContent(),"Предпросмотр видео");assert.equal(await page.evaluate(()=>localStorage.getItem("forte-ui-language")),"ru");assert.deepEqual(errors,[]);assert.deepEqual(untranslated,[],"All controlled interface copy is translated into English");
  console.log("Russian/English UI passed: static and dynamic copy, accessible labels, translated search, preserved review drafts/media/filters, unchanged user text, atomic saves, language-only persistence, and 320–1366px layouts.");
 } finally {await browser.close();}
})().catch(error=>{console.error(error);process.exit(1);});
