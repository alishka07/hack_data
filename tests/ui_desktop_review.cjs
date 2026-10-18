/* Desktop review workflow against synthetic, authenticated endpoints. */
"use strict";
const assert = require("node:assert/strict"), fs = require("node:fs"), path = require("node:path");
const {chromium} = require(process.env.FORTE_PLAYWRIGHT_MODULE || "playwright");
const root = path.resolve(__dirname,"..");
const job = "12345678-1234-4234-8234-123456789012", other = "23456789-2345-4345-8345-234567890123";
const cases = [
  {case_id:"presence-0",type_id:"personal_belongings_in_cash_unit",status:"review_required",requires_review:true,timecode_sec:0,probability:.6,probability_scope:"action_only",checks:[]},
  {case_id:"presence-8",type_id:"personal_belongings_in_cash_unit",status:"review_required",requires_review:true,timecode_sec:8,probability:.8,probability_scope:"action_only",checks:[]},
  {case_id:"screen-20",type_id:"workstation_departure_screen_left_on",status:"confirmed",timecode_sec:20,probability:.91,probability_scope:"joint_checks",checks:[]},
  {case_id:"cash-30",type_id:"previous_transaction_cash_exposed",status:"rejected",timecode_sec:30,action_decision:"rejected",checks:[{stage:"action",name:"cash_exposed",status:"no"}]},
  {case_id:"intake-40",type_id:"active_intake_departure",status:"rejected",timecode_sec:40,action_decision:"uncertain",checks:[]},
  {case_id:"presence-negative",type_id:"personal_belongings_in_cash_unit",status:"rejected",timecode_sec:31,probability_scope:"action_only",checks:[{stage:"action",name:"non_cash_object_visually_supported",status:"no"}]},
  {case_id:"presence-unknown",type_id:"personal_belongings_in_cash_unit",status:"rejected",timecode_sec:32,probability_scope:"action_only",checks:[]}
];
let state={revision:0,reviewer:"",camera_id:"",session_id:"",source_group:"",coverage_complete:false,decisions:[],missed_events:[]};
let staleNext=false, holdSave=false, releaseSave=null, saveStarted=false;
const writes=[], reportKeys=[], historyQueries=[], errors=[];
const metadata = id => ({job_id:id,status:"done",original_name:id===job?"Касса утро.mov":"Касса вечер.mov",review_cases:cases,review_state:structuredClone(state),details:{coverage:{duration_sec:60},case_decision:{cases},detections:[]}});
(async()=>{
 fs.mkdirSync(path.join(root,"tmp","browser-downloads"),{recursive:true});
 const browser=await chromium.launch({headless:true,downloadsPath:path.join(root,"tmp","browser-downloads"),...(process.env.FORTE_BROWSER_EXECUTABLE?{executablePath:process.env.FORTE_BROWSER_EXECUTABLE}:{})});
 try {
  const page=await browser.newPage({viewport:{width:1366,height:900},colorScheme:process.env.FORTE_TEST_COLOR_SCHEME || "light",acceptDownloads:true});
  page.on("pageerror",e=>errors.push(e.message));
  await page.addInitScript(()=>{
   HTMLMediaElement.prototype.load=function(){};
   HTMLMediaElement.prototype.play=function(){this.dataset.playing="true";return Promise.resolve();};
   HTMLMediaElement.prototype.pause=function(){this.dataset.playing="false";};
   Object.defineProperty(HTMLMediaElement.prototype,"readyState",{get:()=>1,configurable:true});
   Object.defineProperty(HTMLMediaElement.prototype,"duration",{get:()=>60,configurable:true});
   Object.defineProperty(HTMLMediaElement.prototype,"currentTime",{get(){return Number(this.dataset.time||0);},set(v){this.dataset.time=String(v);this.dispatchEvent(new Event("timeupdate"));},configurable:true});
  });
  await page.route("**/*",async route=>{
   const req=route.request(),url=new URL(req.url()); assert.equal(url.origin,"http://127.0.0.1:48123");
   if (["/","/app.js","/style.css"].includes(url.pathname)) return route.fulfill({contentType:url.pathname.endsWith("js")?"text/javascript":url.pathname.endsWith("css")?"text/css":"text/html",body:fs.readFileSync(path.join(root,"web",url.pathname==="/"?"index.html":url.pathname.slice(1)))});
   if(url.pathname==="/api/v1/health")return route.fulfill({json:{ready:true,busy:false}});
   assert.equal(req.headers()["x-api-key"],"desktop-test-key","All protected endpoints use an in-memory authenticated header");
   if(url.pathname==="/api/ui/active-job")return route.fulfill({json:{job_id:null}});
   if(url.pathname==="/api/ui/catalog")return route.fulfill({json:{version:"test",items:[...new Set(cases.map(c=>c.type_id))].map(id=>({id,scoring_kind:"event_occurrence",automation:"review_only"}))}});
   if(url.pathname==="/api/ui/history"){
    historyQueries.push(url.searchParams.get("query"));
    const all=[{job_id:job,original_name:"Касса утро.mov",status:"done",review_revision:state.revision},{job_id:other,original_name:"Касса вечер.mov",status:"done",review_revision:0}];
    const found=all.filter(item=>item.original_name.includes(url.searchParams.get("query")||""));
    return route.fulfill({json:{items:found,total:found.length,limit:20,offset:0}});
   }
   if(url.pathname==="/api/ui/quality")return route.fulfill({json:{status:"no_reviewed_labels",records:{total:2,with_review:0,eligible_complete_coverage:0},grouping:{status:"missing_metadata",source_families:0,unknown_group_records:2},per_type:[{type_id:"personal_belongings_in_cash_unit",automatic_decision:{precision:null,false_positive:0,false_negative:0},known_misses:{total:0},event_coverage:{recall:null,denominator_status:"incomplete"},temporal_localization:{pairs:0,mean_iou:null},probability_evaluation:{samples:0,brier_score:null}}],limitations:["Без полной разметки полнота неизвестна."]}});
   if(url.pathname.endsWith("/report.html")){reportKeys.push(req.headers()["x-api-key"]);return route.fulfill({contentType:"text/html; charset=utf-8",body:"<!doctype html><html lang=ru><body>Единый отчёт</body></html>"});}
   if(url.pathname.endsWith("/review")){
    assert.equal(req.method(),"PUT"); const payload=req.postDataJSON();writes.push(payload);saveStarted=true;
    if(holdSave){holdSave=false;await new Promise(resolve=>{releaseSave=resolve;});}
    if(staleNext){staleNext=false;state.revision+=1;return route.fulfill({status:409,json:{detail:"Review revision conflict"}});}
    assert.equal(payload.revision,state.revision);state={...payload,revision:state.revision+1,updated_at:"2026-10-18T12:00:00Z"};return route.fulfill({json:state});
   }
   if(url.pathname.endsWith("/video"))return route.fulfill({contentType:"video/quicktime",body:"synthetic"});
   if(url.pathname.startsWith("/api/v1/jobs/"))return route.fulfill({json:{job_id:url.pathname.split("/").at(-1),status:"done",verdict:"no_violation",violations:[]}});
   if(url.pathname.startsWith("/api/ui/jobs/"))return route.fulfill({json:metadata(url.pathname.split("/").at(-1))});
   return route.fulfill({status:404,body:"Synthetic only"});
  });
  await page.goto("http://127.0.0.1:48123/");
  assert.equal(await page.locator("html").getAttribute("lang"),"ru");
  await page.fill("#api-key","desktop-test-key");
  await page.locator(".existing-job summary").click();
  await page.fill("#job-input",job); await page.locator("#open-job-form button").click();
  await page.waitForFunction(()=>!document.getElementById("results").hidden&&document.getElementById("source-video").src.startsWith("blob:"));
  assert.equal(await page.locator(".case-card").count(),6,"Presence observations are grouped, independent outcomes and contradicted versus unknown results stay separate");
  assert.equal(await page.locator(".case-time .time-button").first().textContent(),"▶ 00:00–00:08");
  assert.deepEqual(await page.locator(".case-status").allTextContents(),["Требует проверки","Подтверждено моделью","Не подтвердилось","Не подтвердилось","Недостаточно данных","Недостаточно данных"]);
  assert.equal(await page.locator("#case-filters button[data-filter=review_required] span").textContent(),"1");
  const bounds=await page.evaluate(()=>({video:document.querySelector(".video-column").getBoundingClientRect().right,results:document.querySelector(".decision-column").getBoundingClientRect().left,sticky:getComputedStyle(document.querySelector(".video-column")).position}));
  assert(bounds.video<bounds.results);assert.equal(bounds.sticky,"sticky");
  assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));
  fs.mkdirSync(path.join(root,"tmp"),{recursive:true});await page.screenshot({path:path.join(root,"tmp","ui-desktop-review-initial"+(process.env.FORTE_TEST_COLOR_SCHEME === "dark" ? "-dark" : "")+".png"),fullPage:false});
  await page.locator("#case-filters button[data-filter=review_required]").click();assert.equal(await page.locator(".case-card").count(),1);
  await page.locator("#case-filters button[data-filter=confirmed]").click();assert.match(await page.locator(".is-selected .check-name").textContent(),/экран/);assert.match(await page.locator("#selected-case-label").textContent(),/экран/);
  await page.fill("#case-search","нет совпадений");assert.equal(await page.locator(".is-selected").count(),0);assert.equal(await page.locator("#play-context").isDisabled(),true);assert.equal(await page.locator("#next-case").isDisabled(),true);
  await page.locator("#event-timeline button").first().click();assert.equal(await page.locator(".case-card").count(),6,"A timeline event outside the filter reveals its row");assert.equal(await page.locator("#case-search").inputValue(),"");assert.equal(await page.locator("#case-filters button[data-filter=all]").getAttribute("aria-pressed"),"true");
  await page.fill("#case-search","экран");assert.equal(await page.locator(".case-card").count(),1);await page.fill("#case-search","");
  await page.locator("#event-timeline button").nth(1).click();
  assert.equal(await page.locator("#source-video").evaluate(v=>v.currentTime),20);assert.match(await page.locator(".is-selected .check-name").textContent(),/экран/);
  await page.locator("#play-context").click();assert.equal(await page.locator("#source-video").evaluate(v=>v.currentTime),17);
  assert.equal(await page.locator("#play-context-status").textContent(),"00:17–00:23");
  await page.locator("#source-video").evaluate(v=>{v.currentTime=23;});assert.equal(await page.locator("#source-video").getAttribute("data-playing"),"false");
  await page.locator("#next-case").click();assert.equal(await page.locator("#source-video").evaluate(v=>v.currentTime),30);
  await page.fill("#reviewer-name","Проверяющий Тест");await page.locator("#review-quality-metadata summary").click();await page.fill("#review-camera","Камера 1");await page.fill("#review-session","Утро");await page.fill("#review-source-group","original-source-A");
  await page.locator(".case-details summary").first().click();
  let editor=page.locator(".case-detail-body").first().locator(".review-editor");
  await editor.locator('input[value="confirmed"]').check();await editor.locator("textarea").fill('Предмет виден на столе. <img src=x onerror="window.injected=1">');
  await editor.locator('input[name="start_sec"]').fill("1");await editor.locator('input[name="end_sec"]').fill("9");
  page.once("dialog",dialog=>dialog.dismiss());
  await page.reload({timeout:3000}).catch(()=>{});
  assert.equal(await page.locator("#api-key").inputValue(),"desktop-test-key","Canceling an unsaved-leave prompt retains the key and source session");
  assert.equal(await editor.locator("textarea").inputValue(),'Предмет виден на столе. <img src=x onerror="window.injected=1">');
  // Filters preserve the draft while swapping row DOM.
  await page.locator("#case-filters button[data-filter=confirmed]").click();await page.locator("#case-filters button[data-filter=all]").click();await page.locator(".case-details summary").first().click();
  editor=page.locator(".case-detail-body").first().locator(".review-editor");assert.equal(await editor.locator("textarea").inputValue(),'Предмет виден на столе. <img src=x onerror="window.injected=1">');
  await page.locator("#open-job-form button").click();assert.equal(await editor.locator("textarea").inputValue(),'Предмет виден на столе. <img src=x onerror="window.injected=1">',"Reopening the same job cannot discard edits");
  await page.setInputFiles("#video-file",{name:"other.mov",mimeType:"video/quicktime",buffer:Buffer.from("test")});assert.equal(await page.locator("#job-id").textContent(),job);assert.equal(await page.locator("#results").isVisible(),true);
  await editor.locator("button[type=submit]").click();
  try { await page.waitForFunction(()=>document.getElementById("review-save-status").textContent.includes("версия 1"),null,{timeout:5000}); }
  catch(error) { console.error({errors,writes,state,status:await page.locator("#review-save-status").textContent(),error:await page.locator("#review-error").textContent()}); throw error; }
  assert.equal(writes[0].decisions.length,2,"One displayed group saves both raw stable IDs atomically");assert.deepEqual(writes[0].decisions.map(v=>v.case_id),["presence-0","presence-8"]);assert(writes[0].decisions.every(v=>v.start_sec===1&&v.end_sec===9));
  assert.equal(await page.locator(".human-status").first().textContent(),"Человек: подтверждено");assert.equal(await page.locator(".case-status").first().textContent(),"Требует проверки","Human decisions do not alter machine outcomes");
  assert.match(await page.locator(".human-interval").textContent(),/00:01–00:09/);assert.equal(await page.evaluate(()=>window.injected),undefined);
  // Reopening a saved job restores the independent human decision and corrected interval.
  await page.locator("#open-job-form button").click();await page.waitForFunction(()=>document.querySelector(".human-status")?.textContent==="Человек: подтверждено");
  await page.locator(".case-details summary").first().click();editor=page.locator(".case-detail-body").first().locator(".review-editor");assert.equal(await editor.locator('input[name="start_sec"]').inputValue(),"1");
  staleNext=true;await editor.locator('input[value="dismissed"]').check();await editor.locator("textarea").fill("Локальный черновик при конфликте");await editor.locator("button[type=submit]").click();
  await page.waitForFunction(()=>!document.getElementById("reload-review").hidden);assert.match(await page.locator("#review-error").textContent(),/другой вкладке/);assert.equal(await editor.locator("textarea").inputValue(),"Локальный черновик при конфликте");
  await page.locator("#reload-review").click();await page.waitForFunction(()=>document.getElementById("reload-review").hidden);assert.match(await page.locator("#review-save-status").textContent(),/версия 2/);
  // In-flight writes lock navigation, filtering and editors until the revision is known.
  holdSave=true;saveStarted=false;await page.locator(".case-details summary").first().click();editor=page.locator(".case-detail-body").first().locator(".review-editor");await editor.locator('input[value="uncertain"]').check();await editor.locator("button[type=submit]").click();
  for(let tries=0;!saveStarted&&tries<100;tries++)await new Promise(resolve=>setTimeout(resolve,20));
  if(!saveStarted)console.error({errors,writes,state,status:await page.locator("#review-save-status").textContent(),error:await page.locator("#review-error").textContent()});
  assert(saveStarted,"The held save must reach the atomic review endpoint");
  assert.equal(await page.locator("#case-search").isDisabled(),true);assert.equal(await page.locator("#case-filters button").first().isDisabled(),true);assert.equal(await editor.locator("textarea").isDisabled(),true);assert.equal(await page.locator("#open-job-form button").isDisabled(),true);releaseSave();
  await page.waitForFunction(()=>document.getElementById("review-save-status").textContent.includes("версия 3"));
  await page.locator("#missed-event-section summary").click();await page.fill("#missed-start","45");await page.fill("#missed-end","48");await page.fill("#missed-comment","На столе виден дополнительный предмет.");
  await page.fill("#job-input",other);await page.locator("#open-job-form button").click();assert.equal(await page.locator("#job-id").textContent(),job,"Unsent manual-event text blocks navigation");
  await page.locator("#missed-event-form button[type=submit]").click();await page.waitForFunction(()=>document.getElementById("review-save-status").textContent.includes("версия 4"));
  assert.equal(state.missed_events.length,1);assert.match(state.missed_events[0].event_id,/^[a-f0-9-]{36}$/);assert.equal(await page.locator("#missed-comment").inputValue(),"");assert.equal(await page.locator("#missed-events li").count(),1);
  await page.locator("#review-coverage").check();await page.locator("#save-review-meta").click();await page.waitForFunction(()=>document.getElementById("review-save-status").textContent.includes("версия 5"));assert.equal(state.coverage_complete,true);
  const downloadEvent=page.waitForEvent("download");await page.locator("#download-report").click();const download=await downloadEvent;assert.match(download.suggestedFilename(),/\.html$/);assert.deepEqual(reportKeys,["desktop-test-key"]);
  const reportText=fs.readFileSync(await download.path(),"utf8");assert.match(reportText,/Единый отчёт/);
  await page.locator("#history-section summary").click();await page.fill("#history-search","вечер");await page.waitForFunction(()=>document.querySelectorAll("#history-items button").length===1&&document.getElementById("history-items").textContent.includes("вечер"));assert(historyQueries.includes("вечер"));
  await page.locator("#history-items button").click();await page.waitForFunction(id=>document.getElementById("job-id").textContent===id,other);assert.equal(await page.locator("#video-name").textContent(),"Касса вечер.mov");
  await page.locator("#quality-section > summary").click();await page.waitForFunction(()=>document.getElementById("quality-status").textContent.includes("недостаточно"));
  assert.match(await page.locator("#quality-items").textContent(),/Точность.*Недостаточно данных/);assert.match(await page.locator("#quality-items").textContent(),/нет полного знаменателя/);assert.doesNotMatch(await page.locator("#quality-items").textContent(),/100\.0%/);
  assert.equal(await page.evaluate(()=>localStorage.getItem("api-key")),null);assert(!await page.evaluate(()=>JSON.stringify({...localStorage,...sessionStorage}).includes("desktop-test-key")));
  await page.locator("#quality-section > summary").click();await page.locator("#history-section summary").click();await page.evaluate(()=>window.scrollTo(0,0));
  fs.mkdirSync(path.join(root,"tmp"),{recursive:true});await page.screenshot({path:path.join(root,"tmp","ui-desktop-review-workflow"+(process.env.FORTE_TEST_COLOR_SCHEME === "dark" ? "-dark" : "")+".png"),fullPage:false});
  assert.deepEqual(errors,[]);console.log("Desktop review workflow passed: Russian layout, source context, filters, atomic decisions, drafts, conflicts, missed events, history, authenticated HTML report and honest quality metrics.");
 } finally {await browser.close();}
})().catch(error=>{console.error(error);process.exit(1);});
