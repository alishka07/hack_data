/* Synthetic browser regression. No service, private jobs, or media are read.
 * Run with Node plus Playwright installed as testing tooling. Optional overrides:
 * FORTE_PLAYWRIGHT_MODULE and FORTE_BROWSER_EXECUTABLE.
 */
"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { chromium } = require(process.env.FORTE_PLAYWRIGHT_MODULE || "playwright");
const root = path.resolve(__dirname, "..");
const ids = {v2:"11111111-1111-4111-8111-111111111111", old:"22222222-2222-4222-8222-222222222222", flagged:"33333333-3333-4333-8333-333333333333", legacy:"44444444-4444-4444-8444-444444444444", unavailable:"55555555-5555-4555-8555-555555555555", clear:"66666666-6666-4666-8666-666666666666", uncertain:"77777777-7777-4777-8777-777777777777", manual:"88888888-8888-4888-8888-888888888888", dense:"99999999-9999-4999-8999-999999999999", dismissed:"aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa", gap:"bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb", empty:"cccccccc-cccc-4ccc-8ccc-cccccccccccc"};
const queue = [{timecode_sec:32.5,type_id:"unsupported_visible_action",reason_codes:["rule_not_mapped","insufficient_context"],message:"Synthetic action requires human review.",observed_action:'Synthetic text <img src=x onerror="window.__injected=true">',rule_id:null,evidence_frames:[]}, {timecode_sec:45,type_id:"unverified_context",reason_codes:["limited_visibility"],message:"Synthetic context is incomplete.",rule_id:"example-rule",evidence_frames:[]}];
const catalog = {version:"synthetic-catalog",items:[
  {id:"event-one",label:"Synthetic event <img src=x>",scoring_kind:"event_occurrence",initial_score:0,automation:"review_only"},
  {item_id:"procedure-one",label:"Synthetic required procedure",scoring_kind:"required_procedure",initial_score:100,automation:"review_only"},
  {id:"procedure-manual",label:"Synthetic manual procedure",scoring_kind:"required_procedure",initial_score:100,automation:"manual"},
  {id:"procedure-na",label:"Synthetic excluded procedure",scoring_kind:"required_procedure",initial_score:100,automation:"excluded"},
  {id:"personal_food_at_workstation",label:"Personal belongings at workplace (food)",scoring_kind:"event_occurrence",initial_score:0,automation:"automatic_when_supported",basis:"workplace_policy",source_url:"/policies/workplace-belongings"}
]};
const assessments = [
  {item_id:"event-one",label:"Synthetic review-only observation",scoring_kind:"event_occurrence",initial_score:0,score:68,model_confidence:0.68,calibrated:false,applicability:"applicable",assessment:"observed",confirmed_finding:false,evidence_refs:[{source_index:10,timestamp_sec:1},{source_index:20,timestamp_sec:2}],scope:"clip_event_summary",automation:"review_only"},
  {item_id:"procedure-one",label:"Synthetic required procedure",scoring_kind:"required_procedure",initial_score:100,score:18,model_confidence:0.82,calibrated:false,applicability:"applicable",assessment:"completion_evidence",confirmed_finding:false,evidence_refs:[{source_index:30,timestamp_sec:3}],scope:"per_episode_summary",scope_certainty:"window_only",known_episode_count:3,unassessed_episode_count:1,episode_coverage_complete:false,episode_id:"transaction-1",episode_count:2,pending_episode_count:2,summary_policy:"maximum_episode_risk",latest_episode_id:"transaction-2",latest_episode_start_sec:2,latest_episode_end_sec:4,automation:"review_only"},
  {item_id:"procedure-manual",label:"Synthetic manual procedure",scoring_kind:"required_procedure",initial_score:100,score:100,model_confidence:null,calibrated:false,applicability:"unknown",assessment:"unassessed",confirmed_finding:false,evidence_refs:[],scope:"per_episode_verification_required",automation:"manual"},
  {item_id:"procedure-na",label:"Synthetic excluded procedure",scoring_kind:"required_procedure",initial_score:100,score:null,model_confidence:null,calibrated:false,applicability:"not_applicable",assessment:"unassessed",confirmed_finding:false,evidence_refs:[]}
];
assessments[1].unscoped_unknown_observation_count = 1;
const assessmentEpisodes = [
  {...assessments[1],scope:"episode_assessment",episode_id:"transaction-1",latest_episode_id:undefined,start_sec:0,end_sec:2},
  {...assessments[1],scope:"episode_assessment",episode_id:"transaction-2",latest_episode_id:undefined,start_sec:2,end_sec:4,score:8,model_confidence:0.92},
];
const detections = [
  {id:"cash-1",label:"Cash <img src=x onerror=alert(1)>",kind:"cash",confidence:.84,timestamp_sec:10,source_index:300,bbox:[.1,.2,.4,.6],evidence_frame:{filename:"detection-001.jpg",timestamp_sec:10}},
  {id:"hand-1",label:"Hand",kind:"hand",confidence:.76,timestamp_sec:10,source_index:300,bbox:[.6,.1,.9,.8],evidence_frame:{filename:"detection-001.jpg",timestamp_sec:10}},
  {id:"invalid",label:"Invalid box",kind:"cash",confidence:.9,timestamp_sec:10,source_index:300,bbox:[.8,.2,.4,.6]},
];
const caseDecision = {status:"review_required",title:"Review required",summary:"An observed action needs transaction-context review.",counts:{confirmed:0,review_required:1,rejected:1,candidates:2,visual_gaps:0,unresolved_assessments:1},cases:[
  {type_id:"active_intake_departure",status:"review_required",requires_review:true,timecode_sec:10,probability:.74,probability_scope:"action_only",probability_calibrated:false,observed_action:"Synthetic action <img src=x>",checks:[{stage:"action",name:"workstation_departure",status:"yes",evidence_refs:[{source_index:300,timestamp_sec:10}]},{stage:"context",name:"unfinished_intake",status:"unknown",evidence_refs:[]}],reasons:["Transaction context remains unknown."]},
  {type_id:"previous_transaction_cash_exposed",status:"rejected",requires_review:false,timecode_sec:null,probability:.12,probability_scope:"joint_checks",probability_calibrated:false,checks:[{stage:"action",name:"cash_exposed",status:"no",evidence_refs:[]}],reasons:["The required action is not supported."]}
]};
const hostQueue = [{...queue[0],type_id:"active_intake_departure",timecode_sec:10,visual_scenario_supported:true},{...queue[1],type_id:"previous_transaction_cash_exposed"}];
const details = {coverage:{duration_sec:142.533,windows_completed:18,windows_planned:18,max_sample_gap_sec:0.25,source_frames_total:4273,context_windows_used:4,visibility_gaps:[]},review_queue:hostQueue,limitations:["Synthetic visibility limitation."],catalog_version:catalog.version,assessments,assessment_episodes:assessmentEpisodes,detections,case_decision:caseDecision,findings:[{type_id:"example_supported_type",timecode_sec:10,rule_name:"Synthetic official rule",paragraph:"22",rule_url:"https://adilet.zan.kz/rus/docs/V1900019680#z114"}]};
const dismissedDetails = {review_queue:[{...hostQueue[0],status:"rejected",requires_review:true,message:"Invalid model blah blah",observed_action:"No cash-like bundle is visibly handled."},{...hostQueue[1],status:"review_required",requires_review:false,message:"Invalid model blah blah",observed_action:"Uncertain model blah blah"}],case_decision:{status:"review_required",counts:{review_required:99},cases:[{...caseDecision.cases[0],status:"rejected",requires_review:true,action_decision:"rejected",observed_action:"No cash-like bundle is visibly handled.",reasons:["The required action is not supported."]},{...caseDecision.cases[1],status:"review_required",requires_review:false,action_decision:"uncertain",observed_action:"Uncertain model blah blah",reasons:["The visible action could not be established."]}]}};
ids.visibility = "dddddddd-dddd-4ddd-8ddd-dddddddddddd";
ids.incomplete = "eeeeeeee-eeee-4eee-8eee-eeeeeeeeeeee";
ids.incompleteFlagged = "ffffffff-ffff-4fff-8fff-ffffffffffff";
ids.incompleteMarker = "01234567-0123-4123-8123-012345678901";
ids.incompleteStatus = "01234567-0123-4123-8123-012345678902";
ids.incompleteCount = "01234567-0123-4123-8123-012345678903";
ids.food = "01234567-0123-4123-8123-012345678904";
ids.unsafePolicy = "01234567-0123-4123-8123-012345678905";
ids.probability = "01234567-0123-4123-8123-012345678906";
ids.screen = "01234567-0123-4123-8123-012345678907";
ids.grouped = "01234567-0123-4123-8123-012345678908";
ids.groupedProbability = "01234567-0123-4123-8123-012345678909";
const probabilityDetails = {assessments:[{...assessments[0],item_id:"active_intake_departure",score:99,model_confidence:.99}],case_decision:{status:"review_required",cases:[
  ...[0,1,null,true,"0.9",NaN,-.01,1.01].map((probability,index) => ({...caseDecision.cases[0],episode_id:"separate-episode-" + index,timecode_sec:index,probability,probability_scope:index === 0 ? "event_occurrence" : "joint_checks"})),
  {...caseDecision.cases[0],episode_id:"procedure-episode",probability:.82,probability_scope:"procedure_completion"},
  {...caseDecision.cases[0],episode_id:"legacy-episode",probability:undefined,probability_scope:undefined}
]}};
let unsafePolicyUrl = "javascript:alert(1)";
const foodDetails = {findings:[{type_id:"personal_food_at_workstation",timecode_sec:12.5,basis:"workplace_policy",rule_name:"User-provided workplace policy",paragraph:"Personal food at the cashier workstation",rule_url:"/policies/workplace-belongings"}],case_decision:{status:"confirmed",cases:[{type_id:"personal_food_at_workstation",status:"confirmed",timecode_sec:12.5,probability:.94,probability_scope:"event_occurrence",probability_calibrated:false,observed_action:"Personal food is visibly present on the cashier work table during work.",checks:[{stage:"action",name:"personal_food_on_work_table",status:"yes",evidence_refs:[{source_index:375,timestamp_sec:12.5}]}],reasons:["Personal food is visibly present at the workplace."]}]}};
const screenDetails = {findings:[{type_id:"workstation_departure_screen_left_on",timecode_sec:16,basis:"workplace_policy",rule_name:"User-provided workplace policy",paragraph:"Switch off the screen when leaving the workstation",rule_url:"/policies/workplace-screen"}],case_decision:{status:"confirmed",cases:[{type_id:"workstation_departure_screen_left_on",status:"confirmed",timecode_sec:16,probability:.91,probability_scope:"joint_checks",probability_calibrated:false,observed_action:"The cashier leaves the workstation; later the same workstation is unattended while its computer screen remains visibly on.",checks:[{stage:"action",name:"workstation_departure_observed",status:"yes",evidence_refs:[{source_index:300,timestamp_sec:10},{source_index:480,timestamp_sec:16}]},{stage:"context",name:"same_screen_visibly_on_after_departure",status:"yes",evidence_refs:[{source_index:480,timestamp_sec:16}]}],reasons:["The departure and subsequent illuminated screen are supported by source frames."]}]}};
const groupedDetails = {case_decision:{status:"review_required",cases:[
  {...caseDecision.cases[1],timecode_sec:2},
  {...screenDetails.case_decision.cases[0],timecode_sec:1,checks:[]},
  ...[0,8,13,17].map((time,index) => ({type_id:"personal_belongings_in_cash_unit",episode_id:"presence-window-" + index,status:"review_required",requires_review:true,timecode_sec:time,probability:null,probability_scope:"action_only",action_decision:"uncertain",observed_action:"Uncertain synthetic text must stay hidden.",checks:[{stage:"action",name:"belongings_present",status:"unknown",evidence_refs:[{source_index:time*30,timestamp_sec:time}]}],reasons:["Synthetic sampled presence check " + (index + 1)]})),
  {...caseDecision.cases[0],episode_id:"departure-one",timecode_sec:5},
  {...caseDecision.cases[0],episode_id:"departure-two",timecode_sec:15},
  {type_id:"personal_belongings_in_cash_unit",episode_id:"presence-rejected",status:"rejected",requires_review:false,timecode_sec:25,probability:.12,probability_scope:"action_only",checks:[],reasons:["Synthetic rejected presence check."]}
]}};
const groupedProbabilityDetails = {case_decision:{status:"review_required",cases:[
  ...[.6,.8].map((probability,index) => ({...groupedDetails.case_decision.cases[2],timecode_sec:3+index*7,probability})),
  {...groupedDetails.case_decision.cases[2],timecode_sec:12,probability:.95,probability_scope:"joint_checks"},
  {...groupedDetails.case_decision.cases[2],timecode_sec:null,probability:null},
  {...groupedDetails.case_decision.cases[2],timecode_sec:null,probability:null}
]}};
const incompleteDetails = {processing_status:"review_incomplete",coverage:{...details.coverage,verification_complete:false,unverified_candidate_count:3},case_decision:{status:"no_supported_violation",cases:[]},review_queue:[{...hostQueue[0],status:"diagnostic",requires_review:false,reason_codes:["verification_budget_exhausted"]}]};
const denseDetections = Array.from({length:50},(_,index) => {
  const group = Math.floor(index / 24), timestamp = 10 + group * 10;
  return {id:"synthetic-dense-" + index,label:index === 49 ? 'Synthetic person <img src=x onerror="window.__denseInjected=true">' : "Synthetic person " + (index + 1),kind:"person",confidence:.71,timestamp_sec:timestamp,source_index:300 + group * 300,bbox:[.1,.2,.4,.6],evidence_frame:{filename:"dense-" + group + ".jpg",timestamp_sec:timestamp}};
});
const denseDetails = {...details,
  detections:denseDetections,
  coverage:{...details.coverage,context_windows_used:["synthetic-a","synthetic-b","synthetic-c"],perception:{frames_analyzed:713,max_gap_sec:.2,complete:true,ending_included:true}},
  transactions:{episodes:[{id:"synthetic-1"},{id:"synthetic-2"},{id:"synthetic-3"}]},
  scene:{zones:[{name:"counter",status:"supported"},{name:"customer_area",status:"unknown"},{name:"storage",status:"contradicted"}]},
  case_decision:{...caseDecision,cases:[{...caseDecision.cases[0],checks:[{stage:"exception_exclusion",name:'synthetic_occlusion_<img src=x onerror="window.__denseInjected=true">',status:"yes",evidence_refs:[{timestamp_sec:null},{source_index:600,timestamp_sec:20},{timestamp_sec:-1},{source_index:601,timestamp_sec:20.25},{source_index:602,timestamp_sec:20.5},{source_index:603,timestamp_sec:20.75}]}]}]}
};
let catalogRequests = 0;
let evidenceRequests = 0;
const denseEvidenceRequests = [];
let unavailableDetails = true;
const runningId = "abcdefab-abcd-4abc-8abc-abcdefabcdef";
let activeJob = null;
let healthBusy = false;
let activeJobRequests = 0;
let analyzeRequests = 0;
let acceptUpload = false;
let runningPolls = 0;
let holdDiscovery = false;
let releaseDiscovery = null;
const syntheticJpeg = "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAIBAQEBAQIBAQECAgICAgQDAgICAgUEBAMEBgUGBgYFBgYGBwkIBgcJBwYGCAsICQoKCgoKBggLDAsKDAkKCgr/2wBDAQICAgICAgUDAwUKBwYHCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgr/wAARCAAIAAgDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwCOiiigD//Z";
const pageErrors = [];

(async () => {
  const browser = await chromium.launch({headless:true,...(process.env.FORTE_BROWSER_EXECUTABLE ? {executablePath:process.env.FORTE_BROWSER_EXECUTABLE} : {})});
  try {
    const page = await browser.newPage({viewport:{width:1280,height:900}});
    page.on("pageerror", error => pageErrors.push(error.message));
    await page.addInitScript(() => {
      // Exercise timestamp controls without loading or decoding any media.
      HTMLMediaElement.prototype.load = function () {};
      Object.defineProperty(HTMLMediaElement.prototype,"readyState",{get:()=>1,configurable:true});
      Object.defineProperty(HTMLMediaElement.prototype,"duration",{get:()=>180,configurable:true});
      Object.defineProperty(HTMLVideoElement.prototype,"videoWidth",{get:()=>1920,configurable:true});
      Object.defineProperty(HTMLVideoElement.prototype,"videoHeight",{get:()=>1080,configurable:true});
      Object.defineProperty(HTMLMediaElement.prototype,"currentTime",{get(){return Number(this.dataset.syntheticTime || 0);},set(value){this.dataset.syntheticTime=String(value);this.dispatchEvent(new Event("timeupdate"));},configurable:true});
    });
    await page.route("**/*", async route => {
      const url = new URL(route.request().url());
      assert.equal(url.origin,"http://forte-synthetic.test","All requests must stay on the synthetic origin");
      const pathname = url.pathname;
      if (["/","/app.js","/style.css"].includes(pathname)) {
        const filename = pathname === "/" ? "index.html" : pathname.slice(1);
        return route.fulfill({status:200,contentType:filename.endsWith("js")?"text/javascript":filename.endsWith("css")?"text/css":"text/html",body:fs.readFileSync(path.join(root,"web",filename))});
      }
      if (pathname === "/api/v1/health") return route.fulfill({json:{worker_ready:true,status:"ready",busy:healthBusy}});
      if (pathname === "/api/ui/active-job") {
        activeJobRequests += 1;
        assert(route.request().headers()["x-api-key"],"Active jobs must only be discovered with the API key");
        if (route.request().headers()["x-api-key"] !== "synthetic-test-key") return route.fulfill({status:401,json:{detail:"Invalid synthetic key"}});
        const payload = activeJob ? {job_id:activeJob,status:"processing",stage:"verifying",progress:.54} : {job_id:null};
        if (holdDiscovery) { holdDiscovery = false; await new Promise(resolve => { releaseDiscovery = resolve; }); }
        return route.fulfill({json:payload});
      }
      if (pathname === "/api/v1/analyze") {
        analyzeRequests += 1; activeJob = runningId; healthBusy = true;
        if (acceptUpload) return route.fulfill({status:202,json:{job_id:runningId,status:"processing"}});
        return route.fulfill({status:429,headers:{"Retry-After":"5"},json:{detail:"Analysis capacity is in use"}});
      }
      if (pathname === "/api/ui/catalog") {
        catalogRequests += 1;
        assert(route.request().headers()["x-api-key"],"The catalog request must be authenticated");
        if (route.request().headers()["x-api-key"] !== "synthetic-test-key") return route.fulfill({status:401,json:{detail:"Invalid synthetic key"}});
        return route.fulfill({json:catalog});
      }
      if (pathname.endsWith("/video")) return route.fulfill({status:200,contentType:"video/quicktime",body:"synthetic-media-placeholder"});
      if (pathname.includes("/evidence/")) {
        evidenceRequests += 1;
        assert.equal(route.request().headers()["x-api-key"],"synthetic-test-key");
        if (pathname.includes("/" + ids.dense + "/")) denseEvidenceRequests.push(pathname.split("/").at(-1));
        return route.fulfill({status:200,contentType:"image/jpeg",body:Buffer.from(syntheticJpeg,"base64")});
      }
      const id = pathname.split("/").at(-1);
      const food = id === ids.food || id === ids.unsafePolicy;
      const screen = id === ids.screen;
      const flagged = id === ids.flagged || id === ids.incompleteFlagged || food || screen;
      if (pathname.startsWith("/api/v1/jobs/")) {
        if (id === runningId) { runningPolls += 1; return route.fulfill({json:{job_id:id,status:"processing"}}); }
        return route.fulfill({json:{job_id:id,status:"done",verdict:flagged?"violation":"no_violation",violations:flagged?[{type:screen?"workstation_departure_screen_left_on":food?"personal_food_at_workstation":"example_supported_type",timecode_sec:screen?16:food?12.5:10,description:screen?"The workstation is unattended after departure and the same computer screen remains on.":food?"Personal food is visibly present on the cashier work table during work.":"Synthetic supported event.",evidence_frames:[{timestamp_sec:screen?16:food?12.5:10,image_base64:syntheticJpeg}]}]:[]}});
      }
      if (pathname.startsWith("/api/ui/jobs/")) {
        if (id === runningId) return route.fulfill({json:{job_id:id,status:"processing",stage:"verifying",progress:.54}});
        if (screen) return route.fulfill({json:{job_id:id,status:"done",stage:"done",progress:1,original_name:"synthetic.mov",details:screenDetails}});
        if (id === ids.groupedProbability) return route.fulfill({json:{job_id:id,status:"done",stage:"done",progress:1,original_name:"synthetic.mov",details:groupedProbabilityDetails}});
        if (food) return route.fulfill({json:{job_id:id,status:"done",stage:"done",progress:1,original_name:"synthetic.mov",details:id === ids.food ? foodDetails : {...foodDetails,findings:[{...foodDetails.findings[0],rule_url:unsafePolicyUrl}]}}});
        if (id === ids.unavailable && unavailableDetails) return route.fulfill({status:500,json:{detail:"Synthetic details failure"}});
        if ([ids.incomplete,ids.incompleteFlagged,ids.incompleteMarker,ids.incompleteStatus,ids.incompleteCount].includes(id)) {
          const incomplete = id === ids.incompleteMarker ? {coverage:{verification_complete:false},case_decision:{cases:[]}} : id === ids.incompleteStatus ? {processing_status:"review_incomplete",case_decision:{cases:[]}} : id === ids.incompleteCount ? {coverage:{unverified_candidate_count:3},case_decision:{cases:[]}} : incompleteDetails;
          return route.fulfill({json:{job_id:id,status:"done",stage:"done",progress:1,original_name:"synthetic.mov",details:incomplete}});
        }
        const jobDetails = id===ids.grouped?groupedDetails:id===ids.probability?probabilityDetails:id===ids.dense?denseDetails:id===ids.dismissed?dismissedDetails:id===ids.visibility?{...dismissedDetails,coverage:{visibility_gaps:[{start_sec:1,end_sec:3}],unresolved_verification_count:0}}:id===ids.gap?{...dismissedDetails,coverage:{visibility_gaps:[],unresolved_verification_count:3}}:id===ids.empty?{...dismissedDetails,case_decision:{status:"review_required",counts:{review_required:99},cases:[]}}:id===ids.old?{}:id===ids.legacy?{review_queue:queue}:id===ids.clear?{detections:[],case_decision:{status:"no_supported_violation",cases:[]}}:id===ids.uncertain?{assessments:[{...assessments[2],automation:"review_only",assessment:"unresolved",unscoped_unknown_observation_count:1}]}:id===ids.manual?{assessments:[assessments[2]]}:details;
        return route.fulfill({json:{job_id:id,status:"done",stage:"done",progress:1,original_name:"synthetic.mov",details:jobDetails}});
      }
      return route.fulfill({status:404,body:"Synthetic route unavailable"});
    });
    await page.goto("http://forte-synthetic.test/");
    assert.equal(catalogRequests,0,"Do not request the protected catalog before a key is entered");
    assert.equal(activeJobRequests,0,"Do not discover a running job before a key is entered");
    assert.equal(await page.locator("#catalog-section").isVisible(),false);
    await page.fill("#api-key","synthetic-test-key");
    await page.waitForFunction(() => document.querySelectorAll("#catalog-items tr").length === 5);
    const catalogDisclosure = page.locator("#catalog-section").locator("xpath=ancestor-or-self::details[1]");
    assert.equal(await catalogDisclosure.evaluate(element => element.open),false,"The reference catalog should be collapsed by default");
    await catalogDisclosure.locator(":scope > summary").click();
    assert.deepEqual(await page.locator("#catalog-section thead th").allTextContents(),["Check","Result","Estimate"]);
    for (const row of await page.locator("#catalog-items tr").all()) {
      assert.equal(await row.locator("th,td").count(),3);
      assert.equal(await row.locator(".score-result").textContent(),"Not checked");
      assert.equal(await row.locator(".score-value").textContent(),"—","Unassessed baselines must not display apparent confidence");
    }
    assert.equal(await page.locator("#catalog-items img").count(),0);
    const policyCatalogLink = page.locator("#catalog-items a");
    assert.equal(await policyCatalogLink.getAttribute("href"),"/policies/workplace-belongings");
    assert.equal(await policyCatalogLink.textContent(),"Read workplace policy ↗");
    assert.match(await page.locator("#catalog-items tr").last().textContent(),/User-provided workplace policy/);
    await catalogDisclosure.locator(":scope > summary").click();
    await page.locator(".existing-job summary").click();
    const open = async id => {
      await page.fill("#job-input",id);
      await page.locator("#open-job-form button[type=submit]").click();
      await page.waitForFunction(() => !document.getElementById("results").hidden);
      await page.waitForFunction(() => document.getElementById("source-video").src.startsWith("blob:"));
    };
    await open(ids.v2);
    assert.equal(await page.locator("#verdict-title").textContent(),"Needs review");
    assert(await page.evaluate(() => Boolean(document.getElementById("results").compareDocumentPosition(document.querySelector(".video-card")) & Node.DOCUMENT_POSITION_FOLLOWING)),"The decision should precede the video in reading order");
    assert.match(await page.locator("#jury-verdict").textContent(),/no_violation.*0 confirmed findings/);
    const technicalDisclosure = page.locator("#jury-verdict").locator("xpath=ancestor::details[1]");
    assert.equal(await technicalDisclosure.evaluate(element => element.open),false,"Raw API and technical details should be collapsed by default");
    assert.equal(await page.locator("#jury-verdict").isVisible(),false);
    assert.equal(await page.locator(".case-card").count(),2);
    assert.deepEqual(await page.locator("#case-decision-section thead th").allTextContents(),["Check","Result","Probability","Time","Details"]);
    for (const row of await page.locator(".case-card").all()) {
      assert.equal(await row.evaluate(element => element.tagName),"TR");
      assert.equal(await row.locator("th,td").count(),5);
      assert.equal(await row.locator(".case-details").evaluate(element => element.open),false);
    }
    assert.match(await page.locator(".case-probability").nth(0).textContent(),/^74%.*Visible action only/);
    assert.match(await page.locator(".case-probability").nth(1).textContent(),/^12%.*All rule checks/,"A rejected proposal's estimate must not be inverted into confidence that the rejection is true");
    assert.equal(await page.locator(".case-detail-row").count(),2);
    for (const row of await page.locator(".case-detail-row").all()) assert.equal(await row.isVisible(),false);
    const detectionsDisclosure = page.locator("#detections-section").locator("xpath=ancestor-or-self::details[1]");
    assert.equal(await detectionsDisclosure.evaluate(element => element.open),false,"Detected item previews should be collapsed by default");
    assert.equal(await page.locator("#detection-controls").isVisible(),true,"The video box toggle stays available when previews are folded");
    await page.evaluate(() => window.scrollTo(0,0));
    await page.screenshot({path:path.join(root,"tmp","ui-simplified-desktop.png"),fullPage:true});
    await page.screenshot({path:path.join(root,"tmp","ui-simplified-desktop-viewport.png")});
    await page.locator(".case-details > summary").first().click();
    assert.equal(await page.locator(".case-details").first().evaluate(element => element.open),true);
    await page.locator(".case-detail-row").first().waitFor({state:"visible"});
    assert.equal(await page.locator(".case-detail-row").first().isVisible(),true);
    assert.equal(await page.locator(".case-detail-body").first().getAttribute("colspan"),"5","Expanded checks need the full result table width");
    assert.equal(await page.locator(".check-status.yes").isVisible(),true);
    await page.locator(".case-details > summary").nth(1).click();
    assert.equal(await page.locator(".case-card .time-button").count(),1,"Null case timestamps must not turn into zero-second seek controls");
    assert.equal(await page.locator(".check-status.yes").textContent(),"Yes");
    assert.equal(await page.locator(".check-status.unknown").textContent(),"Unknown");
    assert.equal(await page.locator(".check-status.no").textContent(),"No");
    assert.equal(await page.locator(".case-observation img").count(),0);
    assert.equal(await page.locator(".detection-card").count(),2,"Malformed boxes must be excluded");
    await detectionsDisclosure.locator(":scope > summary").click();
    await page.waitForFunction(() => document.querySelectorAll(".detection-frame .detection-box").length === 2);
    assert.equal(evidenceRequests,1,"Items sharing a source frame reuse one authenticated JPEG request");
    assert.equal(await page.locator(".detection-card h4 img").count(),0);
    await page.locator(".detection-card .time-button").first().click();
    assert.equal(await page.locator("#source-video").evaluate(video=>video.currentTime),10);
    assert.equal(await page.locator("#detection-overlay .detection-box").count(),2);
    const geometry = await page.locator("#detection-overlay").evaluate(overlay => {
      const video = document.querySelector("#source-video"), scale = Math.min(video.clientWidth / 1920,video.clientHeight / 1080), contentWidth=1920*scale,contentHeight=1080*scale;
      return {width:parseFloat(overlay.style.width),height:parseFloat(overlay.style.height),top:parseFloat(overlay.style.top),left:parseFloat(overlay.style.left),expectedWidth:contentWidth,expectedHeight:contentHeight,expectedTop:(video.clientHeight-contentHeight)/2,expectedLeft:(video.clientWidth-contentWidth)/2};
    });
    assert(Math.abs(geometry.width-geometry.expectedWidth)<1 && Math.abs(geometry.height-geometry.expectedHeight)<1 && Math.abs(geometry.top-geometry.expectedTop)<1 && Math.abs(geometry.left-geometry.expectedLeft)<1,"Boxes must align to the contained video rather than letterboxing: "+JSON.stringify(geometry));
    assert.equal(await page.locator("#detection-overlay .detection-box").first().evaluate(box=>box.style.left),"10%");
    await page.locator("#show-detections").uncheck();
    assert.equal(await page.locator("#detection-overlay").isVisible(),false);
    await page.locator("#show-detections").check();
    assert.equal(await page.locator("#detection-overlay .detection-box").count(),2);
    await page.locator("#source-video").evaluate(video=>{video.currentTime=10.2;});
    assert.equal(await page.locator("#detection-overlay .detection-box").count(),0,"Boxes must not carry forward into unsampled frames");
    await page.locator(".check-evidence button").first().click();
    assert.equal(await page.locator("#source-video").evaluate(video=>video.currentTime),10);
    await page.locator("#detections-section").screenshot({path:path.join(root,"tmp","ui-review-bounding-boxes.png")});
    await page.locator("#case-decision-section").screenshot({path:path.join(root,"tmp","ui-review-case-decisions.png")});
    assert.equal(await page.locator("#finding-count").textContent(),"0 events");
    await technicalDisclosure.locator(":scope > summary").click();
    assert.deepEqual(await page.locator("#assessments-section thead th").allTextContents(),["Check","Result","Estimate"]);
    assert.equal(await page.locator("#assessment-items tr").count(),2,"Only assessed or relevant checks should be shown initially");
    assert.match(await page.locator("#assessment-toggle").textContent(),/Show all checks/);
    await page.locator("#assessment-toggle").click();
    assert.equal(await page.locator("#assessment-items tr").count(),4);
    for (const row of await page.locator("#assessment-items tr").all()) {
      assert.equal(await row.locator("th,td").count(),3);
      assert.equal(await row.locator(".score-details").evaluate(element => element.open),false);
    }
    assert.match(await page.locator("#assessment-items tr").nth(0).locator(".score-value").textContent(),/^68%\s*event support$/);
    assert.match(await page.locator("#assessment-items tr").nth(1).locator(".score-value").textContent(),/^82%\s*completion support$/);
    assert.equal(await page.locator("#assessment-items tr").nth(2).locator(".score-value").textContent(),"—");
    assert.equal(await page.locator("#assessment-items tr").nth(2).locator(".score-result").textContent(),"Not checked");
    assert.equal(await page.locator("#assessment-items tr").nth(3).locator(".score-value").textContent(),"—");
    assert.match(await page.locator("#assessment-items tr").nth(3).locator(".score-result").textContent(),/Not applicable|N\/A/);
    assert(!/100%/.test(await page.locator("#assessment-items").innerText()),"Untouched procedure baselines must not imply a 100% finding or confidence");
    await page.locator("#assessment-items tr").nth(0).locator(".score-details > summary").click();
    await page.locator("#assessment-items tr").nth(1).locator(".score-details > summary").click();
    await page.locator("#assessment-items tr").nth(2).locator(".score-details > summary").click();
    const eventDetails = await page.locator("#assessment-items tr").nth(0).locator(".score-details").textContent();
    for (const fact of [/2 evidence references/, /Event confidence: 68%/, /Strongest event evidence in this clip/, /Review only/]) assert.match(eventDetails,fact);
    const procedureDetails = await page.locator("#assessment-items tr").nth(1).locator(".score-details").textContent();
    for (const fact of [/Correct-procedure confidence: 82%/, /Highest remaining estimate across assessed transactions/, /2 episodes · 2 pending or unknown/, /1 uncertain windows? .*no established procedure trigger; existing completion evidence is retained/, /Incomplete coverage: 1 known episodes? .*no assessment for this check; applicability may be unknown/, /Window-scoped evidence · transaction boundary unconfirmed/, /Latest assessed episode.*00:02–00:04/]) assert.match(procedureDetails,fact);
    const manualDetails = await page.locator("#assessment-items tr").nth(2).locator(".score-details").textContent();
    for (const fact of [/0 evidence references/, /Verify each applicable transaction/, /Manual/]) assert.match(manualDetails,fact);
    assert(!/event-one|procedure-one|procedure-manual/.test(await page.locator("#assessment-items").innerText()),"Raw catalog identifiers must not replace readable check labels");
    await page.locator("#assessment-items tr").nth(1).locator(".check-evidence button").click();
    assert.equal(await page.locator("#source-video").evaluate(video=>video.currentTime),3);
    assert.match(await page.locator("#catalog-status").textContent(),/5 catalog items.*1 event checks eligible/);
    await page.locator(".score-episodes summary").click();
    assert.match(await page.locator(".score-episodes ul").textContent(),/Episode 1.*18%.*correct-procedure confidence 82%.*Episode 2.*8%.*correct-procedure confidence 92%/);
    assert.equal(await page.locator(".finding-card").count(),0);
    assert.equal(await page.locator(".review-item").count(),1,"Rejected queue history must not remain an active review card");
    assert.equal(await page.locator("#review-queue-count").textContent(),"1 item");
    assert.match(await page.locator("#verdict-description").textContent(),/1 case needs a check/);
    assert.equal(await page.locator(".review-item-message").textContent(),"The visible observation needs a human policy check.");
    assert.equal(await page.locator("#coverage-metrics .coverage-metric").count(),6);
    assert.match(await page.locator("#coverage-metrics").textContent(),/4,273/);
    assert.match(await page.locator("#coverage-metrics").textContent(),/0\.25 s/);
    assert.equal(await page.locator(".review-observation img").count(),0);
    assert.match(await page.locator(".review-observation").textContent(),/<img src=x/);
    assert.equal(await page.evaluate(() => window.__injected),undefined);
    await page.locator(".review-item .time-button").first().click();
    assert.equal(await page.locator("#source-video").evaluate(video=>video.currentTime),10);
    await technicalDisclosure.locator(":scope > summary").click();
    await detectionsDisclosure.locator(":scope > summary").click();
    for (const summary of await page.locator(".case-details > summary").all()) await summary.click();
    for (const row of await page.locator(".case-detail-row").all()) await row.waitFor({state:"hidden"});
    await page.setViewportSize({width:375,height:812});
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth+1),"Mobile layout must not overflow horizontally");
    const mobileTime = await page.locator(".case-card .time-button").first().boundingBox();
    const mobileView = await page.locator(".case-card .case-details > summary").first().boundingBox();
    assert(mobileTime && mobileView && mobileTime.x + mobileTime.width + 3 <= mobileView.x,"The probability column must preserve a readable gap between mobile Time and Details controls");
    await page.evaluate(() => window.scrollTo(0,0));
    const mobileVerdict = await page.locator("#verdict-card").boundingBox();
    assert(mobileVerdict && mobileVerdict.y >= 0 && mobileVerdict.y < 812,"The completed decision must appear in the first mobile viewport");
    await page.screenshot({path:path.join(root,"tmp","ui-simplified-mobile.png"),fullPage:true});
    await page.screenshot({path:path.join(root,"tmp","ui-simplified-mobile-viewport.png")});
    await page.locator(".case-details > summary").first().click();
    await page.locator(".case-detail-row").first().waitFor({state:"visible"});
    assert.equal(await page.locator(".case-detail-row").first().isVisible(),true);
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth+1),"Expanded mobile case details must not overflow horizontally");
    await page.locator("#case-decision-section").screenshot({path:path.join(root,"tmp","ui-simplified-mobile-expanded.png")});
    await open(ids.old);
    assert.equal(await page.locator("#verdict-title").textContent(),"No violation detected");
    assert.match(await page.locator("#verdict-description").textContent(),/sampled (?:footage|video)/);
    assert.equal(await page.locator("#review-queue-section").isVisible(),false);
    assert.equal(await page.locator("#coverage-section").isVisible(),false);
    assert.equal(await page.locator("#assessments-section").isVisible(),false);
    assert.equal(await page.locator(".review-item").count(),0);
    assert.equal(await page.locator("#detections-section").isVisible(),false);
    assert.equal(await page.locator("#case-decision-section").isVisible(),false);
    assert.equal(await page.locator("#detection-overlay .detection-box").count(),0);
    await open(ids.legacy);
    assert.equal(await page.locator("#verdict-title").textContent(),"Needs review","Older jobs with an unconfirmed queue must not display a clean headline");
    assert.equal(await page.locator("#case-decision-section").isVisible(),true);
    assert.equal(await page.locator(".case-card").count(),2,"Older review queues should remain actionable in the main case table");
    for (const cell of await page.locator(".case-probability").all()) assert.equal(await cell.textContent(),"Not assessed","A legacy result must not acquire a fabricated probability");
    await open(ids.probability);
    const probabilityCells = page.locator(".case-probability");
    assert.equal(await probabilityCells.count(),10);
    assert.match(await probabilityCells.nth(0).textContent(),/^0%.*Event occurs/,"Zero is a valid model estimate");
    assert.match(await probabilityCells.nth(1).textContent(),/^100%.*All rule checks/);
    for (const index of [2,3,4,5,6,7,9]) assert.equal(await probabilityCells.nth(index).textContent(),"Not assessed","Unknown or invalid values must not borrow the same check's separate episode estimate: " + index);
    assert.match(await probabilityCells.nth(8).textContent(),/^82%.*Procedure completed/,"Required-procedure confidence must identify completion as the target");
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth+1),"The probability column must fit the mobile viewport");
    await open(ids.grouped);
    assert.equal(await page.locator(".case-card").count(),6,"Repeated presence checks should share one row while independent actions and conflicting results remain separate");
    assert.deepEqual(await page.locator(".case-card .case-status").allTextContents(),["Needs review","Needs review","Needs review","Violation","Not supported","Not supported"],"Needs-review checks must precede violations and unsupported results even when the source list starts with a rejected check");
    assert.match(await page.locator("#case-decision-count").textContent(),/^6 checks$/,"The header counts displayed result rows");
    const groupedCard = page.locator(".case-card").first();
    assert.match(await groupedCard.locator(".check-name").textContent(),/^Personal belongings in cash unit/);
    assert.equal(await groupedCard.locator(".case-observation-count").textContent(),"4 observations");
    assert.equal(await groupedCard.locator(".case-time .time-button").textContent(),"▶ 00:00–00:17");
    const rangeGeometry = await groupedCard.locator(".case-time").evaluate(cell => {
      const time = cell.querySelector(".time-button").getBoundingClientRect(), bounds = cell.getBoundingClientRect(), details = cell.nextElementSibling.getBoundingClientRect();
      return {timeLeft:time.left,timeRight:time.right,cellLeft:bounds.left,cellRight:bounds.right,detailsLeft:details.left};
    });
    assert(rangeGeometry.timeLeft >= rangeGeometry.cellLeft-1 && rangeGeometry.timeRight <= rangeGeometry.cellRight+1 && rangeGeometry.timeRight <= rangeGeometry.detailsLeft+1,"The full interval must fit inside the mobile Time cell without overlapping View: " + JSON.stringify(rangeGeometry));
    assert.equal(await groupedCard.locator(".case-probability-value").textContent(),"Not assessed","Grouping must not create a probability for unassessed checks");
    await groupedCard.locator(".case-time .time-button").click();
    assert.equal(await page.locator("#source-video").evaluate(video=>video.currentTime),0,"A grouped interval seeks to its first observation");
    await groupedCard.locator(".case-details > summary").click();
    const groupBody = page.locator(".case-detail-body").first();
    await groupBody.waitFor({state:"visible"});
    assert.equal(await groupBody.locator(".case-occurrence").count(),4,"Every original occurrence remains available in the expanded result");
    assert.match(await groupBody.locator(".case-group-note").textContent(),/sampled|sampling/i);
    assert.match(await groupBody.locator(".case-group-note").textContent(),/continuity|continuous|between/i,"The observed span must not imply uninterrupted presence");
    assert.equal(await groupBody.locator(".case-checks li").count(),4,"Grouping must preserve every occurrence's check details");
    for (const [index,time] of [0,8,13,17].entries()) {
      const occurrence = groupBody.locator(".case-occurrence").nth(index);
      assert.match(await occurrence.textContent(),new RegExp("Synthetic sampled presence check " + (index+1)));
      assert.equal(await occurrence.locator(".case-occurrence-probability").textContent(),"Not assessed");
      await occurrence.locator(".case-occurrence-heading .time-button").click();
      assert.equal(await page.locator("#source-video").evaluate(video=>video.currentTime),time,"Every original observation keeps its own working source timestamp");
    }
    assert.doesNotMatch(await groupBody.textContent(),/Uncertain synthetic text must stay hidden/);
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth+1),"Grouped intervals and expanded observations fit a mobile viewport");
    await page.locator("#case-decision-section").screenshot({path:path.join(root,"tmp","ui-grouped-results-mobile.png")});
    await open(ids.groupedProbability);
    assert.equal(await page.locator(".case-card").count(),4,"Different probability targets and missing timestamps remain separate result rows");
    assert.equal(await page.locator(".case-probability-value").nth(0).textContent(),"60%–80%","A grouped row shows the member estimate range rather than a fabricated aggregate");
    assert.match(await page.locator(".case-probability").nth(0).textContent(),/Visible action only/);
    assert.match(await page.locator(".case-probability").nth(1).textContent(),/^95%.*All rule checks/);
    assert.equal(await page.locator(".case-card .case-time .time-button").count(),2,"Missing timestamps must not become zero-second intervals");
    await page.locator(".case-details > summary").first().click();
    assert.deepEqual(await page.locator(".case-detail-body").first().locator(".case-occurrence-probability").allTextContents(),["60% · Visible action only","80% · Visible action only"],"Member estimates keep their target scope in Details");
    await open(ids.clear);
    assert.equal(await page.locator("#verdict-title").textContent(),"No violation detected");
    assert.match(await page.locator("#detections").textContent(),/No items were localized/);
    await open(ids.unavailable);
    assert.equal(await page.locator("#verdict-title").textContent(),"Needs review");
    assert.equal(await page.locator("#details-warning").isVisible(),true);
    assert.match(await page.locator("#details-warning").textContent(),/could not be loaded/);
    assert.equal(await page.locator("#poll-button").textContent(),"Retry review details");
    unavailableDetails = false;
    await page.locator("#poll-button").click();
    await page.waitForFunction(() => document.getElementById("details-warning").hidden);
    assert.equal(await page.locator(".case-card").count(),2);
    await open(ids.uncertain);
    assert.equal(await page.locator("#verdict-title").textContent(),"Needs review","An unresolved assessed check must remain visibly pending");
    await technicalDisclosure.locator(":scope > summary").click();
    assert.equal(await page.locator("#assessment-items tr").count(),1);
    assert.equal(await page.locator("#assessment-items .score-result").textContent(),"Needs review");
    assert.equal(await page.locator("#assessment-items .score-value").textContent(),"—","An unresolved check has no usable confidence estimate");
    await technicalDisclosure.locator(":scope > summary").click();
    await open(ids.manual);
    assert.equal(await page.locator("#verdict-title").textContent(),"No violation detected","An untouched manual baseline must not create a review case");
    await open(ids.dismissed);
    assert.equal(await page.locator("#verdict-title").textContent(),"No violation detected","Rejected and explicitly dismissed cases must not inherit a stale review status or count");
    assert.equal(await page.locator(".review-item").count(),0);
    assert.deepEqual(await page.locator(".case-status").allTextContents(),["Not supported","Not supported"]);
    assert.equal(await page.locator(".case-observation").count(),0,"Uncertain and negative model prose must not be shown as observed action");
    assert(!/blah blah|No cash-like bundle|99/.test(await page.locator("#results").textContent()));
    await open(ids.gap);
    assert.equal(await page.locator("#verdict-title").textContent(),"Needs review","Dismissing unsupported candidates must preserve checks the evidence could not establish");
    assert.match(await page.locator("#verdict-description").textContent(),/Review details need a check/);
    assert.equal(await page.locator(".review-item").count(),0);
    assert.match(await page.locator("#coverage-metrics").textContent(),/Checks not established3/);
    await open(ids.visibility);
    assert.equal(await page.locator("#verdict-title").textContent(),"Needs review","Actual visibility gaps remain pending even without unresolved verifications or active candidates");
    assert.match(await page.locator("#coverage-metrics").textContent(),/Intervals with limited view1/);
    assert(!/Checks not established/.test(await page.locator("#coverage-metrics").textContent()),"Zero unresolved checks need no extra coverage metric");
    await open(ids.empty);
    assert.equal(await page.locator("#verdict-title").textContent(),"No violation detected","An explicit empty host case list must not revive legacy queued guesses");
    assert.equal(await page.locator(".case-card").count(),0);
    assert.equal(await page.locator(".review-item").count(),0);
    for (const id of [ids.incomplete,ids.incompleteMarker,ids.incompleteStatus,ids.incompleteCount]) {
      await open(id);
      assert.equal(await page.locator("#verdict-title").textContent(),"Needs review","Incomplete verification must not display a clean result");
      assert.equal(await page.locator("#status-title").textContent(),"Review incomplete");
      assert.equal(await page.locator("#status-title").isVisible(),true,"Incomplete status must stay visible above results");
      assert.equal(await page.locator("#status-badge").textContent(),"Incomplete");
      assert.match(await page.locator("#verdict-description").textContent(),/candidate checks were not completed within the processing budget/);
      assert.equal(await page.locator(".case-card").count(),0,"Budget-exhausted diagnostic candidates must not become review cases");
      assert.equal(await page.locator(".review-item").count(),0);
      assert.match(await page.locator("#jury-verdict").textContent(),/no_violation.*0 confirmed findings/);
      if (id === ids.incomplete) {
        assert.match(await page.locator("#coverage-metrics").textContent(),/Candidate checks not completed3/);
        assert.match(await page.locator("#coverage-metrics").textContent(),/Candidate verificationIncomplete/);
      }
    }
    await open(ids.incompleteFlagged);
    assert.equal(await page.locator("#verdict-title").textContent(),"Violation detected","Incomplete checks must preserve independently confirmed findings");
    assert.equal(await page.locator("#status-title").textContent(),"Review incomplete");
    assert.match(await page.locator("#jury-verdict").textContent(),/violation.*1 confirmed finding/);
    await open(ids.dense);
    assert.equal(await page.locator("#detection-count").textContent(),"50 localized items");
    assert.equal(await page.locator(".detection-card").count(),24,"Dense perception results initially render one page of 24 cards");
    await page.waitForFunction(() => document.querySelectorAll(".detection-frame .detection-box").length === 24);
    assert.deepEqual(denseEvidenceRequests,["dense-0.jpg"],"The first page must not fetch JPEGs used only by later pages");
    if (!await detectionsDisclosure.evaluate(element => element.open)) await detectionsDisclosure.locator(":scope > summary").click();
    const showMore = page.getByRole("button",{name:/Show more sampled items/});
    assert.equal(await showMore.textContent(),"Show more sampled items (24 of 50)");
    await showMore.click();
    assert.equal(await page.locator(".detection-card").count(),48,"The next page appends rather than replacing the first cards");
    await page.waitForFunction(() => document.querySelectorAll(".detection-frame .detection-box").length === 48);
    assert.deepEqual(denseEvidenceRequests,["dense-0.jpg","dense-1.jpg"],"Shared images are fetched once per page group and the final page stays deferred");
    assert.equal(await showMore.textContent(),"Show more sampled items (48 of 50)");
    await showMore.click();
    assert.equal(await page.locator(".detection-card").count(),50,"The final partial page contains only its two remaining cards");
    await page.waitForFunction(() => document.querySelectorAll(".detection-frame .detection-box").length === 50);
    assert.deepEqual(denseEvidenceRequests,["dense-0.jpg","dense-1.jpg","dense-2.jpg"],"All 50 cards reuse exactly three authenticated source JPEGs");
    assert.equal(await showMore.isVisible(),false,"Pagination stops after the final item");
    const denseSources = await page.locator(".detection-card img").evaluateAll(images => images.map(image => image.src));
    assert.equal(denseSources.length,50);
    assert(denseSources.every(source => source.startsWith("blob:")),"Protected media must use authenticated blob previews");
    assert.equal(new Set(denseSources).size,3,"Cards sharing a source frame reuse the same cached object URL");
    assert.equal(await page.locator(".detection-card h4 img").count(),0,"Detection labels remain plain text on later pages");
    assert.match(await page.locator(".detection-card h4").last().textContent(),/<img src=x/);
    await page.locator(".detection-card .time-button").last().click();
    assert.equal(await page.locator("#source-video").evaluate(video=>video.currentTime),30);
    assert.equal(await page.locator("#detection-overlay .detection-box").count(),2,"Final-page seeking uses all detections at the exact source moment");
    await page.locator(".case-details > summary").first().click();
    await page.locator(".case-detail-row").first().waitFor({state:"visible"});
    const exception = page.locator(".case-checks li").first();
    assert.match(await exception.locator(".check-label").textContent(),/^Exception excluded · Synthetic occlusion/);
    assert.equal(await exception.locator(".check-status").textContent(),"Yes");
    assert.equal(await exception.locator(".check-label img").count(),0,"Exception names must not inject markup");
    assert.equal(await exception.locator(".check-evidence button").count(),3,"Exception evidence drops invalid timestamps and caps controls at three source references");
    await exception.locator(".check-evidence button").nth(1).click();
    assert.equal(await page.locator("#source-video").evaluate(video=>video.currentTime),20.25,"Exception-exclusion evidence seeks its own source timestamp");
    assert.equal(await page.locator("#detection-overlay .detection-box").count(),0,"Dense source boxes do not persist at an unsampled exception moment");
    const coverageMetrics = await page.locator("#coverage-metrics .coverage-metric").evaluateAll(metrics => Object.fromEntries(metrics.map(metric => [metric.querySelector("dt").textContent,metric.querySelector("dd").textContent])));
    assert.equal(Object.keys(coverageMetrics).length,10);
    assert.equal(coverageMetrics["Motion and person samples"],"713");
    assert.equal(coverageMetrics["Perception sample gap"],"0.20 s");
    assert.equal(coverageMetrics["Tentative transaction episodes"],"3");
    assert.equal(coverageMetrics["Unresolved scene regions"],"2");
    assert.equal(coverageMetrics["Context windows checked"],"3","Array-based context coverage remains compatible");
    assert.equal(await page.evaluate(() => window.__denseInjected),undefined);
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth+1),"Expanded dense detections and exception details must fit the mobile viewport");
    await open(ids.flagged);
    assert.equal(await page.locator("#finding-count").textContent(),"1 event");
    assert.equal(await page.locator(".finding-card").count(),1);
    assert.equal(await page.locator(".review-item").count(),1);
    assert.equal(await page.locator("#verdict-title").textContent(),"Violation detected");
    const sourceLink = page.locator(".finding-card .rule-box a");
    assert.equal(await sourceLink.getAttribute("href"),"https://adilet.zan.kz/rus/docs/V1900019680#z114");
    assert.match(await sourceLink.getAttribute("rel"),/noopener/);
    await page.locator(".finding-card .evidence-button").click();
    assert.equal(await page.locator("#source-video").evaluate(video=>video.currentTime),10);
    await open(ids.food);
    assert.equal(await page.locator("#verdict-title").textContent(),"Violation detected");
    assert.equal(await page.locator(".finding-card h3").textContent(),"Personal belongings at workplace (food)");
    assert.match(await page.locator(".finding-card").textContent(),/Workplace policy violation/);
    assert.match(await page.locator(".finding-card .rule-box").textContent(),/User-provided workplace policy.*Section Personal food at the cashier workstation/);
    assert.doesNotMatch(await page.locator(".finding-card").textContent(),/NBK|Paragraph|official source|regulatory reference/);
    assert.equal(await page.locator(".finding-card .rule-box a").textContent(),"Read workplace policy ↗");
    assert.equal(await page.locator(".finding-card .rule-box a").getAttribute("href"),"/policies/workplace-belongings");
    assert.equal(await page.locator(".finding-card .evidence-button img").count(),1);
    assert.deepEqual(await page.locator("#case-decision-section thead th").allTextContents(),["Check","Result","Probability","Time","Details"]);
    assert.match(await page.locator(".case-card .check-name").textContent(),/^Personal belongings at workplace \(food\)/);
    assert.equal(await page.locator(".case-card .case-status").textContent(),"Violation");
    assert.match(await page.locator(".case-probability").textContent(),/^94%.*Event occurs/);
    assert.equal(await page.locator(".case-card .time-button").count(),1);
    await page.locator(".case-card .time-button").click();
    assert.equal(await page.locator("#source-video").evaluate(video=>video.currentTime),12.5,"A food-presence violation has a working result-table timestamp");
    await page.locator(".finding-card .evidence-button").click();
    assert.equal(await page.locator("#source-video").evaluate(video=>video.currentTime),12.5,"Food presence evidence pinpoints its source moment");
    await open(ids.screen);
    assert.equal(await page.locator("#verdict-title").textContent(),"Violation detected");
    assert.equal(await page.locator(".finding-card h3").textContent(),"Workstation left with computer screen on");
    assert.match(await page.locator(".finding-card").textContent(),/Workplace policy violation/);
    assert.equal(await page.locator(".finding-card .rule-box a").getAttribute("href"),"/policies/workplace-screen");
    assert.equal(await page.locator(".finding-card .rule-box a").textContent(),"Read workplace policy ↗");
    assert.doesNotMatch(await page.locator(".finding-card").textContent(),/NBK|Paragraph|official source|regulatory reference/);
    assert.match(await page.locator(".case-card .check-name").textContent(),/^Workstation left with computer screen on/);
    assert.equal(await page.locator(".case-card .case-status").textContent(),"Violation");
    assert.match(await page.locator(".case-probability").textContent(),/^91%/);
    await page.locator(".case-card .time-button").click();
    assert.equal(await page.locator("#source-video").evaluate(video=>video.currentTime),16,"Screen-on violation pinpoints the post-departure source moment");
    await page.locator(".finding-card .evidence-button").click();
    assert.equal(await page.locator("#source-video").evaluate(video=>video.currentTime),16,"Screen-on evidence opens the post-departure source frame");
    unsafePolicyUrl = "/policies/workplace-eating";
    await open(ids.unsafePolicy);
    assert.equal(await page.locator(".finding-card .rule-box a").getAttribute("href"),"/policies/workplace-eating","Historical workplace policy references remain readable");
    for (const unsafeUrl of ["javascript:alert(1)","//evil.example/policies/workplace-belongings","http://forte-synthetic.test/policies/workplace-belongings","http://forte-synthetic.test:8123/policies/workplace-belongings","/policies/workplace-belongings?redirect=evil","/policies/workplace-belongings#fragment","/policies/../workplace-belongings","/policies/workplace-eating?redirect=evil","/policies/workplace-eating#fragment","/policies/workplace-screen?redirect=evil","/policies/workplace-screen#fragment","/other-policy","https://user:secret@adilet.zan.kz/rus/docs/V1900019680","https://adilet.zan.kz.evil.example/rule"]) {
      unsafePolicyUrl = unsafeUrl;
      await open(ids.unsafePolicy);
      assert.equal(await page.locator(".finding-card .rule-box a").count(),0,"Only the exact local workplace policy route and trusted official hosts may be linked: " + unsafeUrl);
    }
    await page.fill("#api-key","");
    assert.equal(await page.locator("#catalog-section").isVisible(),false);
    assert.equal(await page.locator("#catalog-items tr").count(),0);
    await page.fill("#api-key","wrong-synthetic-key");
    await page.waitForFunction(() => document.getElementById("catalog-status").textContent.includes("not accepted"));
    assert.equal(await page.locator("#catalog-items tr").count(),0);
    // A refreshed browser should reconnect to the existing worker job, rather than upload it again.
    activeJob = runningId; healthBusy = true;
    await page.reload();
    await page.waitForFunction(() => document.getElementById("service-label").textContent === "Analysis running");
    await page.fill("#api-key","synthetic-test-key");
    await page.waitForFunction(id => document.getElementById("job-id").textContent === id && document.getElementById("stage-label").textContent === "Verifying",runningId);
    assert.equal(await page.locator("#progress-track").getAttribute("aria-valuenow"),"54");
    assert.equal(await page.locator("#analyze-button").isDisabled(),true);
    assert.equal(analyzeRequests,0,"Resuming the active job must not submit another upload");
    await page.locator("#poll-button").click();
    const pausedPolls = runningPolls;
    await page.locator("#api-key").dispatchEvent("input");
    await page.waitForTimeout(600);
    assert.equal(runningPolls,pausedPolls,"Active-job discovery must respect paused browser updates");
    assert.equal(await page.locator("#status-title").textContent(),"Updates paused");
    await page.locator(".existing-job summary").click();
    await open(ids.clear);
    await page.locator("#api-key").dispatchEvent("input");
    await page.waitForSelector("#open-active-job");
    assert.equal(await page.locator("#job-id").textContent(),ids.clear,"Discovery must not replace a completed review the user is reading");
    // Discovering work after a file selection preserves that selection until the user opens progress.
    await page.reload();
    await page.waitForFunction(() => document.getElementById("service-label").textContent === "Analysis running");
    await page.setInputFiles("#video-file",{name:"synthetic-new.mov",mimeType:"video/quicktime",buffer:Buffer.from("synthetic media")});
    await page.fill("#api-key","synthetic-test-key");
    await page.waitForSelector("#open-active-job");
    assert.equal(await page.locator("#file-name").textContent(),"synthetic-new.mov");
    assert.equal(await page.locator("#analyze-button").isDisabled(),true);
    assert.match(await page.locator("#admission-note").textContent(),/An analysis is already running/);
    await page.locator("#upload-form").evaluate(form => form.dispatchEvent(new Event("submit",{cancelable:true})));
    assert.equal(analyzeRequests,0,"A busy worker must also reject programmatic form submission locally");
    await page.locator("#open-active-job").click();
    await page.waitForFunction(id => document.getElementById("job-id").textContent === id,runningId);
    // Accepted work blocks another upload even if updates are paused before the next health check.
    activeJob = null; healthBusy = false; acceptUpload = true;
    await page.reload();
    await page.fill("#api-key","synthetic-test-key");
    await page.waitForTimeout(600);
    await page.setInputFiles("#video-file",{name:"synthetic-accepted.mov",mimeType:"video/quicktime",buffer:Buffer.from("synthetic media")});
    await page.locator("#analyze-button").click();
    await page.waitForFunction(id => document.getElementById("job-id").textContent === id,runningId);
    await page.locator("#poll-button").click();
    assert.equal(await page.locator("#status-title").textContent(),"Updates paused");
    assert.equal(await page.locator("#analyze-button").isDisabled(),true,"Pausing accepted work must not reopen admission before the next health check");
    // A second client can occupy capacity after the health check. Recover progress from that 429 too.
    acceptUpload = false;
    activeJob = null; healthBusy = false;
    await page.reload();
    await page.fill("#api-key","synthetic-test-key");
    await page.waitForTimeout(600);
    await page.setInputFiles("#video-file",{name:"synthetic-race.mov",mimeType:"video/quicktime",buffer:Buffer.from("synthetic media")});
    await page.locator("#analyze-button").click();
    await page.waitForSelector("#open-active-job");
    assert.equal(analyzeRequests,2);
    assert.match(await page.locator("#form-error").textContent(),/An analysis is already running/);
    assert.doesNotMatch(await page.locator("#form-error").textContent(),/queue is full/);
    assert.equal(await page.locator("#file-name").textContent(),"synthetic-race.mov");
    assert.equal(await page.locator("#analyze-button").isDisabled(),true);
    const waitForHeldDiscovery = async () => {
      for (let attempt = 0; !releaseDiscovery && attempt < 100; attempt += 1) await new Promise(resolve => setTimeout(resolve,20));
      assert(releaseDiscovery,"The synthetic discovery request must reach its hold point");
    };
    // Opening a saved review while discovery is in flight keeps that deliberate selection.
    activeJob = null; healthBusy = false;
    await page.reload();
    await page.waitForFunction(() => document.getElementById("service-label").textContent === "Worker ready");
    activeJob = runningId; holdDiscovery = true; releaseDiscovery = null;
    await page.fill("#api-key","synthetic-test-key");
    await waitForHeldDiscovery();
    await page.locator(".existing-job summary").click();
    await open(ids.clear);
    releaseDiscovery(); releaseDiscovery = null;
    await page.waitForSelector("#open-active-job");
    assert.equal(await page.locator("#job-id").textContent(),ids.clear,"An in-flight lookup must not replace a manually opened review");
    // A key edit cancels an older authenticated lookup and must not expose its saved job ID.
    activeJob = null; healthBusy = false;
    await page.reload();
    await page.waitForFunction(() => document.getElementById("service-label").textContent === "Worker ready");
    activeJob = runningId; holdDiscovery = true; releaseDiscovery = null;
    await page.fill("#api-key","synthetic-test-key");
    await waitForHeldDiscovery();
    await page.fill("#api-key","");
    releaseDiscovery(); releaseDiscovery = null;
    await page.waitForTimeout(200);
    assert.equal(await page.locator("#job-id").textContent(),"");
    assert.equal(await page.locator("#open-active-job").count(),0,"Stale authenticated discoveries must be discarded after a key edit");
    assert.deepEqual(pageErrors,[]);
    console.log("Synthetic UI checks passed: concise decision-first layout, review-first grouped presence intervals with original observations and estimate scopes retained, folded technical/catalog/item details, support estimates without baseline false certainty, expandable cases, food-belongings and screen-on workplace-policy findings, source links and seeking, legacy/retry behavior, safe authenticated frames, mobile layout, dense pagination, hybrid coverage, busy admission, authenticated running-job recovery, paused/completed/file-selection preservation, 429 races, and stale discovery protection.");
  } finally { await browser.close(); }
})().catch(error=>{console.error(error.stack || error.message);process.exitCode=1;});
