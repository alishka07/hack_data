"use strict";

// Bind interface copy to its source text, so switching languages never rebuilds
// forms, replaces video sources or changes persisted review data.
(() => {
  const preferenceKey = "forte-ui-language";
  const catalog = window.ForteTranslations || {exact:{},patterns:[]};
  const bindings = new Map();
  let language = "ru";
  try { if (localStorage.getItem(preferenceKey) === "en") language = "en"; } catch (_) { /* Storage is optional. */ }

  function english(value) {
    const source = String(value ?? "");
    const text = source.trim();
    if (!text || !/[А-Яа-яЁё]/u.test(text)) return source;
    let result = Object.hasOwn(catalog.exact, text) ? catalog.exact[text] : undefined;
    if (result === undefined) {
      for (const [pattern, replacement] of catalog.patterns) {
        pattern.lastIndex = 0;
        if (pattern.test(text)) { pattern.lastIndex = 0; result = text.replace(pattern, replacement); break; }
      }
    }
    return result === undefined ? source : source.slice(0, source.indexOf(text)) + result + source.slice(source.indexOf(text) + text.length);
  }
  function translate(value) { return language === "en" ? english(value) : String(value ?? ""); }
  function bind(target, key, source, apply) {
    let values = bindings.get(target);
    if (!values) { values = new Map(); bindings.set(target, values); }
    values.set(key, {source:String(source ?? ""),apply});
    apply(translate(source));
  }
  function setText(target, value, localize = true) {
    if (!localize) { bindings.get(target)?.delete("text"); target.textContent = String(value ?? ""); return; }
    // Bind the text node itself: later appends (icons, inputs, time labels) stay
    // intact when the language changes.
    if (target.nodeType === Node.TEXT_NODE) {
      bind(target, "text", value, translated => { target.textContent = translated; });
    } else {
      const text = document.createTextNode(""); target.replaceChildren(text);
      bind(text, "text", value, translated => { text.textContent = translated; });
    }
  }
  function textNode(value, localize = true) {
    const text = document.createTextNode(""); setText(text, value, localize); return text;
  }
  function setAttribute(target, name, value) {
    bind(target, "attr:" + name, value, translated => { target.setAttribute(name, translated); });
  }
  function sourceAttribute(target, name) { return bindings.get(target)?.get("attr:" + name)?.source ?? target.getAttribute(name); }
  function sourceText(target) {
    if (target.nodeType === Node.TEXT_NODE) return bindings.get(target)?.get("text")?.source ?? target.textContent;
    return [...target.childNodes].map(sourceText).join("");
  }
  function registerStaticCopy() {
    const walker = document.createTreeWalker(document.documentElement, NodeFilter.SHOW_TEXT);
    const texts = [];
    while (walker.nextNode()) {
      const text = walker.currentNode;
      if (/[А-Яа-яЁё]/u.test(text.textContent) && !text.parentElement?.closest("script,style,textarea,[data-no-i18n]")) texts.push(text);
    }
    texts.forEach(text => setText(text, text.textContent));
    for (const element of document.querySelectorAll("[placeholder],[aria-label],[title],[alt],[data-label]")) {
      if (element.closest("[data-no-i18n]")) continue;
      for (const name of ["placeholder","aria-label","title","alt","data-label"]) {
        if (element.hasAttribute(name)) setAttribute(element, name, element.getAttribute(name));
      }
    }
  }
  function updateSelection() {
    document.documentElement.lang = language;
    for (const button of document.querySelectorAll("#language-switcher [data-language]")) button.setAttribute("aria-pressed", String(button.dataset.language === language));
  }
  function setLanguage(value) {
    if (!["ru","en"].includes(value) || value === language) return;
    language = value;
    try { localStorage.setItem(preferenceKey, language); } catch (_) { /* Switching still works without persistence. */ }
    for (const [target, values] of bindings) {
      if (!target.isConnected) { bindings.delete(target); continue; }
      for (const binding of values.values()) binding.apply(translate(binding.source));
    }
    updateSelection();
    document.dispatchEvent(new CustomEvent("forte-language-change", {detail:{language}}));
  }
  window.ForteI18n = {translate,english,setText,textNode,setAttribute,sourceAttribute,sourceText,setLanguage,get language(){return language;}};
  registerStaticCopy(); updateSelection();
  // Polling and filters replace many nodes. Discard their bindings without
  // observing field values or retaining old analyses in the translation map.
  function forgetRemoved(target) {
    if (target.isConnected) return;
    bindings.delete(target);
    for (const child of target.childNodes) forgetRemoved(child);
  }
  new MutationObserver(records => {
    for (const record of records) for (const removed of record.removedNodes) forgetRemoved(removed);
  }).observe(document.documentElement, {childList:true,subtree:true});
  for (const button of document.querySelectorAll("#language-switcher [data-language]")) button.addEventListener("click", () => setLanguage(button.dataset.language));
})();
