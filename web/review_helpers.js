(function (root) {
  "use strict";
  const finite = value => value != null && Number.isFinite(Number(value));
  function eventTarget(event, videos, playable) {
    const start = Number(event.start_sec);
    const continuous = videos.find(v => v.key === "source" && !v.sampled && playable(v)) ||
      videos.find(v => v.key === "people" && !v.sampled && playable(v));
    if (continuous && finite(event.start_sec)) {
      const first = Number(continuous.source_start_sec) || 0;
      const last = continuous.source_end_sec;
      if (start >= first && (!finite(last) || start <= Number(last))) {
        let time = start - first;
        const timeline = continuous.source_timeline;
        if (Array.isArray(timeline) && timeline.length) {
          let lo = 0, hi = timeline.length - 1;
          while (lo < hi) { const mid = Math.floor((lo + hi) / 2); if (timeline[mid][0] < start) lo = mid + 1; else hi = mid; }
          const nearest = lo > 0 && Math.abs(timeline[lo-1][0]-start) <= Math.abs(timeline[lo][0]-start) ? lo-1 : lo;
          time = timeline[nearest][1];
        }
        return { video: continuous, time: Math.max(0, time) };
      }
    }
    const own = videos.find(v => v.key === event.mode && playable(v));
    return own && finite(event.playback_sec) ? {video: own, time: Math.max(0, Number(event.playback_sec))} : null;
  }
  function exportResult(job, results) {
    return {schema_version: "1.0", exported_at: new Date().toISOString(),
      video: {id: job.id, name: job.name, status: job.status, ...job.video},
      note: "Возможные события требуют проверки человеком. Ссылки на кадры доступны на компьютере обработки.",
      results};
  }
  const api = {eventTarget, exportResult};
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.ReviewHelpers = api;
})(typeof window !== "undefined" ? window : globalThis);
