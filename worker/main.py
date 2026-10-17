"""Persistent single-model worker with an independent readiness heartbeat."""
from __future__ import annotations
import signal
import threading
import time
import uuid
from app.db import BusyError, JobStore
from app.settings import load_settings
from worker.pipeline import Pipeline
from worker.runtime import LocalRuntime
from worker.video import AnalysisError


def main() -> None:
    settings = load_settings()
    store = JobStore(settings.db_path, settings.worker_stale_sec)
    owner = uuid.uuid4().hex
    stop = threading.Event()
    state = {"ready": False, "detail": "Loading the local model"}
    runtime = LocalRuntime(settings.root, settings.data_dir, stop)
    active_job_id: str | None = None

    def halt(*_args):
        state["ready"] = False
        stop.set()
        runtime.cancel()
    for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), halt)
    store.heartbeat_worker(owner, False, state["detail"])

    def heartbeat():
        while not stop.wait(3):
            try:
                store.heartbeat_worker(owner, state["ready"], state["detail"])
                store.recover_stale_jobs(settings.worker_stale_sec)
            except Exception:
                state["ready"] = False
                stop.set()
                runtime.cancel()
    monitor = threading.Thread(target=heartbeat, daemon=True)
    monitor.start()
    try:
        runtime.start()
        if stop.is_set():
            return
        pipeline = Pipeline(settings.root, settings.data_dir, runtime)
        state.update(ready=True, detail="Local model ready")
        store.heartbeat_worker(owner, True, state["detail"])
        print("Local inference worker is ready", flush=True)
        while not stop.is_set():
            store.recover_stale_jobs(settings.worker_stale_sec)
            job = store.claim_next(owner)
            if not job:
                stop.wait(.4)
                continue
            active_job_id = job["id"]
            try:
                result, details, timings = pipeline.run(job, lambda stage, amount, message: store.update_progress(job["id"], stage, amount, message))
                if stop.is_set():
                    raise AnalysisError("worker_stopped", "The worker stopped before analysis completed")
                if not store.complete_job(job["id"], result, details, timings):
                    store.fail_job(job["id"], "deadline_exceeded", "Processing did not complete before the deadline")
                    print(f"Job {job['id']} failed: deadline_exceeded", flush=True)
                else:
                    print(f"Job {job['id']} finished", flush=True)
            except AnalysisError as exc:
                store.fail_job(job["id"], exc.code, str(exc))
                print(f"Job {job['id']} failed: {exc.code}", flush=True)
                if exc.code.startswith("model_") or exc.code == "deadline_exceeded":
                    state.update(ready=False, detail="Restarting the local model")
                    store.heartbeat_worker(owner, False, state["detail"])
                    runtime.stop()
                    if not stop.is_set():
                        runtime.start()
                        state.update(ready=True, detail="Local model ready")
                        store.heartbeat_worker(owner, True, state["detail"])
            except Exception:
                store.fail_job(job["id"], "analysis_failed", "Local analysis failed; no clean verdict was produced")
                print(f"Job {job['id']} failed: analysis_failed", flush=True)
            finally:
                active_job_id = None
    except AnalysisError:
        if not stop.is_set():
            raise
    finally:
        state["ready"] = False
        stop.set()
        runtime.cancel()
        try:
            if active_job_id is not None:
                store.fail_job(active_job_id, "worker_stopped", "The worker stopped before analysis completed")
            store.heartbeat_worker(owner, False, "Worker stopped")
        except BusyError:
            pass
        finally:
            runtime.stop()
            monitor.join(timeout=4)
            # Do not release while a stalled heartbeat could still write afterward.
            if not monitor.is_alive():
                store.release_worker(owner)


if __name__ == "__main__":
    main()
