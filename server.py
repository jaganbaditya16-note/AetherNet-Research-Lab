"""Local-only dashboard server. Binds to loopback, never to all interfaces."""
from __future__ import annotations
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
import json
import threading

from benchmarks_v2 import run_suite

BASE = Path(__file__).resolve().parent
BENCHMARK_LOCK = threading.Lock()

class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(BASE), **kwargs)

    def do_GET(self):
        if self.path == "/api/status":
            body = json.dumps({"ok": True, "version": "0.2.0", "bind": "127.0.0.1"}).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
            return
        return super().do_GET()

    def do_POST(self):
        if self.path != "/api/benchmark":
            self.send_error(404, "Not found"); return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length > 4000:
                self.send_error(413, "Request too large"); return
            payload = json.loads(self.rfile.read(length) or b"{}")
            repeats = max(1, min(int(payload.get("repeats", 1)), 3))
            samples = max(300, min(int(payload.get("samples", 700)), 2000))
            dataset = payload.get("dataset", "all")
            from benchmarks_v2 import DATASETS
            datasets = list(DATASETS) if dataset == "all" else [dataset]
            if any(d not in DATASETS for d in datasets):
                raise ValueError("Unsupported dataset")
            if not BENCHMARK_LOCK.acquire(blocking=False):
                self.send_error(409, "A benchmark is already running"); return
            try:
                result = run_suite(repeats=repeats, n_samples=samples, dataset_names=datasets)
            finally:
                BENCHMARK_LOCK.release()
            body = json.dumps(result).encode("utf-8")
            self.send_response(200); self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            body = json.dumps({"error": str(exc)}).encode()
            self.send_response(400); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
        except Exception as exc:
            body = json.dumps({"error": f"Benchmark failed: {type(exc).__name__}: {exc}"}).encode()
            self.send_response(500); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

if __name__ == "__main__":
    print("AetherNet Research Lab 0.2.0")
    print("Local only: http://127.0.0.1:8010")
    print("Press Ctrl+C to stop. Benchmark data stays on this computer.")
    ThreadingHTTPServer(("127.0.0.1", 8010), Handler).serve_forever()
