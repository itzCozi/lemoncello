"""
Web Monitor for example.py
===========================
Runs example.py as a subprocess and streams its output to a browser
in real-time via Server-Sent Events (SSE).

Usage (on your Linux VPS):
    python web_monitor.py

Then open http://<your-vps-ip>:8080 in your browser.

Options:
    --port PORT     Port to listen on (default: 8080)
    --host HOST     Host to bind to (default: 0.0.0.0)
"""

import subprocess
import threading
import sys
import os
import json
import time
import argparse
from http.server import HTTPServer, BaseHTTPRequestHandler
from collections import deque

# ---------------- Shared State ----------------

output_lines = deque(maxlen=50000)
process_status = {"running": False, "exit_code": None, "started_at": None}
clients = []  # list of SSE client queues
lock = threading.Lock()


def broadcast(event_type, data):
    """Send an SSE event to all connected clients."""
    with lock:
        dead = []
        for q in clients:
            try:
                q.append((event_type, data))
            except Exception:
                dead.append(q)
        for q in dead:
            clients.remove(q)


def run_script():
    """Run example.py and capture output line-by-line."""
    process_status["running"] = True
    process_status["exit_code"] = None
    process_status["started_at"] = time.time()
    output_lines.clear()

    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "example.py")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    proc = subprocess.Popen(
        [sys.executable, "-u", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=os.path.dirname(os.path.abspath(__file__)),
        env=env,
        bufsize=0,
    )

    buf = b""
    while True:
        chunk = proc.stdout.read(1)
        if not chunk:
            break
        if chunk == b"\n":
            line = buf.decode("utf-8", errors="replace")
            buf = b""
            output_lines.append(line)
            broadcast("line", line)
        elif chunk == b"\r":
            # Handle \r for progress bar overwrites
            line = buf.decode("utf-8", errors="replace")
            buf = b""
            output_lines.append(line)
            broadcast("cr", line)
        else:
            buf += chunk

    if buf:
        line = buf.decode("utf-8", errors="replace")
        output_lines.append(line)
        broadcast("line", line)

    proc.wait()
    process_status["running"] = False
    process_status["exit_code"] = proc.returncode
    broadcast("status", json.dumps({
        "running": False,
        "exit_code": proc.returncode,
    }))


# ---------------- HTML Page ----------------

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Limoncello Optimizer</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    background: #0d1117; color: #c9d1d9;
    font-family: 'SF Mono', 'Cascadia Code', 'Fira Code', 'Consolas', monospace;
    height: 100vh; display: flex; flex-direction: column;
  }
  header {
    background: #161b22; border-bottom: 1px solid #30363d;
    padding: 12px 20px; display: flex; align-items: center;
    justify-content: space-between; flex-shrink: 0;
  }
  header h1 { font-size: 16px; color: #f0e68c; font-weight: 600; }
  .status {
    font-size: 13px; padding: 4px 12px; border-radius: 12px;
    font-weight: 500;
  }
  .status.running { background: #1a3a1a; color: #3fb950; }
  .status.done { background: #1a2a3a; color: #58a6ff; }
  .status.failed { background: #3a1a1a; color: #f85149; }
  .status.waiting { background: #2a2a1a; color: #d29922; }
  #controls {
    background: #161b22; border-bottom: 1px solid #30363d;
    padding: 8px 20px; display: flex; align-items: center;
    gap: 12px; flex-shrink: 0; font-size: 13px;
  }
  #controls label { color: #8b949e; cursor: pointer; user-select: none; }
  #controls input[type=checkbox] { margin-right: 4px; }
  #elapsed { color: #8b949e; margin-left: auto; }
  #terminal {
    flex: 1; overflow-y: auto; padding: 12px 20px;
    font-size: 13px; line-height: 1.5; white-space: pre;
    scrollbar-width: thin;
    scrollbar-color: #30363d #0d1117;
  }
  #terminal::-webkit-scrollbar { width: 8px; }
  #terminal::-webkit-scrollbar-track { background: #0d1117; }
  #terminal::-webkit-scrollbar-thumb { background: #30363d; border-radius: 4px; }
  #terminal .line { min-height: 1.5em; }
  .line-phase { color: #f0e68c; font-weight: bold; }
  .line-score { color: #3fb950; }
  .line-bar { color: #58a6ff; }
  .line-header { color: #f0e68c; }
  .line-result { color: #3fb950; font-weight: bold; }
  .line-error { color: #f85149; }
  .line-dim { color: #6e7681; }
  footer {
    background: #161b22; border-top: 1px solid #30363d;
    padding: 8px 20px; font-size: 12px; color: #484f58;
    flex-shrink: 0; text-align: center;
  }
</style>
</head>
<body>
<header>
  <h1>&#127819; Limoncello Recipe Optimizer</h1>
  <span id="statusBadge" class="status waiting">Connecting...</span>
</header>
<div id="controls">
  <label><input type="checkbox" id="autoScroll" checked> Auto-scroll</label>
  <label><input type="checkbox" id="wordWrap"> Word wrap</label>
  <span id="elapsed"></span>
</div>
<div id="terminal"></div>
<footer>web_monitor.py &mdash; streaming live from server</footer>

<script>
const terminal = document.getElementById('terminal');
const statusBadge = document.getElementById('statusBadge');
const autoScrollCb = document.getElementById('autoScroll');
const wordWrapCb = document.getElementById('wordWrap');
const elapsedEl = document.getElementById('elapsed');
let lineCount = 0;
let startTime = null;
let timerInterval = null;

wordWrapCb.addEventListener('change', () => {
  terminal.style.whiteSpace = wordWrapCb.checked ? 'pre-wrap' : 'pre';
});

function classifyLine(text) {
  if (/^[=]{10,}/.test(text.trim())) return 'line-header';
  if (/Phase \d|RECIPE OPTIMIZER|SA SEARCH|VERIFICATION|CROSS-CHECK|SANITY CHECK/.test(text)) return 'line-phase';
  if (/OPTIMAL RECIPE|SA RESULT|WINNER|DONE/.test(text)) return 'line-result';
  if (/score=|best=|composite=/.test(text)) return 'line-score';
  if (/\[#+\.+\]/.test(text)) return 'line-bar';
  if (/Error|Traceback|Exception/i.test(text)) return 'line-error';
  return '';
}

function appendLine(text, isCR) {
  if (isCR) {
    // Overwrite the last line (carriage return behavior)
    const last = terminal.lastElementChild;
    if (last) {
      last.textContent = text;
      last.className = 'line ' + classifyLine(text);
      maybeScroll();
      return;
    }
  }
  const div = document.createElement('div');
  div.className = 'line ' + classifyLine(text);
  div.textContent = text;
  terminal.appendChild(div);
  lineCount++;
  maybeScroll();
}

function maybeScroll() {
  if (autoScrollCb.checked) {
    terminal.scrollTop = terminal.scrollHeight;
  }
}

function updateElapsed() {
  if (!startTime) return;
  const s = Math.floor((Date.now() - startTime) / 1000);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  elapsedEl.textContent = `${h}h ${m}m ${sec}s — ${lineCount} lines`;
}

function connect() {
  // First, load existing output
  fetch('/api/history').then(r => r.json()).then(data => {
    terminal.innerHTML = '';
    lineCount = 0;
    data.lines.forEach(l => appendLine(l, false));
    if (data.status.running) {
      statusBadge.textContent = 'Running';
      statusBadge.className = 'status running';
      startTime = data.status.started_at * 1000;
    } else if (data.status.exit_code !== null) {
      statusBadge.textContent = data.status.exit_code === 0 ? 'Completed' : `Exited (${data.status.exit_code})`;
      statusBadge.className = data.status.exit_code === 0 ? 'status done' : 'status failed';
    }
  }).catch(() => {});

  // Then connect SSE for live updates
  const es = new EventSource('/stream');

  es.addEventListener('line', e => {
    appendLine(e.data, false);
    if (!startTime) { startTime = Date.now(); }
    statusBadge.textContent = 'Running';
    statusBadge.className = 'status running';
  });

  es.addEventListener('cr', e => {
    appendLine(e.data, true);
  });

  es.addEventListener('status', e => {
    const s = JSON.parse(e.data);
    if (!s.running) {
      statusBadge.textContent = s.exit_code === 0 ? 'Completed' : `Exited (${s.exit_code})`;
      statusBadge.className = s.exit_code === 0 ? 'status done' : 'status failed';
    }
  });

  es.addEventListener('ping', () => {});

  es.onerror = () => {
    es.close();
    statusBadge.textContent = 'Disconnected — retrying...';
    statusBadge.className = 'status failed';
    setTimeout(connect, 3000);
  };
}

timerInterval = setInterval(updateElapsed, 1000);
connect();
</script>
</body>
</html>
"""


# ---------------- HTTP Handler ----------------

class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # suppress default access logs

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self._serve_html()
        elif self.path == "/stream":
            self._serve_sse()
        elif self.path == "/api/history":
            self._serve_history()
        else:
            self.send_error(404)

    def _serve_html(self):
        content = HTML_PAGE.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _serve_history(self):
        data = {
            "lines": list(output_lines),
            "status": dict(process_status),
        }
        body = json.dumps(data).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_sse(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        q = deque()
        with lock:
            clients.append(q)

        try:
            while True:
                if q:
                    event_type, data = q.popleft()
                    # Escape newlines in data for SSE
                    escaped = data.replace("\n", "\ndata: ")
                    msg = f"event: {event_type}\ndata: {escaped}\n\n"
                    self.wfile.write(msg.encode("utf-8"))
                    self.wfile.flush()
                else:
                    # Send keepalive ping every 5 seconds
                    time.sleep(0.05)
                    # Periodic ping to detect disconnection
                    if not q and int(time.time()) % 5 == 0:
                        self.wfile.write(b"event: ping\ndata: \n\n")
                        self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            with lock:
                if q in clients:
                    clients.remove(q)


class ThreadedHTTPServer(HTTPServer):
    """Handle each request in a new thread (needed for SSE)."""
    def process_request(self, request, client_address):
        t = threading.Thread(target=self.process_request_thread,
                             args=(request, client_address), daemon=True)
        t.start()

    def process_request_thread(self, request, client_address):
        try:
            self.finish_request(request, client_address)
        except Exception:
            self.handle_error(request, client_address)
        finally:
            self.shutdown_request(request)


# ---------------- Main ----------------

def main():
    parser = argparse.ArgumentParser(description="Web monitor for example.py")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    # Start the optimization script in a background thread
    script_thread = threading.Thread(target=run_script, daemon=True)
    script_thread.start()

    server = ThreadedHTTPServer((args.host, args.port), Handler)
    print(f"  Monitor running at http://{args.host}:{args.port}")
    print(f"  Open http://<your-vps-ip>:{args.port} in your browser")
    print(f"  Press Ctrl+C to stop\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Shutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
