import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from App.config import WEB_HOST, WEB_PORT
from App.reporting import build_test_report, save_test_report


HTML_PAGE = """
<!doctype html>
<html lang="ru">
  <head>
    <meta charset="utf-8" />
    <title>Matching Test Report</title>
    <style>
      body { font-family: Arial, sans-serif; margin: 24px; }
      h2 { margin-bottom: 8px; }
      .meta { margin-bottom: 18px; }
      table { border-collapse: collapse; width: 100%; }
      th, td { border: 1px solid #ccc; padding: 8px 10px; text-align: left; vertical-align: top; }
      pre { background: #f5f5f5; padding: 16px; border-radius: 8px; }
    </style>
  </head>
  <body>
    <h2>Отчёт по тестированию матчинга</h2>
    <div class="meta">
      <button onclick="refresh()">Обновить</button>
    </div>
    <pre id="result"></pre>
    <script>
      async function refresh() {
        const response = await fetch('/api/report');
        const data = await response.json();
        document.getElementById('result').textContent = JSON.stringify(data, null, 2);
      }
      refresh();
    </script>
  </body>
</html>
"""


class ReportHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path in ('/', '/index.html'):
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode('utf-8'))
            return

        if parsed.path == '/api/report':
            report = build_test_report()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.end_headers()
            self.wfile.write(json.dumps(report, ensure_ascii=False).encode('utf-8'))
            return

        self.send_response(404)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.end_headers()
        self.wfile.write(json.dumps({'status': 'not_found'}, ensure_ascii=False).encode('utf-8'))

    def log_message(self, format, *args):
        return


def run_report_server(host: str | None = None, port: int | None = None):
    host = host or WEB_HOST
    port = int(port or WEB_PORT)
    save_test_report()
    server = ThreadingHTTPServer((host, port), ReportHandler)
    print(f'Report server running on http://{host}:{port}')
    server.serve_forever()


if __name__ == '__main__':
    run_report_server()
