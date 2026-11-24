#!/usr/bin/env python3
# Serves the D4 item tooltip OCR web page and turns uploaded screenshots into
# item tooltip JSON with d4-item-tooltip-ocr (see ocr_adapter.py).
# Standard library only: run it with the OCR project's interpreter, which
# brings Paddle & co.

import argparse
import json
import os
import signal
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from ocr_adapter import UNSUPPORTED_TYPE_MESSAGE, OcrError, StderrRecorder, load_ocr, recognize, sniff_image_type

HERE = Path(__file__).resolve().parent
STATIC_DIR = HERE / 'static'
DEFAULT_OCR_DIR = HERE.parent / 'd4-item-tooltip-ocr'
DEFAULT_PORT = 8044 # "D4"; 8000 and 8080 are often taken
DEFAULT_MAX_UPLOAD_MB = 64

CONTENT_SECURITY_POLICY = "default-src 'self'; img-src 'self' blob:; object-src 'none'; base-uri 'none'; frame-ancestors 'none'"


# Runs all OCR work on one dedicated thread: the OCR holds Paddle predictors
# that must not be used concurrently. Requests wait for their turn.
class OcrEngine():
	def __init__(self, create_ocr):
		StderrRecorder.install()
		self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='ocr')
		self._ocr = self._executor.submit(create_ocr).result()

	def close(self):
		self._executor.shutdown(cancel_futures=True)

	# Returns the item as a dict, or raises OcrError.
	def process(self, image_data, image_type):
		return self._executor.submit(recognize, self._ocr, image_data, image_type).result()


class OcrHTTPServer(ThreadingHTTPServer):
	engine = None
	max_upload_bytes = DEFAULT_MAX_UPLOAD_MB * 1024 * 1024


class RequestHandler(SimpleHTTPRequestHandler):
	server_version = 'd4-lens'

	def __init__(self, *args, **kwargs):
		super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

	def end_headers(self):
		self.send_header('Cache-Control', 'no-cache')
		self.send_header('X-Content-Type-Options', 'nosniff')
		self.send_header('Content-Security-Policy', CONTENT_SECURITY_POLICY)
		super().end_headers()

	# never show directory listings
	def list_directory(self, path):
		self.send_error(HTTPStatus.NOT_FOUND)
		return None

	def do_POST(self):
		if urlsplit(self.path).path != '/api/ocr':
			self.close_connection = True # the body is not read
			self.send_json_error(HTTPStatus.NOT_FOUND, 'Not found.')
			return

		length = self.headers.get('Content-Length')
		try:
			length = int(length) if length is not None else None
		except ValueError:
			length = -1
		if length is None or length < 0:
			self.close_connection = True
			self.send_json_error(HTTPStatus.LENGTH_REQUIRED, 'The upload needs a valid Content-Length.')
			return
		if length > self.server.max_upload_bytes:
			# read the body anyway: browsers report a connection reset rather
			# than the response if the server closes while they still send
			self.discard_body(length)
			limit_mb = self.server.max_upload_bytes / (1024 * 1024)
			self.send_json_error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, f'The image is larger than the {limit_mb:g} MB upload limit.')
			return

		data = self.rfile.read(length)
		if len(data) != length:
			self.close_connection = True
			self.send_json_error(HTTPStatus.BAD_REQUEST, 'The upload was incomplete.')
			return
		image_type = sniff_image_type(data)
		if image_type is None:
			self.send_json_error(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, UNSUPPORTED_TYPE_MESSAGE)
			return

		try:
			item = self.server.engine.process(data, image_type)
		except OcrError as exc:
			self.send_json(HTTPStatus.UNPROCESSABLE_ENTITY, {'error': exc.message, 'details': exc.details})
			return
		except Exception as exc:
			traceback.print_exc()
			self.send_json_error(HTTPStatus.INTERNAL_SERVER_ERROR, f'Server error: {type(exc).__name__}: {exc}')
			return
		self.send_json(HTTPStatus.OK, {'item': item})

	def discard_body(self, length, limit=1024 ** 3):
		if length > limit:
			self.close_connection = True
			return
		while length > 0:
			chunk = self.rfile.read(min(length, 1024 * 1024))
			if not chunk:
				self.close_connection = True
				return
			length -= len(chunk)

	def send_json_error(self, status, message):
		self.send_json(status, {'error': message, 'details': []})

	def send_json(self, status, payload):
		body = json.dumps(payload).encode()
		self.send_response(status)
		self.send_header('Content-Type', 'application/json')
		self.send_header('Content-Length', str(len(body)))
		if self.close_connection:
			self.send_header('Connection', 'close')
		self.end_headers()
		self.wfile.write(body)


def main():
	parser = argparse.ArgumentParser(description='Web page that turns Diablo IV screenshots into item tooltip JSON.')
	parser.add_argument('--host', default='127.0.0.1', help='address to listen on; 0.0.0.0 allows other devices on the LAN (default: %(default)s)')
	parser.add_argument('--port', type=int, default=DEFAULT_PORT, help='port to listen on (default: %(default)s)')
	parser.add_argument('--ocr-dir', type=Path, default=Path(os.environ.get('D4_OCR_DIR', DEFAULT_OCR_DIR)), help='d4-item-tooltip-ocr checkout (default: $D4_OCR_DIR or %(default)s)')
	parser.add_argument('--max-upload-mb', type=float, default=float(os.environ.get('D4_MAX_UPLOAD_MB', DEFAULT_MAX_UPLOAD_MB)), help='largest accepted image in MB (default: $D4_MAX_UPLOAD_MB or %(default)s)')
	opt = parser.parse_args()

	# bind first, so a port conflict shows up before the slow OCR start-up
	server = OcrHTTPServer((opt.host, opt.port), RequestHandler)
	server.max_upload_bytes = int(opt.max_upload_mb * 1024 * 1024)
	print(f'Loading the OCR from {opt.ocr_dir.resolve()} ...', flush=True)
	try:
		server.engine = OcrEngine(lambda: load_ocr(opt.ocr_dir))
	except Exception as exc:
		server.server_close()
		print(f'ERROR: Failed to load the OCR: {exc}', file=sys.stderr)
		sys.exit(1)

	# shut down cleanly on `docker stop` as well
	signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(0))
	host, port = server.server_address[:2]
	print(f'Serving on http://{host}:{port}/ (Ctrl+C to stop)', flush=True)
	try:
		server.serve_forever()
	except KeyboardInterrupt:
		pass
	finally:
		server.server_close()
		server.engine.close()


if __name__ == '__main__':
	main()
