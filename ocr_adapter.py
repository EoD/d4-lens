# Runs d4-item-tooltip-ocr on uploaded images: loads the OCR script from its
# checkout and turns its result or its stderr diagnostics into an item or an
# OcrError. Shared by the server (server.py) and the in-browser build, where it
# runs in Pyodide (browser/). Standard library only.

import contextlib
import importlib.util
import json
import os
import sys
import tempfile
import threading
import traceback
from pathlib import Path

OCR_SCRIPT = 'd4-item-tooltip-ocr.py'
UNSUPPORTED_TYPE_MESSAGE = 'Unsupported file type: only PNG, JPEG and WebP screenshots are supported.'


# Returns 'png', 'jpeg' or 'webp' based on the file signature, or None.
def sniff_image_type(data):
	if data.startswith(b'\x89PNG\r\n\x1a\n'):
		return 'png'
	if data.startswith(b'\xff\xd8\xff'):
		return 'jpeg'
	if len(data) >= 12 and data[:4] == b'RIFF' and data[8:12] == b'WEBP':
		return 'webp'
	return None


class OcrError(Exception):
	def __init__(self, message, details=()):
		super().__init__(message)
		self.message = message
		self.details = list(details)


# Stand-in for sys.stderr that passes everything through and additionally
# records what the current thread writes inside record(). The OCR reports
# failures only as ERROR/WARNING lines on stderr.
class StderrRecorder():
	def __init__(self, stream):
		self._stream = stream
		self._local = threading.local()

	@classmethod
	def install(cls):
		if not isinstance(sys.stderr, cls):
			sys.stderr = cls(sys.stderr)
		return sys.stderr

	def write(self, text):
		chunks = getattr(self._local, 'chunks', None)
		if chunks is not None:
			chunks.append(text)
		return self._stream.write(text)

	def __getattr__(self, name):
		return getattr(self._stream, name)

	@contextlib.contextmanager
	def record(self):
		self._local.chunks = []
		try:
			yield self._local.chunks
		finally:
			self._local.chunks = None


# The ERROR/WARNING lines in the OCR's stderr output, with the temporary file
# name hidden.
def ocr_diagnostics(output, tmp_path):
	lines = (line.strip() for line in output.splitlines())
	return [line.replace(tmp_path, 'uploaded image') for line in lines if line.startswith(('ERROR:', 'WARNING:'))]


# Imports d4-item-tooltip-ocr.py from ocr_dir and creates its OCR object.
def load_ocr(ocr_dir):
	ocr_dir = Path(ocr_dir).resolve()
	script = ocr_dir / OCR_SCRIPT
	if not script.is_file():
		raise FileNotFoundError(f'{script} not found; point --ocr-dir or D4_OCR_DIR to a d4-item-tooltip-ocr checkout')
	spec = importlib.util.spec_from_file_location('d4_item_tooltip_ocr', script)
	module = importlib.util.module_from_spec(spec)
	# the OCR loads its templates and recognition model relative to the working directory
	cwd = os.getcwd()
	os.chdir(ocr_dir)
	try:
		spec.loader.exec_module(module)
		return module.D4ItemTooltipOCR()
	finally:
		os.chdir(cwd)


# Runs the OCR on one image (the bytes of a file of the given image type).
# Returns the item as a dict, or raises OcrError. Not thread-safe: the OCR
# holds model predictors that must not be used concurrently.
def recognize(ocr, image_data, image_type):
	stderr = StderrRecorder.install()
	fd, tmp_path = tempfile.mkstemp(prefix='d4-ocr-', suffix=f'.{image_type}')
	crash = None
	try:
		with os.fdopen(fd, 'wb') as f:
			f.write(image_data)
		with stderr.record() as chunks:
			try:
				jsonstr = ocr.processImage(tmp_path, find_tooltip=True)
			except Exception as exc:
				traceback.print_exc()
				crash = exc
	finally:
		os.unlink(tmp_path)

	details = ocr_diagnostics(''.join(chunks), tmp_path)
	if crash is not None:
		raise OcrError(f'The OCR crashed: {type(crash).__name__}: {crash}', details)
	if jsonstr is None:
		# the first ERROR line becomes the message, the other lines its details
		first_error = next((line for line in details if line.startswith('ERROR:')), None)
		if first_error is None:
			raise OcrError('The OCR returned no result.', details)
		details.remove(first_error)
		raise OcrError(first_error.removeprefix('ERROR:').strip(), details)
	return json.loads(jsonstr)
