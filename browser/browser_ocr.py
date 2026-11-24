# The OCR's entry point in the browser, called by ocr-worker.js in Pyodide:
# runs d4-item-tooltip-ocr.py through ocr_adapter.py like the server does,
# with the stand-ins of ocr_shims.py and the models in onnxruntime-web.

import json
import traceback

import numpy as np
from pyodide.ffi import run_sync, to_js

import ocr_adapter
import ocr_shims
import ort_bridge # registered by ocr-worker.js


# Runs a model in onnxruntime-web. run_sync() waits for its promise, which
# needs WebAssembly JSPI and a call through callPromising().
def infer(model, x):
	output = run_sync(ort_bridge.run(model, to_js(x.ravel()), to_js(list(x.shape))))
	return np.frombuffer(output.data.to_bytes(), dtype=np.float32).reshape(output.dims.to_py())


ocr_shims.install(infer)
ocr = None


def load(ocr_dir):
	global ocr
	ocr = ocr_adapter.load_ocr(ocr_dir)


# Takes the image file as a Uint8Array. Returns the same JSON as the server's
# API: {"item": {...}} or {"error": "...", "details": [...]}.
def recognize(image):
	data = image.to_bytes()
	image_type = ocr_adapter.sniff_image_type(data)
	if image_type is None:
		return json.dumps({'error': ocr_adapter.UNSUPPORTED_TYPE_MESSAGE, 'details': []})
	try:
		return json.dumps({'item': ocr_adapter.recognize(ocr, data, image_type)})
	except ocr_adapter.OcrError as exc:
		return json.dumps({'error': exc.message, 'details': exc.details})
	except Exception as exc: # like the server's 500
		traceback.print_exc()
		return json.dumps({'error': f'The OCR failed: {type(exc).__name__}: {exc}', 'details': []})
