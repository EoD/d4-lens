# Checks that the in-browser OCR gives the same results as Paddle. Needs
# D4_ONNX_MODELS (set by the Docker build); skipped without it.

import contextlib
import importlib.util
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'browser'))
import server
from ocr_adapter import OCR_SCRIPT, load_ocr

OCR_DIR = Path(os.environ.get('D4_OCR_DIR', server.DEFAULT_OCR_DIR)).resolve()
MODELS_DIR = Path(os.environ['D4_ONNX_MODELS']).resolve() if 'D4_ONNX_MODELS' in os.environ else None

MISSING = [name for name in ('cv2', 'onnxruntime', 'paddle', 'pyclipper', 'Levenshtein') if importlib.util.find_spec(name) is None]
MISSING += [str(path) for path in (OCR_DIR / OCR_SCRIPT, MODELS_DIR and MODELS_DIR / 'det.onnx', MODELS_DIR and MODELS_DIR / 'rec.onnx') if path and not path.is_file()]


# The OCR's test tooltips (cropped) and its test screenshots, with their
# find_tooltip argument.
def sample_images(ocr_dir):
	images = [(path, False) for path in sorted((ocr_dir / 'tests').iterdir()) if path.suffix in ('.png', '.jpg')]
	images += [(path, True) for path in sorted((ocr_dir / 'tests' / 'screenshots').glob('*.png'))]
	return images


# The OCR's JSON (or None) for each test image, with Paddle or with the
# in-browser stand-ins and ONNX models. Runs in a separate process per backend:
# both import their own module named paddle.
def ocr_results(backend, ocr_dir, models_dir):
	if backend == 'onnx':
		import onnxruntime
		import ocr_shims
		sessions = {model: onnxruntime.InferenceSession(str(Path(models_dir) / f'{model}.onnx'), providers=['CPUExecutionProvider']) for model in ('det', 'rec')}

		def infer(model, x):
			session = sessions[model]
			return session.run(None, {session.get_inputs()[0].name: x})[0]

		ocr_shims.install(infer)
	ocr_dir = Path(ocr_dir)
	ocr = load_ocr(ocr_dir)
	results = {}
	with contextlib.redirect_stdout(sys.stderr): # the OCR prints its timings
		for path, find_tooltip in sample_images(ocr_dir):
			results[path.relative_to(ocr_dir).as_posix()] = ocr.processImage(str(path), find_tooltip=find_tooltip)
	return results


@unittest.skipIf(MODELS_DIR is None, 'needs D4_ONNX_MODELS, the converted models')
class BrowserShimsTest(unittest.TestCase):
	def setUp(self):
		self.assertEqual(MISSING, [], 'needs the OCR interpreter with onnxruntime, the OCR checkout and the models')

	def run_ocr(self, backend):
		output = subprocess.run([sys.executable, __file__, backend, str(OCR_DIR), str(MODELS_DIR)], check=True, capture_output=True, text=True).stdout
		return json.loads(output)

	def test_same_items_as_paddle(self):
		expected = self.run_ocr('paddle')
		actual = self.run_ocr('onnx')
		self.assertEqual(sorted(actual), sorted(path.relative_to(OCR_DIR).as_posix() for path, _ in sample_images(OCR_DIR)))
		self.assertTrue(any(expected.values()), 'Paddle recognized nothing')
		for name in expected:
			with self.subTest(image=name):
				self.assertEqual(actual[name], expected[name])

	def test_levenshtein_ratio(self):
		import Levenshtein
		import ocr_shims
		words = ['', 'a', 'Strength', 'Strenght', 'Damage', 'damage', 'Critical Strike Chance', 'Crit1cal Str1ke Chanse', 'Upgrades: 4/4', 'Lucky Hit']
		for s1 in words:
			for s2 in words:
				with self.subTest(s1=s1, s2=s2):
					self.assertAlmostEqual(ocr_shims.ratio(s1, s2), Levenshtein.ratio(s1, s2))
					self.assertAlmostEqual(ocr_shims.ratio(s1, s2, score_cutoff=0.8), Levenshtein.ratio(s1, s2, score_cutoff=0.8))


if __name__ == '__main__':
	print(json.dumps(ocr_results(*sys.argv[1:])))
