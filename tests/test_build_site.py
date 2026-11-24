# Tests of browser/build_site.py.

import contextlib
import io
import json
import re
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'browser'))
try:
	import build_site
except ImportError:
	build_site = None

CDN = 'https://cdn.example'
RUNTIMES = {
	'self': {'pyodide': 'pyodide/', 'packages': ['numpy'], 'onnxruntime_wasm': 'onnxruntime-web/ort.wasm', 'cdn': None, 'integrity': {}, 'cdn_bytes': 0},
	'cdn': {'pyodide': f'{CDN}/pyodide/', 'packages': ['numpy'], 'onnxruntime_wasm': f'{CDN}/ort/ort.wasm', 'cdn': CDN, 'integrity': {f'{CDN}/ort/ort.wasm': 'sha256-x'}, 'cdn_bytes': 20_000_000},
}


@unittest.skipIf(build_site is None, 'needs browser/build_site.py')
class BuildSiteTest(unittest.TestCase):
	# cdn_files: what fetch() gets from the CDN, by URL; the same as the OCR's
	# files by default.
	def build(self, mode, source_url=None, cdn_files=None):
		tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
		ocr_dir, models_dir, vendor_dir, out_dir = (tmp / name for name in ('ocr', 'models', 'vendor', 'site'))
		for name in build_site.OCR_FILES + ['templates/a.png']:
			(ocr_dir / name).parent.mkdir(parents=True, exist_ok=True)
			(ocr_dir / name).write_text(name)
		models_dir.mkdir()
		(models_dir / 'rec.onnx').write_bytes(b'\0' * 3_000_000)
		(vendor_dir / 'pyodide').mkdir(parents=True)
		(vendor_dir / 'pyodide' / 'pyodide.mjs').write_text('')
		(vendor_dir / 'runtime.json').write_text(json.dumps(RUNTIMES[mode]))
		(vendor_dir / 'NOTICE.txt').write_text('vendor notice\n')
		if cdn_files is None:
			cdn_files = {build_site.OCR_DIRS_CDN + 'templates/a.png': b'templates/a.png'}
		with contextlib.redirect_stdout(io.StringIO()), mock.patch.object(build_site, 'fetch', cdn_files.get):
			build_site.build(ocr_dir, models_dir, vendor_dir, out_dir, source_url)
		return out_dir

	def archive_names(self, out_dir):
		with zipfile.ZipFile(out_dir / 'ocr.zip') as archive:
			return archive.namelist()

	# The exports of ocr-config.js, which are JSON values.
	def config(self, out_dir):
		source = (out_dir / 'ocr-config.js').read_text()
		return {name: json.loads(value) for name, value in re.findall(r'^export const (\w+) = (.*);$', source, re.M)}

	def policy(self, out_dir):
		return re.search(r'http-equiv="Content-Security-Policy" content="([^"]*)"', (out_dir / 'index.html').read_text())[1]

	def test_self_hosted(self):
		out_dir = self.build('self')
		config = self.config(out_dir)
		self.assertEqual(config['PYODIDE_URL'], 'vendor/pyodide/')
		self.assertEqual(config['ORT_WASM_URL'], 'vendor/onnxruntime-web/ort.wasm')
		self.assertEqual(config['INTEGRITY'], {})
		self.assertNotIn('connect-src', self.policy(out_dir))
		self.assertTrue((out_dir / 'vendor' / 'pyodide' / 'pyodide.mjs').is_file())
		self.assertFalse((out_dir / 'vendor' / 'runtime.json').exists())

	def test_cdn(self):
		out_dir = self.build('cdn')
		config = self.config(out_dir)
		self.assertEqual(config['PYODIDE_URL'], f'{CDN}/pyodide/')
		self.assertEqual(config['ORT_WASM_URL'], f'{CDN}/ort/ort.wasm')
		self.assertLessEqual(RUNTIMES['cdn']['integrity'].items(), config['INTEGRITY'].items())
		self.assertIn(f"connect-src 'self' {CDN}", self.policy(out_dir))
		self.assertNotIn(CDN, self.policy(out_dir).split('; connect-src')[0])

	def test_self_hosted_serves_the_templates(self):
		out_dir = self.build('self')
		self.assertIn('templates/a.png', self.archive_names(out_dir))
		self.assertEqual(self.config(out_dir)['OCR_FILES'], {})
		self.assertNotIn('jsDelivr', (out_dir / 'NOTICE.txt').read_text())

	def test_cdn_loads_the_templates_from_upstream(self):
		out_dir = self.build('cdn')
		url = build_site.OCR_DIRS_CDN + 'templates/a.png'
		self.assertNotIn('templates/a.png', self.archive_names(out_dir))
		self.assertIn('d4-item-tooltip-ocr.py', self.archive_names(out_dir))
		config = self.config(out_dir)
		self.assertEqual(config['OCR_FILES'], {'templates/a.png': url})
		self.assertEqual(config['INTEGRITY'][url], build_site.sri(b'templates/a.png'))
		self.assertIn(build_site.OCR_DIRS_CDN, (out_dir / 'NOTICE.txt').read_text())

	def test_cdn_templates_must_match(self):
		with self.assertRaisesRegex(SystemExit, 'differs'):
			self.build('cdn', cdn_files={build_site.OCR_DIRS_CDN + 'templates/a.png': b'other'})

	def test_footer(self):
		out_dir = self.build('cdn', source_url='https://github.com/example/site')
		page = (out_dir / 'index.html').read_text()
		self.assertIn('<a href="https://github.com/example/site">Source code</a>', page)
		self.assertIn('<a href="NOTICE.txt">Licenses</a>', page)
		notice = (out_dir / 'NOTICE.txt').read_text()
		self.assertIn('d4-item-tooltip-ocr', notice)
		self.assertIn('vendor notice', notice)
		self.assertFalse((out_dir / 'vendor' / 'NOTICE.txt').exists())
		self.assertNotIn('Source code', (self.build('self') / 'index.html').read_text())

	def test_page_uses_the_browser_ocr(self):
		out_dir = self.build('self')
		self.assertEqual((out_dir / 'ocr.js').read_text(), (ROOT / 'browser' / 'ocr.js').read_text())
		self.assertIn("from './ocr-config.js'", (out_dir / 'ocr-worker.js').read_text())


if __name__ == '__main__':
	unittest.main()
