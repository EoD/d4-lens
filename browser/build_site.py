#!/usr/bin/env python3
# Assembles the static site of the in-browser OCR.

import argparse
import base64
import hashlib
import html
import json
import shutil
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OCR_FILES = ['d4-item-tooltip-ocr.py', 'd4-vocabulary.txt', 'LICENSE.txt']
OCR_DIRS = ['templates']
# The OCR's upstream repository on jsDelivr, at a commit with the same
# templates: builds with the runtime from a CDN load them from there instead
# of serving the game's icons themselves. The build checks that they match.
OCR_DIRS_CDN = 'https://cdn.jsdelivr.net/gh/mxtsdev/d4-item-tooltip-ocr@493ac5697fd4a2a5c0074724fd76e0873a914b59/'
PYTHON_FILES = [ROOT / 'ocr_adapter.py', HERE / 'ocr_shims.py', HERE / 'browser_ocr.py']
# The OCR checks that its Paddle recognition model exists; the stand-in for
# Paddle runs models/rec.onnx instead. The OCR's default rec_model_dir.
REC_MODEL_PLACEHOLDERS = ['paddleocr-models/en_PP-OCRv3_rec-d4_tooltip/inference.pdmodel', 'paddleocr-models/en_PP-OCRv3_rec-d4_tooltip/inference.pdiparams']
# The server sends its policy as a header. The worker needs WebAssembly.
CONTENT_SECURITY_POLICY = "default-src 'self'; script-src 'self' 'wasm-unsafe-eval'; img-src 'self' blob:; object-src 'none'; base-uri 'none'"
FOOTER = '<footer class="page-footer">'
NOTICE = '''Third-party software and data of this site

ocr.zip           d4-item-tooltip-ocr, https://github.com/mxtsdev/d4-item-tooltip-ocr,
                  MIT (LICENSE.txt in ocr.zip), with its vocabulary
{templates}models/det.onnx   PaddleOCR's PP-OCRv4_mobile_det text detection model,
                  https://github.com/PaddlePaddle/PaddleOCR, Apache-2.0
models/rec.onnx   d4-item-tooltip-ocr's recognition model for item tooltips
                  (en_PP-OCRv3_rec-d4_tooltip), MIT
'''
NOTICE_TEMPLATES = '''ocr.zip/templates/
                  the OCR's template images, cut from Diablo IV's user
                  interface, (c) Blizzard Entertainment
'''
NOTICE_TEMPLATES_CDN = '''
The OCR's template images, cut from Diablo IV's user interface ((c) Blizzard
Entertainment), are not part of this site. It loads them from
d4-item-tooltip-ocr's repository on jsDelivr:
{url}
'''


def fetch(url):
	with urllib.request.urlopen(url, timeout=120) as response:
		return response.read()


# Subresource Integrity metadata, as fetch() takes it.
def sri(data):
	return 'sha256-' + base64.b64encode(hashlib.sha256(data).digest()).decode()


def build_page(out_dir, runtime, source_url):
	shutil.copytree(ROOT / 'static', out_dir, dirs_exist_ok=True)
	shutil.copy(HERE / 'ocr.js', out_dir / 'ocr.js')
	shutil.copy(HERE / 'ocr-worker.js', out_dir / 'ocr-worker.js')
	index = out_dir / 'index.html'
	page = index.read_text()
	head = '<meta charset="utf-8">'
	if head not in page:
		raise SystemExit(f'ERROR: {head} not found in index.html')
	policy = CONTENT_SECURITY_POLICY
	if runtime['cdn']:
		# the worker fetches the runtime's binaries from there (scripts always come from this
		# site), for browsers that apply the page's policy to its worker; Chromium and Firefox
		# don't: the worker gets the policy of its own response headers, and GitHub Pages sends none
		policy += f"; connect-src 'self' {runtime['cdn']}"
	page = page.replace(head, f'{head}\n\t<meta http-equiv="Content-Security-Policy" content="{policy}">', 1)
	if FOOTER not in page:
		raise SystemExit(f'ERROR: {FOOTER} not found in index.html')
	links = [f'<a href="{html.escape(source_url)}">Source code</a>'] if source_url else []
	links.append('<a href="NOTICE.txt">Licenses</a>')
	index.write_text(page.replace(FOOTER, f'{FOOTER}\n\t\t<p>{" · ".join(links)}</p>', 1))


# Returns the OCR's files that the worker loads from the CDN: {path: URL},
# and their hashes: {URL: hash}.
def build_ocr_archive(ocr_dir, archive, cdn):
	from_cdn, integrity = {}, {}
	with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as zip:
		for name in OCR_FILES:
			zip.write(ocr_dir / name, name)
		for name in OCR_DIRS:
			for path in sorted((ocr_dir / name).rglob('*')):
				if not path.is_file():
					continue
				name = path.relative_to(ocr_dir).as_posix()
				if not cdn:
					zip.write(path, name)
					continue
				url = OCR_DIRS_CDN + name
				print(f'Checking {url}', flush=True)
				data = path.read_bytes()
				if fetch(url) != data:
					raise SystemExit(f'ERROR: {url} differs from {path}; point OCR_DIRS_CDN to a commit with the same {OCR_DIRS}')
				from_cdn[name] = url
				integrity[url] = sri(data)
		for path in PYTHON_FILES:
			zip.write(path, path.name)
		for name in REC_MODEL_PLACEHOLDERS:
			zip.writestr(name, b'')
	return from_cdn, integrity


def build_config(out_dir, runtime, ocr_from_cdn, ocr_integrity):
	settings = {
		# URLs relative to this module (the site's root) or absolute
		'PYODIDE_URL': urllib.parse.urljoin('vendor/', runtime['pyodide']),
		'PACKAGES': runtime['packages'],
		'ORT_WASM_URL': urllib.parse.urljoin('vendor/', runtime['onnxruntime_wasm']),
		# the OCR's files that are not in ocr.zip: path in the OCR directory -> URL
		'OCR_FILES': ocr_from_cdn,
		# URL -> Subresource Integrity hash of the files from the CDN
		'INTEGRITY': runtime['integrity'] | ocr_integrity,
	}
	lines = [f'// Generated by build_site.py: the settings of this build, with the runtime {"from " + runtime["cdn"] if runtime["cdn"] else "served by this site"}.']
	lines += [f'export const {name} = {json.dumps(value)};' for name, value in settings.items()]
	(out_dir / 'ocr-config.js').write_text('\n'.join(lines) + '\n')


# The download size of the page, for the build log.
def print_size(out_dir, runtime):
	hosted = sum(path.stat().st_size for path in out_dir.rglob('*') if path.is_file())
	print(f'{out_dir}: {hosted / 1e6:.1f} MB, plus {runtime["cdn_bytes"] / 1e6:.1f} MB from the CDN', flush=True)


def build(ocr_dir, models_dir, vendor_dir, out_dir, source_url=None):
	runtime = json.loads((vendor_dir / 'runtime.json').read_text())
	out_dir.mkdir(parents=True, exist_ok=True)
	build_page(out_dir, runtime, source_url)
	ocr_from_cdn, ocr_integrity = build_ocr_archive(ocr_dir, out_dir / 'ocr.zip', bool(runtime['cdn']))
	shutil.copytree(models_dir, out_dir / 'models', dirs_exist_ok=True)
	shutil.copytree(vendor_dir, out_dir / 'vendor', dirs_exist_ok=True, ignore=shutil.ignore_patterns('runtime.json', 'NOTICE.txt'))
	notice = NOTICE.format(templates='' if ocr_from_cdn else NOTICE_TEMPLATES)
	notice += NOTICE_TEMPLATES_CDN.format(url=OCR_DIRS_CDN) if ocr_from_cdn else ''
	(out_dir / 'NOTICE.txt').write_text(notice + '\n' + (vendor_dir / 'NOTICE.txt').read_text())
	build_config(out_dir, runtime, ocr_from_cdn, ocr_integrity)
	print_size(out_dir, runtime)


def main():
	parser = argparse.ArgumentParser(description='Assembles the in-browser build of the OCR website.')
	parser.add_argument('--source-url', help='where the page links to its source code')
	for name in ('ocr_dir', 'models_dir', 'vendor_dir', 'out_dir'):
		parser.add_argument(name, type=Path)
	opt = parser.parse_args()
	build(opt.ocr_dir, opt.models_dir, opt.vendor_dir, opt.out_dir, opt.source_url)


if __name__ == '__main__':
	main()
