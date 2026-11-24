#!/usr/bin/env python3
# Downloads Pyodide and onnxruntime-web, pinned by hash. Mode "cdn" keeps only
# their JavaScript; the browser loads the rest from jsDelivr.
# Usage: fetch_vendor.py self|cdn OUT_DIR

import base64
import hashlib
import io
import json
import sys
import tarfile
import urllib.request
from pathlib import Path

PYODIDE_VERSION = '314.0.7'
PYODIDE_TARBALL = (f'https://registry.npmjs.org/pyodide/-/pyodide-{PYODIDE_VERSION}.tgz', '66b7949e5a0cba2250b586d7e368bf12fb482800748d182c4618254b685ffaa0')
PYODIDE_CDN = f'https://cdn.jsdelivr.net/pyodide/v{PYODIDE_VERSION}/full/' # also has the packages
PYODIDE_SCRIPTS = ['pyodide.mjs', 'pyodide.asm.mjs', 'pyodide-lock.json']
PYODIDE_BINARIES = ['pyodide.asm.wasm', 'python_stdlib.zip']
PACKAGES = ['numpy', 'opencv-python', 'pyclipper']

ORT_VERSION = '1.30.0'
ORT_TARBALL = (f'https://registry.npmjs.org/onnxruntime-web/-/onnxruntime-web-{ORT_VERSION}.tgz', 'd2228df7e4616bc3348bf504ee888f3bec43789a273f0a63f3e68d203ce3bf71')
ORT_CDN = f'https://cdn.jsdelivr.net/npm/onnxruntime-web@{ORT_VERSION}/dist/'
ORT_SCRIPTS = ['ort.wasm.min.mjs', 'ort-wasm-simd-threaded.mjs']
ORT_BINARY = 'ort-wasm-simd-threaded.wasm'

CDN = 'https://cdn.jsdelivr.net'
# what browsers accept; jsDelivr compresses the WebAssembly files too
ACCEPT_ENCODING = 'br, gzip'

NOTICE = f'''vendor/pyodide/   Pyodide {PYODIDE_VERSION}, https://pyodide.org/, MPL-2.0, source code
                  at https://github.com/pyodide/pyodide (the bundled CPython
                  and packages keep their own licenses); numpy,
                  opencv-python and pyclipper from the Pyodide distribution
vendor/onnxruntime-web/
                  ONNX Runtime Web {ORT_VERSION}, https://onnxruntime.ai/, MIT,
                  Copyright (c) Microsoft Corporation
'''
NOTICE_CDN = f'''
This build loads their WebAssembly files and the packages from jsDelivr:
{PYODIDE_CDN}
{ORT_CDN}
'''


def download(url, sha256):
	print(f'Downloading {url}', flush=True)
	with urllib.request.urlopen(url, timeout=120) as response:
		data = response.read()
	actual = hashlib.sha256(data).hexdigest()
	if actual != sha256:
		raise SystemExit(f'ERROR: {url} has SHA-256 {actual}, expected {sha256}')
	return data


# The bytes a browser downloads for url: its compressed size.
def transfer_size(url):
	request = urllib.request.Request(url, method='HEAD', headers={'Accept-Encoding': ACCEPT_ENCODING})
	with urllib.request.urlopen(request, timeout=120) as response:
		length = response.headers['Content-Length']
	if length is None:
		raise SystemExit(f'ERROR: {url} has no Content-Length')
	return int(length)


# The members of an npm tarball (paths below package/), by name.
def extract(tarball, members):
	url, sha256 = tarball
	with tarfile.open(fileobj=io.BytesIO(download(url, sha256)), mode='r:gz') as tar:
		return {Path(member).name: tar.extractfile(f'package/{member}').read() for member in members}


# The file names and hashes of the packages and their dependencies, from the
# lock file.
def package_files(lock_data):
	lock = json.loads(lock_data)['packages']
	needed, pending = set(), list(PACKAGES)
	while pending:
		name = pending.pop()
		if name not in needed:
			needed.add(name)
			pending.extend(lock[name]['depends'])
	return [(lock[name]['file_name'], lock[name]['sha256']) for name in sorted(needed)]


# Subresource Integrity metadata, as fetch() takes it.
def sri(data):
	return 'sha256-' + base64.b64encode(hashlib.sha256(data).digest()).decode()


class Vendor():
	def __init__(self, mode, out_dir):
		self.cdn = mode == 'cdn'
		self.out_dir = out_dir
		self.integrity = {} # URL -> hash, for the files from the CDN
		self.cdn_bytes = 0

	def write(self, path, data):
		path = self.out_dir / path
		path.parent.mkdir(parents=True, exist_ok=True)
		path.write_bytes(data)

	# A large file: served by the site, or by the CDN at cdn_url, which must
	# have the same content. Returns where the worker loads it from.
	def add_binary(self, path, data, cdn_url):
		if not self.cdn:
			self.write(path, data)
			return path
		download(cdn_url, hashlib.sha256(data).hexdigest())
		self.integrity[cdn_url] = sri(data)
		self.cdn_bytes += transfer_size(cdn_url)
		return cdn_url

	def fetch_pyodide(self):
		files = extract(PYODIDE_TARBALL, PYODIDE_SCRIPTS + PYODIDE_BINARIES)
		for name in PYODIDE_SCRIPTS:
			self.write(f'pyodide/{name}', files[name])
		for name in PYODIDE_BINARIES:
			self.add_binary(f'pyodide/{name}', files[name], PYODIDE_CDN + name)
		# Pyodide checks them against the hashes in its lock file
		for file_name, sha256 in package_files(files['pyodide-lock.json']):
			data = download(PYODIDE_CDN + file_name, sha256)
			if self.cdn:
				self.cdn_bytes += transfer_size(PYODIDE_CDN + file_name)
			else:
				self.write(f'pyodide/{file_name}', data)
		return PYODIDE_CDN if self.cdn else 'pyodide/'

	def fetch_onnxruntime(self):
		files = extract(ORT_TARBALL, [f'dist/{name}' for name in ORT_SCRIPTS + [ORT_BINARY]])
		for name in ORT_SCRIPTS:
			self.write(f'onnxruntime-web/{name}', files[name])
		return self.add_binary(f'onnxruntime-web/{ORT_BINARY}', files[ORT_BINARY], ORT_CDN + ORT_BINARY)


def main():
	if len(sys.argv) != 3 or sys.argv[1] not in ('self', 'cdn'):
		raise SystemExit(f'usage: {sys.argv[0]} self|cdn OUT_DIR')
	mode, out_dir = sys.argv[1], Path(sys.argv[2])
	vendor = Vendor(mode, out_dir)
	# URLs relative to OUT_DIR, or on the CDN
	runtime = {
		'pyodide': vendor.fetch_pyodide(),
		'packages': PACKAGES,
		'onnxruntime_wasm': vendor.fetch_onnxruntime(),
		'cdn': CDN if vendor.cdn else None,
		'integrity': vendor.integrity,
		'cdn_bytes': vendor.cdn_bytes,
	}
	(out_dir / 'runtime.json').write_text(json.dumps(runtime, indent='\t') + '\n')
	(out_dir / 'NOTICE.txt').write_text(NOTICE + (NOTICE_CDN if vendor.cdn else ''))


if __name__ == '__main__':
	main()
