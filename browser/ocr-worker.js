// Web worker that runs the OCR: Pyodide runs the OCR script, onnxruntime-web its models.
// In: {type: 'recognize', id, image} or {type: 'cancel', id}. Out: {type: 'status', text} or {type: 'result', id, result}.

import { loadPyodide } from './vendor/pyodide/pyodide.mjs';
import * as ort from './vendor/onnxruntime-web/ort.wasm.min.mjs';
import { INTEGRITY, OCR_FILES, ORT_WASM_URL, PACKAGES, PYODIDE_URL } from './ocr-config.js';

const OCR_ARCHIVE = 'ocr.zip'; // the OCR script, its data files and the Python side of this worker
// plus OCR_FILES, the OCR's files that the build left out of the archive: path -> URL
const OCR_DIR = '/ocr';
const MODELS = { det: 'models/det.onnx', rec: 'models/rec.onnx' };

const url = (path) => new URL(path, import.meta.url).href;

// Pyodide fetches its WebAssembly and standard library from indexURL itself,
// with no way to check them, so this worker's fetch() adds the hashes from
// INTEGRITY. Anything else from another origin must come with its own hash
// (Pyodide's packages: the sha256 from its lock file) or is refused.
// Failed downloads also reject loadFailure: Pyodide only warns about packages
// it could not load, and waits forever for a pyodide.asm.wasm that failed.
let reportLoadFailure;
const loadFailure = new Promise((resolve, reject) => {
	reportLoadFailure = reject;
});
const fetchUnchecked = self.fetch;
self.fetch = async (resource, options = {}) => {
	const href = new URL(resource.url ?? resource, self.location.href).href;
	const integrity = options.integrity || resource.integrity || INTEGRITY[href];
	try {
		if (!integrity && new URL(href).origin !== self.location.origin) {
			throw new Error('no integrity hash');
		}
		const response = await fetchUnchecked.call(self, resource, integrity ? { ...options, integrity } : options);
		if (!response.ok) {
			throw new Error(`${response.status} ${response.statusText}`);
		}
		return response;
	} catch (err) {
		const error = new Error(`Loading ${href} failed: ${err.message}`);
		reportLoadFailure(error);
		throw error;
	}
};

// the JavaScript part; the WebAssembly comes from ORT_WASM_URL (loadModels())
ort.env.wasm.wasmPaths = url('vendor/onnxruntime-web/');
// threads would need cross-origin isolation (COOP/COEP headers), which GitHub Pages can't send
ort.env.wasm.numThreads = 1;

const engine = loadEngine();
engine.catch(() => status(null)); // the error is reported with each image
const cancelled = new Set();
let jobs = Promise.resolve(); // one image at a time: the OCR is not reentrant

self.addEventListener('message', ({ data: message }) => {
	if (message.type === 'recognize') {
		jobs = jobs.then(() => recognize(message.id, message.image));
	} else if (message.type === 'cancel') {
		cancelled.add(message.id);
	}
});

async function recognize(id, image) {
	if (cancelled.delete(id)) {
		return;
	}
	let result;
	try {
		const { recognizeImage } = await engine;
		result = JSON.parse(await recognizeImage.callPromising(new Uint8Array(image)));
	} catch (err) {
		console.error(err);
		result = { error: err.message, details: [] };
	}
	cancelled.delete(id);
	self.postMessage({ type: 'result', id, result });
}

async function loadEngine() {
	if (typeof WebAssembly.Suspending !== 'function') {
		throw new Error('This browser cannot run the OCR: it lacks WebAssembly JavaScript Promise Integration (JSPI). Current versions of Chrome, Edge and Firefox have it.');
	}
	status('Loading the OCR');
	const [pyodide, sessions, archive, ocrFiles] = await Promise.race([
		Promise.all([
			loadPython(),
			loadModels(),
			fetch(url(OCR_ARCHIVE)).then((response) => response.arrayBuffer()),
			Promise.all(Object.entries(OCR_FILES).map(async ([path, href]) => [path, await fetch(href).then((response) => response.arrayBuffer())])),
		]),
		loadFailure,
	]);
	pyodide.registerJsModule('ort_bridge', {
		async run(model, data, dims) {
			const session = sessions[model];
			const input = new ort.Tensor('float32', new Float32Array(data), Array.from(dims));
			const outputs = await session.run({ [session.inputNames[0]]: input });
			const output = outputs[session.outputNames[0]];
			return { data: output.data, dims: Array.from(output.dims) };
		},
	});
	pyodide.unpackArchive(archive, 'zip', { extractDir: OCR_DIR });
	for (const [path, data] of ocrFiles) {
		const file = `${OCR_DIR}/${path}`;
		pyodide.FS.mkdirTree(file.slice(0, file.lastIndexOf('/')));
		pyodide.FS.writeFile(file, new Uint8Array(data));
	}
	pyodide.runPython(`import sys; sys.path.insert(0, ${JSON.stringify(OCR_DIR)})`);
	status('Starting the OCR');
	const browserOcr = pyodide.pyimport('browser_ocr');
	browserOcr.load(OCR_DIR);
	status(null);
	return { recognizeImage: browserOcr.recognize };
}

async function loadPython() {
	const { default: createPyodideModule } = await import('./vendor/pyodide/pyodide.asm.mjs');
	const pyodide = await loadPyodide({
		indexURL: url(PYODIDE_URL),
		createPyodideModule, // this site's pyodide.asm.mjs, not indexURL's
		// the lock file of the hash-verified Pyodide release, with the hashes of the packages
		lockFileURL: url('vendor/pyodide/pyodide-lock.json'),
		packageBaseUrl: url(PYODIDE_URL), // the default is the lock file's directory
	});
	await pyodide.loadPackage(PACKAGES, { messageCallback: () => {} });
	return pyodide;
}

async function loadModels() {
	ort.env.wasm.wasmBinary = await fetch(url(ORT_WASM_URL)).then((response) => response.arrayBuffer());
	const sessions = {};
	for (const [model, path] of Object.entries(MODELS)) {
		sessions[model] = await ort.InferenceSession.create(url(path), { executionProviders: ['wasm'] });
	}
	return sessions;
}

function status(text) {
	self.postMessage({ type: 'status', text });
}
