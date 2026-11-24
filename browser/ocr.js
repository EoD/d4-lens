// Runs the OCR in this browser, in a web worker (ocr-worker.js). Replaces
// static/ocr.js, the server version, in the in-browser build.

// the worker runs one OCR at a time and has the image in memory already
export const MAX_PARALLEL_REQUESTS = 1;

const worker = new Worker(new URL('ocr-worker.js', import.meta.url), { type: 'module' });
const jobs = new Map(); // id -> {resolve, onStatus}
let nextId = 1;
let statusText = null; // what the worker is busy with before it can take images
let workerError = null;

worker.addEventListener('message', ({ data: message }) => {
	if (message.type === 'status') {
		statusText = message.text;
		for (const job of jobs.values()) {
			job.onStatus?.(statusText);
		}
	} else if (message.type === 'result') {
		jobs.get(message.id)?.resolve(message.result);
		jobs.delete(message.id);
	}
});

// the worker script did not load or crashed
worker.addEventListener('error', (event) => {
	workerError = `The OCR could not be started${event.message ? `: ${event.message}` : '.'}`;
	for (const job of jobs.values()) {
		job.resolve({ error: workerError, details: [] });
	}
	jobs.clear();
});

// Returns {item} or {error, details}. onStatus(text) reports what has to
// happen before the image's turn (null when done). Rejects if aborted
// through signal.
export async function recognize(file, { signal, onStatus }) {
	if (workerError) {
		return { error: workerError, details: [] };
	}
	const image = await file.arrayBuffer();
	signal?.throwIfAborted();
	const id = nextId++;
	return new Promise((resolve, reject) => {
		jobs.set(id, { resolve, onStatus });
		signal?.addEventListener('abort', () => {
			jobs.delete(id);
			worker.postMessage({ type: 'cancel', id });
			reject(signal.reason);
		}, { once: true });
		worker.postMessage({ type: 'recognize', id, image }, [image]);
		if (statusText) {
			onStatus?.(statusText);
		}
	});
}
