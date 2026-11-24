// Runs the OCR on the server (POST api/ocr). The in-browser build replaces this
// module with browser/ocr.js, which has the same exports.

const OCR_URL = 'api/ocr'; // relative: the page may be served under a subpath

// the server runs one OCR at a time; this keeps the next upload ready
export const MAX_PARALLEL_REQUESTS = 2;

// Returns {item} or {error, details}. Rejects if the request fails or is
// aborted through signal.
export async function recognize(file, { mimeType, signal }) {
	const response = await fetch(OCR_URL, {
		method: 'POST',
		body: file,
		headers: { 'Content-Type': mimeType },
		signal,
	});
	return readOcrResponse(response);
}

async function readOcrResponse(response) {
	if ((response.headers.get('Content-Type') ?? '').startsWith('application/json')) {
		const body = await response.json();
		if (response.ok && body.item) {
			return body;
		}
		if (body.error) {
			return body;
		}
	}
	// not from our server, e.g. a reverse proxy error page
	const hints = {
		413: ' The upload is too large for the server or a proxy in front of it (nginx: client_max_body_size).',
		502: ' Is the OCR server running?',
		504: ' The OCR took too long (nginx: proxy_read_timeout).',
	};
	return { error: `The server answered ${response.status} ${response.statusText}.${hints[response.status] ?? ''}` };
}
