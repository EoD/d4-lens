// Collects screenshots (file picker, drag and drop, clipboard paste), runs the
// OCR on each one and shows the resulting item tooltip JSON. Where the OCR runs
// is up to ocr.js: on the server, or in the browser in the in-browser build.

import * as ocr from './ocr.js';

const JSON_INDENT = 3; // the same formatting as d4-item-tooltip-ocr
const NOTICE_MS = 6000;
const IMAGE_TYPES = { png: 'image/png', jpeg: 'image/jpeg', webp: 'image/webp' };
const STATUS_LABELS = { queued: 'Queued', processing: 'Processing', done: 'Done', failed: 'Failed' };

const $ = (id) => document.getElementById(id);
const fileInput = $('file-input');
const toolbar = $('toolbar');
const summary = $('summary');
const downloadAllButton = $('download-all');
const copyAllButton = $('copy-all');
const clearAllButton = $('clear-all');
const notice = $('notice');
const cardList = $('cards');
const cardTemplate = $('card-template');

const cards = []; // in upload order
const queue = [];
let runningRequests = 0;
let intake = Promise.resolve(); // adds files one batch after another, so duplicate checks see earlier batches
let noticeTimer = null;

// --- adding screenshots ---

function addFiles(files, { pasted = false } = {}) {
	files = Array.from(files);
	if (files.length === 0) {
		return;
	}
	intake = intake.then(() => addFilesInOrder(files, pasted)).catch((err) => console.error(err));
}

async function addFilesInOrder(files, pasted) {
	const skipped = [];
	for (const file of files) {
		const name = pasted ? `Pasted image (${new Date().toLocaleTimeString()})` : file.name;
		let bytes;
		try {
			bytes = new Uint8Array(await file.arrayBuffer());
		} catch (err) {
			createCard({ name }).fail(`Could not read the file: ${err.message}`);
			continue;
		}
		const imageType = sniffImageType(bytes);
		if (imageType === null) {
			createCard({ name }).fail('Unsupported file type: only PNG, JPEG and WebP screenshots are supported.');
			continue;
		}
		const key = `${bytes.length}:${contentHash(bytes)}`;
		if (cards.some((card) => card.key === key)) {
			skipped.push(name);
			continue;
		}
		const card = createCard({ name, file, key, mimeType: IMAGE_TYPES[imageType] });
		queue.push(card);
		pumpQueue();
	}
	if (skipped.length > 0) {
		showNotice(`Skipped ${skipped.length === 1 ? 'a screenshot that is' : `${skipped.length} screenshots that are`} already in the list: ${skipped.join(', ')}`);
	}
}

// The same check as the server: the file signature decides, not the name.
function sniffImageType(bytes) {
	const startsWith = (offset, signature) => signature.every((b, i) => bytes[offset + i] === b);
	if (startsWith(0, [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a])) {
		return 'png';
	}
	if (startsWith(0, [0xff, 0xd8, 0xff])) {
		return 'jpeg';
	}
	if (startsWith(0, [0x52, 0x49, 0x46, 0x46]) && startsWith(8, [0x57, 0x45, 0x42, 0x50])) { // RIFF....WEBP
		return 'webp';
	}
	return null;
}

// Fast 53-bit hash (cyrb53) for duplicate detection. crypto.subtle is not
// available on plain-HTTP LAN origins, and a collision only skips a file.
function contentHash(bytes) {
	let h1 = 0xdeadbeef;
	let h2 = 0x41c6ce57;
	for (let i = 0; i < bytes.length; i++) {
		h1 = Math.imul(h1 ^ bytes[i], 2654435761);
		h2 = Math.imul(h2 ^ bytes[i], 1597334677);
	}
	h1 = Math.imul(h1 ^ (h1 >>> 16), 2246822507);
	h1 ^= Math.imul(h2 ^ (h2 >>> 13), 3266489909);
	h2 = Math.imul(h2 ^ (h2 >>> 16), 2246822507);
	h2 ^= Math.imul(h1 ^ (h1 >>> 13), 3266489909);
	return (4294967296 * (2097151 & h2) + (h1 >>> 0)).toString(36);
}

// --- cards ---

function createCard({ name, file = null, key = null, mimeType = null }) {
	const element = cardTemplate.content.firstElementChild.cloneNode(true);
	const card = {
		name,
		file,
		key,
		mimeType,
		status: 'queued',
		item: null,
		jsonText: null,
		controller: null,
		objectUrl: file ? URL.createObjectURL(file) : null,
		element,
		fail(message, details = []) {
			setFailed(card, message, details);
			return card;
		},
	};

	element.querySelector('.card-name').textContent = name;
	const thumb = element.querySelector('.thumb');
	const img = thumb.querySelector('img');
	if (card.objectUrl) {
		thumb.href = card.objectUrl;
		img.src = card.objectUrl;
	} else {
		img.hidden = true;
	}
	element.querySelector('.copy').addEventListener('click', (event) => copyJson(card.jsonText, event.currentTarget));
	element.querySelector('.remove').addEventListener('click', () => removeCard(card));

	cards.push(card);
	cardList.append(element);
	setStatus(card, 'queued');
	return card;
}

function setStatus(card, status) {
	card.status = status;
	card.element.dataset.status = status;
	card.element.querySelector('.status').textContent = STATUS_LABELS[status];
	updateToolbar();
}

// a more specific text for the current status, e.g. while the OCR loads
function setStatusText(card, text) {
	card.element.querySelector('.status').textContent = text;
}

function setDone(card, item) {
	card.item = item;
	card.jsonText = JSON.stringify(item, null, JSON_INDENT);
	const pre = card.element.querySelector('.json');
	renderJson(pre, card.jsonText);
	pre.hidden = false;
	card.element.querySelector('.copy').disabled = false;
	setStatus(card, 'done');
}

function setFailed(card, message, details = []) {
	const error = card.element.querySelector('.error');
	error.querySelector('.error-message').textContent = message;
	error.querySelector('.error-details').replaceChildren(...details.map((line) => {
		const li = document.createElement('li');
		li.textContent = line;
		return li;
	}));
	error.hidden = false;
	setStatus(card, 'failed');
}

function removeCard(card) {
	const index = cards.indexOf(card);
	if (index === -1) {
		return;
	}
	cards.splice(index, 1);
	const queued = queue.indexOf(card);
	if (queued !== -1) {
		queue.splice(queued, 1);
	}
	card.controller?.abort();
	if (card.objectUrl) {
		URL.revokeObjectURL(card.objectUrl);
	}
	card.element.remove();
	updateToolbar();
}

function clearAll() {
	for (const card of [...cards]) {
		removeCard(card);
	}
	hideNotice();
}

function updateToolbar() {
	const counts = { queued: 0, processing: 0, done: 0, failed: 0 };
	for (const card of cards) {
		counts[card.status]++;
	}
	const parts = [`${cards.length} screenshot${cards.length === 1 ? '' : 's'}`];
	for (const status of ['done', 'processing', 'queued', 'failed']) {
		if (counts[status] > 0) {
			parts.push(`${counts[status]} ${STATUS_LABELS[status].toLowerCase()}`);
		}
	}
	summary.textContent = parts.join(' · ');
	downloadAllButton.textContent = counts.done > 0 ? `Download all (${counts.done})` : 'Download all';
	downloadAllButton.disabled = counts.done === 0;
	copyAllButton.disabled = counts.done === 0;
	toolbar.hidden = cards.length === 0;
	document.body.classList.toggle('has-cards', cards.length > 0);
}

// --- OCR requests ---

function pumpQueue() {
	while (runningRequests < ocr.MAX_PARALLEL_REQUESTS && queue.length > 0) {
		const card = queue.shift();
		runningRequests++;
		runOcr(card).finally(() => {
			runningRequests--;
			pumpQueue();
		});
	}
}

async function runOcr(card) {
	card.controller = new AbortController();
	setStatus(card, 'processing');
	try {
		const result = await ocr.recognize(card.file, {
			mimeType: card.mimeType,
			signal: card.controller.signal,
			onStatus: (text) => setStatusText(card, text ?? STATUS_LABELS[card.status]),
		});
		if (result.item) {
			setDone(card, result.item);
		} else {
			setFailed(card, result.error, result.details ?? []);
		}
	} catch (err) {
		if (err.name !== 'AbortError') {
			setFailed(card, `The request failed: ${err.message}`);
		}
	} finally {
		card.controller = null;
	}
}

// --- output ---

const JSON_TOKEN = /("(?:\\.|[^"\\])*")(\s*:)?|\b(?:true|false|null)\b|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?/g;

// Light syntax highlighting; text only, never HTML.
function renderJson(pre, text) {
	const nodes = [];
	let last = 0;
	for (const match of text.matchAll(JSON_TOKEN)) {
		nodes.push(text.slice(last, match.index));
		const span = document.createElement('span');
		if (match[1] !== undefined) {
			span.className = match[2] ? 'key' : 'string';
			span.textContent = match[1];
			nodes.push(span, match[2] ?? '');
		} else {
			span.className = 'literal';
			span.textContent = match[0];
			nodes.push(span);
		}
		last = match.index + match[0].length;
	}
	nodes.push(text.slice(last));
	pre.replaceChildren(...nodes);
}

async function copyJson(text, button) {
	try {
		await copyText(text);
		flashButton(button, 'Copied');
	} catch (err) {
		console.error(err);
		flashButton(button, 'Copy failed');
	}
}

// navigator.clipboard only exists in secure contexts (HTTPS or localhost),
// so fall back to the legacy copy command on plain-HTTP LAN addresses.
async function copyText(text) {
	if (navigator.clipboard && window.isSecureContext) {
		await navigator.clipboard.writeText(text);
		return;
	}
	const textarea = document.createElement('textarea');
	textarea.value = text;
	textarea.setAttribute('readonly', '');
	textarea.style.position = 'fixed';
	textarea.style.opacity = '0';
	document.body.append(textarea);
	textarea.select();
	const copied = document.execCommand('copy');
	textarea.remove();
	if (!copied) {
		throw new Error('the browser refused to copy');
	}
}

function flashButton(button, label) {
	const original = button.dataset.label ?? button.textContent;
	button.dataset.label = original;
	button.textContent = label;
	clearTimeout(button.flashTimer);
	button.flashTimer = setTimeout(() => {
		button.textContent = original;
	}, 1500);
}

// The items of all successful cards as one JSON array, in upload order;
// null if there are none.
function allItemsJson() {
	const items = cards.filter((card) => card.status === 'done').map((card) => card.item);
	return items.length > 0 ? JSON.stringify(items, null, JSON_INDENT) + '\n' : null;
}

function copyAll() {
	const text = allItemsJson();
	if (text === null) {
		return;
	}
	copyJson(text, copyAllButton);
}

function downloadAll() {
	const text = allItemsJson();
	if (text === null) {
		return;
	}
	const blob = new Blob([text], { type: 'application/json' });
	const url = URL.createObjectURL(blob);
	const link = document.createElement('a');
	link.href = url;
	link.download = `d4-items-${timestamp()}.json`;
	document.body.append(link);
	link.click();
	link.remove();
	setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function timestamp() {
	const d = new Date();
	const pad = (n) => String(n).padStart(2, '0');
	return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}_${pad(d.getHours())}-${pad(d.getMinutes())}-${pad(d.getSeconds())}`;
}

function showNotice(message) {
	notice.textContent = message;
	notice.hidden = false;
	clearTimeout(noticeTimer);
	noticeTimer = setTimeout(hideNotice, NOTICE_MS);
}

function hideNotice() {
	clearTimeout(noticeTimer);
	notice.hidden = true;
}

// --- input wiring ---

fileInput.addEventListener('change', () => {
	addFiles(fileInput.files);
	fileInput.value = ''; // allow choosing the same file again after removing it
});

document.addEventListener('paste', (event) => {
	const files = event.clipboardData?.files;
	if (files && files.length > 0) {
		event.preventDefault();
		addFiles(files, { pasted: true });
	}
});

const dragsFiles = (event) => event.dataTransfer?.types.includes('Files');
let dragDepth = 0;

window.addEventListener('dragenter', (event) => {
	if (dragsFiles(event)) {
		dragDepth++;
		document.body.classList.add('dragging');
	}
});

window.addEventListener('dragleave', (event) => {
	if (dragsFiles(event) && --dragDepth <= 0) {
		dragDepth = 0;
		document.body.classList.remove('dragging');
	}
});

window.addEventListener('dragover', (event) => {
	if (dragsFiles(event)) {
		event.preventDefault();
		event.dataTransfer.dropEffect = 'copy';
	}
});

window.addEventListener('drop', (event) => {
	if (dragsFiles(event)) {
		event.preventDefault();
		dragDepth = 0;
		document.body.classList.remove('dragging');
		addFiles(event.dataTransfer.files);
	}
});

downloadAllButton.addEventListener('click', downloadAll);
copyAllButton.addEventListener('click', copyAll);
clearAllButton.addEventListener('click', clearAll);
updateToolbar();
