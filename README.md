# D4 Lens

A web page that turns Diablo IV screenshots (1080p to 4K) into item tooltip
JSON, using [d4-item-tooltip-ocr](https://github.com/mxtsdev/d4-item-tooltip-ocr).
The OCR runs in your browser: screenshots never leave your device.

**Use it at <https://eod.github.io/d4-lens/>.**

- Add screenshots with the file picker, drag and drop, or <kbd>Ctrl</kbd>+<kbd>V</kbd>.
- Each screenshot gets a card with the item JSON (with a copy button) or the
  reason it failed.
- "Download all" and "Copy all" give the items of all cards as one JSON array.

It needs a browser with WebAssembly JSPI: current Chrome, Edge or Firefox.
The first visit downloads about 40 MB, which the browser caches.

## How it works

[Pyodide](https://pyodide.org/) runs the unmodified OCR script,
[onnxruntime-web](https://onnxruntime.ai/) its models, converted to ONNX.
The GitHub Pages build loads the large runtime files and the OCR's template
images from [jsDelivr](https://www.jsdelivr.com/), checked against hashes
recorded at build time.

## Building

The site is built with Docker from a d4-item-tooltip-ocr checkout:

```sh
docker build --build-context ocr=../d4-item-tooltip-ocr --target pages --output site .
python3 -m http.server -d site 8046   # http://127.0.0.1:8046/
```

The build converts the models and fails unless the in-browser OCR gives the
same JSON as the original on the OCR's test images. By default the site
serves everything itself (55 MB, also works offline); with
`--build-arg VENDOR=cdn` it loads the runtime and the template images from
jsDelivr (15 MB).

`.github/workflows/pages.yml` builds the site with `VENDOR=cdn` and publishes
it with GitHub Pages (Settings > Pages > Source: GitHub Actions). The
repository variables `OCR_REPOSITORY` and `OCR_REF` choose the OCR to build
with.

## License

[GNU Affero General Public License v3.0](LICENSE). The site's third-party
software and data keep their own licenses, listed on the site's
`NOTICE.txt`.
