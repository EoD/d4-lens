# syntax=docker/dockerfile:1
# Built from a d4-item-tooltip-ocr checkout, the named build context "ocr"
# (--build-context ocr=PATH_OR_GIT_URL). Two targets:
#   server  the website with the OCR on the server (the last stage, the default):
#             docker build --build-context ocr=../d4-item-tooltip-ocr -t d4-item-tooltip-website .
#   pages   the in-browser build, a static site, written to a directory:
#             docker build --build-context ocr=../d4-item-tooltip-ocr --target pages --output site .
#           --build-arg VENDOR=cdn loads the large runtime files and the OCR's
#           template images from jsDelivr instead of the site (default:
#           VENDOR=self, the site serves them)
#           --build-arg SOURCE_URL=https://... links the page to its source code

# --- the OCR with its dependencies and models ---

# Python 3.12: the OCR needs numpy<2, which has no wheels for 3.13
FROM python:3.12-slim-trixie AS base

# libgomp for Paddle; GL and GLib for the non-headless OpenCV wheels that
# paddlex depends on
RUN apt-get update \
	&& apt-get install -y --no-install-recommends libgomp1 libgl1 libglib2.0-0t64 \
	&& rm -r /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
	PYTHONUNBUFFERED=1 \
	PIP_NO_CACHE_DIR=1 \
	PIP_DISABLE_PIP_VERSION_CHECK=1 \
	PIP_ROOT_USER_ACTION=ignore

ENV D4_OCR_DIR=/opt/d4-item-tooltip-ocr
WORKDIR $D4_OCR_DIR
COPY --from=ocr requirements.txt ./
COPY constraints.txt /tmp/
RUN pip install -r requirements.txt -c /tmp/constraints.txt

COPY --from=ocr LICENSE.txt d4-item-tooltip-ocr.py d4-vocabulary.txt ./
COPY --from=ocr templates templates/
COPY --from=ocr paddleocr-models paddleocr-models/
COPY --from=ocr tests tests/

RUN useradd --create-home --uid 1000 app
WORKDIR /app
COPY ocr_adapter.py ./
USER app
# download the text detection model now, so the container needs no network
# access and starts quickly
RUN python -c "import ocr_adapter; ocr_adapter.load_ocr('$D4_OCR_DIR')"
# the model is there; don't let paddlex probe the model hosters on start-up
ENV PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True

# --- the in-browser build (target "pages") ---

# The models converted to ONNX, checked against Paddle on the OCR's test images.
FROM base AS models
USER root
# paddle2onnx 2.1.0 writes a recognizer that onnxruntime refuses to load
RUN pip install paddle2onnx==2.0.2rc3 onnx==1.17.0 onnxslim==0.1.98 onnxruntime==1.31.0 -c /tmp/constraints.txt \
	&& install -d -o app /models
# as app: paddlex has the detection model in app's home
USER app
COPY browser/convert_models.py browser/ocr_shims.py browser/
COPY server.py ./
COPY tests/test_browser_shims.py tests/
RUN python browser/convert_models.py /home/app/.paddlex/official_models/PP-OCRv4_mobile_det $D4_OCR_DIR/paddleocr-models/en_PP-OCRv3_rec-d4_tooltip /models
RUN D4_ONNX_MODELS=/models python -m unittest -v tests/test_browser_shims.py

# Pyodide and onnxruntime-web, pinned; with VENDOR=cdn only their JavaScript
FROM python:3.12-slim-trixie AS vendor
ARG VENDOR=self
COPY browser/fetch_vendor.py /
RUN python /fetch_vendor.py "$VENDOR" /vendor

FROM python:3.12-slim-trixie AS site
ARG SOURCE_URL=
WORKDIR /src
COPY --from=ocr LICENSE.txt d4-item-tooltip-ocr.py d4-vocabulary.txt /ocr/
COPY --from=ocr templates /ocr/templates/
COPY ocr_adapter.py ./
COPY static static/
COPY browser browser/
COPY --from=models /models /models
COPY --from=vendor /vendor /vendor
RUN python browser/build_site.py ${SOURCE_URL:+--source-url "$SOURCE_URL"} /ocr /models /vendor /site

FROM scratch AS pages
COPY --from=site /site /

# --- the website with the OCR on the server (target "server", the default) ---

FROM base AS server
COPY server.py ./
COPY static static/
COPY tests tests/

EXPOSE 8044
HEALTHCHECK --start-period=60s --interval=30s --timeout=5s \
	CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8044/', timeout=4)"]
# inside the container, listen on all interfaces; the port mapping decides who can reach the port
CMD ["python", "server.py", "--host", "0.0.0.0"]
