# Stand-ins for the modules the OCR imports but Pyodide lacks (paddle,
# paddleocr, Levenshtein), running the ONNX models.

import math
import sys
import types

import cv2
import numpy as np
import pyclipper

DETECTION_MODEL = 'PP-OCRv4_mobile_det'

_infer = None # (model, input) -> output, model is 'det' or 'rec'


def install(infer):
	global _infer
	_infer = infer
	for name, module in (('paddle', paddle), ('paddle.inference', paddle.inference), ('paddleocr', paddleocr), ('Levenshtein', Levenshtein)):
		sys.modules[name] = module


# --- paddle ---

class _Config():
	def __init__(self, model_file=None, params_file=None):
		self.model_file = model_file

	# disable_gpu(), enable_mkldnn(), enable_new_ir(), ...: nothing to configure
	def __getattr__(self, name):
		return lambda *args, **kwargs: None


class _Tensor():
	def __init__(self, predictor):
		self._predictor = predictor

	def reshape(self, shape):
		pass

	def copy_from_cpu(self, array):
		self._predictor.input = array

	def copy_to_cpu(self):
		return self._predictor.output


# The recognizer is the only Paddle Inference model of the OCR.
class _Predictor():
	def __init__(self, config):
		self.input = None
		self.output = None

	def get_input_names(self):
		return ['x']

	def get_output_names(self):
		return ['softmax_2.tmp_0']

	def get_input_handle(self, name):
		return _Tensor(self)

	def get_output_handle(self, name):
		return _Tensor(self)

	def run(self):
		self.output = _infer('rec', np.ascontiguousarray(self.input, dtype=np.float32))


paddle = types.ModuleType('paddle')
paddle.inference = types.ModuleType('paddle.inference')
paddle.inference.Config = _Config
paddle.inference.create_predictor = _Predictor


# --- paddleocr ---

# paddlex 3.7.2 TextDetection with the PP-OCRv4_mobile_det configuration
# (its inference.yml); unclip_ratio and the thresholds can be overridden like
# in paddlex.
class TextDetection():
	def __init__(self, model_name=DETECTION_MODEL, device=None, enable_mkldnn=None, thresh=0.3, box_thresh=0.6, unclip_ratio=1.5, limit_side_len=960):
		if model_name != DETECTION_MODEL:
			raise ValueError(f'the in-browser OCR only has the {DETECTION_MODEL} text detection model, not {model_name}')
		self.limit_side_len = limit_side_len
		self.max_side_limit = 4000
		self.mean = [0.485, 0.456, 0.406]
		self.std = [0.229, 0.224, 0.225]
		self.scale = 1 / 255.
		self.thresh = thresh
		self.box_thresh = box_thresh
		self.unclip_ratio = unclip_ratio
		self.max_candidates = 1000
		self.min_size = 3

	def predict(self, img, batch_size=1):
		src_h, src_w = img.shape[:2]
		pred = _infer('det', self._normalize(self._resize(img)))
		boxes, scores = self._postprocess(pred, src_h, src_w)
		return [{'dt_polys': boxes, 'dt_scores': scores}]

	# DetResizeForTest.resize_image_type0 with limit_type 'max'
	def _resize(self, img):
		h, w = img.shape[:2]
		if h + w < 64:
			padded = np.zeros((max(32, h), max(32, w), img.shape[2]), np.uint8)
			padded[:h, :w] = img
			img = padded
			h, w = img.shape[:2]
		ratio = float(self.limit_side_len) / max(h, w) if max(h, w) > self.limit_side_len else 1.0
		resize_h, resize_w = int(h * ratio), int(w * ratio)
		if max(resize_h, resize_w) > self.max_side_limit:
			ratio = float(self.max_side_limit) / max(resize_h, resize_w)
			resize_h, resize_w = int(resize_h * ratio), int(resize_w * ratio)
		resize_h = max(int(round(resize_h / 32) * 32), 32)
		resize_w = max(int(round(resize_w / 32) * 32), 32)
		if resize_h == h and resize_w == w:
			return img
		return cv2.resize(img, (int(resize_w), int(resize_h)))

	# NormalizeImage (HWC), ToCHWImage and ToBatch
	def _normalize(self, img):
		if img.shape[2] == 4:
			img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
		alpha = [self.scale / s for s in self.std]
		beta = [-m / s for m, s in zip(self.mean, self.std)]
		channels = list(cv2.split(img))
		for c in range(img.shape[2]):
			channels[c] = channels[c].astype(np.float32) * alpha[c] + beta[c]
		return cv2.merge(channels).transpose((2, 0, 1))[None].astype(np.float32)

	# DBPostProcess with box_type 'quad', score_mode 'fast' and no dilation
	def _postprocess(self, pred, src_h, src_w):
		pred = pred[0, 0]
		bitmap = pred > self.thresh
		height, width = bitmap.shape
		contours, _ = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
		boxes, scores = [], []
		for contour in contours[:self.max_candidates]:
			points, short_side = self._mini_box(contour)
			if short_side < self.min_size:
				continue
			points = np.array(points)
			score = self._box_score_fast(pred, points.reshape(-1, 2))
			if self.box_thresh > score:
				continue
			box = self._unclip(points).reshape(-1, 1, 2)
			box, short_side = self._mini_box(box)
			if short_side < self.min_size + 2:
				continue
			box = np.array(box)
			for i in range(box.shape[0]):
				box[i, 0] = max(0, min(round(box[i, 0] * src_w / width), src_w))
				box[i, 1] = max(0, min(round(box[i, 1] * src_h / height), src_h))
			boxes.append(box.astype(np.int16))
			scores.append(score)
		return np.array(boxes, dtype=np.int16), scores

	@staticmethod
	def _mini_box(contour):
		bounding_box = cv2.minAreaRect(contour)
		points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda p: p[0])
		index_1, index_4 = (0, 1) if points[1][1] > points[0][1] else (1, 0)
		index_2, index_3 = (2, 3) if points[3][1] > points[2][1] else (3, 2)
		return [points[index_1], points[index_2], points[index_3], points[index_4]], min(bounding_box[1])

	@staticmethod
	def _box_score_fast(bitmap, box):
		h, w = bitmap.shape[:2]
		box = box.copy()
		xmin = max(0, min(math.floor(box[:, 0].min()), w - 1))
		xmax = max(0, min(math.ceil(box[:, 0].max()), w - 1))
		ymin = max(0, min(math.floor(box[:, 1].min()), h - 1))
		ymax = max(0, min(math.ceil(box[:, 1].max()), h - 1))
		mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
		box[:, 0] -= xmin
		box[:, 1] -= ymin
		cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
		return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]

	def _unclip(self, box):
		distance = cv2.contourArea(box) * self.unclip_ratio / cv2.arcLength(box, True)
		offset = pyclipper.PyclipperOffset()
		offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
		try:
			return np.array(offset.Execute(distance))
		except ValueError:
			return np.array(offset.Execute(distance)[0])


paddleocr = types.ModuleType('paddleocr')
paddleocr.TextDetection = TextDetection


# --- Levenshtein ---

# Normalized indel similarity, like python-Levenshtein's ratio().
def ratio(s1, s2, *, processor=None, score_cutoff=None):
	if processor is not None:
		s1, s2 = processor(s1), processor(s2)
	if not s1 and not s2:
		similarity = 1.0
	else:
		# length of the longest common subsequence
		previous = [0] * (len(s2) + 1)
		for a in s1:
			current = [0]
			for j, b in enumerate(s2, 1):
				current.append(previous[j - 1] + 1 if a == b else max(previous[j], current[j - 1]))
			previous = current
		similarity = 2.0 * previous[-1] / (len(s1) + len(s2))
	if score_cutoff is not None and similarity < score_cutoff:
		return 0.0
	return similarity


Levenshtein = types.ModuleType('Levenshtein')
Levenshtein.ratio = ratio
