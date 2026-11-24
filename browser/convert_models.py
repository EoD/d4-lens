#!/usr/bin/env python3
# Converts the OCR's Paddle models to ONNX (det.onnx, rec.onnx).
# Usage: convert_models.py DET_MODEL_DIR REC_MODEL_DIR OUT_DIR

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

OPSET = 11
# onnxslim folds the batch normalizations into the weights, which turns the
# recognizer's denormal weights (35% of all) into tiny normal ones that still
# produce denormal intermediate results. WebAssembly can't flush denormals to
# zero, and computing with them made the recognizer 3x slower. Zeroing weights
# below this threshold doesn't change the OCR's results.
TINY_WEIGHT = 1e-12


def convert(model_dir, model_file, out_file):
	subprocess.run([
		'paddle2onnx',
		'--model_dir', str(model_dir),
		'--model_filename', model_file,
		'--params_filename', 'inference.pdiparams',
		'--save_file', str(out_file),
		'--opset_version', str(OPSET),
		'--optimize_tool', 'None',
		'--enable_onnx_checker', 'True',
	], check=True)


def zero_tiny_weights(model):
	count = 0
	tensors = list(model.graph.initializer)
	tensors += [attr.t for node in model.graph.node if node.op_type == 'Constant' for attr in node.attribute if attr.name == 'value']
	for tensor in tensors:
		weights = numpy_helper.to_array(tensor)
		if weights.dtype != np.float32:
			continue
		tiny = (weights != 0) & (np.abs(weights) < TINY_WEIGHT)
		if tiny.any():
			weights = weights.copy()
			weights[tiny] = 0
			tensor.CopyFrom(numpy_helper.from_array(weights, tensor.name))
			count += int(tiny.sum())
	return count


def build(model_dir, model_file, out_file):
	with tempfile.TemporaryDirectory() as tmp:
		# paddle2onnx may write next to the model, which can be read-only
		model_copy = Path(tmp) / 'model'
		shutil.copytree(model_dir, model_copy, ignore=shutil.ignore_patterns('.*'))
		converted = Path(tmp) / 'converted.onnx'
		slimmed = Path(tmp) / 'slimmed.onnx'
		convert(model_copy, model_file, converted)
		subprocess.run(['onnxslim', str(converted), str(slimmed)], check=True)
		model = onnx.load(slimmed)
	count = zero_tiny_weights(model)
	onnx.checker.check_model(model)
	onnx.save(model, out_file)
	print(f'{out_file}: {out_file.stat().st_size} bytes, zeroed {count} weights below {TINY_WEIGHT:g}', flush=True)


def main():
	if len(sys.argv) != 4:
		raise SystemExit(f'usage: {sys.argv[0]} DET_MODEL_DIR REC_MODEL_DIR OUT_DIR')
	det_dir, rec_dir, out_dir = map(Path, sys.argv[1:])
	out_dir.mkdir(parents=True, exist_ok=True)
	build(det_dir, 'inference.json', out_dir / 'det.onnx')
	build(rec_dir, 'inference.pdmodel', out_dir / 'rec.onnx')


if __name__ == '__main__':
	main()
