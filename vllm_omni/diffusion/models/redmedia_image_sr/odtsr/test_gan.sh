#!/usr/bin/env bash
export qwen_path=/workspace/models/RedMedia-Image-SR-1.0/Qwen-Image

EXP_DATE="20251030-035453"
ALIGN_METHOD="wavelet" # adain no

CUDA_VISIBLE_DEVICES="0" python infer.py \
  --input_path /sgl-workspace/ODTSR/input \
  --output_path /sgl-workspace/vllm-omni/output \
  --trained_ckpt /workspace/models/RedMedia-Image-SR-1.0/ODTSR/weight.pth \
  --scale 2.0 \
  --cfg 1.0 \
  --align_method ${ALIGN_METHOD}
