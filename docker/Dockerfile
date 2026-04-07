ARG VLLM_BASE_IMAGE=artifactory.devops.xiaohongshu.com/dockerhub/vllm/vllm-openai
ARG VLLM_BASE_TAG=v0.19.0
FROM ${VLLM_BASE_IMAGE}:${VLLM_BASE_TAG}
ARG APP_DIR=/workspace/vllm-omni
WORKDIR ${APP_DIR}
COPY . .

# Install system dependencies
RUN apt-get update && \
    apt-get install -y espeak-ng ffmpeg git sox libsox-fmt-all jq && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN uv pip install --system ".[dev]" && \
    cd /workspace/vllm-omni/ODTSR && \
    uv pip install setuptools==81.0.0 && \
    uv pip install -e . --no-build-isolation && \
    uv pip install basicsr && \
    uv pip install nvitop && \
    sed -i '8s/from torchvision.transforms.functional_tensor import rgb_to_grayscale/from torchvision.transforms._functional_tensor import rgb_to_grayscale/' /usr/local/lib/python3.12/dist-packages/basicsr/data/degradations.py




RUN ln -sf /usr/bin/python3 /usr/bin/python

ENTRYPOINT []
