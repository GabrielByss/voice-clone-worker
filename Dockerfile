# =============================================================================
# Voice clone worker — RunPod Serverless (queue-based, handler.py)
# Silniki (pole `engine` w zleceniu):
#   voxcpm2   OpenBMB VoxCPM2 2B (Apache-2.0, 30 języków, 48 kHz)         – klon z próbki, opcjonalnie z transkrypcją
#   qwen3tts  Qwen3-TTS-12Hz-1.7B-Base (Apache-2.0, 10 języków, 24 kHz)   – klon z próbki (x-vector) lub z transkrypcją
# Baza: pytorch/pytorch 2.8 CUDA 12.8 (Python 3.11). Pakiety modeli z --no-deps, żeby nie podmienić torcha i nie ciągnąć
# gradio/modelscope/funasr. Wagi (~10 GB) wypiekane w 3 warstwach do cache HF (/app/hf); handler ładuje offline.
#
# Build (x86_64!): docker build --platform linux/amd64 -t <registry>/voice-clone-worker:v1 runpod/voice
# =============================================================================
ARG BASE_IMAGE=pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime
FROM ${BASE_IMAGE}

ARG BAKE_MODELS=true
ARG HF_TOKEN=""
ARG VOXCPM_SPEC="voxcpm==2.0.3"
ARG QWEN_TTS_SPEC="qwen-tts==0.1.1"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONFAULTHANDLER=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HOME=/app/hf

# ffmpeg: mp3 + torchcodec; sox: zależność qwen-tts (tokenizer 25 Hz, nieużywany, ale import jest twardy)
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg sox libsox-fmt-all git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt \
    && pip install --no-deps "${VOXCPM_SPEC}" "${QWEN_TTS_SPEC}" \
    && python -c "import torch, torchaudio, transformers, voxcpm, qwen_tts; print('torch', torch.__version__, 'torchaudio', torchaudio.__version__, 'transformers', transformers.__version__)"

COPY download_models.py /app/download_models.py
RUN if [ "$BAKE_MODELS" = "true" ]; then HF_TOKEN="$HF_TOKEN" python /app/download_models.py voxcpm2; else echo "BAKE_MODELS=false"; fi
RUN if [ "$BAKE_MODELS" = "true" ]; then HF_TOKEN="$HF_TOKEN" python /app/download_models.py qwen_base; fi
RUN if [ "$BAKE_MODELS" = "true" ]; then HF_TOKEN="$HF_TOKEN" python /app/download_models.py qwen_tok; fi

COPY handler.py /app/handler.py

# VOICE_BAKED=true => HF_HUB_OFFLINE=1 (wagi z obrazu). Dla network volume: VOICE_BAKED=false, HF_HOME=/runpod-volume/hf
ENV VOICE_MODE=serverless \
    VOICE_BAKED=${BAKE_MODELS} \
    VOICE_ENGINES=voxcpm2,qwen3tts \
    VOICE_PRELOAD=all \
    VOICE_VOXCPM_MODEL_ID=openbmb/VoxCPM2 \
    VOICE_QWEN_MODEL_ID=Qwen/Qwen3-TTS-12Hz-1.7B-Base \
    VOXCPM_OPTIMIZE=false \
    VOICE_MAX_TEXT_CHARS=1000 \
    VOICE_REF_MAX_SECONDS=30 \
    VOICE_OUTPUT_DIR=/tmp/voice-output

CMD ["python", "-u", "/app/handler.py"]
