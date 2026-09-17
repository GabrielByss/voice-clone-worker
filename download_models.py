"""Pobiera wagi obu silników do cache HF (HF_HOME) podczas `docker build`; każda część = osobna warstwa.

    python download_models.py voxcpm2 | qwen_base | qwen_tok | all

voxcpm2   openbmb/VoxCPM2                 ~5,0 GB (model.safetensors 4,6 GB + audiovae.pth 0,4 GB)
qwen_base Qwen/Qwen3-TTS-12Hz-1.7B-Base   ~4,5 GB (model 3,9 GB + speech_tokenizer/ 0,7 GB)
qwen_tok  Qwen/Qwen3-TTS-Tokenizer-12Hz   ~0,7 GB (osobny tokenizer; model Base ma swój w speech_tokenizer/, ale trzymamy dla pewności)
Pełne repozytoria (bez allow_patterns), żeby snapshot w cache był kompletny i działał z local_files_only=True.
"""
import os
import sys

from huggingface_hub import snapshot_download

PARTS = {
    "voxcpm2": os.environ.get("VOICE_VOXCPM_MODEL_ID", "openbmb/VoxCPM2"),
    "qwen_base": os.environ.get("VOICE_QWEN_MODEL_ID", "Qwen/Qwen3-TTS-12Hz-1.7B-Base"),
    "qwen_tok": os.environ.get("VOICE_QWEN_TOKENIZER_ID", "Qwen/Qwen3-TTS-Tokenizer-12Hz"),
}


def main() -> None:
    part = (sys.argv[1] if len(sys.argv) > 1 else "all").strip().lower()
    names = list(PARTS) if part == "all" else [part]
    token = os.environ.get("HF_TOKEN") or None
    for name in names:
        repo = PARTS.get(name)
        if not repo:
            print(f"[bake] nieznana część: {name} (dostępne: {', '.join(PARTS)})", file=sys.stderr)
            sys.exit(2)
        print(f"[bake] {name}: {repo} -> HF_HOME={os.environ.get('HF_HOME', '~/.cache/huggingface')}")
        path = snapshot_download(repo_id=repo, token=token, max_workers=4)
        print(f"[bake] ok: {path}")
    print("[bake] MODELS_BAKED")


if __name__ == "__main__":
    main()
