"""
Voice clone worker — handler RunPod Serverless (kolejkowy: /run, /runsync, /status).

Dwa silniki TTS z klonowaniem głosu z krótkiej próbki (pole `engine`), ten sam kontrakt wyjścia co workery
ACE-Step / MiniMax (audio_base64 | upload_url), żeby backend finalizował joby jednym kodem:
  voxcpm2   OpenBMB VoxCPM2 2B (Apache-2.0, 30 języków, 48 kHz). Klon z samej próbki (reference cloning) albo
            z próbką + jej transkrypcją (prompt/continuation cloning, wierniejszy).
  qwen3tts  Qwen3-TTS-12Hz-1.7B-Base (Apache-2.0, 10 języków, 24 kHz). Klon z próbką + transkrypcją, albo bez
            transkrypcji w trybie x_vector_only (słabszy).

Wejście (job["input"]):
  engine            str   "voxcpm2" | "qwen3tts"                                        [wymagane]
  text              str   tekst do przeczytania, <= VOICE_MAX_TEXT_CHARS (1000)          [wymagane]
  language          str   kod ISO ("en", "de", "fr", ...). VoxCPM2 wykrywa język sam; Qwen dostaje nazwę
                          języka z mapy QWEN_LANGUAGES, nieznany kod => "Auto"          domyślnie "en"
  ref_audio_base64  str   próbka głosu (wav/mp3/m4a, dowolna częstotliwość)              jedno z dwóch
  ref_audio_url     str   URL próbki (HTTPS)                                             wymagane
  ref_text          str   transkrypcja próbki; jeśli podana, oba silniki używają trybu wierniejszego
  seed              int   -1 = losowy
  audio_format      str   mp3 | wav                                                       domyślnie mp3
  mp3_bitrate       str   np. "128k"                                                      domyślnie "128k"
  return_base64     bool  dołącz audio_base64 (domyślnie true, gdy brak upload_url)
  upload_url / upload_content_type / public_url    presigned PUT jak w pozostałych workerach
  cfg_value         float (voxcpm2) skala CFG, referencyjnie 2.0
  inference_timesteps int (voxcpm2) kroki CFM, referencyjnie 10
  normalize         bool  (voxcpm2) normalizacja tekstu (liczby itp., wetext – zh/en)   domyślnie false
  ping              bool  health-check: zwraca gpu, vram, załadowane silniki

Wyjście: audio_base64? | audio_url?, format, mime, sample_rate, duration, size_bytes, engine, model, attribution,
         language, ref_mode ("transcript" | "reference_only"), ref_seconds, text_chars, seed, generation_seconds,
         uploaded, upload_status?
"""
from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
import urllib.request

import numpy as np

MODE = os.environ.get("VOICE_MODE", "serverless").strip().lower()
OUTPUT_DIR = os.environ.get("VOICE_OUTPUT_DIR", "/tmp/voice-output")
VOXCPM_ID = os.environ.get("VOICE_VOXCPM_MODEL_ID", "openbmb/VoxCPM2").strip()
QWEN_ID = os.environ.get("VOICE_QWEN_MODEL_ID", "Qwen/Qwen3-TTS-12Hz-1.7B-Base").strip()

ENGINE_VOXCPM = "voxcpm2"
ENGINE_QWEN = "qwen3tts"
ENGINE_INFO = {
    ENGINE_VOXCPM: {"model": "voxcpm2-2b", "attribution": "VoxCPM2 (OpenBMB, Apache-2.0)"},
    ENGINE_QWEN: {"model": "qwen3-tts-12hz-1.7b-base", "attribution": "Qwen3-TTS (Alibaba, Apache-2.0)"},
}
MIME = {"mp3": "audio/mpeg", "wav": "audio/wav"}

# Qwen3-TTS przyjmuje nazwy języków; poza listą => "Auto"
QWEN_LANGUAGES = {
    "en": "English", "de": "German", "fr": "French", "es": "Spanish", "it": "Italian", "pt": "Portuguese",
    "ru": "Russian", "zh": "Chinese", "ja": "Japanese", "ko": "Korean",
}


def log(msg: str) -> None:
    print(f"[voice-worker] {msg}", flush=True)


def env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except ValueError:
        return default


BAKED = env_bool("VOICE_BAKED", True)
MAX_TEXT_CHARS = int(env_float("VOICE_MAX_TEXT_CHARS", 1000))
REF_MAX_SECONDS = env_float("VOICE_REF_MAX_SECONDS", 30.0)
REF_MAX_BYTES = int(env_float("VOICE_REF_MAX_MB", 25.0) * 1024 * 1024)
ENGINES_ENABLED = {e.strip().lower() for e in os.environ.get("VOICE_ENGINES", "voxcpm2,qwen3tts").split(",") if e.strip()}
PRELOAD_RAW = os.environ.get("VOICE_PRELOAD", "all").strip().lower()

if BAKED and not os.environ.get("HF_HUB_OFFLINE"):
    os.environ["HF_HUB_OFFLINE"] = "1"  # wagi z obrazu: zero ruchu do HF przy starcie
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

BOOT_T0 = time.time()

import torch  # noqa: E402

if not torch.cuda.is_available():
    raise RuntimeError("Voice worker wymaga CUDA (torch.cuda.is_available() == False)")

VRAM_GB = torch.cuda.get_device_properties(0).total_memory / 1e9
GPU_NAME = torch.cuda.get_device_name(0)
log(f"boot: gpu={GPU_NAME} vram={VRAM_GB:.1f}GB engines={sorted(ENGINES_ENABLED)} preload={PRELOAD_RAW} "
    f"offline={os.environ.get('HF_HUB_OFFLINE', '0')} hf_home={os.environ.get('HF_HOME', '')}")

_MODELS: dict = {}


def _snapshot_dir(repo_id: str, probe: str = "config.json") -> str | None:
    """Katalog snapshotu w cache HF (przez ścieżkę zcache'owanego pliku); None gdy brak."""
    try:
        from huggingface_hub import hf_hub_download

        return os.path.dirname(hf_hub_download(repo_id, probe, local_files_only=True))
    except Exception as exc:  # pragma: no cover
        log(f"snapshot dir for {repo_id} not resolved ({type(exc).__name__}: {exc})")
        return None


def _load_voxcpm():
    from voxcpm import VoxCPM

    src = (_snapshot_dir(VOXCPM_ID) if BAKED else None) or VOXCPM_ID
    optimize = env_bool("VOXCPM_OPTIMIZE", False)  # torch.compile przy każdym cold starcie = minuty bilowanego GPU
    model = VoxCPM.from_pretrained(src, load_denoiser=False, optimize=optimize, device="cuda", local_files_only=BAKED)
    sr = int(getattr(model.tts_model, "sample_rate", 48000))
    log(f"voxcpm2 loaded from {src}, sample_rate={sr}, optimize={optimize}")
    return model


def _load_qwen():
    from qwen_tts import Qwen3TTSModel

    attn = os.environ.get("QWEN_ATTN", "sdpa").strip() or "sdpa"
    # id repo (nie ścieżka): qwen_tts sam dociąga speech_tokenizer przez snapshot_download i honoruje HF_HUB_OFFLINE
    model = Qwen3TTSModel.from_pretrained(QWEN_ID, device_map="cuda:0", dtype=torch.bfloat16, attn_implementation=attn)
    log(f"qwen3tts loaded ({QWEN_ID}, attn={attn})")
    return model


_LOADERS = {ENGINE_VOXCPM: _load_voxcpm, ENGINE_QWEN: _load_qwen}


def get_model(engine: str):
    if engine not in ENGINES_ENABLED:
        raise ValueError(f"Silnik '{engine}' wyłączony na tym endpoincie (VOICE_ENGINES={sorted(ENGINES_ENABLED)}).")
    if engine not in _MODELS:
        t0 = time.time()
        _MODELS[engine] = _LOADERS[engine]()
        log(f"{engine}: model ready in {time.time() - t0:.1f}s")
    return _MODELS[engine]


# Preload przy starcie (domyślnie oba) – błąd ładowania ma zatrzymać workera od razu.
_preload = set() if PRELOAD_RAW in {"", "none", "false", "0"} else (
    set(ENGINES_ENABLED) if PRELOAD_RAW == "all" else {e.strip() for e in PRELOAD_RAW.split(",") if e.strip()})
for _eng in sorted(_preload):
    if _eng in ENGINES_ENABLED:
        get_model(_eng)
log(f"boot done in {time.time() - BOOT_T0:.1f}s, loaded={sorted(_MODELS)}, "
    f"vram_used={torch.cuda.memory_allocated() / 1e9:.1f}GB")


# --------------------------------------------------------------------------------------
# Pomocnicze
# --------------------------------------------------------------------------------------
def _clamp(value, lo, hi):
    return max(lo, min(hi, value))


def _run(cmd: list[str], timeout: int = 120) -> None:
    subprocess.run(cmd, check=True, timeout=timeout, capture_output=True)


def _probe_seconds(path: str) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", path],
        check=True, capture_output=True, text=True, timeout=60,
    ).stdout.strip()
    try:
        return float(out)
    except ValueError:
        return 0.0


def _put_upload(url: str, data: bytes, content_type: str) -> int:
    req = urllib.request.Request(url, data=data, method="PUT")
    req.add_header("Content-Type", content_type)
    req.add_header("Content-Length", str(len(data)))
    with urllib.request.urlopen(req, timeout=120) as resp:  # nosec - URL podaje nasz backend
        return int(resp.status)


def fetch_reference(inp: dict, work_dir: str) -> tuple[str, float]:
    """Zapisuje próbkę (base64 albo URL), konwertuje ffmpeg-iem do mono WAV 24 kHz, przycina do REF_MAX_SECONDS."""
    src = os.path.join(work_dir, "ref_src")
    b64 = inp.get("ref_audio_base64")
    url = inp.get("ref_audio_url")
    if b64:
        data = base64.b64decode(str(b64), validate=False)
    elif url:
        url = str(url)
        if not url.lower().startswith(("https://", "http://")):
            raise ValueError("ref_audio_url musi być adresem http(s).")
        with urllib.request.urlopen(url, timeout=60) as resp:  # nosec - URL z naszego backendu/aplikacji
            data = resp.read(REF_MAX_BYTES + 1)
    else:
        raise ValueError("Podaj próbkę głosu: 'ref_audio_base64' albo 'ref_audio_url'.")
    if len(data) > REF_MAX_BYTES:
        raise ValueError(f"Próbka głosu za duża (> {REF_MAX_BYTES // (1024 * 1024)} MB).")
    if len(data) < 1000:
        raise ValueError("Próbka głosu jest pusta albo uszkodzona.")
    with open(src, "wb") as fh:
        fh.write(data)
    ref = os.path.join(work_dir, "ref.wav")
    _run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-vn", "-ac", "1", "-ar", "24000",
          "-t", str(REF_MAX_SECONDS), "-c:a", "pcm_s16le", ref])
    seconds = _probe_seconds(ref)
    if seconds < 1.0:
        raise ValueError(f"Próbka głosu za krótka ({seconds:.1f} s); potrzeba co najmniej 1 s, najlepiej 5–15 s.")
    return ref, seconds


def build_request(inp: dict) -> dict:
    engine = str(inp.get("engine") or "").strip().lower()
    if engine not in ENGINE_INFO:
        raise ValueError(f"Podaj 'engine': {' | '.join(ENGINE_INFO)}.")
    text = str(inp.get("text") or "").strip()
    if not text:
        raise ValueError("Podaj 'text' do przeczytania.")
    if len(text) > MAX_TEXT_CHARS:
        raise ValueError(f"'text' za długi ({len(text)} znaków, limit {MAX_TEXT_CHARS}).")
    language = str(inp.get("language") or "en").strip().lower()[:5]
    ref_text = str(inp.get("ref_text") or "").strip()[:2000] or None
    seed = int(inp.get("seed", -1))
    if seed < 0:
        seed = int.from_bytes(os.urandom(4), "little")
    audio_format = str(inp.get("audio_format") or "mp3").strip().lower()
    if audio_format not in MIME:
        audio_format = "mp3"
    return {
        "engine": engine,
        "text": text,
        "language": language,
        "ref_text": ref_text,
        "seed": seed,
        "format": audio_format,
        "mp3_bitrate": str(inp.get("mp3_bitrate") or "128k"),
        "cfg_value": float(inp.get("cfg_value", 2.0) or 2.0),
        "inference_timesteps": int(_clamp(int(inp.get("inference_timesteps", 10) or 10), 4, 50)),
        "normalize": bool(inp.get("normalize", False)),
    }


def run_voxcpm(model, req: dict, ref_path: str) -> tuple[np.ndarray, int]:
    torch.manual_seed(req["seed"])
    kwargs = dict(
        text=req["text"],
        reference_wav_path=ref_path,
        cfg_value=req["cfg_value"],
        inference_timesteps=req["inference_timesteps"],
        normalize=req["normalize"],
        retry_badcase=True,
    )
    if req["ref_text"]:
        kwargs.update(prompt_wav_path=ref_path, prompt_text=req["ref_text"])
    wav = model.generate(**kwargs)
    return np.asarray(wav, dtype=np.float32), int(getattr(model.tts_model, "sample_rate", 48000))


def run_qwen(model, req: dict, ref_path: str) -> tuple[np.ndarray, int]:
    torch.manual_seed(req["seed"])
    language = QWEN_LANGUAGES.get(req["language"], "Auto")
    wavs, sr = model.generate_voice_clone(
        text=req["text"],
        language=language,
        ref_audio=ref_path,
        ref_text=req["ref_text"],
        x_vector_only_mode=not bool(req["ref_text"]),
    )
    wav = wavs[0] if isinstance(wavs, (list, tuple)) else wavs
    if isinstance(wav, torch.Tensor):
        wav = wav.float().cpu().numpy()
    return np.asarray(wav, dtype=np.float32), int(sr)


RUNNERS = {ENGINE_VOXCPM: run_voxcpm, ENGINE_QWEN: run_qwen}


def to_mono_1d(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        # [channels, samples] -> średnia; [samples, channels] -> średnia po osi 1
        audio = audio.mean(axis=0) if audio.shape[0] <= 8 and audio.shape[0] < audio.shape[1] else audio.mean(axis=1)
    return np.clip(audio.reshape(-1), -1.0, 1.0)


def encode(audio: np.ndarray, sr: int, req: dict, work_dir: str) -> tuple[bytes, str]:
    import soundfile as sf

    wav_path = os.path.join(work_dir, "out.wav")
    sf.write(wav_path, audio, sr, subtype="PCM_16")
    if req["format"] == "wav":
        with open(wav_path, "rb") as fh:
            return fh.read(), "wav"
    mp3_path = os.path.join(work_dir, "out.mp3")
    _run(["ffmpeg", "-y", "-loglevel", "error", "-i", wav_path, "-codec:a", "libmp3lame", "-b:a", req["mp3_bitrate"], mp3_path], timeout=300)
    with open(mp3_path, "rb") as fh:
        return fh.read(), "mp3"


def generate(inp: dict, job_id: str) -> dict:
    req = build_request(inp)
    model = get_model(req["engine"])
    work_dir = os.path.join(OUTPUT_DIR, job_id)
    os.makedirs(work_dir, exist_ok=True)
    try:
        ref_path, ref_seconds = fetch_reference(inp, work_dir)
        ref_mode = "transcript" if req["ref_text"] else "reference_only"
        log(f"job {job_id}: engine={req['engine']} lang={req['language']} text_chars={len(req['text'])} "
            f"ref={ref_seconds:.1f}s mode={ref_mode} seed={req['seed']}")
        t0 = time.time()
        audio, sr = RUNNERS[req["engine"]](model, req, ref_path)
        audio = to_mono_1d(audio)
        elapsed = time.time() - t0
        if audio.shape[0] < sr // 10:
            raise RuntimeError("Model zwrócił pusty dźwięk.")
        data, fmt = encode(audio, sr, req, work_dir)
        duration = audio.shape[0] / sr
        content_type = str(inp.get("upload_content_type") or MIME[fmt])
        out = {
            "format": fmt,
            "mime": content_type,
            "sample_rate": sr,
            "duration": round(duration, 3),
            "size_bytes": len(data),
            "engine": req["engine"],
            "model": ENGINE_INFO[req["engine"]]["model"],
            "attribution": ENGINE_INFO[req["engine"]]["attribution"],
            "language": req["language"],
            "ref_mode": ref_mode,
            "ref_seconds": round(ref_seconds, 2),
            "text_chars": len(req["text"]),
            "seed": req["seed"],
            "generation_seconds": round(elapsed, 3),
            "uploaded": False,
        }
        upload_url = inp.get("upload_url")
        if upload_url:
            status = _put_upload(str(upload_url), data, content_type)
            out["uploaded"] = 200 <= status < 300
            out["upload_status"] = status
            if out["uploaded"] and inp.get("public_url"):
                out["audio_url"] = str(inp["public_url"])
        want_b64 = inp.get("return_base64")
        if want_b64 is None:
            want_b64 = not bool(upload_url)
        if want_b64:
            out["audio_base64"] = base64.b64encode(data).decode("ascii")
        log(f"job {job_id}: done in {elapsed:.1f}s, audio {duration:.1f}s @ {sr} Hz, {len(data)} bytes")
        return out
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
        torch.cuda.empty_cache()


def handler(job: dict) -> dict:
    inp = job.get("input") or {}
    if inp.get("ping"):
        return {
            "ok": True, "engines": sorted(ENGINES_ENABLED), "loaded": sorted(_MODELS), "gpu": GPU_NAME,
            "vram_gb": round(VRAM_GB, 1), "vram_used_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
            "uptime_s": round(time.time() - BOOT_T0, 1),
        }
    try:
        return generate(inp, str(job.get("id") or f"local-{int(time.time())}"))
    except ValueError as exc:
        return {"error": str(exc)}
    except torch.cuda.OutOfMemoryError as exc:  # pragma: no cover
        torch.cuda.empty_cache()
        return {"error": f"CUDA OOM ({VRAM_GB:.0f} GB VRAM): {exc}"}
    except Exception as exc:  # pragma: no cover
        log(traceback.format_exc())
        return {"error": f"{type(exc).__name__}: {exc}"}


if __name__ == "__main__":
    if MODE in {"selftest", "pod"}:
        test_path = os.environ.get("VOICE_TEST_INPUT", os.path.join(os.path.dirname(__file__), "test_input_voxcpm_en.json"))
        with open(test_path, "r", encoding="utf-8") as fh:
            job = json.load(fh)
        job.setdefault("id", "selftest")
        res = handler(job)
        b64 = res.pop("audio_base64", None)
        print(json.dumps(res, indent=2, ensure_ascii=False))
        if b64:
            out_file = os.path.join(os.getcwd(), f"selftest.{res.get('format', 'mp3')}")
            with open(out_file, "wb") as fh:
                fh.write(base64.b64decode(b64))
            print(f"zapisano {out_file}")
    else:
        import runpod  # noqa: E402

        runpod.serverless.start({"handler": handler})
