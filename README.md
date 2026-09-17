# voice-clone-worker

Worker RunPod Serverless: TTS z klonowaniem głosu z krótkiej próbki, dwa silniki w jednym obrazie do bake-offu
(pole `engine`): **VoxCPM2 2B** (OpenBMB, Apache-2.0, 30 języków, 48 kHz) i **Qwen3-TTS-12Hz-1.7B-Base**
(Alibaba, Apache-2.0, 10 języków, 24 kHz). Ten sam kontrakt wyjścia co workery ACE-Step / MiniMax.

Źródło prawdy: katalog `runpod/voice` w repo `ringtones_repo` (to repo jest kopią do budowania obrazu).
Opis, pomiary i decyzje: `ringtones_repo/runpod/README.md`.

## Build

Actions → **Build voice clone worker image** → tag `v1`, `v2`, … → `ghcr.io/gabrielbyss/voice-clone-worker:<tag>`.
Obraz ~16 GB (wagi ~10 GB w 3 warstwach). Nowa wersja = nowy tag.

## Kontrakt

`POST https://api.runpod.ai/v2/<ENDPOINT_ID>/run`

```json
{ "input": { "engine": "voxcpm2", "text": "Hey, pick up the phone!", "language": "en",
             "ref_audio_url": "https://…/voice.wav", "ref_text": "opcjonalna transkrypcja próbki",
             "audio_format": "mp3", "return_base64": true } }
```

Zamiast `ref_audio_url` można podać `ref_audio_base64`. Bez `ref_text` VoxCPM2 działa w trybie reference cloning,
Qwen3-TTS w trybie x-vector (słabszy). Odpowiedź: `audio_base64` | `audio_url`, `format`, `sample_rate`, `duration`,
`engine`, `model`, `ref_mode`, `generation_seconds`. `{"input":{"ping":true}}` → `{ok, engines, loaded, gpu, vram_gb}`.

Env: `VOICE_ENGINES` (które silniki), `VOICE_PRELOAD` (`all` | `none` | lista), `VOXCPM_OPTIMIZE` (`false`: bez
torch.compile przy starcie), `QWEN_ATTN` (`sdpa`), `VOICE_MAX_TEXT_CHARS` (1000), `VOICE_REF_MAX_SECONDS` (30).
