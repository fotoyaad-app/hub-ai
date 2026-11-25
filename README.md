# hub-ai

Background AI for the Hub, run on GitHub Actions (CPU runners).

| Job | Model | Output to the Hub |
| --- | --- | --- |
| `sigblur` | pixel signature + Laplacian | 100%-similar fingerprints, blurry/dark score |
| `music` | `laion/larger_clap_music` (CLAP) | 7 moods, genre/vocals/instrument/tempo tags, sound embedding for "similar songs" |
| `caption` | `microsoft/Florence-2-large` (0.77B, int8 dynamic) | one-sentence caption per photo (search) |
| `tags` | `google/siglip-so400m-patch14-384` (0.9B) | zero-shot photo tags (search) |
| `video` | SigLIP + Florence-2 on 3 keyframes of the 720p preview | video tags + caption (search) |
| `docs` | pypdf / python-docx / plain text (+ Florence-2 OCR for scanned PDFs) | document text (search) |

* The runner gets only numeric ids; the Hub resolves them to media, so file names and paths never reach GitHub.
* Media is read through the Hub's key-checked endpoints (`/bgai/gh/*`); results are posted back. Nothing is stored in this repo, in artifacts or in logs (logs print counts only).
* Read-only on media: nothing is moved or deleted in pCloud or R2.
* Locked, locked-album, Hidden and Safe photos are never sent.
* Resumable: the Hub keeps what is done (D1 `ai_done`, `px_cap`, `px_sig`, `px_blur`, `music_ai`); every run continues where the last stopped.
* Progress: each shard reports every 5 minutes; the Hub shows it under Cloudflare › Background AI (percent per job, last update, speed, time left).
* Budget: the Hub's `bgai_cfg.monthMin` caps runner minutes per month; while work is left and minutes remain, each run starts the next (chain).

Secrets: `HUB_URL`, `HUB_AI_KEY` (the Hub stores only its SHA-256). Both come only from repository secrets and are masked in logs.

Schedule: every 6 h (new items only); while a backlog is left each run starts the next one. Standard `ubuntu-latest` runners, at most 4 in parallel, model downloads cached with `actions/cache`. Only `actions/checkout`, `actions/setup-python` and `actions/cache` are used.
