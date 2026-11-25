# hub-ai

Background AI for the Hub, run on GitHub Actions (CPU runners).

| Job | Model | Output to the Hub |
| --- | --- | --- |
| `sigblur` | pixel signature + Laplacian | 100%-similar fingerprints, blurry/dark score |
| `music` | `laion/larger_clap_music` (CLAP) | 7 moods, genre/vocals/instrument/tempo tags, sound embedding for "similar songs" |
| `caption` | `microsoft/Florence-2-large` (0.77B, int8 dynamic) | one-sentence caption per photo (search) |
| `tags` | `google/siglip-so400m-patch14-384` (0.9B) | zero-shot photo tags (search) |

* Media is read through the Hub's key-checked endpoints (`/bgai/gh/*`); results are posted back. Nothing is stored in this repo, in artifacts or in logs (logs print counts only).
* Read-only on media: nothing is moved or deleted in pCloud or R2.
* Locked, locked-album, Hidden and Safe photos are never sent.
* Resumable: the Hub keeps what is done (D1 `ai_done`, `px_cap`, `px_sig`, `px_blur`, `music_ai`); every run continues where the last stopped.
* Progress: each shard reports every 5 minutes; the Hub shows it under Cloudflare › Background AI (percent per job, last update, speed, time left).
* Budget: the Hub's `bgai_cfg.monthMin` caps runner minutes per month; while work is left and minutes remain, each run starts the next (chain).

Secrets: `HUB_URL`, `HUB_AI_KEY` (the Hub stores only its SHA-256).
