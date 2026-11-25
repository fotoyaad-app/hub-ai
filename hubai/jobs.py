"""The four Hub AI jobs. Each job: load() once, then run(items, blobs) -> list of result dicts (id + fields, or id + err)."""
import io, os, math, base64, subprocess, tempfile
import numpy as np
from PIL import Image

THREADS = int(os.environ.get('TORCH_THREADS', '0')) or os.cpu_count() or 2


def _torch():
    import torch
    torch.set_num_threads(THREADS)
    return torch


def open_img(b):
    im = Image.open(io.BytesIO(b))
    im.load()
    return im


# ---------------- sigblur: same maths as the hub (pxui sameSig + box blurfill.py) ----------------
class SigBlur:
    kind = 't'; batch = 64; workers = 16

    def load(self):
        pass

    @staticmethod
    def sig(im):
        a = np.asarray(im.convert('RGB'), dtype=np.float64)
        g = 0.299 * a[:, :, 0] + 0.587 * a[:, :, 1] + 0.114 * a[:, :, 2]
        H, W = g.shape
        out = []
        for i in range(16):
            y0 = (i * H) // 16; y1 = max(y0 + 1, ((i + 1) * H) // 16)
            for j in range(16):
                x0 = (j * W) // 16; x1 = max(x0 + 1, ((j + 1) * W) // 16)
                v = int(math.floor(g[y0:y1, x0:x1].mean() + 0.5))
                out.append('%02x' % min(255, v))
        return ''.join(out)

    @staticmethod
    def blur(im):
        im = im.convert('L'); w, h = im.size; s = 256 / max(w, h)
        if s < 1:
            im = im.resize((max(8, int(w * s)), max(8, int(h * s))), Image.BILINEAR)
        g = np.asarray(im, dtype=np.float64)
        L = (-4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:])
        H, W = L.shape; v = []
        for i in range(4):
            for j in range(4):
                t = L[i * H // 4:(i + 1) * H // 4, j * W // 4:(j + 1) * W // 4]
                if t.size:
                    v.append(t.var())
        return round(float(np.percentile(v, 90)), 2)

    def run(self, items, blobs):
        out = []
        for it, b in zip(items, blobs):
            try:
                im = open_img(b)
                out.append({'id': it['id'], 'sig': self.sig(im), 'blur': self.blur(im)})
            except Exception as e:
                out.append({'id': it['id'], 'err': 'decode ' + type(e).__name__})
        return out


# ---------------- caption: Florence-2 large ----------------
class Caption:
    kind = 'p'; batch = 8; workers = 8
    MODEL = os.environ.get('CAPTION_MODEL', 'microsoft/Florence-2-large')
    TASK = '<DETAILED_CAPTION>'

    def load(self):
        torch = _torch()
        from unittest.mock import patch
        from transformers import AutoProcessor, AutoModelForCausalLM
        from transformers.dynamic_module_utils import get_imports

        def fixed(filename):
            imp = get_imports(filename)
            return [x for x in imp if x != 'flash_attn']
        with patch('transformers.dynamic_module_utils.get_imports', fixed):
            self.model = AutoModelForCausalLM.from_pretrained(self.MODEL, trust_remote_code=True, torch_dtype=torch.float32, attn_implementation='eager').eval()
            self.proc = AutoProcessor.from_pretrained(self.MODEL, trust_remote_code=True)
        if os.environ.get('QUANT', '1') == '1':
            self.model = torch.quantization.quantize_dynamic(self.model, {torch.nn.Linear}, dtype=torch.qint8)
        self.torch = torch

    @staticmethod
    def clean(t):
        t = ' '.join(str(t).replace('<pad>', ' ').replace('</s>', ' ').replace('<s>', ' ').split())
        for p in ('The image shows ', 'The image is ', 'In this image we can see ', 'In this image, ', 'This image shows ', 'The picture shows '):
            if t.startswith(p):
                t = t[len(p):]; t = t[:1].upper() + t[1:]
        return t[:300]

    def run(self, items, blobs):
        out, ims, ok = [], [], []
        for it, b in zip(items, blobs):
            try:
                ims.append(open_img(b).convert('RGB')); ok.append(it)
            except Exception as e:
                out.append({'id': it['id'], 'err': 'decode ' + type(e).__name__})
        if not ims:
            return out
        with self.torch.inference_mode():
            x = self.proc(text=[self.TASK] * len(ims), images=ims, return_tensors='pt')
            ids = self.model.generate(input_ids=x['input_ids'], pixel_values=x['pixel_values'], max_new_tokens=64, num_beams=1, do_sample=False)
            txt = self.proc.batch_decode(ids, skip_special_tokens=True)
        for it, t in zip(ok, txt):
            t = self.clean(t)
            out.append({'id': it['id'], 'text': t} if t else {'id': it['id'], 'err': 'empty'})
        return out


# ---------------- tags: SigLIP so400m zero-shot ----------------
class Tags:
    kind = 't'; batch = 16; workers = 12
    MODEL = os.environ.get('TAG_MODEL', 'google/siglip-so400m-patch14-384')
    MIN_P = float(os.environ.get('TAG_MIN_P', '0.10'))

    def load(self):
        torch = _torch()
        from transformers import AutoModel, AutoProcessor
        from .labels import PHOTO_TAGS, PROMPT
        self.model = AutoModel.from_pretrained(self.MODEL, torch_dtype=torch.float32).eval()
        self.proc = AutoProcessor.from_pretrained(self.MODEL)
        self.labels = PHOTO_TAGS
        with torch.inference_mode():
            t = self.proc(text=[PROMPT.format(l) for l in PHOTO_TAGS], padding='max_length', return_tensors='pt')
            f = self.model.get_text_features(**t)
            self.txt = f / f.norm(dim=-1, keepdim=True)
        self.torch = torch

    def run(self, items, blobs):
        out, ims, ok = [], [], []
        for it, b in zip(items, blobs):
            try:
                ims.append(open_img(b).convert('RGB')); ok.append(it)
            except Exception as e:
                out.append({'id': it['id'], 'err': 'decode ' + type(e).__name__})
        if not ims:
            return out
        T = self.torch
        with T.inference_mode():
            f = self.model.get_image_features(**self.proc(images=ims, return_tensors='pt'))
            f = f / f.norm(dim=-1, keepdim=True)
            p = T.sigmoid(f @ self.txt.T * self.model.logit_scale.exp() + self.model.logit_bias).numpy()
        for it, row in zip(ok, p):
            idx = np.argsort(-row)[:8]
            tags = [[self.labels[i], round(float(row[i]), 3)] for i in idx if row[i] >= self.MIN_P]
            out.append({'id': it['id'], 'tags': tags})
        return out


# ---------------- music: CLAP (larger_clap_music) moods, tags and a sound embedding ----------------
class Music:
    kind = 'music'; batch = 4; workers = 4
    MODEL = os.environ.get('MUSIC_MODEL', 'laion/larger_clap_music_and_speech')
    MAX_BYTES = 30 * 1024 * 1024

    def load(self):
        torch = _torch()
        from transformers import ClapModel, ClapProcessor
        from .labels import MUSIC_MOODS, MUSIC_TAG_GROUPS, MOOD_CAL
        self.cal = MOOD_CAL
        self.model = ClapModel.from_pretrained(self.MODEL).eval()
        self.proc = ClapProcessor.from_pretrained(self.MODEL)
        self.moods = MUSIC_MOODS; self.groups = MUSIC_TAG_GROUPS
        texts = []
        for k, (a, b) in MUSIC_MOODS.items():
            texts += [a, b]
        for g, ls in MUSIC_TAG_GROUPS.items():
            texts += ls
        with torch.inference_mode():
            t = self.proc(text=texts, return_tensors='pt', padding=True)
            f = self.model.get_text_features(**t)
            f = f / f.norm(dim=-1, keepdim=True)
        self.tx = dict(zip(texts, f))
        self.scale = float(self.model.logit_scale_a.exp()) if hasattr(self.model, 'logit_scale_a') else 100.0
        self.torch = torch

    def decode(self, b, ext):
        with tempfile.NamedTemporaryFile(suffix='.' + (ext or 'mp3')) as f:
            f.write(b); f.flush()
            r = subprocess.run(['ffmpeg', '-v', 'quiet', '-nostdin', '-i', f.name, '-t', '900', '-ac', '1', '-ar', '48000', '-f', 'f32le', 'pipe:1'], capture_output=True, timeout=300)
        a = np.frombuffer(r.stdout, dtype=np.float32)
        return a if a.size > 48000 * 3 else None

    def run(self, items, blobs):
        out = []
        T = self.torch
        for it, b in zip(items, blobs):
            try:
                if b is None:
                    out.append({'id': it['id'], 'err': 'no R2 copy'}); continue
                a = self.decode(b, it.get('ext'))
                if a is None:
                    out.append({'id': it['id'], 'err': 'decode'}); continue
                win = 48000 * 10; n = a.size
                clips = [a[max(0, int(n * c) - win // 2):][:win] for c in (0.2, 0.5, 0.8)] if n > win * 2 else [a[:win]]
                with T.inference_mode():
                    x = self.proc(audios=[c.astype(np.float32) for c in clips], sampling_rate=48000, return_tensors='pt')
                    f = self.model.get_audio_features(**x)
                    f = f / f.norm(dim=-1, keepdim=True)
                    e = f.mean(0); e = e / e.norm()
                moods = {}
                for k, (pa, pb) in self.moods.items():
                    mu, sd = self.cal.get(k, (0.0, 0.15))
                    z = (float(e @ self.tx[pa] - e @ self.tx[pb]) - mu) / (sd or 0.15)
                    moods[k] = round(1 / (1 + math.exp(-2 * (z - 0.9))), 3)
                tags = []
                for g, ls in self.groups.items():
                    s = T.stack([e @ self.tx[l] for l in ls]) * self.scale
                    p = T.softmax(s, 0)
                    i = int(p.argmax())
                    if float(p[i]) >= 0.35:
                        tags.append([ls[i], round(float(p[i]), 3)])
                q = np.clip(np.round(e.numpy() * 127), -127, 127).astype(np.int8)
                out.append({'id': it['id'], 'moods': moods, 'tags': tags, 'emb': base64.b64encode(q.tobytes()).decode()})
            except Exception as ex:
                out.append({'id': it['id'], 'err': 'music ' + type(ex).__name__})
        return out


JOBS = {'sigblur': SigBlur, 'caption': Caption, 'tags': Tags, 'music': Music}
