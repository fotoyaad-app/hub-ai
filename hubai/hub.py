"""Talks to the Hub Worker. Only counts are ever printed: no keys, file names or paths (the repo may become public)."""
import os, time, json
import requests

BASE = os.environ.get('HUB_URL', '').rstrip('/')
KEY = os.environ.get('HUB_AI_KEY', '')
S = requests.Session()
S.headers.update({'x-hub-ai-key': KEY, 'user-agent': 'hub-ai-runner/1'})


def _call(method, path, **kw):
    last = None
    for i in range(5):
        try:
            r = S.request(method, BASE + path, timeout=kw.pop('timeout', 120), **kw)
            if r.status_code in (429, 500, 502, 503, 504):
                last = 'http %d' % r.status_code
                time.sleep(2 + i * 4)
                continue
            return r
        except requests.RequestException as e:
            last = type(e).__name__
            time.sleep(2 + i * 4)
    raise RuntimeError('hub unreachable (%s)' % last)


def get_json(path, **params):
    r = _call('GET', path, params=params)
    if r.status_code != 200:
        raise RuntimeError('hub %s -> %d' % (path.split('?')[0], r.status_code))
    return r.json()


def post_json(path, body):
    r = _call('POST', path, data=json.dumps(body), headers={'content-type': 'application/json'})
    if r.status_code != 200:
        raise RuntimeError('hub %s -> %d' % (path, r.status_code))
    return r.json()


def blob(key, max_bytes=None):
    h = {'range': 'bytes=0-%d' % (max_bytes - 1)} if max_bytes else {}
    r = _call('GET', '/bgai/gh/blob', params={'k': key}, headers=h, timeout=180)
    if r.status_code not in (200, 206):
        return None
    return r.content
