"""Talks to the Hub Worker. Only counts are ever printed: no keys, file names or paths (the repo may become public)."""
import os, sys, time, json, logging, warnings
import requests
from requests.adapters import HTTPAdapter

# Public logs: library loggers/warnings can print hosts or URLs, so they are silenced; uncaught errors print only their type.
logging.getLogger('urllib3').setLevel(logging.CRITICAL)
logging.getLogger('requests').setLevel(logging.CRITICAL)
warnings.filterwarnings('ignore')
sys.excepthook = lambda t, v, tb: print('error', t.__name__, file=sys.stderr, flush=True)

BASE = os.environ.get('HUB_URL', '').rstrip('/')
KEY = os.environ.get('HUB_AI_KEY', '')
S = requests.Session()
S.mount('https://', HTTPAdapter(pool_connections=4, pool_maxsize=32))
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


def blob(key=None, max_bytes=None, job=None, id=None):
    """Media bytes from R2 via the Hub. Newer Hubs serve by job + id, so the runner never sees file names or paths."""
    h = {'range': 'bytes=0-%d' % (max_bytes - 1)} if max_bytes else {}
    params = {'k': key} if key else {'job': job, 'id': id}
    r = _call('GET', '/bgai/gh/blob', params=params, headers=h, timeout=300)
    if r.status_code not in (200, 206):
        return None
    return r.content
