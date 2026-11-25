"""Hub background AI runner (GitHub Actions). Usage: python -m hubai.run --jobs caption,music --shard 0 --of 4 --minutes 50
Prints only counts and speeds (safe for public logs)."""
import argparse, gc, os, sys, time, concurrent.futures as cf
from . import hub
from .jobs import JOBS, rss_mb

T0 = float(os.environ.get('JOB_T0') or time.time())


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def fetch(it, kind):
    if not it.get('k'):
        return None
    try:
        if kind == 'music':
            mb = 30 * 1024 * 1024
            return hub.blob(it['k'], mb if (it.get('size') or 0) > mb else None)
        return hub.blob(it['k'])
    except Exception:
        return None


def run_job(name, a, deadline):
    J = JOBS[name](); t_load = time.time()
    first = hub.get_json('/bgai/gh/todo', job=name, n=1, shard=a.shard, of=a.of, after=0)
    if not first:
        log(name, 'nothing to do'); hub.post_json('/bgai/gh/progress', {'job': name, 'shard': a.shard, 'run': a.run, 'state': 'done', 'n': 0, 'final': True}); return 0
    log(name, 'loading model'); J.load(); log(name, 'model ready in %.0fs, peak memory %d MB' % (time.time() - t_load, rss_mb()))
    after, n, fail, skipped, t0, last_p, limit = 0, 0, 0, 0, time.time(), time.time() - 240, a.limit or 10 ** 9
    state = 'running'
    pool = cf.ThreadPoolExecutor(J.workers)
    while True:
        if time.time() > deadline:
            state = 'paused (time)'; break
        if n + fail >= limit:
            state = 'paused (limit)'; break
        items = hub.get_json('/bgai/gh/todo', job=name, n=min(J.batch * 4, limit - n - fail), shard=a.shard, of=a.of, after=after)
        if not items:
            state = 'done'; break
        after = items[-1]['id']
        blobs = list(pool.map(lambda it: fetch(it, J.kind), items))
        for i in range(0, len(items), J.batch):
            its, bs = items[i:i + J.batch], blobs[i:i + J.batch]
            keep = [(it, b) for it, b in zip(its, bs) if b is not None or J.kind == 'music' and not it.get('k')]
            skipped += len(its) - len(keep)  # download failed: retried next run (not marked)
            if not keep:
                continue
            res = J.run([x[0] for x in keep], [x[1] for x in keep])
            r = hub.post_json('/bgai/gh/ingest', {'job': name, 'items': res})
            n += r.get('ok', 0); fail += r.get('fail', 0)
            if time.time() - last_p > 300:
                rate = n / max(1e-6, (time.time() - t0) / 60)
                hub.post_json('/bgai/gh/progress', {'job': name, 'shard': a.shard, 'run': a.run, 'state': 'running', 'n': n, 'fail': fail, 'sec': time.time() - t0, 'rate': rate})
                log(name, 'done %d, failed %d, skipped %d, %.1f/min, peak memory %d MB' % (n, fail, skipped, rate, rss_mb())); last_p = time.time()
            if time.time() > deadline:
                break
    pool.shutdown(wait=False)
    sec = time.time() - t0; rate = n / max(1e-6, sec / 60)
    hub.post_json('/bgai/gh/progress', {'job': name, 'shard': a.shard, 'run': a.run, 'state': state, 'n': n, 'fail': fail, 'sec': sec, 'rate': rate, 'final': True})
    log(name, state, 'done %d, failed %d, skipped %d in %.1f min (%.1f/min)' % (n, fail, skipped, sec / 60, rate))
    with open(os.environ.get('GITHUB_STEP_SUMMARY', '/dev/null'), 'a') as f:
        f.write('| %s | shard %d/%d | %d done | %d failed | %d skipped | %.1f min | %.1f/min | %s |\n' % (name, a.shard, a.of, n, fail, skipped, sec / 60, rate, state))
    del J; gc.collect()
    return n


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--jobs', default='sigblur,music,caption,tags'); p.add_argument('--shard', type=int, default=0); p.add_argument('--of', type=int, default=1)
    p.add_argument('--minutes', type=float, default=50); p.add_argument('--limit', type=int, default=0); p.add_argument('--run', default=os.environ.get('GITHUB_RUN_ID', 'local')); p.add_argument('--child', action='store_true')
    a = p.parse_args()
    if not hub.BASE or not hub.KEY:
        log('HUB_URL / HUB_AI_KEY missing'); sys.exit(1)
    deadline = (time.time() if a.child else T0) + a.minutes * 60
    if not a.child:
        log('cpus', os.cpu_count(), 'shard %d/%d' % (a.shard, a.of), 'budget %.0f min' % a.minutes)
    names = [j for j in a.jobs.split(',') if j in JOBS]
    if len(names) > 1:  # one process per job: each model's memory is fully freed before the next one loads (2-core runners have 7 GB)
        import subprocess
        for name in names:
            if time.time() > deadline - 120:
                break
            r = subprocess.run([sys.executable, '-m', 'hubai.run', '--jobs', name, '--shard', str(a.shard), '--of', str(a.of), '--minutes', str(max(1.0, (deadline - time.time()) / 60)), '--limit', str(a.limit), '--run', a.run, '--child'])
            if r.returncode:
                log(name, 'process ended with code', r.returncode)
                try:
                    hub.post_json('/bgai/gh/progress', {'job': name, 'shard': a.shard, 'run': a.run, 'state': 'error', 'err': 'exit %d (memory?)' % r.returncode, 'final': True})
                except Exception:
                    pass
        names = []
    for name in names:
        if time.time() > deadline - 120:
            break
        try:
            run_job(name, a, deadline)
        except Exception as e:
            log(name, 'error', type(e).__name__, str(e)[:120] if 'hub' in str(e) else '')
            try:
                hub.post_json('/bgai/gh/progress', {'job': name, 'shard': a.shard, 'run': a.run, 'state': 'error', 'err': type(e).__name__, 'final': True})
            except Exception:
                pass
    if a.child:
        return
    used = (time.time() - T0) / 60
    try:
        hub.post_json('/bgai/gh/progress', {'job': '_min', 'run': a.run, 'minutes': used})
    except Exception:
        pass
    log('runner minutes %.1f' % used)


if __name__ == '__main__':
    main()
