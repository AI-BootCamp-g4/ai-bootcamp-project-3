import argparse
import getpass
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import openai
from openai import OpenAI

BASE_URL = 'https://193.93.169.12/v1'
PRICES = {
    'openai/gpt-4o-mini': (0.15, 0.60),
    'openai/gpt-4.1-mini': (0.40, 1.60),
}
NO_RETRY = tuple(c for c in (getattr(openai, n, None) for n in (
    'AuthenticationError', 'PermissionDeniedError', 'NotFoundError', 'BadRequestError'))
    if isinstance(c, type))
CONN_ERR = getattr(openai, 'APIConnectionError', Exception)


def make_client(base_url, key, insecure):
    kw = dict(base_url=base_url, api_key=key, max_retries=0, timeout=60)
    if insecure:
        if hasattr(openai, 'DefaultHttpxClient'):
            kw['http_client'] = openai.DefaultHttpxClient(verify=False, timeout=60)
        else:
            try:
                import httpx2 as hx
            except ImportError:
                import httpx as hx
            kw['http_client'] = hx.Client(verify=False, timeout=60)
    return OpenAI(**kw)


def parse_label(raw):
    s = (raw or '').strip().lower().replace('-', '_').replace(' ', '_')
    for lab in ('not_recommended', 'no_idea', 'recommended'):
        if lab in s:
            return lab
    return None


def build_messages(job, text, mode):
    msgs = [{'role': 'system', 'content': job['system']}]
    if mode == 'few':
        for ex in job['few']:
            msgs.append({'role': 'user', 'content': 'Review: ' + ex['text']})
            msgs.append({'role': 'assistant', 'content': ex['label']})
    msgs.append({'role': 'user', 'content': 'Review: ' + text})
    return msgs


def ask(client, model, msgs, tries=5):
    err = None
    for attempt in range(tries):
        t0 = time.perf_counter()
        try:
            r = client.chat.completions.create(model=model, messages=msgs,
                                               temperature=0, max_tokens=8)
            u = r.usage
            return {'raw': r.choices[0].message.content,
                    'sec': round(time.perf_counter() - t0, 3),
                    'tok_in': getattr(u, 'prompt_tokens', None),
                    'tok_out': getattr(u, 'completion_tokens', None),
                    'cost_usd': getattr(u, 'cost', None),
                    'cost_toman': getattr(u, 'total_cost_toman', None),
                    'attempts': attempt + 1, 'err': None}
        except NO_RETRY:
            raise
        except Exception as e:
            err = f'{type(e).__name__}: {e}'[:300]
            if attempt < tries - 1:
                time.sleep(min(2 ** attempt, 30))
    return {'raw': None, 'sec': None, 'tok_in': None, 'tok_out': None,
            'cost_usd': None, 'cost_toman': None, 'attempts': tries, 'err': err}


def load_done(path):
    done = set()
    if os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if d.get('err') is None:
                    done.add((d['model'], d['mode'], d['id']))
    return done


def load_dotenv():
    here = os.path.dirname(os.path.abspath(__file__))
    for folder in (os.getcwd(), os.path.dirname(here), here):
        path = os.path.join(folder, '.env')
        if os.path.exists(path):
            with open(path, encoding='utf-8-sig') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#') or '=' not in line:
                        continue
                    k, v = line.split('=', 1)
                    k, v = k.strip(), v.strip().strip('"\'')
                    if k and v and k not in os.environ:
                        os.environ[k] = v
            return path
    return None


def read_key():
    key, src = os.environ.get('QBC13_API_KEY', ''), 'QBC13_API_KEY'
    if not key:
        key, src = getpass.getpass('QBC13 API key ro paste kon va Enter bezan: '), 'getpass'
    key = ''.join(ch for ch in key if ch.isprintable()).strip().strip('"\'').strip()
    return key, src


def connection_hint(e):
    s = repr(e) + repr(getattr(e, '__cause__', None))
    if 'CERTIFICATE' in s.upper() or 'SSL' in s.upper():
        return 'khata-ye SSL: dastoor ro ba --insecure dobare ejra kon.'
    return 'server javab nadad: VPN khamoosh-e? internet vasl-e?'


def main():
    ap = argparse.ArgumentParser(description='LLM zero/few-shot roo nazar-haye test')
    ap.add_argument('--job', default='llm_job.json')
    ap.add_argument('--out', default='llm_cache.jsonl')
    ap.add_argument('--model', default='openai/gpt-4o-mini')
    ap.add_argument('--modes', nargs='+', default=['zero', 'few'], choices=['zero', 'few'])
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--limit', type=int, default=None, help='faghat N nazar-e aval')
    ap.add_argument('--insecure', action='store_true', help='govahi-e SSL server check nashe')
    ap.add_argument('--ping', action='store_true', help='faghat ye darkhast-e OK')
    ap.add_argument('--base-url', default=None)
    args = ap.parse_args()

    env_path = load_dotenv()
    if env_path:
        print('.env khoonde shod:', env_path)
    base_url = args.base_url or os.environ.get('QBC13_API_BASE_URL') or BASE_URL

    key, src = read_key()
    if not key:
        sys.exit('API key khali-e.')
    print(f'kelid az {src}: {key[:3]}... ({len(key)} character)')
    if not key.startswith('sk-'):
        print('hoshdar: kelid ba sk- shoroo nemishe.')
    client = make_client(base_url, key, args.insecure)

    if args.ping:
        t0 = time.perf_counter()
        try:
            r = client.chat.completions.create(
                model=args.model, max_tokens=5,
                messages=[{'role': 'user', 'content': 'Reply with the single word OK'}])
        except CONN_ERR as e:
            sys.exit(f'{type(e).__name__}: {e}\n{connection_hint(e)}')
        print(r.choices[0].message.content, '|', r.usage, '|',
              round(time.perf_counter() - t0, 2), 's')
        return

    with open(args.job, encoding='utf-8') as f:
        job = json.load(f)
    items = job['items'][:args.limit] if args.limit else job['items']
    done = load_done(args.out)
    tasks = [(m, it) for m in args.modes for it in items
             if (args.model, m, it['id']) not in done]
    print(f'{len(items)} nazar | halat-ha: {args.modes} | model: {args.model}')
    print(f'{len(tasks)} darkhast baghi munde ({len(done)} ta ghablan anjam shode)')
    if not tasks:
        print(f'hame anjam shode. {args.out} ro too Drive upload kon.')
        return

    stats = {m: {'n': 0, 'bad': 0, 'err': 0, 'tin': 0, 'tout': 0, 'sec': [], 'usd': 0.0, 'toman': 0.0} for m in args.modes}
    t_start = time.perf_counter()
    ex = ThreadPoolExecutor(max_workers=args.workers)
    futs = {ex.submit(ask, client, args.model, build_messages(job, it['text'], m)): (m, it['id'])
            for m, it in tasks}
    try:
        with open(args.out, 'a', encoding='utf-8') as fout:
            for k, fu in enumerate(as_completed(futs), 1):
                m, i = futs[fu]
                res = fu.result()
                lab = parse_label(res['raw'])
                rec = {'model': args.model, 'mode': m, 'id': i, 'label': lab, **res,
                       'ts': time.strftime('%Y-%m-%dT%H:%M:%S')}
                fout.write(json.dumps(rec, ensure_ascii=False) + '\n')
                fout.flush()
                st = stats[m]
                st['n'] += 1
                if res['err'] is not None:
                    st['err'] += 1
                else:
                    st['bad'] += lab is None
                    st['tin'] += res['tok_in'] or 0
                    st['tout'] += res['tok_out'] or 0
                    st['sec'].append(res['sec'])
                    st['usd'] += res['cost_usd'] or 0
                    st['toman'] += res['cost_toman'] or 0
                if k % 50 == 0 or k == len(futs):
                    errs = sum(s['err'] for s in stats.values())
                    print(f'  {k}/{len(futs)} | {time.perf_counter() - t_start:.0f} s | khata: {errs}',
                          flush=True)
    except NO_RETRY as e:
        ex.shutdown(wait=False, cancel_futures=True)
        sys.exit(f'\n{type(e).__name__}: {e}\nin khata ba tekrar dorost nemishe '
                 '(kelid, dastresi ya esm-e model). javab-haye ghabli too cache mounde.')
    except KeyboardInterrupt:
        ex.shutdown(wait=False, cancel_futures=True)
        sys.exit('\nmotevaghef shod. dobare ejra koni az hamin ja edame mide.')
    ex.shutdown(wait=True)

    p_in, p_out = PRICES.get(args.model, (float('nan'), float('nan')))
    print(f'\ntamam: {time.perf_counter() - t_start:.0f} s (faghat hamin ejra)')
    total = 0.0
    for m, st in stats.items():
        cost = (st['tin'] * p_in + st['tout'] * p_out) / 1e6
        total += cost
        sec = sorted(st['sec'])
        mean = sum(sec) / len(sec) if sec else float('nan')
        p95 = sec[int(0.95 * (len(sec) - 1))] if sec else float('nan')
        print(f'  {m:4s} | n={st["n"]} | namotabar={st["bad"]} | khata={st["err"]} | '
              f'tok_in={st["tin"]} tok_out={st["tout"]} | sec mean={mean:.2f} p95={p95:.2f} | '
              f'takhmini ${cost:.4f} | server ${st["usd"]:.4f} = {st["toman"]:.0f} toman')
    print(f'  hazine-ye in ejra: takhmini ${total:.4f} | server '
          f'${sum(st["usd"] for st in stats.values()):.4f} = '
          f'{sum(st["toman"] for st in stats.values()):.0f} toman')
    if any(st['err'] for st in stats.values()):
        print('baazi darkhast-ha khata dadan; dobare ejra kon ta faghat hamoon-ha ferestade beshan.')
    else:
        print(f'{args.out} ro too Drive, poshe-ye digikala_q3/outputs upload kon.')


if __name__ == '__main__':
    main()
