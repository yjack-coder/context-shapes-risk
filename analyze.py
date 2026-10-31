#!/usr/bin/env python3
"""Score results/raw/*.jsonl per DESIGN.md and write results/summary.json. Stdlib only."""
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent
BOOTSTRAP = 10000


def load_cases():
    with open(ROOT / 'data' / 'contrast_pairs.csv', encoding='utf-8', newline='') as f:
        return {r['sample_id']: r for r in csv.DictReader(f)}


def truth(row):
    return 'CONCERNING' if row['risk_label'] == '1' else 'BENIGN'


def rate(by_pair, pairs):
    """Pooled proportion over the given pairs; by_pair maps pair -> [hits, total]."""
    hits = sum(by_pair[p][0] for p in pairs)
    total = sum(by_pair[p][1] for p in pairs)
    return hits / total if total else 0.0


def ci(stat, pair_ids, seed=0):
    """95% percentile bootstrap, resampling pairs."""
    rng = random.Random(seed)
    draws = sorted(stat(rng.choices(pair_ids, k=len(pair_ids))) for _ in range(BOOTSTRAP))
    return draws[int(.025 * BOOTSTRAP)], draws[int(.975 * BOOTSTRAP)]


def tally(trials, cases, cond, want, keep=lambda r: True):
    """pair -> [trials where label == want(row), trials] for one condition."""
    out = defaultdict(lambda: [0, 0])
    for t in trials:
        row = cases[t['sample_id']]
        if t['condition'] == cond and keep(row):
            out[t['pair_id']][0] += t['label'] == want(row)
            out[t['pair_id']][1] += 1
    return out


def pair_accuracy(trials, cases, cond='context'):
    """pair -> [(both cases correct, averaged over repetitions), 1], as hits/total."""
    per_rep = defaultdict(dict)
    for t in trials:
        if t['condition'] == cond:
            per_rep[(t['pair_id'], t['rep'])][t['sample_id']] = t['label'] == truth(cases[t['sample_id']])
    out = defaultdict(lambda: [0, 0])
    for (pair, _), got in per_rep.items():
        out[pair][0] += len(got) == 2 and all(got.values())
        out[pair][1] += 1
    return out


def summarize(trials, cases):
    pairs = sorted({t['pair_id'] for t in trials})
    set_of = {r['pair_id']: r['set'] for r in cases.values()}
    is_benign = lambda r: r['risk_label'] == '0'
    is_concerning = lambda r: r['risk_label'] == '1'
    s = dict(models=sorted({t['model'] for t in trials}), trials=len(trials),
             invalid=sum(t['label'] == 'INVALID' for t in trials))

    def point(by_pair, ids=pairs):
        return dict(rate=rate(by_pair, ids), ci=ci(lambda d: rate(by_pair, d), ids))

    pa = pair_accuracy(trials, cases)
    s['pair_accuracy'] = point(pa)
    simple = [p for p in pairs if set_of[p] == 'simple']
    infer = [p for p in pairs if set_of[p] == 'inferential']
    s['pair_accuracy_simple'] = point(pa, simple)
    s['pair_accuracy_inferential'] = point(pa, infer)

    s['no_context_insufficient'] = point(tally(trials, cases, 'no_context', lambda r: 'INSUFFICIENT'))
    s['no_context_concerning'] = rate(tally(trials, cases, 'no_context', lambda r: 'CONCERNING'), pairs)
    s['no_context_benign'] = rate(tally(trials, cases, 'no_context', lambda r: 'BENIGN'), pairs)
    s['context_abstain'] = rate(tally(trials, cases, 'context', lambda r: 'INSUFFICIENT'), pairs)

    def effect(cond, keep, name):
        """Error rate under cond and under the neutral control, and their difference."""
        want = truth
        treat = tally(trials, cases, cond, want)
        ctrl = tally(trials, cases, 'memory_neutral', want, keep)
        err = lambda t, ids: 1 - rate(t, ids)
        lo, hi = ci(lambda d: err(treat, d) - err(ctrl, d), pairs)
        s[name] = dict(treated=err(treat, pairs), control=err(ctrl, pairs),
                       diff=err(treat, pairs) - err(ctrl, pairs), ci=[lo, hi])

    effect('memory_alarming', is_benign, 'alarming_false_alarms')
    effect('memory_reassuring', is_concerning, 'reassuring_misses')

    # Which label the errors went to, for reading the memory effects.
    errs = defaultdict(lambda: defaultdict(int))
    for t in trials:
        if t['condition'] != 'no_context' and t['label'] != truth(cases[t['sample_id']]):
            errs[t['condition']][t['label']] += 1
    s['error_labels'] = {k: dict(v) for k, v in errs.items()}

    worst = defaultdict(lambda: [0, 0])
    for t in trials:
        if t['condition'] != 'no_context':
            worst[t['sample_id']][0] += t['label'] != truth(cases[t['sample_id']])
            worst[t['sample_id']][1] += 1
    s['case_errors'] = {k: v for k, v in sorted(worst.items(), key=lambda kv: -kv[1][0]) if v[0]}

    # Exploratory, not in the analysis plan.
    err_b = tally(trials, cases, 'context', truth, is_benign)
    err_c = tally(trials, cases, 'context', truth, is_concerning)
    s['exploratory'] = dict(
        benign_error=point({p: [v[1] - v[0], v[1]] for p, v in err_b.items()}),
        concerning_error=point({p: [v[1] - v[0], v[1]] for p, v in err_c.items()}),
        pair_accuracy_without_P40=point(pa, [p for p in pairs if p != 'P40']),
    )
    single = [t for t in trials if ',' not in t['model']]
    s['multi_model_trials'] = len(trials) - len(single)
    if len(single) < len(trials):
        # Per-case accuracy, because dropping calls would leave some pairs incomplete.
        acc = lambda ts: sum(t['label'] == truth(cases[t['sample_id']]) for t in ts) / len(ts)
        is_scored = lambda t: t['condition'] == 'context'
        s['exploratory']['case_accuracy_all_calls'] = acc([t for t in trials if is_scored(t)])
        s['exploratory']['case_accuracy_single_model_calls'] = acc([t for t in single if is_scored(t)])
    return s


def main():
    cases = load_cases()
    out = {}
    for path in sorted((ROOT / 'results' / 'raw').glob('*.jsonl')):
        trials = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        out[path.stem] = summarize(trials, cases)
    (ROOT / 'results' / 'summary.json').write_text(json.dumps(out, indent=1))
    for name, s in out.items():
        print(f"\n{name}  {s['models']}  trials={s['trials']} invalid={s['invalid']}")
        for k in ('pair_accuracy', 'pair_accuracy_simple', 'pair_accuracy_inferential', 'no_context_insufficient'):
            print(f"  {k:28} {s[k]['rate']:.3f}  [{s[k]['ci'][0]:.3f}, {s[k]['ci'][1]:.3f}]")
        print(f"  no_context labels            benign={s['no_context_benign']:.3f} concerning={s['no_context_concerning']:.3f}")
        print(f"  context abstain              {s['context_abstain']:.3f}")
        for k in ('alarming_false_alarms', 'reassuring_misses'):
            e = s[k]
            print(f"  {k:28} {e['treated']:.3f} vs {e['control']:.3f}  diff={e['diff']:+.3f} [{e['ci'][0]:+.3f}, {e['ci'][1]:+.3f}]")
        print('  errors by condition', s['error_labels'])
        print('  top cases', list(s['case_errors'].items())[:10])


if __name__ == '__main__':
    main()
