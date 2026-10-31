# When Context Changes the Risk

**Read the report: https://yjack-coder.github.io/context-shapes-risk/**

An assistant sees only a fragment of what a person means. A joke can be stored as lasting evidence about someone, and a concerning statement can look harmless without its background. This small evaluation asks whether language models use the context in front of them, say so when it is missing, and ignore saved memory that has nothing to do with the situation.

It uses 40 contrast pairs: the same sentence in a benign and a concerning context. Four models and 2,800 trials. The [analysis plan](DESIGN.md) was committed before the first scored call.

## Results

| | Opus 5.5 | Sonnet 5.5 | Haiku 5.5 | Qwen3 8B |
|---|---|---|---|---|
| Pair accuracy with context | 95% | 91% | 63% | 75% |
| Says INSUFFICIENT with no context | 95% | 88% | 87% | 2% |
| Added benign-case errors from alarming memory (points) | +0 | -2 | **+18** [+5, +31] | +5 [-8, +18] |
| Added concerning-case errors from reassuring memory (points) | +0 | +0 | -1 | +2 |

Brackets are 95% bootstrap intervals over pairs. Haiku's memory effect went entirely to INSUFFICIENT, not to CONCERNING.

These numbers describe these models on these synthetic cases. The labels were not independently validated, the stimuli were drafted with Claude, and at least one label (P40) looks questionable. The report lists the limits in full.

## Reproduce

Python 3.9+, standard library only.

```bash
python3 -m unittest test_analyze          # tests
python3 analyze.py                        # raw outputs -> results/summary.json
python3 build_site.py                     # summary -> docs/index.html

# Re-run the models (appends to results/raw; remove those files first)
python3 run_eval.py --backend claude --model opus --reps 3      # needs Claude Code
python3 run_eval.py --backend ollama --model qwen3:8b --reps 1  # needs Ollama
```

## Layout

- `data/contrast_pairs.csv`: the stimuli and labels
- `DESIGN.md`: conditions, measures, hypotheses, and deviations
- `run_eval.py`: prompts and model calls
- `analyze.py`: scoring and bootstrap intervals
- `results/raw/`: every model reply
- `docs/`: the report

## How this was made

This is one evening's work, built with heavy AI assistance. I chose the question and approach. Claude Code wrote the code, drafted the additional stimuli and the report, and ran the experiment. The first 12 pairs came from an earlier AI-assisted draft.

MIT License.
