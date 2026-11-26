# Product comparison and manager analytics

> **In this repository** the package lives at `src/digikala_assistant/comparison_analytics`.
> Install with `pip install -e ".[comparison]"` and run modules as
> `python -m digikala_assistant.comparison_analytics.<module>` (for example
> `...data_prep`, `...eval.run_eval`); the dashboard is
> `streamlit run src/digikala_assistant/comparison_analytics/app.py`.
> Settings come from the repo `.env` (`QBC13_API_KEY`, `QBC13_API_BASE_URL`,
> `DIGIKALA_DATA_DIR`). Entry points: `digikala_assistant.comparison.compare`
> and `digikala_assistant.analytics.ask`.


Parts 3 and 4 of the assistant (Section 2), plus their evaluation (Section 4).

## Design in one paragraph

Numbers are computed by code on **all** comments; the LLM never calculates.
The LLM only (a) picks the analysis when rules are unsure, (b) explains the
computed results in Persian, and (c) summarises review evidence that it must
cite by comment id. Every answer is checked by code before it is shown:
invented comment ids are dropped and numbers that are not in the data are
flagged. A zero-shot baseline runs through the same functions so the
evaluation can show what each design choice adds.

```
question ─► router (rules → LLM fallback) ─► category match (difflib / title vote)
         ─► exact analysis on all comments (pandas, Kruskal-Wallis)
         ─► LLM report from the numbers only ─► number check ─► answer
products ─► facts by code ─► evidence comments with ids (aspect-aware)
         ─► LLM structured JSON (findings + cited ids + inference)
         ─► citation & number check ─► 3-layer answer (data / reviews / inference)
```

## Files

| File | Role |
|---|---|
| `data_prep.py` | One-time cleaning → `data/clean/*.parquet` (rules and numbers in `DATA_NOTES.md`) |
| `stats.py` | Exact product / brand / category statistics |
| `aspects.py` | Groups free-text pros/cons into 18 fixed aspects (≈76% coverage) |
| `compare.py` | Product comparison, `mode="baseline"` or `"full"` |
| `manager.py` | Manager analytics: complaints, praises, low recommendation, brands, trend |
| `prompts.py` | All prompt templates |
| `grounding.py` | Number and citation checks (used by the pipelines and the evaluation) |
| `llm_client.py` | The only place that calls the API; logs latency and tokens to `logs/llm_calls.jsonl` |
| `eval/build_eval_set.py` | Fixed question set with gold answers computed from the data |
| `eval/run_eval.py` | Runs baseline and full, LLM judge, writes `eval/results/summary.md` |
| `eval/judge.py` | LLM-as-a-judge rubric (different model family from the generator) |

## Setup

```powershell
uv venv --python 3.11 .venv
uv pip install --python .venv\Scripts\python.exe -r comparison_analytics\requirements.txt
```

Put the API key in the environment (`QBC13_API_KEY`) or in a `.env` file in the
project root (`QBC13_API_KEY=...`). `.env` is git-ignored; never commit the key.

Download the fixed dataset revision into `data/`, then clean once (about 1.5 min):

```powershell
.venv\Scripts\python.exe -X utf8 -m comparison_analytics.data_prep
```

## Use

```python
from comparison_analytics import compare, manager

r = compare.compare([1644810, 262958], "کدوم برای پوست چرب بهتره؟")
print(r["answer"])

r = manager.ask("پرتکرارترین شکایت کاربران در دسته‌ی مراقبت پوست چیست؟")
print(r["answer"])
```

## Dashboard

```powershell
.venv\Scripts\streamlit run comparison_analytics\app.py
```

Three tabs: manager analytics (free Persian question, chart of the computed
results next to the report), product comparison (title search, pick 2-3
products, optional question) and the evaluation tables. Each answer shows its
latency, tokens, how many numbers matched the data and how many citations
were valid.

## Caching

`llm_client` can reuse answers of deterministic calls (temperature 0) from
`cache/llm_cache.sqlite`; manager analyses are cached per process. The app
turns the cache on; the evaluation keeps it off so latency and tokens are
real. `python -m comparison_analytics.eval.bench_cache` runs the same six
questions twice: median latency 7.3 s → 0.0 s, tokens 11,584 → 0, 6/6 cache
hits on the second pass (`eval/results/bench_cache.json`).

## Configuration

Settings are read from environment variables or `.env` with the names of the
team repo's `.env.example`: `QBC13_API_KEY`, `QBC13_API_BASE_URL`,
`DIGIKALA_DATA_DIR` (folder with the two CSV files; cleaned parquet files go
to its `clean/` subfolder).

## Team integration

Product search can pass its selected `product_id` values directly to
`compare.compare()`. The comparison can use the team's review retriever
through the optional `retrieve_comment_ids(query, product_id, k)` callback.
Return original Digikala comment IDs in retrieval order. The comparison
loads the review text for those IDs, checks product ownership, and falls
back to its local selector for a product when no valid IDs are returned.
Exact statistics still use the full cleaned review table.

```python
result = compare.compare(
    selected_product_ids,
    question,
    retrieve_comment_ids=sepehr_retrieve_comment_ids,
)
```

The example assumes a team adapter has been provided. For integration,
ask for the retriever's import path or endpoint, startup instructions,
the meaning of its product and comment IDs, and one example call with a
real `product_id`. If the team is also providing a shared table for
manager analytics, ask for its schema and whether it contains all
reviews in the agreed dataset or a sample used for search. The manager
uses the full cleaned tables, including category, brand, and review
fields. Vector search results represent selected reviews and suit the
comparison evidence, while category percentages need a defined full
population in the shared dataset.

The general comparison reports the top two purchase recommendation
shares as close if they differ by less than 3 percentage points. It
also qualifies the recommendation when a product has fewer than 30
reviews with a recommendation label. These are reporting rules.

## Evaluate

Everything below writes to `eval/results/final/`; finished questions are
skipped on a rerun, so an API outage only pauses the run.

```powershell
.venv\Scripts\python.exe -X utf8 -m comparison_analytics.eval.build_eval_set
sh comparison_analytics/eval/run_final.sh
```

`run_final.sh` runs, for both the zero-shot baseline and the full system:
20 product comparisons, 53 manager questions and 16 reviewed manager
questions written by other model families (Gemini, DeepSeek, Llama), then
prints `final/summary.md` and runs the retriever test below.

| What | How it is measured |
|---|---|
| Numbers (Grounding) | every number in the answer is matched against the computed facts |
| Citations | every cited comment id must be one of the reviews sent for that product |
| Key finding / winner | compared with the gold answer computed from the data |
| Retrieval quality | an LLM judge labels each retrieved review relevant or not (precision@8) |
| Answer quality | LLM judge from another model family, 1-5 (relevance, usefulness, faithfulness) |
| Latency, cost | measured per question from the API log (tokens) |

### Retriever test

`python -m comparison_analytics.eval.ab_retriever` compares three ways of
finding evidence reviews on 18 product pairs from the team's sample:
keyword aspects, the team's BM25 retriever, and Yalda vectors (the model and
settings of the team's vector database). Yalda gave the most relevant
evidence and is the default (`COMPARE_RETRIEVER=keywords` switches it off).

### Judge validation (human check)

`python -m comparison_analytics.eval.human_check build` writes a blind Excel
sheet with 20 answers from the final run (12 full, 8 baseline, shuffled) and
the computed facts for each. Usefulness was rated by a human; faithfulness was
checked claim by claim against the facts. Agreement with the LLM judge
(`eval/results/human_agreement.json`):

| | Usefulness (human vs judge) | Faithfulness (fact check vs judge) |
|---|---|---|
| Spearman | 0.58 (p = 0.007) | 0.82 (p < 0.001) |
| Mean absolute difference | 0.40 | 0.55 |
| Within one point | 95% | 85% |

Both raters rank the full system clearly above the baseline. The judge's one
large miss: a two-part question ("oily skin" and "absorption") answered only
in part got 5 from the judge and 2 from the human, so the judge does not check
that every part of the question is covered.

### Failure analysis

- The model declared a winner for products whose recommendation rates were
  less than 3 points apart. The gap rule is now applied by code.
- The model once called a 20-point gap "under 3 points". The gap and the
  close-call flag are now computed by code and passed to the model.
- A manager report used the total number of comments as the base of a
  complaint percentage. The prompt now names each denominator.
- Two comparison answers cited a comment from the other product; the
  citation check drops such findings before the user sees them.
- «چرا از اسباب‌بازی‌های لوکس ما کمتر خرید می‌شود؟» is a sales question but is
  routed to low recommendation: the one routing error on the reviewed set.

Limits: the LLM judge has not been checked against human ratings; the 16
reviewed questions were also used to fix two routing rules; monetary cost
needs the bootcamp key's price per token.
