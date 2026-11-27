"""A/B/C test of the comparison's evidence retriever: ours vs the team's BM25 vs Yalda vectors.

Same products, same questions, same answer model; only the review retriever
changes. Products come from the team's development sample so both retrievers
cover them.

Retrieval quality is judged independently of both retrievers: an LLM judge
(different model family) labels every retrieved review as relevant or not to
the question (precision@k). Our own aspect keywords are NOT used to score,
because our retriever is built on them and that would be circular.

Run:  python -m comparison_analytics.eval.ab_retriever
"""

from __future__ import annotations

import json
import random
import statistics

from .. import compare as C
from .. import llm_client, stats
from ..team import adapter, yalda
from .judge import JUDGE_MODEL, judge
from .run_eval import RESULTS

QUESTIONS = {
    "مراقبت پوست": "کدوم برای پوست چرب بهتره و جذبش چطوره؟",
    "شامپو و مراقبت مو": "کدوم بوی بهتری داره و موها رو نرم‌تر می‌کنه؟",
    "لباس زنانه": "سایز و جنس پارچه‌ی کدوم بهتره؟",
    "اسباب بازی": "کیفیت ساخت و دوام کدوم بیشتره؟",
    "نوشت افزار": "کدوم قیمتش به کیفیتش می‌ارزه؟",
    "لباس مردانه": "اندازه و جنس کدوم بهتره؟",
}
PAIRS_PER_CATEGORY = 3
ARMS = ("ours", "team_bm25", "yalda")
SEED = 7

RELEVANCE = """برای هر نظر مشخص کن آیا به سؤال کاربر مربوط است و اطلاعاتی درباره‌ی همان ویژگی‌ای که پرسیده شده دارد.
سؤال: {question}

نظرها:
{reviews}

فقط JSON: {{"relevant": [0 یا 1 برای هر نظر به همان ترتیب]}}"""


def build_pairs() -> list[dict]:
    rng = random.Random(SEED)
    c = stats.comments()
    cov = adapter.covered_products()
    lab = c[c["product_id"].isin(cov) & c["recommendation_status"].notna()]
    per = lab.groupby("product_id").agg(n=("recommendation_status", "size"), cat=("Category1", "first"))
    pairs = []
    for cat, q in QUESTIONS.items():
        ids = sorted(per[(per["cat"] == cat) & (per["n"] >= 40)].index)
        rng.shuffle(ids)
        for i in range(0, min(len(ids) - 1, 2 * PAIRS_PER_CATEGORY), 2):
            pairs.append({"id": f"AB-{len(pairs)}", "category": cat, "question": q,
                          "products": [int(ids[i]), int(ids[i + 1])]})
    return pairs


def relevance(question: str, texts: list[str]) -> list[int]:
    if not texts:
        return []
    body = "\n".join(f"{i + 1}. {t[:300]}" for i, t in enumerate(texts))
    out = llm_client.chat_json([{"role": "user", "content": RELEVANCE.format(question=question, reviews=body)}],
                               model=JUDGE_MODEL, task="judge", max_tokens=300)
    rel = [int(bool(x)) for x in out.get("relevant", [])]
    return rel[:len(texts)] + [0] * (len(texts) - len(rel))


def run_one(pair: dict, arm: str) -> dict:
    llm_client.CALLS.clear()
    kwargs = {"team_bm25": {"retrieve_comment_ids": adapter.retrieve_comment_ids},
              "yalda": {"retrieve_comment_ids": yalda.retrieve_comment_ids}}.get(arm, {})
    out = C.compare(pair["products"], pair["question"], **kwargs)
    texts = [e["text"] for e in out["evidence"]]
    rel = relevance(pair["question"], texts)
    cited = {(f["product_id"], c) for f in out["findings"] for c in f["comment_ids"]}
    rel_by_id = {(e["product_id"], e["comment_id"]): r for e, r in zip(out["evidence"], rel)}
    cited_rel = [rel_by_id[c] for c in cited if c in rel_by_id]
    j = judge(out["question"], {"products": out["facts"], "reviews": out["evidence"]}, out["answer"], "یک خریدار")
    gen = [c for c in llm_client.CALLS if c.get("task") != "judge"]
    return {"id": pair["id"], "arm": arm, "n_evidence": len(texts),
            "precision_at_k": round(sum(rel) / len(rel), 3) if rel else None,
            "cited_relevant": round(sum(cited_rel) / len(cited_rel), 3) if cited_rel else None,
            "citation_precision": out["checks"]["citations"]["citation_precision"],
            "faithfulness": j.get("faithfulness"), "usefulness": j.get("usefulness"),
            "latency_s": out["latency_s"],
            "tokens": sum((c.get("prompt_tokens") or 0) + (c.get("completion_tokens") or 0) for c in gen)}


def main() -> None:
    path = RESULTS / "ab_retriever.jsonl"
    done = set()
    if path.exists():
        done = {(json.loads(l)["id"], json.loads(l)["arm"]) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()}
    pairs = build_pairs()
    for pair in pairs:
        for arm in ARMS:
            if (pair["id"], arm) in done:
                continue
            try:
                row = run_one(pair, arm)
            except Exception as e:
                print(pair["id"], arm, "FAILED", str(e)[:120]); continue
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(pair["id"], arm, row["precision_at_k"], row["latency_s"], flush=True)
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    summary = {}
    for arm in ARMS:
        r = [x for x in rows if x["arm"] == arm]
        if not r:
            continue
        m = lambda k: round(statistics.mean([x[k] for x in r if isinstance(x.get(k), (int, float))]), 3)
        summary[arm] = {"pairs": len(r), "precision_at_k": m("precision_at_k"), "cited_relevant": m("cited_relevant"),
                        "citation_precision": m("citation_precision"), "faithfulness": m("faithfulness"),
                        "usefulness": m("usefulness"), "median_latency_s": statistics.median(x["latency_s"] for x in r),
                        "mean_tokens": m("tokens")}
    (RESULTS / "ab_retriever_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
