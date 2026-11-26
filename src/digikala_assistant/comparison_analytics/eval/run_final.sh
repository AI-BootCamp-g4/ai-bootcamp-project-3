#!/bin/sh
# Final evaluation after aligning the recommendation rate with the team.
# Safe to re-run: finished questions are skipped, so an API outage only pauses it.
# Run from the project root:  sh comparison_analytics/eval/run_final.sh
D=comparison_analytics/eval/results/final
PY=".venv/Scripts/python.exe -X utf8 -m"
for part in compare manager manager_heldout; do
  for mode in baseline full; do
    echo "== $part $mode"; $PY comparison_analytics.eval.run_eval --part $part --mode $mode --results-dir $D
  done
done
$PY comparison_analytics.eval.run_eval --summary --results-dir $D
echo "== retriever A/B (ours vs team BM25)"
$PY comparison_analytics.eval.ab_retriever
