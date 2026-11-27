"""Human check of the LLM judge (project: compare LLM-as-a-judge with human evaluation).

1. Build:  python -m comparison_analytics.eval.human_check build
   Writes an Excel sheet with 20 answers from the final evaluation (mixed
   baseline and full, comparison and manager, shuffled, mode hidden), the
   reference facts for each answer, and two 1-5 columns to fill in.
2. Score:  python -m comparison_analytics.eval.human_check score
   Reads the filled sheet and compares human scores with the judge's scores
   for the same answers: Spearman correlation (taught in the statistics
   workshop), mean absolute difference and agreement within one point.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

from .. import compare as C
from ..manager import analyse
from .run_eval import HERE, RESULTS

FINAL = RESULTS / "final"
SHEET = Path.home() / "Desktop" / "برگه ارزیابی انسانی - پروژه سوم.xlsx"
KEY = RESULTS / "human_key.json"
PLAN = [("manager_full", 6), ("manager_baseline", 4), ("compare_full", 6), ("compare_baseline", 4)]
METRICS = [("faithfulness", "وفاداری به داده (۱ تا ۵)"), ("usefulness", "مفید بودن (۱ تا ۵)")]
SEED = 13


def _rows(name: str) -> list[dict]:
    p = FINAL / f"{name}.jsonl"
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def _facts_manager(q: dict) -> str:
    res = analyse(q["intent"], q["category"])
    ov = res["overview"]
    head = (f"دسته: {q['category']} | نظرها: {ov['n_comments']:,} | درصد پیشنهاد خرید: {ov['recommended_pct']}٪ "
            f"| میانگین امتیاز: {ov['mean_rate']}")
    if "aspects" in res:
        body = "، ".join(f"{a['label']} {a['share_pct']}٪" for a in res["aspects"][:5])
        return f"{head}\nپایه: {res['base_comments_with_points']:,} نظر دارای نکته\nجنبه‌های پرتکرار: {body}"
    if "products" in res:
        body = "\n".join(f"- {p['title'][:60]} | پیشنهاد {p['recommended_pct']}٪ | امتیاز {p['mean_rate']}"
                         for p in res["products"][:4])
        return f"{head}\nکم‌پیشنهادترین محصولات پرنظر:\n{body}"
    if "brands" in res:
        body = "\n".join(f"- {b['brand']} | امتیاز {b['mean_rate']} | پیشنهاد {b['recommended_pct']}٪"
                         for b in res["brands"][:6])
        t = res.get("test", {})
        sig = "" if "p_value" not in t else f"\nآزمون کروسکال‌والیس: {'تفاوت معنادار' if t['significant'] else 'تفاوت معنادار نیست'}"
        return f"{head}\nبرندها:\n{body}{sig}"
    body = "، ".join(f"{y['year']}: {y['recommended_pct']}٪" for y in res["years"])
    return f"{head}\nدرصد پیشنهاد به تفکیک سال: {body}"


def _facts_compare(q: dict) -> str:
    lines = []
    for pid in q["products"]:
        f = C.product_facts(pid)
        praised = "، ".join(a["aspect"] for a in f["top_praised_aspects"][:2]) or "-"
        weak = "، ".join(a["aspect"] for a in f["top_complained_aspects"][:2]) or "-"
        lines.append(f"- {f['title'][:70]}\n  قیمت {f['price_rial'] // 10 if f['price_rial'] else '-':,} تومان | "
                     f"پیشنهاد {f['recommended_pct']}٪ از {f['n_labelled']} نظر | امتیاز {f['mean_rate_out_of_5']} | "
                     f"قوت: {praised} | ضعف: {weak}")
    return "\n".join(lines)


def _lines(text: str, width: int) -> int:
    """Rough number of wrapped lines a cell needs."""
    return sum(max(1, -(-len(line) // width)) for line in str(text).split("\n"))


def build() -> None:
    rng = random.Random(SEED)
    qs = {q["id"]: q for part in ("manager", "compare")
          for q in json.loads((HERE / f"questions_{part}.json").read_text(encoding="utf-8"))}
    picked = []
    for name, n in PLAN:
        rows = [r for r in _rows(name) if isinstance((r.get("judge") or {}).get("faithfulness"), int)
                and qs[r["id"]].get("intent") != "unsupported"]
        for r in rng.sample(rows, n):
            picked.append((name, r))
    rng.shuffle(picked)

    wb = Workbook()
    guide = wb.active
    guide.title = "راهنما"
    guide.sheet_view.rightToLeft = True
    text = [
        "برگهٔ ارزیابی انسانی پاسخ‌های دستیار",
        "",
        "هدف: مقایسهٔ نمرهٔ شما با نمرهٔ داور خودکار (مدل زبانی) روی همین پاسخ‌ها.",
        "برای هر ردیف در برگهٔ «پاسخ‌ها»: سؤال، دادهٔ درست (محاسبه‌شده با کد) و جواب سیستم آمده است.",
        "دو ستون زرد را با عدد ۱ تا ۵ پر کنید. ستون توضیح اختیاری است.",
        "",
        "وفاداری به داده: ادعاها و عددهای جواب با «دادهٔ درست» سازگارند؟",
        "  ۵ = همه سازگار   ۳ = یک یا دو مورد بی‌پشتوانه یا اشتباه   ۱ = ادعای غلط یا ساختگی جدی",
        "مفید بودن: برای تصمیم‌گیری (خریدار یا مدیر) چقدر به درد می‌خورد؟",
        "  ۵ = روشن و عملی   ۳ = کلی   ۱ = بی‌فایده",
        "",
        "نوع سیستمی که جواب را داده عمداً نوشته نشده تا قضاوت بی‌طرف باشد. زمان تقریبی: ۱۵ تا ۲۰ دقیقه.",
        "اگر متن یک خانه کامل دیده نمی‌شود، روی خانه کلیک کنید تا متن کامل در نوار بالای اکسل (Formula Bar) نمایش داده شود.",
        "بعد از پر کردن، فایل را ذخیره کنید (همین نام و همین جا).",
    ]
    for i, t in enumerate(text, 1):
        guide.cell(row=i, column=1, value=t).font = Font(name="Tahoma", size=12, bold=(i == 1))
    guide.column_dimensions["A"].width = 110

    ws = wb.create_sheet("پاسخ‌ها")
    ws.sheet_view.rightToLeft = True
    headers = ["شماره", "سؤال", "دادهٔ درست (برای چک کردن)", "جواب سیستم", METRICS[0][1], METRICS[1][1], "توضیح (اختیاری)"]
    widths = [7, 26, 48, 130, 12, 12, 20]
    yellow = PatternFill("solid", fgColor="FFF4C2")
    head_fill = PatternFill("solid", fgColor="DCE8F2")
    for c, (h, w) in enumerate(zip(headers, widths), 1):
        cell = ws.cell(row=1, column=c, value=h)
        cell.font = Font(name="Tahoma", bold=True, size=11)
        cell.fill = head_fill
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        ws.column_dimensions[cell.column_letter].width = w
    dv = DataValidation(type="whole", operator="between", formula1="1", formula2="5", allow_blank=True,
                        showErrorMessage=True, errorTitle="عدد ۱ تا ۵", error="فقط یک عدد صحیح از ۱ تا ۵")
    ws.add_data_validation(dv)

    key = []
    for i, (name, r) in enumerate(picked, 1):
        q = qs[r["id"]]
        facts = _facts_manager(q) if name.startswith("manager") else _facts_compare(q)
        question = r["question"]
        values = [f"H{i:02d}", question, facts, r["answer"], None, None, None]
        for c, v in enumerate(values, 1):
            cell = ws.cell(row=i + 1, column=c, value=v)
            cell.font = Font(name="Tahoma", size=9)
            cell.alignment = Alignment(wrap_text=True, vertical="top", horizontal="right")
        for c in (5, 6):
            ws.cell(row=i + 1, column=c).fill = yellow
            ws.cell(row=i + 1, column=c).alignment = Alignment(horizontal="center", vertical="center")
            dv.add(ws.cell(row=i + 1, column=c))
        need = max(_lines(r["answer"], 150), _lines(facts, 60), _lines(question, 30))
        ws.row_dimensions[i + 1].height = min(409, 14 * need + 10)  # 409 = Excel's maximum
        key.append({"code": f"H{i:02d}", "source": name, "id": r["id"],
                    "judge": {m: r["judge"].get(m) for m, _ in METRICS}})
    ws.freeze_panes = "B2"
    wb.active = 1
    wb.save(SHEET)
    KEY.write_text(json.dumps(key, ensure_ascii=False, indent=1), encoding="utf-8")
    print(SHEET)


def score() -> None:
    from scipy.stats import spearmanr

    key = {k["code"]: k for k in json.loads(KEY.read_text(encoding="utf-8"))}
    ws = load_workbook(SHEET)["پاسخ‌ها"]
    pairs = {m: [] for m, _ in METRICS}
    for row in ws.iter_rows(min_row=2, values_only=True):
        code, human = row[0], {"faithfulness": row[4], "usefulness": row[5]}
        if code not in key:
            continue
        for m in pairs:
            if isinstance(human[m], (int, float)) and isinstance(key[code]["judge"][m], (int, float)):
                pairs[m].append((int(human[m]), int(key[code]["judge"][m]), key[code]["source"]))
    out = {}
    for m, ps in pairs.items():
        if len(ps) < 3:
            out[m] = {"rated": len(ps)}
            continue
        h, j = [p[0] for p in ps], [p[1] for p in ps]
        rho, pval = spearmanr(h, j)
        by_mode = {}
        for mode in ("full", "baseline"):
            sel = [p for p in ps if p[2].endswith(mode)]
            if sel:
                by_mode[mode] = {"human_mean": round(sum(p[0] for p in sel) / len(sel), 2),
                                 "judge_mean": round(sum(p[1] for p in sel) / len(sel), 2)}
        out[m] = {"rated": len(ps), "spearman": round(float(rho), 3), "p_value": float(pval),
                  "mean_abs_diff": round(sum(abs(a - b) for a, b in zip(h, j)) / len(ps), 2),
                  "within_1_pct": round(100 * sum(abs(a - b) <= 1 for a, b in zip(h, j)) / len(ps)),
                  "by_mode": by_mode}
    (RESULTS / "human_agreement.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    {"build": build, "score": score}[sys.argv[1] if len(sys.argv) > 1 else "build"]()
