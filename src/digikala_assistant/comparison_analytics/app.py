"""Interactive dashboard for product comparison and manager analytics.

Run:  .venv\\Scripts\\streamlit run comparison_analytics\\app.py
(bonus: UI / dashboard for the store manager and the buyer)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from comparison_analytics import compare, llm_client, manager, stats  # noqa: E402

llm_client.enable_cache(True)

st.set_page_config(page_title="دستیار دیجی‌کالا", page_icon="🛒", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@400;600;700&display=swap');
html, body, [class*="css"], .stMarkdown, .stTextInput, .stSelectbox, .stButton, p, li, td, th, label, h1, h2, h3 {
  font-family: 'Vazirmatn', Tahoma, sans-serif !important; }
.block-container { direction: rtl; text-align: right; }
.stMarkdown table { direction: rtl; }
[data-testid="stMetricValue"] { direction: ltr; text-align: right; }
code, pre { direction: ltr; text-align: left; }
</style>""", unsafe_allow_html=True)

COLORS = ["#2a6f97", "#61a5c2", "#89c2d9", "#a9d6e5", "#468faf", "#01497c", "#014f86", "#2c7da0"]
INTENT_FA = {"complaints": "شکایت‌های پرتکرار", "praises": "نقاط قوت پرتکرار",
             "low_recommendation": "پرنظر ولی کم‌پیشنهاد", "brands": "مقایسه‌ی برندها",
             "trend": "روند سالانه", "unsupported": "خارج از دامنه"}


@st.cache_resource(show_spinner="بارگذاری داده (حدود ۱۰ ثانیه، فقط بار اول)...")
def warm_up() -> list[str]:
    stats.comments(); stats.products()
    compare.default_retriever()  # loads the Yalda model once, if installed
    return manager.category_names()


def usage_badges(t0: float, calls: list[dict], checks: dict | None, extra: dict | None = None) -> None:
    cols = st.columns(4)
    cols[0].metric("زمان پاسخ", f"{time.perf_counter() - t0:.1f} s")
    tokens = sum((c.get("prompt_tokens") or 0) + (c.get("completion_tokens") or 0) for c in calls)
    cached = sum(bool(c.get("cached")) for c in calls)
    cols[1].metric("توکن مصرفی", f"{tokens:,}", f"{cached} از کش" if cached else None)
    if checks and checks.get("numbers", {}).get("numbers"):
        n = checks["numbers"]
        cols[2].metric("عددهای منطبق با داده", f"{n['numbers'] - len(n['unsupported'])}/{n['numbers']}")
    if extra:
        cols[3].metric(*extra)


# ----------------------------------------------------------------- manager

def manager_tab(categories: list[str]) -> None:
    st.subheader("تحلیل برای مدیر دسته")
    examples = ["پرتکرارترین شکایت کاربران در دسته‌ی مراقبت پوست چیست؟",
                "برندای شامپو و مراقبت مو رو از نظر رضایت مشتری مقایسه کن",
                "کدام محصولات اسباب بازی نظر زیادی دارند اما درصد پیشنهاد خرید آن‌ها پایین است؟",
                "رضایت مشتریای لباس زنانه طی این چند سال بهتر شده یا بدتر؟"]
    pick = st.selectbox("نمونه‌ی سؤال", ["(سؤال خودم)"] + examples)
    question = st.text_input("سؤال", value="" if pick == "(سؤال خودم)" else pick,
                             placeholder="مثلاً: مشتری‌ها بیشتر از چی محصولات نوشت افزار ناراضی‌اند؟")
    if not st.button("تحلیل کن", type="primary", disabled=not question.strip()):
        return
    llm_client.CALLS.clear(); t0 = time.perf_counter()
    with st.spinner("در حال تحلیل..."):
        out = manager.ask(question)
    r = out["routing"]
    st.caption(f"نوع تحلیل: **{INTENT_FA.get(r['intent'], r['intent'])}** · دسته: **{r.get('category') or '-'}**"
               f" · تشخیص با: {'قواعد' if r.get('router') == 'rules' else 'مدل'}")
    usage_badges(t0, list(llm_client.CALLS), out.get("checks"))
    if out["results"] is None:
        st.warning(out["answer"]); return
    left, right = st.columns([1.1, 1])
    with left:
        st.markdown(out["answer"])
    with right:
        draw_results(r["intent"], out["results"])
    with st.expander("نتایج محاسبه‌شده (JSON) که به مدل داده شد"):
        st.json(out["results"])


def draw_results(intent: str, res: dict) -> None:
    ov = res["overview"]
    c = st.columns(3)
    c[0].metric("نظرهای دسته", f"{ov['n_comments']:,}")
    c[1].metric("درصد پیشنهاد خرید", f"{ov['recommended_pct']}٪")
    c[2].metric("میانگین امتیاز", ov["mean_rate"])
    if intent in ("complaints", "praises"):
        df = pd.DataFrame(res["aspects"])
        fig = px.bar(df.iloc[::-1], x="share_pct", y="label", orientation="h", text="share_pct",
                     color_discrete_sequence=[COLORS[0] if intent == "praises" else "#c8553d"],
                     labels={"share_pct": "٪ نظرهای دارای نکته", "label": ""})
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10),
                          title=f"پایه: {res['base_comments_with_points']:,} نظر")
        st.plotly_chart(fig, use_container_width=True)
    elif intent == "low_recommendation":
        df = pd.DataFrame(res["products"])
        st.dataframe(df.rename(columns={"title": "محصول", "brand": "برند", "n_labelled": "نظر برچسب‌دار",
                                        "recommended_pct": "٪ پیشنهاد", "mean_rate": "امتیاز",
                                        "top_complaint": "شکایت اصلی"}).drop(columns="product_id"),
                     hide_index=True, use_container_width=True)
        st.caption(f"میانگین درصد پیشنهاد دسته: {res['category_recommended_pct']}٪")
    elif intent == "brands":
        df = pd.DataFrame(res["brands"])
        if not df.empty:
            fig = px.scatter(df, x="mean_rate", y="recommended_pct", size="n_comments", text="brand",
                             labels={"mean_rate": "میانگین امتیاز", "recommended_pct": "٪ پیشنهاد"},
                             color_discrete_sequence=[COLORS[0]])
            fig.update_traces(textposition="top center")
            fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)
        t = res.get("test", {})
        if "p_value" in t:
            st.caption(f"آزمون کروسکال‌والیس روی امتیازها: p = {t['p_value']:.2g} → "
                       + ("تفاوت برندها معنادار است" if t["significant"] else "تفاوت معناداری دیده نشد"))
    elif intent == "trend":
        df = pd.DataFrame(res["years"])
        fig = px.line(df, x="year", y="recommended_pct", markers=True,
                      labels={"year": "سال", "recommended_pct": "٪ پیشنهاد خرید"},
                      color_discrete_sequence=[COLORS[0]])
        fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
    for note in res.get("data_notes", []):
        st.caption("• " + note)


# ----------------------------------------------------------------- compare

def compare_tab() -> None:
    st.subheader("مقایسه‌ی محصولات")
    query = st.text_input("جستجوی محصول با عنوان", placeholder="مثلاً: کرم ضد آفتاب سینره")
    options: dict[str, int] = st.session_state.setdefault("options", {})
    if query.strip():
        hits = stats.find_products(query, limit=15)
        for pid, row in hits.iterrows():
            options[f"{row['title_fa']} · {row['n_comments']} نظر · #{pid}"] = int(pid)
    chosen = st.multiselect("محصول‌ها (دو یا سه تا)", list(options), max_selections=3)
    question = st.text_input("سؤال (اختیاری)", placeholder="مثلاً: کدوم برای پوست چرب بهتره؟")
    if not st.button("مقایسه کن", type="primary", disabled=len(chosen) < 2):
        return
    llm_client.CALLS.clear(); t0 = time.perf_counter()
    with st.spinner("در حال مقایسه..."):
        out = compare.compare([options[c] for c in chosen], question)
    cit = out["checks"]["citations"]
    usage_badges(t0, list(llm_client.CALLS), out["checks"],
                 ("ارجاع‌های معتبر", f"{cit['citations'] - len(cit['invalid'])}/{cit['citations']}"))
    st.markdown(out["answer"])
    with st.expander("نظرهایی که به مدل نشان داده شد (شواهد)"):
        st.dataframe(pd.DataFrame(out["evidence"]), hide_index=True, use_container_width=True)


# -------------------------------------------------------------- evaluation

def eval_tab() -> None:
    st.subheader("نتایج ارزیابی")
    base = Path(__file__).resolve().parent / "eval" / "results"
    for name, path in (("ارزیابی نهایی", base / "final" / "summary.md"),):
        if path.exists():
            with st.expander(name, expanded=True):
                st.markdown(path.read_text(encoding="utf-8").split("\n", 1)[1])
    bench = base / "bench_cache.json"
    if bench.exists():
        b = json.loads(bench.read_text(encoding="utf-8"))
        st.markdown("**اثر کش** (همان سؤال‌ها، بار دوم)")
        c = st.columns(3)
        c[0].metric("زمان میانه", f"{b['warm_median_s']} s", f"{b['warm_median_s'] - b['cold_median_s']:.1f} s")
        c[1].metric("توکن", b["warm_tokens"], -b["cold_tokens"])
        c[2].metric("فراخوانی از کش", b["warm_cache_hits"])


def main() -> None:
    categories = warm_up()
    st.title("دستیار خرید و تحلیل دیجی‌کالا")
    st.caption("عددها با کد روی کل نظرات حساب می‌شوند؛ مدل زبانی فقط توضیح می‌دهد و هر ادعا با داده چک می‌شود.")
    t1, t2, t3 = st.tabs(["تحلیل مدیر", "مقایسه‌ی محصولات", "ارزیابی"])
    with t1:
        manager_tab(categories)
    with t2:
        compare_tab()
    with t3:
        eval_tab()


main()
