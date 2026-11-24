import json
import io
import numpy as np
import pandas as pd
import altair as alt
import joblib
import streamlit as st

from text_utils import clean_text

st.set_page_config(page_title="Review Sentiment Analyzer", page_icon="☕", layout="wide")

# ---------------------------------------------------------------- styling
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 1.6rem; max-width: 1150px; }
.hero {
    background: linear-gradient(135deg, #0b3d2e 0%, #14705a 55%, #1f9d7a 100%);
    border-radius: 20px; padding: 34px 38px; color: #fff; margin-bottom: 22px;
    box-shadow: 0 10px 30px rgba(11,61,46,.25);
}
.hero h1 { margin: 0 0 6px 0; font-size: 2.1rem; font-weight: 800; letter-spacing: -.02em; }
.hero p { margin: 0; opacity: .88; font-size: 1.02rem; }
.pill { display:inline-block; background:rgba(255,255,255,.16); border-radius:999px; padding:4px 12px;
        font-size:.78rem; margin:14px 8px 0 0; font-weight:500; }
.result { border-radius: 18px; padding: 26px 28px; border: 1px solid; margin-top: 6px; }
.result.pos { background:#eff6ff; border-color:#bfdbfe; }
.result.neg { background:#fff7ed; border-color:#fed7aa; }
.result .big { font-size: 2rem; font-weight: 800; margin: 0; letter-spacing:-.02em; }
.result.pos .big { color:#1d4ed8; } .result.neg .big { color:#c2410c; }
.result .sub { color:#475569; margin: 2px 0 14px 0; font-size:.95rem; }
.bar { height: 12px; border-radius: 999px; background:#e2e8f0; overflow:hidden; }
.bar > div { height: 100%; border-radius: 999px; }
.chip { display:inline-block; padding:5px 12px; border-radius:999px; margin:4px 6px 0 0; font-size:.86rem; font-weight:600; }
.chip.p { background:#dbeafe; color:#1e40af; } .chip.n { background:#ffedd5; color:#9a3412; }
.card { background: var(--secondary-background-color); border-radius:16px; padding:18px 20px; }
.small { color:#64748b; font-size:.85rem; }
div.stButton > button { border-radius: 12px; font-weight: 600; }
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------- model
@st.cache_resource
def load_assets():
    model = joblib.load("sentiment_model.joblib")
    with open("model_info.json") as f:
        info = json.load(f)
    return model, info


model, info = load_assets()
POS_C, NEG_C = "#2563eb", "#ea580c"


def score_positive(texts):
    """Return P(positive) for a list of already-cleaned texts."""
    clf = model.named_steps["clf"]
    if hasattr(clf, "predict_proba"):
        return model.predict_proba(texts)[:, 1]
    return 1 / (1 + np.exp(-2 * model.decision_function(texts)))


def explain(cleaned, top_k=8):
    """Words that pushed the decision (linear models): weight x tf-idf value."""
    vec, clf = model.named_steps["tfidf"], model.named_steps["clf"]
    if not hasattr(clf, "coef_"):
        return []
    X = vec.transform([cleaned])
    names = vec.get_feature_names_out()
    coef = clf.coef_[0]
    contrib = [(names[i], float(X[0, i] * coef[i])) for i in X.nonzero()[1]]
    return sorted(contrib, key=lambda t: -abs(t[1]))[:top_k]


# ---------------------------------------------------------------- header
fm = info["final_metrics"]
st.markdown(
    f"""
<div class="hero">
  <h1>☕ Review Sentiment Analyzer</h1>
  <p>Paste a customer review and the model tells you whether it is <b>Positive</b> or <b>Negative</b> — and which words decided it.</p>
  <span class="pill">Model: {info['model_name']}</span>
  <span class="pill">Trained on {info['n_train']} Starbucks reviews</span>
  <span class="pill">Test accuracy {fm['Accuracy']:.0%}</span>
  <span class="pill">Macro-F1 {fm['Macro F1']:.2f}</span>
</div>
""",
    unsafe_allow_html=True,
)

EXAMPLES = {
    "— pick an example —": "",
    "😊 Happy customer": "The barista was friendly and my latte was perfect. I love this place and will definitely come back!",
    "😡 Angry customer": "Terrible service. The coffee was cold, the order was wrong and the staff was rude to me.",
    "🔁 Negation: not good": "The coffee was not good and the service was not great either.",
    "🤔 Tricky: not bad": "Not bad at all, the drink was really good!",
    "📦 Wrong order": "I would never go back, they got my order wrong again and nobody cared.",
}

tab1, tab2, tab3 = st.tabs(["🔍 Analyze a review", "📂 Batch (CSV)", "📊 Model insights"])

# ---------------------------------------------------------------- tab 1
with tab1:
    left, right = st.columns([1.15, 1], gap="large")
    with left:
        def _load_example():
            ex = EXAMPLES[st.session_state["example_choice"]]
            if ex:
                st.session_state["review_text"] = ex

        st.selectbox("Try an example", list(EXAMPLES), key="example_choice", on_change=_load_example)
        text = st.text_area("Your review", key="review_text", height=190,
                            placeholder="Type or paste a review here…")
        go = st.button("Analyze sentiment", type="primary", width="stretch")
        st.caption("Tip: negations are understood — “not good” and “good” give opposite answers.")

    with right:
        if go or text.strip():
            cleaned = clean_text(text)
            if not text.strip():
                st.info("Write a review on the left to get started.")
            elif not cleaned:
                st.warning("I could not find any meaningful words in that text. Try a longer review.")
            else:
                p_pos = float(score_positive([cleaned])[0])
                is_pos = p_pos >= 0.5
                conf = p_pos if is_pos else 1 - p_pos
                cls, emoji, label = ("pos", "😊", "Positive") if is_pos else ("neg", "😞", "Negative")
                color = POS_C if is_pos else NEG_C
                st.markdown(
                    f"""
<div class="result {cls}">
  <p class="big">{emoji} {label}</p>
  <p class="sub">Model confidence: <b>{conf:.0%}</b></p>
  <div class="bar"><div style="width:{conf*100:.0f}%; background:{color}"></div></div>
  <p class="small" style="margin-top:10px">P(positive) = {p_pos:.0%} &nbsp;·&nbsp; P(negative) = {1-p_pos:.0%}</p>
</div>
""",
                    unsafe_allow_html=True,
                )
                words = explain(cleaned)
                if words:
                    st.markdown("**Why this prediction?**")
                    chips = "".join(
                        f'<span class="chip {"p" if w >= 0 else "n"}">{"＋" if w >= 0 else "－"} {t}</span>'
                        for t, w in words
                    )
                    st.markdown(chips, unsafe_allow_html=True)
                    st.caption("Blue words pushed towards Positive, orange towards Negative (strongest first).")
                with st.expander("See the cleaned text the model actually reads"):
                    st.code(cleaned, language=None)
        else:
            st.markdown(
                '<div class="card"><b>Result appears here</b><br><span class="small">Pick an example or write your own review, then press <i>Analyze sentiment</i>.</span></div>',
                unsafe_allow_html=True,
            )

# ---------------------------------------------------------------- tab 2
with tab2:
    st.markdown("Upload a CSV with a column of reviews to score them all at once.")
    up = st.file_uploader("CSV file", type=["csv"])
    if up is not None:
        data = pd.read_csv(up)
        guess = [c for c in data.columns if c.lower() in ("review", "text", "comment", "content")]
        col = st.selectbox("Which column contains the review text?", data.columns,
                           index=list(data.columns).index(guess[0]) if guess else 0)
        if st.button("Score all reviews", type="primary"):
            texts = data[col].astype(str)
            cleaned = texts.apply(clean_text)
            ok = cleaned.str.strip() != ""
            p = np.full(len(data), np.nan)
            if ok.any():
                p[ok.values] = score_positive(cleaned[ok].tolist())
            out = data.copy()
            out["sentiment"] = np.where(np.isnan(p), "Unknown", np.where(p >= 0.5, "Positive", "Negative"))
            out["confidence"] = np.where(np.isnan(p), np.nan, np.maximum(p, 1 - p)).round(3)
            c1, c2, c3 = st.columns(3)
            c1.metric("Reviews scored", int(ok.sum()))
            c2.metric("Positive", int((out.sentiment == "Positive").sum()))
            c3.metric("Negative", int((out.sentiment == "Negative").sum()))
            dist = out.sentiment.value_counts().rename_axis("sentiment").reset_index(name="count")
            st.altair_chart(
                alt.Chart(dist).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6).encode(
                    x=alt.X("sentiment:N", title="Sentiment", sort=["Negative", "Positive", "Unknown"]),
                    y=alt.Y("count:Q", title="Number of reviews"),
                    color=alt.Color("sentiment:N", legend=None,
                                    scale=alt.Scale(domain=["Negative", "Positive", "Unknown"], range=[NEG_C, POS_C, "#9ca3af"])),
                    tooltip=["sentiment", "count"],
                ).properties(height=260, title="Predicted sentiment"),
                width="stretch",
            )
            st.dataframe(out, width="stretch", height=320)
            st.download_button("⬇️ Download results (CSV)", out.to_csv(index=False).encode("utf-8"),
                               "sentiment_results.csv", "text/csv")

# ---------------------------------------------------------------- tab 3
with tab3:
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy", f"{fm['Accuracy']:.1%}")
    m2.metric("Precision (Positive)", f"{fm['Precision (Pos)']:.2f}")
    m3.metric("Recall (Positive)", f"{fm['Recall (Pos)']:.2f}")
    m4.metric("F1 (Positive)", f"{fm['F1 (Pos)']:.2f}")
    m5.metric("Macro F1", f"{fm['Macro F1']:.2f}")
    st.caption(
        f"Test set: {info['n_test']} reviews the model never saw. Data: {info['label_counts'].get('Negative', 0)} negative vs "
        f"{info['label_counts'].get('Positive', 0)} positive — imbalanced, so accuracy alone is not enough; look at F1 and recall."
    )

    c1, c2 = st.columns(2, gap="large")
    with c1:
        cm = np.array(info["confusion_matrix"])
        cmdf = pd.DataFrame(
            [(a, p, int(cm[i, j])) for i, a in enumerate(["Negative", "Positive"]) for j, p in enumerate(["Negative", "Positive"])],
            columns=["Actual", "Predicted", "Reviews"],
        )
        base = alt.Chart(cmdf).encode(x=alt.X("Predicted:N", sort=["Negative", "Positive"]), y=alt.Y("Actual:N", sort=["Negative", "Positive"]))
        heat = base.mark_rect(cornerRadius=6).encode(color=alt.Color("Reviews:Q", scale=alt.Scale(scheme="blues"), legend=None))
        txt = base.mark_text(fontSize=22, fontWeight="bold").encode(text="Reviews:Q",
                                                                   color=alt.condition(alt.datum.Reviews > cm.max() / 2, alt.value("white"), alt.value("#0f172a")))
        st.altair_chart((heat + txt).properties(height=280, title="Confusion matrix (test set)"), width="stretch")
    with c2:
        comp = pd.DataFrame(info["comparison"])
        comp = comp[["Model", "Accuracy", "Macro F1"]].melt("Model", var_name="Metric", value_name="Score")
        st.altair_chart(
            alt.Chart(comp).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
                x=alt.X("Score:Q", scale=alt.Scale(domain=[0, 1])),
                y=alt.Y("Model:N", sort="-x", title=None),
                yOffset="Metric:N",
                color=alt.Color("Metric:N", scale=alt.Scale(range=["#9ca3af", POS_C])),
                tooltip=["Model", "Metric", alt.Tooltip("Score:Q", format=".3f")],
            ).properties(height=280, title="All models: accuracy vs macro-F1"),
            width="stretch",
        )

    if info.get("top_positive"):
        w1, w2 = st.columns(2, gap="large")
        for col, key, color, title in [(w1, "top_negative", NEG_C, "Words that signal NEGATIVE"), (w2, "top_positive", POS_C, "Words that signal POSITIVE")]:
            d = pd.DataFrame({"word": list(info[key].keys()), "weight": list(info[key].values())}).head(12)
            col.altair_chart(
                alt.Chart(d).mark_bar(color=color, cornerRadiusTopRight=4, cornerRadiusBottomRight=4).encode(
                    x=alt.X("weight:Q", title="Model weight"), y=alt.Y("word:N", sort="-x", title=None),
                    tooltip=["word", alt.Tooltip("weight:Q", format=".2f")],
                ).properties(height=300, title=title),
                width="stretch",
            )

    st.markdown("##### Known limits")
    st.markdown(
        "- Trained on **a few hundred Starbucks reviews** — it may be less accurate on other products or on sarcasm.\n"
        "- Only two classes: 3-star (mixed) reviews were excluded from training.\n"
        "- The data is imbalanced (mostly negative), so the model is more reliable on negative reviews."
    )
