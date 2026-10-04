"""
Streamlit front end.

Run:  streamlit run ui/streamlit_app.py
Talks to the FastAPI backend over HTTP rather than importing the pipeline
directly -- that way the UI proves the API works, which is the thing you
actually want to demo.
"""
import os

import requests
import streamlit as st

API = os.getenv("PAPERRAG_API", "http://127.0.0.1:8000")

st.set_page_config(page_title="PaperRAG", layout="wide")
st.title("PaperRAG")
st.caption("Answers grounded in indexed papers, with page-level citations. "
           "Refuses to answer when retrieval is too weak.")

with st.sidebar:
    st.subheader("Retrieval settings")
    try:
        h = requests.get(f"{API}/health", timeout=5).json()
    except Exception:
        h = None
    # Start the slider at the server's threshold (chosen by the evaluation), not
    # at a number of our own: whatever the slider says is sent with every question.
    start = float(h.get("threshold", 0.50)) if h else 0.50
    top_k = st.slider("Top-k passages", 1, 15, 5)
    threshold = st.slider(
        "Abstention threshold (cosine)", -0.2, 0.9, start, 0.01,
        help="Below this top-1 similarity, the system refuses to answer.",
    )
    mode = st.radio("Generation", ["extractive", "mistral"], index=0)
    st.divider()
    if h:
        st.success(f"API up · {h['chunks']} chunks indexed")
    else:
        st.error("API unreachable. Start it with `uvicorn app.api:app`.")

question = st.text_input(
    "Question", placeholder="What visual prompt length does TMPT report as optimal?"
)

if st.button("Ask", type="primary") and question:
    try:
        r = requests.post(
            f"{API}/ask",
            json={"question": question, "top_k": top_k,
                  "threshold": threshold, "mode": mode},
            timeout=90,
        )
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        st.error(f"Request failed: {e}")
        st.stop()

    c1, c2, c3 = st.columns(3)
    c1.metric("Top similarity", f"{data['top_score']:.3f}")
    c2.metric("Threshold", f"{data['threshold']:.3f}")
    c3.metric("Latency", f"{data['latency_ms']:.0f} ms")

    if data["abstained"]:
        st.warning(data["answer"])
        st.caption("This is the guard working, not a bug. Lower the threshold "
                   "in the sidebar to see what it was holding back.")
    else:
        st.markdown(data["answer"])
        st.subheader("Sources")
        for c in data["citations"]:
            st.markdown(
                f"**[{c['n']}]** `{c['source']}` — page {c['page']} "
                f"· similarity {c['score']:.3f}"
            )
