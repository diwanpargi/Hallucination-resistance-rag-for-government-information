 
 
import os
import streamlit as st

st.set_page_config(page_title="Govt Scholarship Assistant", page_icon="🎓")
st.title("🎓 Government Scholarship Assistant")
st.write("Type your question below. The answer comes only from the PDF documents. If it is not there, the app will say so.")

#  Step 1: keys 
 
KEY_NAMES = ["QDRANT_URL", "QDRANT_API_KEY", "GEMINI_API_KEY"]

with st.sidebar:
    st.header("Keys")
    for name in KEY_NAMES:
        if not os.getenv(name):
            value = st.text_input(name, type="password")
            if value:
                os.environ[name] = value

if not all(os.getenv(n) for n in KEY_NAMES):
    st.info("Please enter all the keys in the sidebar first (press Enter after typing each one).")
    st.stop()
 
import config
from embedding_store import load_embed_model, get_qdrant_client
from generation import get_gemini_client, NOT_FOUND
from rag_pipeline import answer_question


# tep 2: load model, Qdrant and Gemini only once  
@st.cache_resource
def load_everything():
    model = load_embed_model()
    client = get_qdrant_client()
    gemini = get_gemini_client()
    return model, client, gemini


with st.spinner("Loading models (the first time takes a while)..."):
    model, client, gemini = load_everything()

#  Step 3: settings 
with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Number of chunks to fetch", 1, 10, config.TOP_K)
    threshold = st.slider("Minimum score", 0.0, 1.0, config.SCORE_THRESHOLD, 0.05)

#   Step 4: question and answer  
question = st.text_input("Your question")

if st.button("Get answer") and question.strip():

    # 1) retrieve, check the context, generate and validate the answer
    with st.spinner("Searching, generating and checking the answer..."):
        out = answer_question(client, model, gemini, question, top_k=top_k, threshold=threshold)
    answer, hits = out["answer"], out["hits"]
    st.caption(f"Best match score: {out['top_score']:.2f}  |  Chunks used: {len(hits)}")

    st.subheader("Answer")
    if answer.strip() == NOT_FOUND:
        st.warning(answer)
    else:
        st.success(answer)

    if out["reason"]:
        st.caption("Reason: " + out["reason"])

    # 2) show the sources
    if hits:
        st.subheader("Sources")
        for i, p in enumerate(hits, start=1):
            d = p.payload
            title = f"{i}. {d.get('source')} | page {d.get('page_number')} | score {p.score:.2f}"
            with st.expander(title):
                st.write(d.get("text"))
