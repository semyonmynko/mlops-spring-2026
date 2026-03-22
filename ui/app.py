import os
import time
import requests
import streamlit as st

st.set_page_config(page_title="Sentiment Classifier", page_icon="💬", layout="centered")

with st.sidebar:
    st.header("Settings")
    url = st.text_input(
        "Endpoint URL",
        value=os.getenv("SERVING_URL", "http://localhost:8082/serve/sentiment"),
    )
    timeout = st.slider("Timeout (s)", 1, 30, 10)
    st.caption("Model is hosted via ClearML Serving and loaded from Model Registry.")

st.title("💬 Sentiment Classifier")
st.write("Enter a movie review and click **Predict**.")

text = st.text_area("Text", placeholder="Paste your review here...", height=150)
btn = st.button("Predict", type="primary", use_container_width=True)

if btn:
    if not text.strip():
        st.warning("Enter some text first.")
    elif not url.strip():
        st.error("Set the endpoint URL in the sidebar.")
    else:
        with st.spinner("Sending request..."):
            t0 = time.perf_counter()
            try:
                resp = requests.post(url, json={"text": text}, timeout=timeout)
                latency = (time.perf_counter() - t0) * 1000
                resp.raise_for_status()
                result = resp.json()
            except requests.exceptions.ConnectionError:
                st.error("Can't reach the endpoint. Is clearml-serving running?")
                st.stop()
            except requests.exceptions.Timeout:
                st.error(f"Timed out after {timeout}s.")
                st.stop()
            except Exception as e:
                st.error(str(e))
                st.stop()

        label = result.get("label", "?")
        is_pos = result.get("label_id") == 1

        st.markdown("---")
        col1, col2 = st.columns(2)
        col1.metric("Sentiment", "✅ Positive" if is_pos else "❌ Negative")
        col2.metric("Latency", f"{latency:.1f} ms")

        color = "green" if is_pos else "red"
        st.markdown(
            f"<p style='color:{color};font-size:18px'><b>{label.upper()}</b></p>",
            unsafe_allow_html=True,
        )
        with st.expander("Raw response"):
            st.json(result)

st.markdown("---")
st.caption("MLOps Spring 2026 · ClearML · TF-IDF + LogReg")
