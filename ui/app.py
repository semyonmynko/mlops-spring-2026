"""
ЭТАП 5 — Streamlit UI for sentiment classification.

Sends HTTP requests to the ClearML Serving endpoint.
Does NOT load any model directly.

Usage:
    streamlit run ui/app.py

Set the endpoint URL via the sidebar or the env variable:
    SERVING_URL=http://localhost:8080/serve/sentiment
"""

import os
import time

import requests
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sentiment Classifier",
    page_icon="💬",
    layout="centered",
)

# ── Sidebar — endpoint configuration ─────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    default_url = os.getenv(
        "SERVING_URL", "http://localhost:8080/serve/sentiment"
    )
    endpoint_url = st.text_input("Serving endpoint URL", value=default_url)
    timeout_sec = st.slider("Request timeout (s)", min_value=1, max_value=30, value=10)
    st.markdown("---")
    st.caption(
        "The model is hosted by **ClearML Serving** and loaded from the "
        "Model Registry. This UI communicates with it over HTTP."
    )

# ── Main UI ───────────────────────────────────────────────────────────────────
st.title("💬 Sentiment Classifier")
st.write("Enter a movie review (or any text) and click **Predict** to classify it.")

text_input = st.text_area(
    label="Input text",
    placeholder="Type or paste your text here...",
    height=150,
)

predict_clicked = st.button("Predict", type="primary", use_container_width=True)

if predict_clicked:
    if not text_input.strip():
        st.warning("Please enter some text before predicting.")
    elif not endpoint_url.strip():
        st.error("Endpoint URL is empty — configure it in the sidebar.")
    else:
        with st.spinner("Sending request to serving endpoint..."):
            start = time.perf_counter()
            try:
                response = requests.post(
                    endpoint_url,
                    json={"text": text_input},
                    timeout=timeout_sec,
                )
                latency_ms = (time.perf_counter() - start) * 1000
                response.raise_for_status()
                result = response.json()

            except requests.exceptions.ConnectionError:
                st.error(
                    "Could not connect to the serving endpoint. "
                    "Make sure `clearml-serving` is running and the URL is correct."
                )
                st.stop()
            except requests.exceptions.Timeout:
                st.error(
                    f"Request timed out after {timeout_sec} s. "
                    "Try increasing the timeout in the sidebar."
                )
                st.stop()
            except requests.exceptions.HTTPError as exc:
                st.error(f"Endpoint returned an error: {exc}")
                st.stop()
            except Exception as exc:
                st.error(f"Unexpected error: {exc}")
                st.stop()

        # ── Display results ───────────────────────────────────────────────────
        label = result.get("label", "unknown")
        label_id = result.get("label_id", -1)

        is_positive = label_id == 1
        emoji = "✅ Positive" if is_positive else "❌ Negative"
        color = "green" if is_positive else "red"

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Sentiment", value=emoji)
        with col2:
            st.metric(label="Latency", value=f"{latency_ms:.1f} ms")

        st.markdown(
            f"<p style='color:{color}; font-size:18px; font-weight:bold;'>"
            f"Prediction: <strong>{label.upper()}</strong></p>",
            unsafe_allow_html=True,
        )

        with st.expander("Raw response"):
            st.json(result)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("MLOps Spring 2026 · ClearML · TF-IDF + Logistic Regression")
