"""
ЭТАП 4 — ClearML Serving preprocessing script.

This file is passed to `clearml-serving model add --preprocess serving/preprocess.py`.

Contract (clearml-serving):
  - preprocess(body, state, ...)  → model input  (what model.predict() receives)
  - postprocess(data, state, ...) → HTTP response (JSON-serialisable dict)

The sklearn pipeline is loaded automatically by the serving engine from the
Model Registry — do NOT load it manually here.
"""

from __future__ import annotations

import re
from typing import Any


def _clean(text: str) -> str:
    """Minimal text cleaning: strip HTML tags and collapse whitespace."""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


class Preprocess:
    """Stateless preprocessing/postprocessing for sentiment inference."""

    # label_map matches the IMDB dataset encoding used during training
    _label_map = {0: "negative", 1: "positive"}

    def preprocess(
        self,
        body: dict,
        state: dict,
        collect_custom_statistics_fn=None,
    ) -> Any:
        """
        Transform the raw HTTP request body into model input.

        Expected request body:
            {"text": "The movie was great!"}

        Returns a list with one cleaned string — the sklearn Pipeline
        receives it as X and calls tfidf.transform() then clf.predict().
        """
        raw_text: str = body.get("text", "")
        if not isinstance(raw_text, str) or not raw_text.strip():
            raise ValueError("Request body must contain a non-empty 'text' field.")
        return [_clean(raw_text)]

    def postprocess(
        self,
        data: Any,
        state: dict,
        collect_custom_statistics_fn=None,
    ) -> dict:
        """
        Transform the model's raw predict() output to an HTTP response dict.

        `data` is whatever sklearn pipeline.predict() returns, typically a
        numpy array like array([0]) or array([1]).
        """
        try:
            prediction = int(data[0])
        except (TypeError, IndexError, ValueError):
            prediction = int(data)

        label = self._label_map.get(prediction, str(prediction))
        return {
            "label": label,
            "label_id": prediction,
        }
