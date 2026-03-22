import re


def _clean(text):
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


class Preprocess:
    _labels = {0: "negative", 1: "positive"}

    def preprocess(self, body, state, collect_custom_statistics_fn=None):
        text = body.get("text", "")
        if not text or not text.strip():
            raise ValueError("'text' field is required and must be non-empty")
        return [_clean(text)]

    def postprocess(self, data, state, collect_custom_statistics_fn=None):
        try:
            pred = int(data[0])
        except (TypeError, IndexError):
            pred = int(data)
        return {"label": self._labels.get(pred, str(pred)), "label_id": pred}
