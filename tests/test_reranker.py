import unittest

from services.core.reranker import rerank


class DummyModel:
    def __init__(self, scores):
        self._scores = scores

    def predict(self, pairs):
        return self._scores


class TestReranker(unittest.TestCase):
    def test_rerank_rows_enabled_reorders(self):
        original_enabled = rerank.RERANKER_ENABLED
        original_get_model = rerank._get_model
        try:
            rerank.RERANKER_ENABLED = True
            rerank._get_model = lambda: DummyModel([0.2, 0.9, 0.1])
            rows = [
                {"id": 1, "text": "alpha"},
                {"id": 2, "text": "beta"},
                {"id": 3, "text": "gamma"},
            ]
            ranked = rerank.rerank_rows("query", rows, top_k=3)
            ranked_ids = [row["id"] for row in ranked]
            self.assertEqual(ranked_ids, [2, 1, 3])
        finally:
            rerank.RERANKER_ENABLED = original_enabled
            rerank._get_model = original_get_model

    def test_rerank_rows_disabled_returns_input(self):
        original_enabled = rerank.RERANKER_ENABLED
        try:
            rerank.RERANKER_ENABLED = False
            rows = [
                {"id": 1, "text": "alpha"},
                {"id": 2, "text": "beta"},
            ]
            ranked = rerank.rerank_rows("query", rows, top_k=2)
            self.assertEqual(ranked, rows)
        finally:
            rerank.RERANKER_ENABLED = original_enabled


if __name__ == "__main__":
    unittest.main()
