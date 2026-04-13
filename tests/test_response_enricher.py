import unittest

from services.api.app.response_enricher import infer_recommended_actions


class TestResponseEnricher(unittest.TestCase):
    def test_timeout_incident_prefers_answer_specific_actions(self):
        query = (
            "How to fix: document-archival-service timing out with 504 timeout "
            "responses."
        )
        answer_text = (
            "To fix the document-archival-service timing out with 504 timeout "
            "responses, follow these steps:\n\n"
            "1. Confirm the affected service is the document-archival-service.\n"
            "2. Check for timeout errors (ReadTimeout, ConnectTimeout, 504 Gateway "
            "Timeout).\n"
            "3. Retry the archival request using an exponential backoff strategy, "
            "with a maximum of 3 attempts.\n"
            "4. Temporarily increase the downstream timeout for the archival path.\n"
            "5. Validate the health of the service and monitor the latency trend "
            "after each retry loop.\n"
            "6. Ensure that 504/timeout errors stop for new archival requests and "
            "that queue depth and latency return toward baseline.\n"
            "7. Restart the document-archival-service only if retries continue to fail."
        )

        actions = infer_recommended_actions(query, answer_text)

        self.assertEqual(
            actions,
            [
                "Check for timeout errors (ReadTimeout, ConnectTimeout, 504 Gateway "
                "Timeout).",
                "Retry the archival request using an exponential backoff strategy, "
                "with a maximum of 3 attempts.",
                "Temporarily increase the downstream timeout for the archival path.",
            ],
        )

    def test_latency_only_query_keeps_generic_latency_actions(self):
        actions = infer_recommended_actions(
            "How to investigate latency spikes?",
            "Monitor request latency and compare it with last deployment.",
        )

        self.assertEqual(
            actions,
            [
                "Check cache hit rate",
                "Inspect DB connection pool",
                "Review recent deployment changes",
            ],
        )


if __name__ == "__main__":
    unittest.main()
