#!/usr/bin/env python3
"""
Test script to query the RAG system.
Run the API server first:
PYTHONPATH=. uvicorn services.api.app.main:app --reload --port 8000
"""

import requests
import sys

API_URL = "http://localhost:8000/query"


def test_query(query_text, top_k=3):
    """Send a query to the RAG API and print results."""
    print(f"\n{'='*80}")
    print(f"Query: {query_text}")
    print(f"{'='*80}\n")

    try:
        response = requests.post(
            API_URL, json={"queryText": query_text, "top_k": top_k}, timeout=30
        )
        response.raise_for_status()

        results = response.json()
        hits = results.get("hits", [])

        if not hits:
            print("No results found.")
            return

        for i, hit in enumerate(hits, 1):
            print(f"{i}. {hit['title']}")
            print(f"   Distance: {hit['distance']:.4f}")
            print()

    except requests.exceptions.ConnectionError:
        print("✗ Error: Could not connect to API. Is the server running?")
        print(
            "  Start it with: PYTHONPATH=. uvicorn services.api.app.main:app "
            "--reload --port 8000"
        )
        sys.exit(1)
    except requests.exceptions.RequestException as e:
        print(f"✗ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Test queries
    queries = [
        "How do I deploy the system?",
        "What is pgvector?",
        "How do I troubleshoot database connection issues?",
        "What are the API endpoints?",
        "Tell me about the technical architecture",
    ]

    for query in queries:
        test_query(query, top_k=3)

    print("\n" + "=" * 80)
    print("✓ All test queries completed")
    print("=" * 80)
