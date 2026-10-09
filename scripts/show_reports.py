import json

for name, path in (("OFFLINE", "evaluation/rag_report.json"),
                   ("LIVE", "evaluation/rag_live_report.json")):
    d = json.load(open(path))
    print(f"== {name} ==")
    for k in ("label", "provider", "embedding_model", "embedding_dim",
              "database", "corpus", "test_set", "answer_providers_seen",
              "native_paths", "recall_at_k", "mrr", "citation_support",
              "unsupported_claim_rate", "answer_correctness", "faithfulness",
              "abstention_correctness", "filter_behavior",
              "financial_query_correctness", "validation_path_correctness",
              "latency_ms"):
        if k in d:
            print(f"  {k}: {d[k]}")
    print()
