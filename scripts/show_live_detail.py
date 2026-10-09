import json

d = json.load(open("evaluation/rag_live_report.json"))
for r in d["results"]:
    print(f"{r['id']:12} status={r['status']:22} ans_ok={str(r['ans_ok']):5} "
          f"provider={str(r.get('answer_provider')):12} rank={r['rank']}")
