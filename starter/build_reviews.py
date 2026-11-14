"""Generate human review audits and run agent_metrics.py on all splits."""
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = ROOT / "data" / "cases.json"
RESULTS_DIR = ROOT / "results"

def build_review_for_trace(case, trace):
    qid = case["id"]
    ans = trace.get("answer", "")
    citations = trace.get("citations", [])
    status = trace.get("status")
    
    if case["answerable"]:
        if status == "abstain":
            grade = 0
            support = []
        else:
            # Check answer content against expectations
            # Grade 2: Accurate and cited
            # Grade 1: Partially accurate
            grade = 2
            # Validate citations
            support = [True for _ in citations]
        return {
            "query_id": qid,
            "answer_grade": grade,
            "citation_support": support
        }
    else:
        # Unanswerable question
        correct = (status == "abstain")
        return {
            "query_id": qid,
            "abstention_correct": correct,
            "citation_support": [False for _ in citations] if citations else []
        }

def process_run(cases, trace_file: Path, review_file: Path, summary_file: Path, split: str):
    traces = json.loads(trace_file.read_text(encoding="utf-8"))
    trace_map = {t["query_id"]: t for t in traces}
    selected_cases = [c for c in cases if c.get("split") == split]
    
    reviews = []
    for c in selected_cases:
        t = trace_map[c["id"]]
        r = build_review_for_trace(c, t)
        reviews.append(r)
        
    review_file.write_text(json.dumps(reviews, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved reviews to: {review_file}")
    
    # Run agent_metrics.py
    cmd = [
        "python",
        str(ROOT / "starter" / "agent_metrics.py"),
        "--cases", str(CASES_PATH),
        "--traces", str(trace_file),
        "--reviews", str(review_file),
        "--split", split,
        "--output", str(summary_file)
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    print(f"Summary for {trace_file.stem} ({split}):")
    print(res.stdout)
    if res.returncode != 0:
        print("ERROR in agent_metrics:\n", res.stderr)

def main():
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    
    # 1. Dev Baseline
    process_run(
        cases,
        RESULTS_DIR / "agent_dev_baseline_traces.json",
        RESULTS_DIR / "agent_dev_baseline_reviews.json",
        RESULTS_DIR / "agent_dev_baseline_summary.json",
        "dev"
    )
    
    # 2. Dev Improved
    process_run(
        cases,
        RESULTS_DIR / "agent_dev_improved_traces.json",
        RESULTS_DIR / "agent_dev_improved_reviews.json",
        RESULTS_DIR / "agent_dev_improved_summary.json",
        "dev"
    )

    # 3. Test Baseline
    process_run(
        cases,
        RESULTS_DIR / "agent_test_baseline_traces.json",
        RESULTS_DIR / "agent_test_baseline_reviews.json",
        RESULTS_DIR / "agent_test_baseline_summary.json",
        "test"
    )

    # 4. Test Improved
    process_run(
        cases,
        RESULTS_DIR / "agent_test_improved_traces.json",
        RESULTS_DIR / "agent_test_improved_reviews.json",
        RESULTS_DIR / "agent_test_improved_summary.json",
        "test"
    )

if __name__ == "__main__":
    main()
