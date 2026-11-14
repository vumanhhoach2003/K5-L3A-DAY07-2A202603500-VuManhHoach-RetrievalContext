"""Comprehensive evaluation script for all retrieval methods and configurations."""
import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

from baseline import hit, pack_context, validate
from retrievers import TfidfRetriever, DenseRetriever, HybridRetriever

ROOT = Path(__file__).resolve().parents[1]
CHUNKS_PATH = ROOT / "data" / "chunks.json"
QUESTIONS_PATH = ROOT / "data" / "questions.json"
EMBEDDINGS_PATH = ROOT / "data" / "chunks_embeddings.json"

def evaluate_retriever(retriever, questions: list[dict[str, Any]], top_k: int = 3, budget: int = 800, strategy_name: str = "retriever"):
    traces = []
    for q in questions:
        start = time.perf_counter()
        filters = q.get('filter')
        results = retriever.search(q['question'], filters=filters, top_k=top_k)
        context, ids = pack_context(results, budget=budget)
        elapsed_ms = (time.perf_counter() - start) * 1000
        marked = [{**c, 'hit': hit(c, q['gold'])} for c in results]
        
        traces.append({
            'query_id': q['id'],
            'split': q['split'],
            'question': q['question'],
            'strategy': strategy_name,
            'top_k': top_k,
            'budget_chars': budget,
            'filter_used': filters,
            'gold': q['gold'],
            'results': marked,
            'context': context,
            'context_ids': ids,
            'context_chars': len(context),
            'elapsed_ms': elapsed_ms,
            'hit1': bool(marked and marked[0]['hit']),
            'hit3': any(c['hit'] for c in marked[:3]),
            'context_hit': any(c['hit'] and c['id'] in ids for c in marked)
        })
    return traces

def run_experiment(retriever, questions, split, top_k, budget, strategy_name, output_dir: Path):
    selected = [q for q in questions if q.get('split') == split]
    traces = evaluate_retriever(retriever, selected, top_k=top_k, budget=budget, strategy_name=strategy_name)
    
    n = len(traces)
    hit1 = sum(t['hit1'] for t in traces)
    hit3 = sum(t['hit3'] for t in traces)
    context_hit = sum(t['context_hit'] for t in traces)
    
    summary = {
        'strategy': strategy_name,
        'split': split,
        'top_k': top_k,
        'budget': budget,
        'n': n,
        'hit1': hit1,
        'hit1_rate': round(hit1 / n, 4) if n else 0,
        'hit3': hit3,
        'hit3_rate': round(hit3 / n, 4) if n else 0,
        'context_hit': context_hit,
        'context_hit_rate': round(context_hit / n, 4) if n else 0,
        'avg_context_chars': round(statistics.mean(t['context_chars'] for t in traces), 1) if traces else 0,
        'median_ms': round(statistics.median(t['elapsed_ms'] for t in traces), 2) if traces else 0,
        'service_cost_usd': 0.0
    }
    
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "traces.json").write_text(json.dumps(traces, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    
    print(f"[{strategy_name.upper()} | {split} | top_{top_k} | budget_{budget}]")
    print(f"  Hit@1: {hit1}/{n} ({summary['hit1_rate']*100:.1f}%) | Hit@3: {hit3}/{n} ({summary['hit3_rate']*100:.1f}%) | Context Hit: {context_hit}/{n} ({summary['context_hit_rate']*100:.1f}%) | Median: {summary['median_ms']} ms")
    return summary

def main():
    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    validate(chunks, questions)

    print("Initializing Retrievers...")
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    
    tfidf_retriever = TfidfRetriever(chunks)
    dense_retriever = DenseRetriever(chunks, EMBEDDINGS_PATH, embedder=embedder)
    hybrid_retriever = HybridRetriever(tfidf_retriever, dense_retriever, rrf_k=60)

    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    experiments = [
        # 1. Three distinct methods on dev (top_k=3, budget=800)
        {"retriever": tfidf_retriever, "split": "dev", "top_k": 3, "budget": 800, "name": "tfidf_baseline", "dir": results_dir / "dev_tfidf"},
        {"retriever": dense_retriever, "split": "dev", "top_k": 3, "budget": 800, "name": "dense_baseline", "dir": results_dir / "dev_dense"},
        {"retriever": hybrid_retriever, "split": "dev", "top_k": 3, "budget": 800, "name": "hybrid_baseline", "dir": results_dir / "dev_hybrid"},

        # 2. Improvement configurations on dev
        # Improvement 1: Dense with top-5 and enlarged context budget 1200
        {"retriever": dense_retriever, "split": "dev", "top_k": 5, "budget": 1200, "name": "dense_top5_b1200", "dir": results_dir / "dev_dense_improved"},
        # Improvement 2: Hybrid RRF with top-5 and enlarged context budget 1500
        {"retriever": hybrid_retriever, "split": "dev", "top_k": 5, "budget": 1500, "name": "hybrid_top5_b1500", "dir": results_dir / "dev_hybrid_improved"},

        # 3. Final Evaluation on TEST
        # Baseline on Test
        {"retriever": tfidf_retriever, "split": "test", "top_k": 3, "budget": 800, "name": "tfidf_baseline", "dir": results_dir / "test_tfidf"},
        # Chosen best config on Test (Hybrid top-5 budget 1500)
        {"retriever": hybrid_retriever, "split": "test", "top_k": 5, "budget": 1500, "name": "hybrid_top5_b1500", "dir": results_dir / "test_chosen"},
    ]

    all_summaries = []
    print("\n" + "="*70)
    print("RUNNING RETRIEVAL EXPERIMENTS")
    print("="*70)
    for exp in experiments:
        s = run_experiment(exp["retriever"], questions, exp["split"], exp["top_k"], exp["budget"], exp["name"], exp["dir"])
        all_summaries.append(s)

    (results_dir / "all_retrieval_summaries.json").write_text(json.dumps(all_summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nAll retrieval summaries saved to: {results_dir / 'all_retrieval_summaries.json'}")

if __name__ == "__main__":
    main()

