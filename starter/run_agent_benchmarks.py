"""Run all Agent benchmark experiments for Baseline and Improved configurations."""
import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any

from agentic import run_agent
from retrievers import TfidfRetriever, DenseRetriever, HybridRetriever

ROOT = Path(__file__).resolve().parents[1]
CHUNKS_PATH = ROOT / "data" / "chunks.json"
QUESTIONS_PATH = ROOT / "data" / "questions.json"
CASES_PATH = ROOT / "data" / "cases.json"
EMBEDDINGS_PATH = ROOT / "data" / "chunks_embeddings.json"

AGY_EXE = r"C:\Users\Admin\.gemini\bin\agy.exe"

class RealGeminiController:
    def __init__(self, model: str = "gemini-3.8-flash-low"):
        self.model = model

    def decide(self, question: str, history: list[dict[str, Any]]) -> dict[str, Any]:
        if not history:
            return {"action": "search", "query": question}

        ctx = history[0].get("context", "")
        ids = history[0].get("context_ids", [])
        num_searches = len(history[0].get("queries", []))

        prompt = (
            "Bạn là trợ lý AI trả lời câu hỏi dựa trên các đoạn trích từ Đề án mở ngành AI của PTIT.\n"
            f"Câu hỏi: \"{question}\"\n\n"
            f"Context hiện có:\n{ctx}\n\n"
            "QUY TẮC BẮT BUỘC:\n"
            "1. Nếu trong Context CÓ THÔNG TIN để trả lời câu hỏi:\n"
            "   - Trả lời đúng trọng tâm câu hỏi.\n"
            "   - BẮT BUỘC viết mã trích dẫn dạng [chunk_id] vào trong câu trả lời (ví dụ: 'Năm 1997 [de_an_ai_ptit_2024_p4_para001]').\n"
            "   - Điền đúng các mã chunk_id đó vào mảng citations.\n"
            '   - JSON mẫu: {"action": "answer", "answer": "Học viện thành lập năm 1997 [de_an_ai_ptit_2024_p4_para001].", "citations": ["de_an_ai_ptit_2024_p4_para001"]}\n'
            "2. Nếu trong Context KHÔNG CÓ THÔNG TIN để trả lời câu hỏi:\n"
        )
        if num_searches < 2:
            prompt += (
                "   - Có thể tìm kiếm lại 1 lần nữa với từ khóa ngắn gọn hơn (3-5 từ cốt lõi).\n"
                '   - JSON mẫu: {"action": "search", "query": "từ khóa ngắn gọn"}\n'
                "   - Hoặc nếu câu hỏi hỏi điều hoàn toàn không liên quan, từ chối: "
                '{"action": "answer", "answer": "Không đủ bằng chứng để trả lời.", "citations": []}\n'
            )
        else:
            prompt += (
                "   - Không tìm thấy thông tin trong tài liệu, từ chối trả lời:\n"
                '   - JSON mẫu: {"action": "answer", "answer": "Không đủ bằng chứng để trả lời.", "citations": []}\n'
            )
        prompt += "\nChỉ trả về DUY NHẤT một chuỗi JSON hợp lệ (không kèm markdown):"

        res = subprocess.run(
            [AGY_EXE, "-p", prompt, "--model", self.model],
            capture_output=True,
            text=True,
            encoding="utf-8"
        )
        out = res.stdout.strip()
        out = re.sub(r"^```(?:json)?\s*", "", out, flags=re.MULTILINE)
        out = re.sub(r"\s*```$", "", out, flags=re.MULTILINE).strip()

        try:
            return json.loads(out)
        except Exception:
            return {"action": "answer", "answer": "Không đủ bằng chứng để trả lời.", "citations": []}


def run_benchmark(cases, q_map, controller, retriever, top_k, budget, strategy_name, split):
    selected_cases = [c for c in cases if c.get("split") == split]
    print(f"\n>>> Running Agent [{strategy_name}] on split '{split}' ({len(selected_cases)} cases)...")
    traces = []
    
    for idx, c in enumerate(selected_cases, start=1):
        qid = c["id"]
        q_text = c["question"]
        q_obj = q_map.get(qid, {})
        filters = q_obj.get("filter")
        
        t0 = time.perf_counter()
        result = run_agent(
            question=q_text,
            controller=controller,
            retriever=retriever,
            filters=filters,
            max_searches=2,
            top_k=top_k,
            budget=budget
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        
        trace = {
            "query_id": qid,
            "split": split,
            "question": q_text,
            "answerable": c["answerable"],
            "strategy": strategy_name,
            "top_k": top_k,
            "budget": budget,
            "filter_used": filters,
            "status": result["status"],
            "answer": result.get("answer", ""),
            "citations": result.get("citations", []),
            "events": result.get("events", []),
            "elapsed_ms": elapsed_ms
        }
        traces.append(trace)
        print(f"  [{idx}/{len(selected_cases)}] {qid}: {trace['status']} | Citations: {trace['citations']} | {elapsed_ms:.1f}ms")
        
    return traces


def main():
    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    q_map = {q["id"]: q for q in questions}

    print("Initializing Retrievers and Agent Controller...")
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    
    tfidf_retriever = TfidfRetriever(chunks)
    dense_retriever = DenseRetriever(chunks, EMBEDDINGS_PATH, embedder=embedder)
    hybrid_retriever = HybridRetriever(tfidf_retriever, dense_retriever, rrf_k=60)
    controller = RealGeminiController(model="gemini-3.8-flash-low")

    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Dev: Baseline Agent (TFIDF, top_3, budget 800)
    dev_baseline_traces = run_benchmark(cases, q_map, controller, tfidf_retriever, top_k=3, budget=800, strategy_name="tfidf_baseline", split="dev")
    (results_dir / "agent_dev_baseline_traces.json").write_text(json.dumps(dev_baseline_traces, ensure_ascii=False, indent=2), encoding="utf-8")

    # 2. Dev: Improved Agent (Hybrid RRF, top_5, budget 1500)
    dev_improved_traces = run_benchmark(cases, q_map, controller, hybrid_retriever, top_k=5, budget=1500, strategy_name="hybrid_improved", split="dev")
    (results_dir / "agent_dev_improved_traces.json").write_text(json.dumps(dev_improved_traces, ensure_ascii=False, indent=2), encoding="utf-8")

    # 3. Test: Baseline Agent (TFIDF, top_3, budget 800)
    test_baseline_traces = run_benchmark(cases, q_map, controller, tfidf_retriever, top_k=3, budget=800, strategy_name="tfidf_baseline", split="test")
    (results_dir / "agent_test_baseline_traces.json").write_text(json.dumps(test_baseline_traces, ensure_ascii=False, indent=2), encoding="utf-8")

    # 4. Test: Improved Agent (Hybrid RRF, top_5, budget 1500)
    test_improved_traces = run_benchmark(cases, q_map, controller, hybrid_retriever, top_k=5, budget=1500, strategy_name="hybrid_improved", split="test")
    (results_dir / "agent_test_improved_traces.json").write_text(json.dumps(test_improved_traces, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nAll 4 benchmark runs completed successfully!")

if __name__ == "__main__":
    main()
