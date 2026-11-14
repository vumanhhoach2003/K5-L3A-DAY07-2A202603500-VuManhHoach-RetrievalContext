"""Run Agent on benchmark questions using real Gemini model and real retriever."""
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
CASES_PATH = ROOT / "data" / "cases.json"
EMBEDDINGS_PATH = ROOT / "data" / "chunks_embeddings.json"

AGY_EXE = r"C:\Users\Admin\.gemini\bin\agy.exe"

class RealGeminiController:
    def __init__(self, model: str = "gemini-3.8-flash-low"):
        self.model = model

    def decide(self, question: str, history: list[dict[str, Any]]) -> dict[str, Any]:
        if not history:
            # Turn 1: Search directly with the user's question
            return {"action": "search", "query": question}

        # History has results
        ctx = history[0].get("context", "")
        ids = history[0].get("context_ids", [])
        num_searches = len(history[0].get("queries", []))

        prompt = (
            "Bạn là trợ lý AI trả lời câu hỏi dựa trên các đoạn văn bản (context) trích xuất từ Đề án mở ngành AI của PTIT.\n"
            f"Câu hỏi: \"{question}\"\n\n"
            f"Context các đoạn văn bản hiện có:\n{ctx}\n\n"
            "QUY TẮC:\n"
            "1. Nếu trong Context CÓ ĐỦ BẰNG CHỨNG để trả lời câu hỏi:\n"
            "   - Trả lời ngắn gọn, chính xác.\n"
            "   - BẮT BUỘC chèn đúng mã [chunk_id] vào câu trả lời ngay sau mệnh đề chứa thông tin đó.\n"
            "   - Liệt kê các chunk_id đó vào mảng 'citations'.\n"
            '   - Định dạng: {"action": "answer", "answer": "<câu trả lời kèm [chunk_id]>", "citations": ["chunk_id"]}\n'
            "2. Nếu trong Context CHƯA CÓ BẰNG CHỨNG:\n"
        )
        if num_searches < 2:
            prompt += (
                "   - Bạn có thể thực hiện thêm 1 lần tìm kiếm với từ khóa khác cụ thể hơn.\n"
                '   - Định dạng: {"action": "search", "query": "<từ khóa tìm kiếm mới>"}\n'
                "   - Hoặc nếu câu hỏi chắc chắn không liên quan đến đề án, từ chối: "
                '{"action": "answer", "answer": "Không đủ bằng chứng để trả lời.", "citations": []}\n'
            )
        else:
            prompt += (
                "   - Đã hết lượt tìm kiếm, hãy từ chối trả lời:\n"
                '   - Định dạng: {"action": "answer", "answer": "Không đủ bằng chứng để trả lời.", "citations": []}\n'
            )
        prompt += "\nTrả về DUY NHẤT một JSON hợp lệ (không kèm markdown):"

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
            action = json.loads(out)
            return action
        except Exception:
            return {"action": "answer", "answer": "Không đủ bằng chứng để trả lời.", "citations": []}


def run_benchmark_for_split(split: str, controller, retriever, top_k: int = 5, budget: int = 1500):
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    selected_cases = [c for c in cases if c.get("split") == split]
    
    print(f"\n--- Running Agent for split '{split}' ({len(selected_cases)} cases) ---")
    traces = []
    
    for idx, c in enumerate(selected_cases, start=1):
        qid = c["id"]
        q_text = c["question"]
        print(f"[{idx}/{len(selected_cases)}] {qid}: {q_text[:60]}...")
        
        t0 = time.perf_counter()
        result = run_agent(
            question=q_text,
            controller=controller,
            retriever=retriever,
            filters=None,
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
            "status": result["status"],
            "answer": result.get("answer", ""),
            "citations": result.get("citations", []),
            "events": result.get("events", []),
            "elapsed_ms": elapsed_ms
        }
        traces.append(trace)
        print(f"   -> Status: {trace['status']} | Citations: {trace['citations']} | Time: {elapsed_ms:.1f}ms")
        
    return traces


def main():
    print("Loading chunks and building retrievers...")
    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    
    tfidf_retriever = TfidfRetriever(chunks)
    dense_retriever = DenseRetriever(chunks, EMBEDDINGS_PATH, embedder=embedder)
    hybrid_retriever = HybridRetriever(tfidf_retriever, dense_retriever, rrf_k=60)
    
    controller = RealGeminiController(model="gemini-3.8-flash-low")
    
    # 1. Run dev (17 cases: 15 answerable, 2 unanswerable)
    dev_traces = run_benchmark_for_split("dev", controller, hybrid_retriever, top_k=5, budget=1500)
    dev_trace_path = ROOT / "results" / "agent_dev_traces.json"
    dev_trace_path.write_text(json.dumps(dev_traces, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved Dev Traces to: {dev_trace_path}")
    
    # 2. Run test (7 cases: 5 answerable, 2 unanswerable)
    test_traces = run_benchmark_for_split("test", controller, hybrid_retriever, top_k=5, budget=1500)
    test_trace_path = ROOT / "results" / "agent_test_traces.json"
    test_trace_path.write_text(json.dumps(test_traces, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved Test Traces to: {test_trace_path}")

if __name__ == "__main__":
    main()
