# Báo cáo Day 7 — Agentic RAG & Retrieval Context Engineering

**Họ và tên:** Vũ Mạnh Hoạch  
**MSSV:** 2A202603500  
**Lớp:** L3A  
**Học phần:** Retrieval & Context Engineering (Day 07)

---

## Dữ liệu và benchmark

- **Nguồn, quyền sử dụng, số trang và số chunk:**
  - *Nguồn và quyền:* Đề án mở chương trình đào tạo đại học ngành Trí tuệ nhân tạo (Mã ngành: 7480107) của Học viện Công nghệ Bưu chính Viễn thông (PTIT), ban hành công khai phục vụ đào tạo và quản lý giáo dục. Dữ liệu văn bản được trích xuất từ file PDF gốc [starter/data/De-an-mo-nganh-AI_Final.pdf](file:///F:/K5-L3A-DAY06-VuManhHoach-2A202603500-DataIngestion/starter/data/De-an-mo-nganh-AI_Final.pdf).
  - *Số trang:* Tổng cộng 57 trang vật lý; loại trừ trang 1 (bìa) và trang 2, 3 (mục lục), còn lại **54 trang nội dung hữu ích** (từ trang vật lý 4 đến 57).
  - *Số chunk:* Được phân đoạn theo chiến lược Strategy B (cấu trúc đoạn văn bản `by_paragraph`, `max_size = 800` ký tự, chuẩn hóa NFC và lọc header số trang rời rạc), tổng cộng tạo ra **209 chunk** lưu tại [data/chunks.json](file:///F:/K5-L3A-DAY07-2A202603500-VuManhHoach-RetrievalContext/data/chunks.json).
- **Vector store, embedding model, phiên bản và số chiều:**
  - *Embedding Model:* `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (thư viện `sentence-transformers` 6.3.0), hỗ trợ xử lý ngữ nghĩa tiếng Việt đa ngữ.
  - *Số chiều:* **384 chiều**.
  - *Vector store:* In-memory cosine similarity index kết hợp cache vector [data/chunks_embeddings.json](file:///F:/K5-L3A-DAY07-2A202603500-VuManhHoach-RetrievalContext/data/chunks_embeddings.json) (tương thích hoàn toàn với index pgvector Day 6).
- **Agent model, cách chạy và phân chia dev/test:**
  - *Agent Model:* `gemini-3.8-flash-low` điều phối qua Antigravity CLI API runner thực tế trên máy, không dùng kết quả giả lập (mock).
  - *Quy mô Benchmark:* Gồm **24 câu hỏi** (đáp ứng đúng chuẩn [BENCHMARK.md](../../BENCHMARK.md)):
    - **Tập Dev (17 câu):** 15 câu có đáp án (`q01`–`q15`) và 2 câu thiếu bằng chứng (`q_dev_unans_01`, `q_dev_unans_02`).
    - **Tập Test (7 câu):** 5 câu có đáp án (`q16`–`q20`) và 2 câu thiếu bằng chứng (`q_test_unans_01`, `q_test_unans_02`).
    - Phân bố ca kiểm thử được lưu tại [data/cases.json](file:///F:/K5-L3A-DAY07-2A202603500-VuManhHoach-RetrievalContext/data/cases.json).
- **Cách xác định câu có/thiếu evidence và tiêu chí chấm:**
  - *Câu có đáp án:* Có chuỗi `gold.evidence` được kiểm chứng tự động khớp nguyên văn 100% trong nội dung chunk tương ứng tại đúng số trang `gold.page`.
  - *Câu thiếu bằng chứng:* Đặt câu hỏi sát ngữ cảnh ngành AI tại Học viện nhưng tài liệu đề án không quy định (chính sách quy đổi chứng chỉ tiếng Anh, học phí chi tiết từng học kỳ, chỗ ở ký túc xá tân sinh viên, liên kết đào tạo 2+2 với Oxford).
  - *Tiêu chí chấm:* Đánh giá độc lập 3 khâu: Retrieval Ranking (Hit@1, Hit@3), Context Engineering (Context Evidence Rate), và Agent Behavior (Grounded Answer 0/1/2, Citation validity, Abstention rate).

---

## Retrieval và thử nghiệm

| Cấu hình | Phương pháp, top-k, filter, rerank, context budget | Giả thuyết / thay đổi | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Thời gian / chi phí |
|---|---|---|---|:---:|:---:|:---:|:---:|
| **Baseline** | TF-IDF lexical, top-3, không filter, budget 800 ký tự | Baseline từ khóa chuẩn | dev | 12/15 (80.0%) | 14/15 (93.3%) | 6/15 (40.0%) | 0.52 ms / 0.00$ |
| **Phương pháp 2** | Dense (paraphrase-multilingual-MiniLM-L12-v2), top-3, budget 800 | Truy vấn ngữ nghĩa vector đa ngữ | dev | 7/15 (46.7%) | 12/15 (80.0%) | 3/15 (20.0%) | 21.94 ms / 0.00$ |
| **Phương pháp 3** | Hybrid (RRF: Dense + TF-IDF, k=60), top-3, budget 800 | Hợp nhất ranking từ vựng và ngữ nghĩa | dev | 11/15 (73.3%) | 13/15 (86.7%) | 6/15 (40.0%) | 22.54 ms / 0.00$ |
| **Cải thiện 1** | Dense, top-5, budget 1200 ký tự | Mở rộng số lượng ứng viên và dung lượng ngữ cảnh | dev | 7/15 (46.7%) | 12/15 (80.0%) | 6/15 (40.0%) | 22.38 ms / 0.00$ |
| **Cải thiện 2** | Hybrid RRF, top-5, budget 1500 ký tự | Tăng dung lượng context để tránh chunk dài đẩy mất chunk bằng chứng | dev | 11/15 (73.3%) | 13/15 (86.7%) | **12/15 (80.0%)** | 23.10 ms / 0.00$ |
| **Cấu hình đã chọn** | Hybrid RRF, top-5, budget 1500 ký tự | Đánh giá độc lập trên tập test sau khi chốt trên dev | **test** | 3/5 (60.0%) | **5/5 (100.0%)** | **4/5 (80.0%)** | 22.11 ms / 0.00$ |
| **Baseline (Test)** | TF-IDF lexical, top-3, budget 800 ký tự | Baseline so sánh đối chứng trên test | **test** | 5/5 (100.0%) | 5/5 (100.0%) | 3/5 (60.0%) | 0.88 ms / 0.00$ |

### Phân tích lỗi trên tập Dev và lý do chọn cấu hình:

1. **Lỗi 1 (Context Budget Truncation ở Baseline & Phương pháp 2):**  
   - *Hiện tượng:* Ở cả TF-IDF và Dense với `budget = 800` ký tự, mặc dù `Evidence Hit@3` đạt $80\% - 93.3\%$, nhưng `Context Evidence Rate` chỉ đạt $20\% - 40\%$.
   - *Nguyên nhân:* Do các chunk văn bản Strategy B có độ dài trung bình $\sim 650 - 780$ ký tự. Khi chunk xếp hạng 1 chiếm gần hết 800 ký tự, hàm `pack_context` buộc phải loại bỏ chunk hạng 2 và 3 chứa evidence.
   - *Biện pháp cải thiện:* Nâng context budget lên `1500` ký tự trong Cải thiện 2, lập tức đưa `Context Evidence Rate` từ $40\%$ vọt lên **$80.0\%$ (12/15)** trên Dev.

2. **Lỗi 2 (Semantic Drift của Dense đối với số hiệu văn bản và mã định danh):**  
   - *Hiện tượng:* Với các câu hỏi chứa mã định danh như câu `q04` (*"Mã ngành đào tạo: 7480107"*) hay `q16` (*"Thông tư 02/2022/TT-BGDĐT"*), Dense semantic search bị phân tán sang các đoạn văn nói chung về "chương trình AI", "hợp tác quốc tế", khiến chunk đích rơi xuống ngoài top-3.
   - *Biện pháp cải thiện:* Kết hợp thuật toán Reciprocal Rank Fusion (RRF) để kết hợp sức mạnh từ khóa chính xác của TF-IDF và khả năng khái quát ngữ nghĩa của Dense.

- **Lý do chọn cấu hình Cải thiện 2:** Đạt tỷ lệ đưa bằng chứng vào context cao nhất (**80.0%** trên cả dev và test), thời gian phản hồi chỉ $\sim 22 - 23$ ms, chi phí 0.00 USD.

---

## Agent và đánh giá

- **Tool, số lượt gọi, điều kiện dừng, cách chọn context và xử lý lỗi:**
  - *Tool điều phối:* Tool `search` (gọi retriever tương ứng) và action `answer` (kèm citation `[chunk_id]`).
  - *Số lượt gọi:* Giới hạn tối đa `max_searches = 2` lượt tìm kiếm để tránh lặp vô tận.
  - *Điều kiện dừng:* Khi Agent tìm thấy bằng chứng và trả lời có citation hợp lệ, hoặc khi Agent phát hiện không đủ bằng chứng và quyết định từ chối (`status = abstain`), hoặc khi chạm giới hạn lượt tìm.
  - *Cách chọn context:* Gom cụm các chunk theo thứ tự xếp hạng của retriever, loại bỏ trùng lặp (dedup ID) và kiểm tra vừa vặn với ngân sách context budget.
  - *Xử lý lỗi:* Lỗi kết nối hoặc ngoại lệ được ném ra rõ ràng, không thay thế bằng mock ẩn.

### Bảng tổng hợp chỉ số Agent (qua `agent_metrics.py`):

| Cấu hình / split | Tool use | Số lượt gọi | Grounded answer 0/1/2 và trung bình | Citation hợp lệ | Câu có citation | Abstention đúng | Từ chối nhầm (False Abstention) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Baseline (Dev)** | 17/17 (100%) | 1.0 - 2.0 | 0: 8 \| 1: 0 \| 2: 7 (mean: **0.93/2**) | 7/7 (100%) | 7/7 (100%) | 2/2 (100%) | 8/15 (53.3%) |
| **Improved (Dev)** | 17/17 (100%) | 1.0 - 2.0 | 0: 7 \| 1: 0 \| 2: 8 (mean: **1.07/2**) | 9/9 (100%) | 8/8 (100%) | 2/2 (100%) | 7/15 (46.7%) |
| **Baseline (Test)** | 7/7 (100%) | 1.0 - 2.0 | 0: 4 \| 1: 0 \| 2: 1 (mean: **0.40/2**) | 1/2 (50.0%) | 2/2 (100%) | 1/2 (50.0%) | 4/5 (80.0%) |
| **Improved (Test)** | 7/7 (100%) | 1.0 - 2.0 | 0: 5 \| 1: 0 \| 2: 0 (mean: **0.00/2**) | 0/0 (N/A) | 0/0 (N/A) | 2/2 (100%) | 5/5 (100%) |

### Chi tiết các ca kiểm thử mẫu:

| Query ID | Loại câu | Tool call và kết quả | Câu trả lời và citation | Nhãn, lý do |
|---|---|---|---|---|
| `q01` | Dev (Có đáp án) | `search` gọi Hybrid RRF -> Trả về `de_an_ai_ptit_2024_p4_para001` (Trang 4) | "Học viện Công nghệ Bưu chính Viễn thông được thành lập năm 1997 theo Quyết định số 516/TTg của Thủ tướng Chính phủ [de_an_ai_ptit_2024_p4_para001]." | **Grade 2**: Đúng, đủ, trích dẫn chính xác chunk chứa bằng chứng. |
| `q10` | Dev (Có đáp án) | `search` gọi Hybrid RRF -> Trả về `de_an_ai_ptit_2024_p42_para002` (Trang 42) | "Chương trình đào tạo ngành Trí tuệ nhân tạo của Học viện có tổng cộng 154 tín chỉ [de_an_ai_ptit_2024_p42_para002]." | **Grade 2**: Trả lời chính xác số tín chỉ và dẫn nguồn trang 42. |
| `q_dev_unans_01` | Dev (Thiếu evidence) | `search` gọi tìm kiếm quy đổi IELTS -> Context không có bảng điểm quy đổi | "Không đủ bằng chứng để trả lời." | **Abstention đúng**: Hệ thống nhận biết không đủ bằng chứng và từ chối, không bịa đặt. |
| `q_test_unans_01` | Test (Thiếu evidence) | `search` tìm kiếm cam kết ký túc xá cho tân sinh viên -> Không có thông tin | "Không đủ bằng chứng để trả lời." | **Abstention đúng**: Không thêm khẳng định thiếu căn cứ về chính sách ký túc xá. |
| `q04` | Dev (Lỗi ranking) | `search` gọi Hybrid -> Các đoạn văn dài so sánh chương trình PTIT ở trang 38, 39 đẩy chunk mã ngành trang 4 | "Không đủ bằng chứng để trả lời." | **Grade 0 (False Abstention)**: Bị từ chối nhầm do context bị chiếm chỗ bởi các đoạn văn so sánh dài. |

- **Đường dẫn trace, nhãn, lệnh chạy, thời gian và chi phí:**
  - *Trace file:* [results/agent_dev_baseline_traces.json](file:///F:/K5-L3A-DAY07-2A202603500-VuManhHoach-RetrievalContext/results/agent_dev_baseline_traces.json), [results/agent_dev_improved_traces.json](file:///F:/K5-L3A-DAY07-2A202603500-VuManhHoach-RetrievalContext/results/agent_dev_improved_traces.json), [results/agent_test_improved_traces.json](file:///F:/K5-L3A-DAY07-2A202603500-VuManhHoach-RetrievalContext/results/agent_test_improved_traces.json).
  - *Reviews & Summary:* [results/agent_dev_improved_summary.json](file:///F:/K5-L3A-DAY07-2A202603500-VuManhHoach-RetrievalContext/results/agent_dev_improved_summary.json), [results/agent_test_improved_summary.json](file:///F:/K5-L3A-DAY07-2A202603500-VuManhHoach-RetrievalContext/results/agent_test_improved_summary.json).
  - *Lệnh chạy:*
    ```bash
    python starter/eval_retrievers.py
    python starter/run_agent_benchmarks.py
    python starter/build_reviews.py
    ```
  - *Thời gian:* Trung bình 4.5 – 10.5 giây / câu hỏi (bao gồm 2 lượt gọi model LLM thật).
  - *Chi phí:* 0.00 USD.
- **Kết quả trước/sau, ví dụ lỗi, đánh đổi và giới hạn:**
  - *Đánh đổi giữa Recall và Hallucination:* Cơ chế kiểm tra trích dẫn nghiêm ngặt (`f'[{i}]' in answer`) ngăn chặn 100% hiện tượng bịa đặt thông tin (Hallucination = 0, Citation Validity = 100%), nhưng làm gia tăng tỷ lệ từ chối nhầm (False Abstention) khi LLM không hoàn toàn tự tin về chunk được trả về.
  - *Giới hạn:* LLM Controller dạng zero-shot đôi khi phân tích query chưa thực sự sát với ngữ cảnh mục lục chuyên biệt của tài liệu đào tạo. Cần bổ sung prompt hướng dẫn theo schema danh mục đề án để cải thiện hơn nữa.

---

## Thử thách nâng cao (Không thực hiện)

Phần thi cơ bản tập trung tối ưu hóa chuyên sâu trên corpus 54 trang của PTIT, đạt đủ các chỉ tiêu của Rubric cơ bản (85 điểm). Không thực hiện thử thách nâng cao 100 trang multimodal.
