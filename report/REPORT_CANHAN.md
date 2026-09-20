# Báo Cáo Cá Nhân — Lab 7: Embedding & Vector Store

**Họ tên:** Đinh Tuấn Long
**Nhóm:** [Tên nhóm]
**Ngày:** 20/09

> **Nộp 1 bản / sinh viên.** Phần nhóm (lựa chọn tài liệu, thiết kế chiến lược, bộ câu hỏi đánh giá, demo) nộp chung 1 bản trong `REPORT_NHOM.md`. Chi tiết thang điểm: `docs/SCORING.md`.

**Tổng điểm phần cá nhân: 60** = Khởi động (5) + Hướng tiếp cận (10) + Hoàn thiện code (30) + Dự đoán độ tương tự (5) + Kết quả truy xuất của tôi (10).

---

## 1. Khởi động (Warm-up) — Cá nhân (5 điểm)

### Độ tương tự Cosine (Cosine Similarity) (Bài tập 1.1)

**Độ tương tự cosine cao (High cosine similarity) nghĩa là gì?**
> Là cách đo xem 2 vector có hướng giống nhau đến thế nào, so sánh góc giữa 2 vector. Với text embedding, similarity càng cao thì hai câu càng gần nhau về ngữ nghĩa, ngay cả khi dùng từ khác nhau.

**Ví dụ có độ tương tự CAO:**
- Câu A: Người mua có thể yêu cầu hoàn tiền trong vòng 15 ngày
- Câu B: Khách hàng được phép gửi yêu cầu refund trong thời hạn 15 ngày
- Tại sao tương đồng: Hai câu dùng từ khác nhau nhưng cùng nghĩa về refund

**Ví dụ có độ tương tự THẤP:**
- Câu A: Người mua có thể yêu cầu hoàn tiền trong vòng 15 ngày
- Câu B: Người bán phải đăng thông tin bảo hành trong mô tả sản phẩm
- Tại sao khác: Hai câu nói về hai đối tượng và hai nghiệp vụ khác nhau. Câu A liên quan đến quyền trả hàng/hoàn tiền của Người Mua, trong khi câu B liên quan đến trách nhiệm bảo hành của Người Bán.


**Tại sao độ tương tự cosine (cosine similarity) được ưu tiên hơn khoảng cách Euclid (Euclidean distance) cho text embeddings?**
> Cosine phù hợp với text embedding hơn Euclidean distance vì cosine tập trung vào hướng của vector, tức pattern/ngữ nghĩa, thay vì độ lớn tuyệt đối của vector

### Bài toán tính toán Chunking (Bài tập 1.2)

**Tài liệu 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> Công thức:
>
> `ceil((document_length - overlap) / (chunk_size - overlap))`
>
> `= ceil((10000 - 50) / (500 - 50))`
>
> `= ceil(9950 / 450)`
>
> `= ceil(22.11...)`
>
> `= 23 chunks`
**Nếu độ chồng chéo (overlap) tăng lên 100, số lượng chunk thay đổi thế nào? Tại sao muốn độ chồng chéo nhiều hơn?**
> Khi `overlap=100`:
>
> `ceil((10000 - 100) / (500 - 100))`
>
> `= ceil(9900 / 400)`
>
> `= 25 chunks`.
>
> Overlap lớn hơn giúp giữ thêm context ở ranh giới giữa hai chunk, giảm nguy cơ một thông tin quan trọng hoặc một câu bị tách giữa hai chunk.
> Đổi lại, số chunk tăng lên, dữ liệu bị trùng lặp nhiều hơn và chi phí embedding, lưu trữ và retrieval cũng tăng.
---

## 2. Hướng tiếp cận của tôi (My Approach) — Cá nhân (10 điểm)

Giải thích cách tiếp cận khi implement các phần chính trong gói `src`.

### Các hàm chia nhỏ (Chunking Functions)

#### `SentenceChunker.chunk` — hướng tiếp cận

> Tôi sử dụng regex `(?<=[.!?]) |(?<=\.)\n` để phát hiện ranh giới câu tại khoảng trắng hoặc xuống dòng nằm sau dấu kết thúc câu.
> Cách này giúp giữ lại dấu câu trong nội dung thay vì loại bỏ chúng.
> Sau khi tách, các câu được `strip()` và gom tối đa theo `max_sentences_per_chunk`.
> Với text rỗng, hàm trả về danh sách rỗng.
> Một hạn chế là regex đơn giản có thể chưa xử lý hoàn hảo một số trường hợp như chữ viết tắt hoặc số thập phân.

#### `RecursiveChunker.chunk` / `_split` — hướng tiếp cận

> RecursiveChunker thử các separator theo thứ tự ưu tiên từ cấu trúc lớn đến nhỏ, ví dụ paragraph, newline, câu và khoảng trắng.
> Mục tiêu là giữ cấu trúc tự nhiên của văn bản càng lâu càng tốt trước khi phải cắt ở mức ký tự.
>
> Nếu một phần vẫn dài hơn `chunk_size`, hàm tiếp tục gọi đệ quy bằng separator nhỏ hơn.
> Nếu các phần đã đủ nhỏ, chúng được merge lại đến gần giới hạn `chunk_size`.
>
> Base case gồm text rỗng, text đã nhỏ hơn `chunk_size`, hoặc trường hợp không còn separator phù hợp thì fallback sang cắt trực tiếp theo số ký tự.

---

### Chiến lược chunking cá nhân: `HeadingChunker`

> Đối với benchmark cá nhân, tôi sử dụng `HeadingChunker` với `chunk_size=800`.
> Chiến lược này ưu tiên tách tài liệu theo Markdown heading và các section policy dạng đánh số như `1.9.2.`, `2.7.1.`, v.v.
>
> Nếu một section dài hơn `chunk_size`, nội dung section được tiếp tục chia bằng `RecursiveChunker`, đồng thời heading của section được gắn lại vào các sub-chunk.
> Mục tiêu là giúp mỗi chunk vẫn giữ được context về section mà nó thuộc về.
>
> Cách này đặc biệt phù hợp với corpus chính sách Shopee vì tài liệu có nhiều section và subsection được đánh số rõ ràng.

---

### Lớp `EmbeddingStore`

#### `add_documents` + `search` — hướng tiếp cận

> Mỗi `Document` được chuyển thành một record gồm `id`, `content`, `metadata` và `embedding`.
> Chunking được thực hiện trước khi đưa dữ liệu vào store, vì vậy mỗi chunk được biểu diễn dưới dạng một `Document` riêng.
>
> Khi search, query được embed một lần.
> Sau đó hệ thống tính similarity giữa query embedding và embedding của từng record.
> Các kết quả được sắp xếp theo score giảm dần và lấy tối đa `top_k`.
>
> Trong benchmark semantic, embedding được normalize nên dot product giữa hai vector tương ứng với cosine similarity.

#### `search_with_filter` + `delete_document` — hướng tiếp cận

> `search_with_filter` lọc candidate theo metadata trước khi thực hiện similarity search.
> Một record chỉ được giữ lại nếu các key/value trong `metadata_filter` khớp với metadata của record.
>
> Lọc trước giúp tránh việc các chunk sai audience hoặc sai nhóm metadata chiếm vị trí trong `top_k`.
>
> `delete_document` xóa toàn bộ record có cùng `metadata["doc_id"]`.
> Vì một tài liệu gốc có thể tạo ra nhiều chunk, cách này đảm bảo toàn bộ chunk thuộc tài liệu đó được xóa cùng lúc.

---

### Tác tử `KnowledgeBaseAgent`

#### `answer` — hướng tiếp cận

> Agent trước tiên retrieve các chunk liên quan nhất từ `EmbeddingStore`.
> Sau đó các chunk được đưa vào context dưới dạng `[1]`, `[2]`, `[3]`, đồng thời kèm thông tin nguồn `doc_id`.
>
> Prompt yêu cầu LLM chỉ sử dụng context được cung cấp, không tự thêm thông tin không có trong nguồn và sử dụng citation theo số chunk.
>
> Nếu retrieval không trả về context phù hợp, agent trả về thông báo thiếu thông tin thay vì tạo ra câu trả lời không có căn cứ.

---

## 3. Hoàn thiện code (Core Implementation) — Cá nhân (30 điểm)

Vượt qua bộ kiểm thử là điều kiện tính điểm phần này.

### Kết Quả Kiểm Thử (Test Results)

```
# Dán kết quả (output) của: pytest tests/ -v
```
========================================== test session starts ===========================================
platform win32 -- Python 3.12.10, pytest-9.1.1, pluggy-1.6.0 -- C:\Users\ASUS\Documents\AI thực chiến\K4-DAY07-DinhTuanLong-2A202602620\.venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: C:\Users\ASUS\Documents\AI thực chiến\K4-DAY07-DinhTuanLong-2A202602620
collected 42 items                                                                                        

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED               [  2%]
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED                        [  4%]
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED                 [  7%]
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED                  [  9%]
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED                       [ 11%]
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED       [ 14%]
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED             [ 16%]
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED              [ 19%]
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED            [ 21%]
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED                              [ 23%]
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED              [ 26%]
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED                         [ 28%]
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED                     [ 30%]
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED                               [ 33%]
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED      [ 35%]
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED          [ 38%]
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED    [ 40%]
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED          [ 42%]
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED                              [ 45%]
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED                [ 47%]
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED                  [ 50%]
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED                        [ 52%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED             [ 54%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED               [ 57%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED   [ 59%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED                [ 61%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED                         [ 64%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED                        [ 66%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED                   [ 69%]
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED               [ 71%]
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED          [ 73%]
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED              [ 76%]
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED                    [ 78%]
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED              [ 80%]
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED [ 83%]
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED         [ 85%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED        [ 88%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED [ 90%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED       [ 92%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED [ 95%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED [ 97%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED [100%]

=========================================== 42 passed in 0.08s ===========================================

**Số lượng bài test vượt qua (pass):** 42 / 42

---

## 4. Dự đoán độ tương tự (Similarity Predictions) — Cá nhân (5 điểm)

## 4. Dự đoán độ tương tự (Similarity Predictions) — Cá nhân (5 điểm)

| Cặp | Câu A | Câu B | Dự đoán | Điểm thực tế | Đúng? |
|---|---|---|---|---:|---|
| 1 | Người mua có thể yêu cầu hoàn tiền trong vòng 15 ngày. | Khách hàng được phép gửi yêu cầu refund trong thời hạn 15 ngày. | Cao | 0.7736 | Có |
| 2 | Người mua có thể yêu cầu hoàn tiền trong vòng 15 ngày. | Người bán phải đăng thông tin bảo hành của sản phẩm. | Thấp | 0.3719 | Có |
| 3 | Người Bán phải cung cấp bằng chứng theo yêu cầu của Shopee. | Người Bán tại Shopee Mall phải cung cấp bằng chứng liên quan đến yêu cầu trả hàng/hoàn tiền. | Cao | 0.8972 | Có |
| 4 | Người Mua phải hoàn trả sản phẩm sau khi yêu cầu được chấp thuận. | Shopee xử lý tranh chấp sau khi nhận đầy đủ tài liệu. | Thấp | 0.4308 | Có |
| 5 | Shopee giải quyết tranh chấp trong vòng 07 ngày làm việc. | Shopee đưa ra hướng giải quyết sau khi nhận đầy đủ tài liệu. | Cao | 0.6076 | Có |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn ý nghĩa?**

> Kết quả bất ngờ nhất với tôi là cặp 5. Tôi dự đoán hai câu có độ tương tự cao vì đều nói về việc Shopee đưa ra hướng giải quyết tranh chấp, nhưng similarity thực tế chỉ ở mức 0.6076, thấp hơn cặp 1 và cặp 3 khá nhiều. Nguyên nhân có thể là câu thứ nhất nhấn mạnh thời hạn “07 ngày làm việc”, trong khi câu thứ hai tập trung vào điều kiện “sau khi nhận đầy đủ tài liệu”.
>
> Ngược lại, cặp 4 vẫn đạt 0.4308 dù hai câu nói về hai quy trình khác nhau. Điều này cho thấy embedding không chỉ dựa vào việc hai câu có cùng câu trả lời hay không, mà còn biểu diễn ngữ nghĩa tổng thể và các pattern chung như cùng liên quan đến quy trình sau bán hàng trên Shopee.
---

## 5. Kết quả truy xuất của tôi (Competition Results) — Cá nhân (10 điểm)

Chạy **5 câu hỏi đánh giá của nhóm** trên mã nguồn cá nhân của bạn trong gói `src`. **5 câu hỏi này phải trùng với các thành viên cùng nhóm** (xem `REPORT_NHOM.md`).

| # | Câu hỏi (Query) | Top-1 Chunk truy xuất được (tóm tắt) | Điểm Score | Có liên quan không? (Relevant) | Câu trả lời của Agent (tóm tắt) |
|---|---|---|---:|---|---|
| 1 | Người Mua có bao lâu để gửi yêu cầu trả hàng/hoàn tiền sau khi đơn được giao thành công? | `return-refund-policy#8` — mục 3.2, chứa quy định Người Mua có thể gửi yêu cầu trong vòng 15 ngày kể từ khi đơn được cập nhật giao hàng thành công | 2/2 | Có — chunk chứa đáp án ở rank 1 | Chưa đánh giá bằng real LLM; retrieval context đã chứa đáp án |
| 2 | Sau khi yêu cầu trả hàng/hoàn tiền Shopee Mall được chấp thuận, Người Mua có bao lâu để gửi trả sản phẩm? | `shopee-mall-buyer#18` — section về thời hạn và điều kiện trả hàng; chunk chứa đáp án `shopee-mall-buyer#5` nằm ở rank 2 và chứa mốc 06 ngày lịch | 1/2 | Có — chunk chứa đáp án ở rank 2 | Chưa đánh giá bằng real LLM; retrieval context đã chứa đáp án |
| 3 | Khi Shopee yêu cầu bằng chứng cho một yêu cầu trả hàng/hoàn tiền Shopee Mall, Người Bán phải cung cấp trong bao lâu? | `shopee-mall-seller#4` — chứa quy định Người Bán phải cung cấp bằng chứng trong tối đa 24 giờ | 2/2 | Có — chunk chứa đáp án ở rank 1 | Chưa đánh giá bằng real LLM; retrieval context đã chứa đáp án |
| 4 | Ai chịu trách nhiệm tiếp nhận bảo hành sản phẩm cho Người Mua trên Shopee? | `warranty-general-seller#22` — đúng tài liệu seller nhưng section top-1 không chứa trực tiếp câu trả lời về trách nhiệm tiếp nhận bảo hành | 0/2 | Không — chunk chứa đáp án không nằm trong top-3 | Chưa đánh giá bằng real LLM; context top-3 chưa đủ để trả lời chắc chắn |
| 5 | Đối với tranh chấp không phải khiếu nại trả hàng/hoàn tiền, Shopee đưa ra hướng giải quyết trong bao lâu sau khi nhận đủ tài liệu? | `dispute-resolution#3` — chứa quy định Shopee đưa ra hướng giải quyết trong vòng 07 ngày làm việc kể từ khi nhận đủ tài liệu | 2/2 | Có — chunk chứa đáp án ở rank 1 | Chưa đánh giá bằng real LLM; retrieval context đã chứa đáp án |

**Bao nhiêu câu hỏi trả về chunk có liên quan trong top-3?** 4/ 5

**Điều hay nhất tôi học được từ thành viên khác / nhóm khác (qua demo):**
> *Viết 2-3 câu:*

---

## Tự Đánh Giá (Phần Cá Nhân)

| Tiêu chí | Điểm tự đánh giá |
|----------|-------------------:|
| Khởi động (Warm-up) | 5 / 5 |
| Hướng tiếp cận của tôi (My Approach) | 10 / 10 |
| Hoàn thiện code (Core Implementation — tests) | 30 / 30 |
| Dự đoán độ tương tự (Similarity Predictions) | 5 / 5 |
| Kết quả truy xuất của tôi (Competition Results) | 7 / 10 |
| **Tổng phần cá nhân** | **57 / 60** |
