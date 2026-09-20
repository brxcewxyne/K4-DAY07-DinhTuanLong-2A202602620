from src import LocalEmbedder, compute_similarity

embedder = LocalEmbedder()

pairs = [
    (
        "Người mua có thể yêu cầu hoàn tiền trong vòng 15 ngày.",
        "Khách hàng được phép gửi yêu cầu refund trong thời hạn 15 ngày.",
        "CAO",
    ),
    (
        "Người mua có thể yêu cầu hoàn tiền trong vòng 15 ngày.",
        "Người bán phải đăng thông tin bảo hành của sản phẩm.",
        "THẤP",
    ),
    (
        "Người Bán phải cung cấp bằng chứng theo yêu cầu của Shopee.",
        "Người Bán tại Shopee Mall phải cung cấp bằng chứng liên quan đến yêu cầu trả hàng/hoàn tiền.",
        "CAO",
    ),
    (
        "Người Mua phải hoàn trả sản phẩm sau khi yêu cầu được chấp thuận.",
        "Shopee xử lý tranh chấp sau khi nhận đầy đủ tài liệu.",
        "THẤP",
    ),
    (
        "Shopee giải quyết tranh chấp trong vòng 07 ngày làm việc.",
        "Shopee đưa ra hướng giải quyết sau khi nhận đầy đủ tài liệu.",
        "CAO",
    ),
]

for i, (a, b, prediction) in enumerate(pairs, 1):
    va = embedder(a)
    vb = embedder(b)
    score = compute_similarity(va, vb)

    print(f"Pair {i}")
    print(f"Prediction: {prediction}")
    print(f"Similarity: {score:.4f}")
    print()