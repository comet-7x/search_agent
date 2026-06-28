"""
embeddings 模块示例

演示：
  1. aembed_text        — 纯文本编码
  2. aembed_image       — 纯图片编码
  3. aembed_multimodal  — 文本+图片联合编码
  4. aembed_list        — 批量并行编码
  5. get_output_dim     — 获取向量维度
"""
import asyncio
import os

from dotenv import find_dotenv, load_dotenv
from minio import Minio

from src.basic.embeddings.qwen3_vl_embedding import (
    EmbeddingData,
    Qwen3VLEmbeddingModel,
)
from src.basic.loaders import MinioLoader

load_dotenv(find_dotenv())

# 配置
EMBEDDING_URL   = os.environ["EMBEDDING_BASE_URL"]
EMBEDDING_MODEL = os.environ["EMBEDDING_MODEL"]
EMBEDDING_KEY   = os.environ["EMBEDDING_API_KEY"]

MINIO_ENDPOINT  = os.environ["MINIO_ENDPOINT"]
MINIO_ACCESS    = os.environ["MINIO_ACCESS_KEY"]
MINIO_SECRET    = os.environ["MINIO_SECRET_KEY"]
MINIO_SECURE    = os.getenv("MINIO_SECURE", "false").lower() == "true"
MINIO_IMAGE_URI = "minio://test-bucket/test_multi_rag/output/1/卡尔曼滤波算法/page_images/page_0.jpg"

QUERY = "2000年我国水产品对外贸易总量"


async def main() -> None:
    model = Qwen3VLEmbeddingModel(
        base_url=EMBEDDING_URL, model_name=EMBEDDING_MODEL, api_key=EMBEDDING_KEY
    )

    # 加载测试图片
    loader = MinioLoader(Minio(MINIO_ENDPOINT, access_key=MINIO_ACCESS, secret_key=MINIO_SECRET, secure=MINIO_SECURE))
    blob = await loader.load(MINIO_IMAGE_URI)
    assert blob is not None and blob.is_image
    image_data_url = blob.to_data_url()
    print(f"测试图片: {len(blob.content)} bytes, {blob.content_type}\n")

    # 1. 纯文本
    print("── 1. aembed_text ─────────────────────────────────────")
    vec = await model.aembed_text(QUERY)
    print(f"  dim={len(vec)},  前3维={[round(v, 6) for v in vec[:3]]}")

    # 2. 纯图片
    print("\n── 2. aembed_image ────────────────────────────────────")
    vec = await model.aembed_image(image_data_url)
    print(f"  dim={len(vec)}, 前3维={[round(v, 6) for v in vec[:3]]}")

    # 3. 文本+图片
    print("\n── 3. aembed_multimodal ───────────────────────────────")
    vec = await model.aembed_multimodal(QUERY, image_data_url)
    print(f"  dim={len(vec)}, 前3维={[round(v, 6) for v in vec[:3]]}")

    # 4. 批量并行（aembed_list）
    print("\n── 4. aembed_list（批量并行）──────────────────────────")
    texts = ["水产品对外贸易", "卡尔曼滤波算法", "深度学习模型"]
    vecs = await model.aembed_list(
        [EmbeddingData(text=t) for t in texts]
    )
    for t, v in zip(texts, vecs):
        print(f"  [{t}] dim={len(v)}")  # type: ignore[arg-type]

    # 5. 三路并行，验证维度一致
    print("\n── 5. 三路并行编码维度验证 ────────────────────────────")
    t_vec, i_vec, m_vec = await asyncio.gather(
        model.aembed_text(QUERY),
        model.aembed_image(image_data_url),
        model.aembed_multimodal(QUERY, image_data_url),
    )
    assert len(t_vec) == len(i_vec) == len(m_vec), "维度不一致！"
    print(f"  embed_text / embed_image / embed_multimodal 维度均为 {len(t_vec)} ✓")

    # 6. 向量维度
    print("\n── 6. get_output_dim ──────────────────────────────────")
    print(f"  output_dim = {model.get_output_dim()}")


if __name__ == "__main__":
    asyncio.run(main())
