"""
loaders 模块示例

演示：
  1. HttpLoader  — 加载 HTTP/HTTPS 资源
  2. LocalLoader — 加载本地文件
  3. MinioLoader — 加载 MinIO 对象
  4. CompositeLoader — 按优先级自动分派
  5. guess_mime_type — 魔数 MIME 检测
"""
import asyncio
import os

from dotenv import find_dotenv, load_dotenv
from minio import Minio

from src.basic.loaders import (
    BlobData,
    CompositeLoader,
    HttpLoader,
    LocalLoader,
    MinioLoader,
    guess_mime_type,
)

load_dotenv(find_dotenv())

# 配置
MINIO_ENDPOINT = os.environ["MINIO_ENDPOINT"]
MINIO_ACCESS   = os.environ["MINIO_ACCESS_KEY"]
MINIO_SECRET   = os.environ["MINIO_SECRET_KEY"]
MINIO_SECURE   = os.getenv("MINIO_SECURE", "false").lower() == "true"

HTTP_URL    = "https://raw.githubusercontent.com/comet-7x/My-NoteBook/main/images/202606182358786.png"
LOCAL_PATH  = "/mnt/steins/zhihao/MyPractice_Project/search_agent/poc/loader/doc/my_student_info.png"
MINIO_URI   = "minio://test-bucket/test_multi_rag/output/1/卡尔曼滤波算法/page_images/page_0.jpg"


def _print(label: str, blob: BlobData | None) -> None:
    if blob is None:
        print(f"  {label}: 加载失败")
        return
    print(f"  {label}:")
    print(f"    content_type : {blob.content_type}")
    print(f"    is_image     : {blob.is_image}")
    print(f"    size         : {len(blob.content)} bytes")


async def main() -> None:
    minio_client = Minio(MINIO_ENDPOINT, access_key=MINIO_ACCESS, secret_key=MINIO_SECRET, secure=MINIO_SECURE)

    # 1. HttpLoader
    print("\n── 1. HttpLoader ───────────────────────────────────────")
    http_loader = HttpLoader()
    print(f"  can_handle({HTTP_URL[:40]}…): {http_loader.can_handle(HTTP_URL)}")
    blob = await http_loader.load(HTTP_URL)
    _print("result", blob)

    # 2. LocalLoader
    print("\n── 2. LocalLoader ──────────────────────────────────────")
    local_loader = LocalLoader()
    print(f"  can_handle: {local_loader.can_handle(LOCAL_PATH)}")
    blob = await local_loader.load(LOCAL_PATH)
    _print("result", blob)

    # 3. MinioLoader
    print("\n── 3. MinioLoader ──────────────────────────────────────")
    minio_loader = MinioLoader(client=minio_client)
    print(f"  can_handle: {minio_loader.can_handle(MINIO_URI)}")
    blob = await minio_loader.load(MINIO_URI)
    _print("result", blob)

    # 4. CompositeLoader — 自动路由
    print("\n── 4. CompositeLoader ──────────────────────────────────")
    composite = CompositeLoader([HttpLoader(), LocalLoader(), MinioLoader(client=minio_client)])
    for label, ref in [("HTTP", HTTP_URL), ("Local", LOCAL_PATH), ("MinIO", MINIO_URI)]:
        blob = await composite.load(ref)
        _print(label, blob)

    # 5. guess_mime_type — 魔数检测
    print("\n── 5. guess_mime_type ──────────────────────────────────")
    cases = [
        ("PNG  魔数", b"\x89PNG\r\n\x1a\n" + b"\x00" * 4),
        ("JPEG 魔数", b"\xff\xd8\xff\xe0" + b"\x00" * 8),
        ("PDF  魔数", b"%PDF-1.4 rest"),
        ("ZIP  魔数", b"PK\x03\x04" + b"\x00" * 9),
        ("扩展名兜底", b"\x00" * 16),
    ]
    for name, data in cases:
        mime = guess_mime_type(data, fallback_url="file.png" if name == "扩展名兜底" else None)
        print(f"  {name:12}: {mime}")

    # 6. BlobData.from_data_url 往返验证
    print("\n── 6. BlobData 往返（to_data_url → from_data_url）─────")
    original = BlobData(content=b"\x89PNG\r\n\x1a\nDATA", content_type="image/png")
    restored = BlobData.from_data_url(original.to_data_url())
    assert original.content == restored.content and original.content_type == restored.content_type
    print(f"  往返一致: content_type={restored.content_type}, size={len(restored.content)} bytes ✓")


if __name__ == "__main__":
    asyncio.run(main())
