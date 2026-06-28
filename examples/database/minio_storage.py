"""
database.object.minio 模块示例

演示：
  1. list_buckets      — 列举所有 Bucket
  2. bucket_exists     — 判断 Bucket 是否存在
  3. make_bucket       — 幂等创建 Bucket
  4. put_object        — 写入字节对象
  5. stat_object       — 查看对象元数据
  6. list_objects      — 列举 Bucket 内对象
  7. get_object        — 下载对象内容
  8. presigned_get_url — 生成预签名下载 URL
  9. remove_object     — 删除对象（清理）
"""
import asyncio
import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

from src.basic.database.object.minio import AsyncMinioStorage

load_dotenv(find_dotenv())

# ── 配置 ──────────────────────────────────────────────────────────────────────
MINIO_ENDPOINT = os.environ["MINIO_ENDPOINT"]
MINIO_ACCESS   = os.environ["MINIO_ACCESS_KEY"]
MINIO_SECRET   = os.environ["MINIO_SECRET_KEY"]
MINIO_SECURE   = os.getenv("MINIO_SECURE", "false").lower() == "true"

TEST_BUCKET = "examples-test"
TEST_OBJECT = "examples/hello.txt"
TEST_BYTES  = b"Hello from search_agent examples!"

LIST_BUCKET  = "test-bucket"
LIST_PREFIX  = "test_multi_rag/output/1/卡尔曼滤波算法/page_images/"


async def main() -> None:
    async with AsyncMinioStorage(
        endpoint=MINIO_ENDPOINT,
        access_key=MINIO_ACCESS,
        secret_key=MINIO_SECRET,
        secure=MINIO_SECURE,
    ) as storage:
        # 1. list_buckets
        print("── 1. list_buckets ──────────────────────────────────")
        buckets = await storage.list_buckets()
        print(f"  {buckets}")

        # 2. bucket_exists
        print("\n── 2. bucket_exists ─────────────────────────────────")
        print(f"  '{LIST_BUCKET}' 存在: {await storage.bucket_exists(LIST_BUCKET)}")
        print(f"  'no-such-bucket' 存在: {await storage.bucket_exists('no-such-bucket')}")

        # 3. make_bucket（幂等）
        print("\n── 3. make_bucket（幂等）────────────────────────────")
        await storage.make_bucket(TEST_BUCKET)
        print(f"  '{TEST_BUCKET}' 已就绪: {await storage.bucket_exists(TEST_BUCKET)}")

        # 4. put_object
        print("\n── 4. put_object ────────────────────────────────────")
        name = await storage.put_object(
            bucket=TEST_BUCKET,
            name=TEST_OBJECT,
            data=TEST_BYTES,
            content_type="text/plain",
        )
        print(f"  已写入: {name} ({len(TEST_BYTES)} bytes)")

        # 5. stat_object
        print("\n── 5. stat_object ───────────────────────────────────")
        info = await storage.stat_object(TEST_BUCKET, TEST_OBJECT)
        if info:
            print(f"  name          : {info.name}")
            print(f"  size          : {info.size} bytes")
            print(f"  content_type  : {info.content_type}")
            print(f"  last_modified : {info.last_modified}")
            print(f"  etag          : {info.etag}")

        # 6. list_objects
        print("\n── 6. list_objects ──────────────────────────────────")
        objects = await storage.list_objects(LIST_BUCKET, prefix=LIST_PREFIX)
        print(f"  '{LIST_PREFIX}' 下共 {len(objects)} 个对象")
        for obj in objects[:5]:
            print(f"    {Path(obj.name).name:30} {obj.size:>8} bytes  {obj.content_type}")

        # 7. get_object
        print("\n── 7. get_object ────────────────────────────────────")
        content = await storage.get_object(TEST_BUCKET, TEST_OBJECT)
        assert content == TEST_BYTES
        print(f"  内容: {content.decode()} ✓")

        # 8. presigned_get_url
        print("\n── 8. presigned_get_url ─────────────────────────────")
        url = await storage.presigned_get_url(
            bucket=TEST_BUCKET,
            name=TEST_OBJECT,
            expires_seconds=3600,
        )
        print(f"  {url[:80]}…")

        # 9. remove_object（清理）
        print("\n── 9. remove_object（清理）──────────────────────────")
        await storage.remove_object(TEST_BUCKET, TEST_OBJECT)
        stat_after = await storage.stat_object(TEST_BUCKET, TEST_OBJECT)
        print(f"  删除后 stat 返回 None: {stat_after is None} ✓")


if __name__ == "__main__":
    asyncio.run(main())
