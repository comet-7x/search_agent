import asyncio
import io
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from minio import Minio
from minio.error import S3Error

from ...exceptions import MinioError

logger = logging.getLogger(__name__)


@dataclass
class ObjectInfo:
    """MinIO 对象元数据，屏蔽 SDK 内部类型。"""

    bucket: str
    name: str
    size: int
    last_modified: datetime
    content_type: str
    etag: str


class AsyncMinioStorage:
    """异步 MinIO 对象存储类，通过 asyncio.to_thread 包装同步 SDK。

    Args:
        endpoint:   MinIO 服务地址，例如 "localhost:9000"
        access_key: Access Key
        secret_key: Secret Key
        secure:     是否使用 HTTPS，默认 True
        kwargs:     透传给 Minio 构造器的其余参数（region、http_client 等）
    """

    def __init__(
        self,
        endpoint: str,
        access_key: str,
        secret_key: str,
        secure: bool = True,
        **kwargs: Any,
    ) -> None:
        self.client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure,
            **kwargs,
        )

    async def __aenter__(self) -> "AsyncMinioStorage":
        return self

    async def __aexit__(self, *_: object) -> None:
        pass

    # Bucket 管理
    async def bucket_exists(self, bucket: str) -> bool:
        """检查 Bucket 是否存在。"""
        try:
            return await asyncio.to_thread(self.client.bucket_exists, bucket)
        except S3Error as e:
            logger.error(f"S3Error checking bucket '{bucket}': {e}", exc_info=True)
            return False
        except Exception as e:
            logger.error(f"Unexpected error checking bucket '{bucket}': {e}", exc_info=True)
            return False

    async def list_buckets(self) -> list[str]:
        """列出所有 Bucket 名称。"""
        try:
            buckets = await asyncio.to_thread(self.client.list_buckets)
            return [b.name for b in buckets]
        except S3Error as e:
            logger.error(f"S3Error listing buckets: {e}", exc_info=True)
            return []
        except Exception as e:
            logger.error(f"Unexpected error listing buckets: {e}", exc_info=True)
            return []

    async def make_bucket(self, bucket: str, **kwargs: Any) -> None:
        """创建 Bucket（幂等，已存在时静默跳过）。

        Raises:
            MinioError: 权限不足或服务异常时抛出
        """
        try:
            exists = await self.bucket_exists(bucket)
            if exists:
                logger.debug(f"Bucket '{bucket}' already exists.")
                return
            await asyncio.to_thread(self.client.make_bucket, bucket, **kwargs)
            logger.info(f"Created bucket '{bucket}'.")
        except S3Error as e:
            if e.code == "BucketAlreadyOwnedByYou":
                logger.debug(f"Bucket '{bucket}' concurrently created.")
                return
            logger.error(f"S3Error creating bucket '{bucket}': {e}", exc_info=True)
            raise MinioError(f"Failed to create bucket '{bucket}'.") from e
        except MinioError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error creating bucket '{bucket}': {e}", exc_info=True)
            raise MinioError(f"Unexpected exception in make_bucket('{bucket}').") from e

    async def remove_bucket(self, bucket: str) -> None:
        """删除 Bucket（Bucket 必须为空）。

        Raises:
            MinioError: Bucket 非空、不存在或服务异常时抛出
        """
        try:
            await asyncio.to_thread(self.client.remove_bucket, bucket)
            logger.info(f"Removed bucket '{bucket}'.")
        except S3Error as e:
            logger.error(f"S3Error removing bucket '{bucket}': {e}", exc_info=True)
            raise MinioError(f"Failed to remove bucket '{bucket}': {e.code}") from e
        except Exception as e:
            logger.error(f"Unexpected error removing bucket '{bucket}': {e}", exc_info=True)
            raise MinioError(f"Unexpected exception in remove_bucket('{bucket}').") from e

    # 对象写入
    async def put_object(
        self,
        bucket: str,
        name: str,
        data: bytes,
        content_type: str = "application/octet-stream",
        metadata: Any = None,
    ) -> str:
        """将 bytes 写入指定对象，返回对象名称。

        Raises:
            MinioError: 写入失败时抛出
        """
        try:
            stream = io.BytesIO(data)
            await asyncio.to_thread(
                self.client.put_object,
                bucket,
                name,
                stream,
                len(data),
                content_type=content_type,
                metadata=metadata,
            )
            logger.debug(f"put_object: bucket='{bucket}', name='{name}', size={len(data)}")
            return name
        except S3Error as e:
            logger.error(f"S3Error in put_object '{bucket}/{name}': {e}", exc_info=True)
            raise MinioError(f"Failed to put object '{bucket}/{name}'.") from e
        except Exception as e:
            logger.error(f"Unexpected error in put_object '{bucket}/{name}': {e}", exc_info=True)
            raise MinioError(f"Unexpected exception in put_object('{bucket}/{name}').") from e

    async def fput_object(
        self,
        bucket: str,
        name: str,
        file_path: str | Path,
        content_type: str = "application/octet-stream",
        metadata: Any = None,
    ) -> str:
        """从本地文件上传对象，返回对象名称。

        Raises:
            MinioError: 上传失败时抛出
        """
        try:
            path_str = str(file_path)
            await asyncio.to_thread(
                self.client.fput_object,
                bucket,
                name,
                path_str,
                content_type=content_type,
                metadata=metadata,
            )
            logger.debug(f"fput_object: bucket='{bucket}', name='{name}', file='{path_str}'")
            return name
        except S3Error as e:
            logger.error(f"S3Error in fput_object '{bucket}/{name}': {e}", exc_info=True)
            raise MinioError(f"Failed to upload file to '{bucket}/{name}'.") from e
        except Exception as e:
            logger.error(f"Unexpected error in fput_object '{bucket}/{name}': {e}", exc_info=True)
            raise MinioError(f"Unexpected exception in fput_object('{bucket}/{name}').") from e

    # 对象读取
    async def get_object(self, bucket: str, name: str) -> bytes:
        """下载对象内容并以 bytes 返回。

        Raises:
            MinioError: 对象不存在或服务异常时抛出
        """

        def _get() -> bytes:
            response = self.client.get_object(bucket, name)
            try:
                return response.read()
            finally:
                response.close()
                response.release_conn()

        try:
            content = await asyncio.to_thread(_get)
            logger.debug(f"get_object: bucket='{bucket}', name='{name}', size={len(content)}")
            return content
        except S3Error as e:
            logger.error(f"S3Error in get_object '{bucket}/{name}': {e}", exc_info=True)
            raise MinioError(f"Failed to get object '{bucket}/{name}'.") from e
        except Exception as e:
            logger.error(f"Unexpected error in get_object '{bucket}/{name}': {e}", exc_info=True)
            raise MinioError(f"Unexpected exception in get_object('{bucket}/{name}').") from e

    async def fget_object(
        self,
        bucket: str,
        name: str,
        file_path: str | Path,
    ) -> None:
        """将对象下载到本地文件。

        Raises:
            MinioError: 对象不存在或服务异常时抛出
        """
        try:
            await asyncio.to_thread(self.client.fget_object, bucket, name, str(file_path))
            logger.debug(f"fget_object: bucket='{bucket}', name='{name}' → '{file_path}'")
        except S3Error as e:
            logger.error(f"S3Error in fget_object '{bucket}/{name}': {e}", exc_info=True)
            raise MinioError(f"Failed to download object '{bucket}/{name}'.") from e
        except Exception as e:
            logger.error(f"Unexpected error in fget_object '{bucket}/{name}': {e}", exc_info=True)
            raise MinioError(f"Unexpected exception in fget_object('{bucket}/{name}').") from e

    async def stat_object(self, bucket: str, name: str) -> ObjectInfo | None:
        """获取对象元数据，不存在时返回 None。"""
        try:
            stat = await asyncio.to_thread(self.client.stat_object, bucket, name)
            return ObjectInfo(
                bucket=bucket,
                name=name,
                size=stat.size or 0,
                last_modified=stat.last_modified or datetime.min,
                content_type=stat.content_type or "application/octet-stream",
                etag=stat.etag or "",
            )
        except S3Error as e:
            if e.code == "NoSuchKey":
                return None
            logger.error(f"S3Error in stat_object '{bucket}/{name}': {e}", exc_info=True)
            raise MinioError(f"Failed to stat object '{bucket}/{name}'.") from e
        except Exception as e:
            logger.error(f"Unexpected error in stat_object '{bucket}/{name}': {e}", exc_info=True)
            raise MinioError(f"Unexpected exception in stat_object('{bucket}/{name}').") from e

    # 对象删除
    async def remove_object(self, bucket: str, name: str) -> None:
        """删除单个对象（幂等，不存在时静默跳过）。

        Raises:
            MinioError: 服务异常时抛出
        """
        try:
            await asyncio.to_thread(self.client.remove_object, bucket, name)
            logger.debug(f"remove_object: bucket='{bucket}', name='{name}'")
        except S3Error as e:
            if e.code == "NoSuchKey":
                logger.debug(f"Object '{bucket}/{name}' not found, skip remove.")
                return
            logger.error(f"S3Error in remove_object '{bucket}/{name}': {e}", exc_info=True)
            raise MinioError(f"Failed to remove object '{bucket}/{name}'.") from e
        except Exception as e:
            logger.error(f"Unexpected error in remove_object '{bucket}/{name}': {e}", exc_info=True)
            raise MinioError(f"Unexpected exception in remove_object('{bucket}/{name}').") from e

    async def remove_objects(self, bucket: str, names: list[str]) -> list[str]:
        """批量删除对象，返回删除失败的对象名称列表。"""
        if not names:
            return []

        from minio.deleteobjects import DeleteObject

        def _remove() -> list[str]:
            delete_list = [DeleteObject(n) for n in names]
            errors = list(self.client.remove_objects(bucket, iter(delete_list)))
            return [str(e.name) for e in errors]

        try:
            failed = await asyncio.to_thread(_remove)
            if failed:
                logger.warning(f"remove_objects: {len(failed)} failed in bucket '{bucket}'.")
            else:
                logger.debug(f"remove_objects: deleted {len(names)} objects from '{bucket}'.")
            return failed
        except Exception as e:
            logger.error(f"Unexpected error in remove_objects for '{bucket}': {e}", exc_info=True)
            raise MinioError(f"Unexpected exception in remove_objects('{bucket}').") from e


    # 对象列举
    async def list_objects(
        self,
        bucket: str,
        prefix: str = "",
        recursive: bool = False,
    ) -> list[ObjectInfo]:
        """列出 Bucket 中的对象（非递归时仅返回当前层级）。"""

        def _list() -> list[ObjectInfo]:
            objects = self.client.list_objects(bucket, prefix=prefix, recursive=recursive)
            results = []
            for obj in objects:
                results.append(
                    ObjectInfo(
                        bucket=bucket,
                        name=obj.object_name or "",
                        size=obj.size or 0,
                        last_modified=obj.last_modified or datetime.min,
                        content_type=obj.content_type or "application/octet-stream",
                        etag=obj.etag or "",
                    )
                )
            return results

        try:
            return await asyncio.to_thread(_list)
        except S3Error as e:
            logger.error(f"S3Error in list_objects '{bucket}/{prefix}': {e}", exc_info=True)
            raise MinioError(f"Failed to list objects in '{bucket}'.") from e
        except Exception as e:
            logger.error(f"Unexpected error in list_objects '{bucket}': {e}", exc_info=True)
            raise MinioError(f"Unexpected exception in list_objects('{bucket}').") from e

    # 预签名 URL
    async def presigned_get_url(
        self,
        bucket: str,
        name: str,
        expires_seconds: int = 3600,
    ) -> str:
        """生成对象的预签名下载 URL。

        Raises:
            MinioError: 服务异常时抛出
        """
        try:
            url = await asyncio.to_thread(
                self.client.presigned_get_object,
                bucket,
                name,
                expires=timedelta(seconds=expires_seconds),
            )
            return url
        except S3Error as e:
            logger.error(f"S3Error in presigned_get_url '{bucket}/{name}': {e}", exc_info=True)
            raise MinioError(f"Failed to generate download URL for '{bucket}/{name}'.") from e
        except Exception as e:
            logger.error(
                f"Unexpected error in presigned_get_url '{bucket}/{name}': {e}", exc_info=True
            )
            raise MinioError(
                f"Unexpected exception in presigned_get_url('{bucket}/{name}')."
            ) from e

    async def presigned_put_url(
        self,
        bucket: str,
        name: str,
        expires_seconds: int = 3600,
    ) -> str:
        """生成对象的预签名上传 URL。

        Raises:
            MinioError: 服务异常时抛出
        """
        try:
            url = await asyncio.to_thread(
                self.client.presigned_put_object,
                bucket,
                name,
                expires=timedelta(seconds=expires_seconds),
            )
            return url
        except S3Error as e:
            logger.error(f"S3Error in presigned_put_url '{bucket}/{name}': {e}", exc_info=True)
            raise MinioError(f"Failed to generate upload URL for '{bucket}/{name}'.") from e
        except Exception as e:
            logger.error(
                f"Unexpected error in presigned_put_url '{bucket}/{name}': {e}", exc_info=True
            )
            raise MinioError(
                f"Unexpected exception in presigned_put_url('{bucket}/{name}')."
            ) from e


class MinioStorage:
    """同步 MinIO 对象存储类（与 AsyncMinioStorage 接口一致）。"""

    def __init__(
        self,
        endpoint: str,
        access_key: str,
        secret_key: str,
        secure: bool = True,
        **kwargs: Any,
    ) -> None:
        self.client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure,
            **kwargs,
        )

    def __enter__(self) -> "MinioStorage":
        return self

    def __exit__(self, *_: object) -> None:
        pass


    # Bucket 管理
    def bucket_exists(self, bucket: str) -> bool:
        try:
            return self.client.bucket_exists(bucket)
        except S3Error as e:
            logger.error(f"S3Error checking bucket '{bucket}': {e}", exc_info=True)
            return False
        except Exception as e:
            logger.error(f"Unexpected error checking bucket '{bucket}': {e}", exc_info=True)
            return False

    def list_buckets(self) -> list[str]:
        try:
            return [b.name for b in self.client.list_buckets()]
        except Exception as e:
            logger.error(f"Unexpected error listing buckets: {e}", exc_info=True)
            return []

    def make_bucket(self, bucket: str, **kwargs: Any) -> None:
        """创建 Bucket（幂等）。

        Raises:
            MinioError: 权限不足或服务异常时抛出
        """
        try:
            if self.bucket_exists(bucket):
                logger.debug(f"Bucket '{bucket}' already exists.")
                return
            self.client.make_bucket(bucket, **kwargs)
            logger.info(f"Created bucket '{bucket}'.")
        except S3Error as e:
            if e.code == "BucketAlreadyOwnedByYou":
                return
            logger.error(f"S3Error creating bucket '{bucket}': {e}", exc_info=True)
            raise MinioError(f"Failed to create bucket '{bucket}'.") from e
        except MinioError:
            raise
        except Exception as e:
            raise MinioError(f"Unexpected exception in make_bucket('{bucket}').") from e

    def remove_bucket(self, bucket: str) -> None:
        """删除 Bucket（必须为空）。

        Raises:
            MinioError: 失败时抛出
        """
        try:
            self.client.remove_bucket(bucket)
            logger.info(f"Removed bucket '{bucket}'.")
        except S3Error as e:
            raise MinioError(f"Failed to remove bucket '{bucket}': {e.code}") from e
        except Exception as e:
            raise MinioError(f"Unexpected exception in remove_bucket('{bucket}').") from e

    # 对象写入
    def put_object(
        self,
        bucket: str,
        name: str,
        data: bytes,
        content_type: str = "application/octet-stream",
        metadata: Any = None,
    ) -> str:
        """将 bytes 写入对象，返回对象名称。

        Raises:
            MinioError: 写入失败时抛出
        """
        try:
            self.client.put_object(
                bucket,
                name,
                io.BytesIO(data),
                len(data),
                content_type=content_type,
                metadata=metadata,
            )
            logger.debug(f"put_object: bucket='{bucket}', name='{name}', size={len(data)}")
            return name
        except S3Error as e:
            raise MinioError(f"Failed to put object '{bucket}/{name}'.") from e
        except Exception as e:
            raise MinioError(f"Unexpected exception in put_object('{bucket}/{name}').") from e

    def fput_object(
        self,
        bucket: str,
        name: str,
        file_path: str | Path,
        content_type: str = "application/octet-stream",
        metadata: Any = None,
    ) -> str:
        """从本地文件上传对象，返回对象名称。

        Raises:
            MinioError: 上传失败时抛出
        """
        try:
            self.client.fput_object(
                bucket,
                name,
                str(file_path),
                content_type=content_type,
                metadata=metadata,
            )
            logger.debug(f"fput_object: bucket='{bucket}', name='{name}'")
            return name
        except S3Error as e:
            raise MinioError(f"Failed to upload file to '{bucket}/{name}'.") from e
        except Exception as e:
            raise MinioError(f"Unexpected exception in fput_object('{bucket}/{name}').") from e

    # 对象读取
    def get_object(self, bucket: str, name: str) -> bytes:
        """下载对象内容并以 bytes 返回。

        Raises:
            MinioError: 对象不存在或服务异常时抛出
        """
        try:
            response = self.client.get_object(bucket, name)
            try:
                content = response.read()
            finally:
                response.close()
                response.release_conn()
            logger.debug(f"get_object: bucket='{bucket}', name='{name}', size={len(content)}")
            return content
        except S3Error as e:
            raise MinioError(f"Failed to get object '{bucket}/{name}'.") from e
        except MinioError:
            raise
        except Exception as e:
            raise MinioError(f"Unexpected exception in get_object('{bucket}/{name}').") from e

    def fget_object(self, bucket: str, name: str, file_path: str | Path) -> None:
        """将对象下载到本地文件。

        Raises:
            MinioError: 失败时抛出
        """
        try:
            self.client.fget_object(bucket, name, str(file_path))
            logger.debug(f"fget_object: '{bucket}/{name}' → '{file_path}'")
        except S3Error as e:
            raise MinioError(f"Failed to download object '{bucket}/{name}'.") from e
        except Exception as e:
            raise MinioError(f"Unexpected exception in fget_object('{bucket}/{name}').") from e

    def stat_object(self, bucket: str, name: str) -> ObjectInfo | None:
        """获取对象元数据，不存在时返回 None。"""
        try:
            stat = self.client.stat_object(bucket, name)
            return ObjectInfo(
                bucket=bucket,
                name=name,
                size=stat.size or 0,
                last_modified=stat.last_modified or datetime.min,
                content_type=stat.content_type or "application/octet-stream",
                etag=stat.etag or "",
            )
        except S3Error as e:
            if e.code == "NoSuchKey":
                return None
            raise MinioError(f"Failed to stat object '{bucket}/{name}'.") from e
        except Exception as e:
            raise MinioError(f"Unexpected exception in stat_object('{bucket}/{name}').") from e

    # 对象删除
    def remove_object(self, bucket: str, name: str) -> None:
        """删除单个对象（幂等）。

        Raises:
            MinioError: 服务异常时抛出
        """
        try:
            self.client.remove_object(bucket, name)
            logger.debug(f"remove_object: '{bucket}/{name}'")
        except S3Error as e:
            if e.code == "NoSuchKey":
                return
            raise MinioError(f"Failed to remove object '{bucket}/{name}'.") from e
        except Exception as e:
            raise MinioError(f"Unexpected exception in remove_object('{bucket}/{name}').") from e

    def remove_objects(self, bucket: str, names: list[str]) -> list[str]:
        """批量删除对象，返回删除失败的对象名称列表。"""
        if not names:
            return []
        from minio.deleteobjects import DeleteObject

        try:
            delete_list = [DeleteObject(n) for n in names]
            errors = list(self.client.remove_objects(bucket, iter(delete_list)))
            failed = [str(e.name) for e in errors]
            if failed:
                logger.warning(f"remove_objects: {len(failed)} failed in '{bucket}'.")
            return failed
        except Exception as e:
            raise MinioError(f"Unexpected exception in remove_objects('{bucket}').") from e

    # 对象列举
    def list_objects(
        self,
        bucket: str,
        prefix: str = "",
        recursive: bool = False,
    ) -> list[ObjectInfo]:
        """列出 Bucket 中的对象。"""
        try:
            objects = self.client.list_objects(bucket, prefix=prefix, recursive=recursive)
            return [
                ObjectInfo(
                    bucket=bucket,
                    name=obj.object_name or "",
                    size=obj.size or 0,
                    last_modified=obj.last_modified or datetime.min,
                    content_type=obj.content_type or "application/octet-stream",
                    etag=obj.etag or "",
                )
                for obj in objects
            ]
        except S3Error as e:
            raise MinioError(f"Failed to list objects in '{bucket}'.") from e
        except Exception as e:
            raise MinioError(f"Unexpected exception in list_objects('{bucket}').") from e

    # 预签名 URL
    def presigned_get_url(
        self,
        bucket: str,
        name: str,
        expires_seconds: int = 3600,
    ) -> str:
        """生成对象的预签名下载 URL。

        Raises:
            MinioError: 服务异常时抛出
        """
        try:
            return self.client.presigned_get_object(
                bucket, name, expires=timedelta(seconds=expires_seconds)
            )
        except S3Error as e:
            raise MinioError(f"Failed to generate download URL for '{bucket}/{name}'.") from e
        except Exception as e:
            raise MinioError(
                f"Unexpected exception in presigned_get_url('{bucket}/{name}')."
            ) from e

    def presigned_put_url(
        self,
        bucket: str,
        name: str,
        expires_seconds: int = 3600,
    ) -> str:
        """生成对象的预签名上传 URL。

        Raises:
            MinioError: 服务异常时抛出
        """
        try:
            return self.client.presigned_put_object(
                bucket, name, expires=timedelta(seconds=expires_seconds)
            )
        except S3Error as e:
            raise MinioError(f"Failed to generate upload URL for '{bucket}/{name}'.") from e
        except Exception as e:
            raise MinioError(
                f"Unexpected exception in presigned_put_url('{bucket}/{name}')."
            ) from e
