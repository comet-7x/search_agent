from typing import Any, ClassVar, cast

from pydantic import BaseModel
from pymilvus import (
    AsyncMilvusClient,
    MilvusClient,
    MilvusException,
    Function,
    DataType,
    AnnSearchRequest,
)
from pymilvus.client.abstract import BaseRanker
import logging

from ...exceptions import MilvusError

logger = logging.getLogger(__name__)


class VectorSearchContent(BaseModel):
    record: dict[str, Any]
    score: float


class VectorQueryContent(BaseModel):
    record: dict[str, Any]


class BaseVectorRecord(BaseModel):
    """
    所有 Milvus 集合 Schema 的基类,子类通过 Pydantic Field 声明字段,
    create_collection 会自动解析并建表,无需手动拼装 Milvus 配置。

    使用方式:
        1. 定义 schema_config(必须)
        2. 每个字段用 json_schema_extra 声明 Milvus 配置

    完整示例:

        class MyRecord(BaseVectorRecord):
            schema_config: ClassVar[dict] = {
                "auto_id": False,
                "enable_dynamic_field": True,
                "description": "我的集合",
            }

            id: int = Field(..., json_schema_extra={
                "datatype": DataType.INT64,
                "is_primary": True,
            })

            text: str = Field(..., json_schema_extra={
                "datatype": DataType.VARCHAR,
                "max_length": 65535,
            })

            # 稠密向量字段:必须配置 index
            embedding: list[float] | None = Field(default=None, json_schema_extra={
                "datatype": DataType.FLOAT_VECTOR,
                "dim": 1536,
                "index": {
                    "metric_type": "COSINE",
                    "index_type": "HNSW",
                    "index_name": "embedding_index",
                    "params": {"M": 64, "efConstruction": 100},
                },
            })

            # BM25 稀疏向量字段:配置 function,由服务端自动生成,insert 时不传值
            sparse: dict | None = Field(default=None, json_schema_extra={
                "datatype": DataType.SPARSE_FLOAT_VECTOR,
                "function": {
                    "name": "sparse_func",
                    "input_field_names": ["text"],
                    "function_type": FunctionType.BM25,
                },
                "index": {
                    "metric_type": "BM25",
                    "index_type": "SPARSE_INVERTED_INDEX",
                    "index_name": "sparse_index",
                    "params": {"inverted_index_algo": "DAAT_MAXSCORE"},
                },
            })

    json_schema_extra 支持的 key:
        datatype (必须)       : Milvus DataType 枚举值
        is_primary            : 是否主键,仅一个字段可设为 True
        max_length            : VARCHAR 类型的最大长度
        dim                   : 向量字段维度
        nullable              : 是否允许为空
        enable_analyzer       : 是否启用文本分析器(全文检索)
        multi_analyzer_params : 多语言分析器配置
        index                 : 向量字段索引配置(FLOAT_VECTOR 必须)
        function              : 服务端函数配置(BM25 稀疏向量使用)
    """

    schema_config: ClassVar[dict]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

        if not hasattr(cls, "schema_config") or not isinstance(cls.schema_config, dict):
            raise TypeError(
                f"{cls.__name__} 必须定义 schema_config: ClassVar[dict],"
                f"例如:\n"
                f"    schema_config: ClassVar[dict] = {{\n"
                f'        "auto_id": False,\n'
                f'        "enable_dynamic_field": True,\n'
                f'        "description": "...",\n'
                f"    }}"
            )

        for field_name, field_info in cls.model_fields.items():
            extra = field_info.json_schema_extra

            if not isinstance(extra, dict):
                raise TypeError(
                    f"{cls.__name__}.{field_name} 必须通过 json_schema_extra 声明 Milvus 字段配置,"
                    f"例如:\n"
                    f"    {field_name} = Field(..., json_schema_extra={{'datatype': DataType.VARCHAR, 'max_length': 1024}})"
                )

            if "datatype" not in extra:
                raise TypeError(
                    f"{cls.__name__}.{field_name} 的 json_schema_extra 缺少 'datatype' 键,"
                    f"请指定 DataType,例如:DataType.VARCHAR、DataType.INT64、DataType.FLOAT_VECTOR"
                )

            vector_types = {
                DataType.FLOAT_VECTOR,
                DataType.BINARY_VECTOR,
                DataType.FLOAT16_VECTOR,
                DataType.BFLOAT16_VECTOR,
                DataType.SPARSE_FLOAT_VECTOR,
                DataType.INT8_VECTOR,
            }
            if extra["datatype"] in vector_types and "index" not in extra:
                raise TypeError(
                    f"{cls.__name__}.{field_name} 是向量字段({extra['datatype']}),"
                    f"必须在 json_schema_extra 中配置 'index',例如:\n"
                    f"    'index': {{\n"
                    f'        "metric_type": "COSINE",\n'
                    f'        "index_type": "HNSW",\n'
                    f'        "index_name": "{field_name}_index",\n'
                    f'        "params": {{"M": 64, "efConstruction": 100}},\n'
                    f"    }}"
                )

            func_cfg = extra.get("function")
            if func_cfg is not None:
                if not isinstance(func_cfg, dict):
                    raise TypeError(
                        f"{cls.__name__}.{field_name} 的 function 配置必须是 dict,"
                        f"当前类型为 {type(func_cfg).__name__}"
                    )
                missing = {
                    "name",
                    "input_field_names",
                    "function_type",
                } - func_cfg.keys()
                if missing:
                    raise TypeError(
                        f"{cls.__name__}.{field_name} 的 function 配置缺少键:{missing},"
                        f"完整结构应为:\n"
                        f"    'function': {{\n"
                        f'        "name": "my_func",\n'
                        f'        "input_field_names": ["text"],\n'
                        f'        "function_type": FunctionType.BM25,\n'
                        f"    }}"
                    )


def _parse_hits(hits: list[list[dict]]) -> list[VectorSearchContent]:
    """将 pymilvus search/hybrid_search 返回的原始 hits 转换为 VectorSearchContent 列表。"""
    results: list[VectorSearchContent] = []
    if not hits:
        return results
    for hit in hits[0]:
        record = dict(hit.get("entity", {}))
        record["id"] = hit.get("id")
        results.append(
            VectorSearchContent(
                record=record,
                score=float(hit.get("distance", 0.0)),
            )
        )
    return results


class AsyncMilvusStorage:
    """异步 Milvus 存储类"""

    def __init__(
        self,
        uri: str = "http://localhost:19530",
        user: str = "",
        password: str = "",
        db_name: str = "",
        **kwargs,
    ) -> None:
        self.db_name = db_name
        self.client = AsyncMilvusClient(
            uri=uri, user=user, password=password, db_name=db_name, **kwargs
        )

    async def __aenter__(self) -> "AsyncMilvusStorage":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()

    async def close(self) -> None:
        """关闭与 Milvus 的连接。"""
        await self.client.close()

    # 数据库管理
    async def create_db(self, timeout: float | None = None, **kwargs) -> bool:
        """创建 Milvus 数据库（幂等，已存在时直接返回 True）。

        Args:
            timeout: 请求超时时间（秒）

        Returns:
            bool: 成功返回 True，失败返回 False
        """
        target_db = self.db_name
        try:
            exist_dbs = await self.list_dbs()
            if target_db in exist_dbs:
                logger.debug(f"Milvus database '{target_db}' already exists.")
                return True

            await self.client.create_database(db_name=target_db, timeout=timeout, **kwargs)
            logger.info(f"Successfully created Milvus database: '{target_db}'")
            return True

        except MilvusException as e:
            if "already exists" in str(e).lower() or e.code == 65535:
                logger.debug(f"Milvus database '{target_db}' was created concurrently.")
                return True
            logger.error(f"Failed to create Milvus database '{target_db}': {e}", exc_info=True)
            return False
        except Exception as e:
            logger.error(
                f"Unexpected error while creating database '{target_db}': {e}", exc_info=True
            )
            return False

    async def list_dbs(self, timeout: float | None = None, **kwargs) -> list[str]:
        """列出所有 Milvus 数据库名称。

        Returns:
            list[str]: 数据库名称列表，异常时返回 []
        """
        try:
            return await self.client.list_databases(timeout=timeout, **kwargs)
        except MilvusException as e:
            logger.error(f"MilvusException while listing databases: {e}", exc_info=True)
            return []
        except Exception as e:
            logger.error(f"Unexpected error while listing databases: {e}", exc_info=True)
            return []

    # 集合管理
    async def has_collection(self, collection: str, timeout: float | None = None, **kwargs) -> bool:
        """检查集合是否存在。

        Returns:
            bool: 存在返回 True，不存在或异常返回 False
        """
        try:
            return await self.client.has_collection(
                collection_name=collection, timeout=timeout, **kwargs
            )
        except MilvusException as e:
            logger.error(
                f"MilvusException while checking collection '{collection}': {e}", exc_info=True
            )
            return False
        except Exception as e:
            logger.error(
                f"Unexpected error while checking collection '{collection}': {e}", exc_info=True
            )
            return False

    async def list_collections(self, timeout: float | None = None, **kwargs) -> list[str]:
        """列出当前数据库下的所有集合名称。

        Returns:
            list[str]: 集合名称列表，异常时返回 []
        """
        try:
            return await self.client.list_collections(timeout=timeout, **kwargs)
        except MilvusException as e:
            logger.error(f"MilvusException while listing collections: {e}", exc_info=True)
            return []
        except Exception as e:
            logger.error(f"Unexpected error while listing collections: {e}", exc_info=True)
            return []

    async def describe_collection(
        self,
        collection: str,
        timeout: float | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """获取集合的 Schema 及元数据信息。

        Returns:
            dict: 集合详情，包含字段、索引、一致性级别等信息

        Raises:
            MilvusError: 集合不存在或集群异常时抛出
        """
        try:
            return await self.client.describe_collection(
                collection_name=collection, timeout=timeout, **kwargs
            )
        except MilvusException as e:
            logger.error(
                f"MilvusException while describing collection '{collection}': {e}", exc_info=True
            )
            raise MilvusError(f"Failed to describe collection '{collection}'.") from e
        except Exception as e:
            logger.error(
                f"Unexpected error while describing collection '{collection}': {e}", exc_info=True
            )
            raise MilvusError(
                f"Unexpected exception in {self.__class__.__name__}.describe_collection"
            ) from e

    async def get_collection_stats(
        self,
        collection: str,
        timeout: float | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """获取集合统计信息（行数等）。

        Returns:
            dict: 例如 {"row_count": 1000}

        Raises:
            MilvusError: 集合不存在或集群异常时抛出
        """
        try:
            return await self.client.get_collection_stats(
                collection_name=collection, timeout=timeout, **kwargs
            )
        except MilvusException as e:
            logger.error(
                f"MilvusException while getting stats for '{collection}': {e}", exc_info=True
            )
            raise MilvusError(f"Failed to get stats for collection '{collection}'.") from e
        except Exception as e:
            logger.error(
                f"Unexpected error while getting stats for '{collection}': {e}", exc_info=True
            )
            raise MilvusError(
                f"Unexpected exception in {self.__class__.__name__}.get_collection_stats"
            ) from e

    async def get_load_state(
        self,
        collection: str,
        partition_names: list[str] | None = None,
        timeout: float | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """获取集合（或分区）的加载状态。

        Returns:
            dict: 例如 {"state": LoadState.Loaded} 或 {"state": LoadState.Loading, "progress": 80}

        Raises:
            MilvusError: 集合不存在或集群异常时抛出
        """
        try:
            return await self.client.get_load_state(
                collection_name=collection,
                partition_names=partition_names,
                timeout=timeout,
                **kwargs,
            )
        except MilvusException as e:
            logger.error(
                f"MilvusException while getting load state for '{collection}': {e}", exc_info=True
            )
            raise MilvusError(f"Failed to get load state for collection '{collection}'.") from e
        except Exception as e:
            logger.error(
                f"Unexpected error while getting load state for '{collection}': {e}", exc_info=True
            )
            raise MilvusError(
                f"Unexpected exception in {self.__class__.__name__}.get_load_state"
            ) from e

    async def load_collection(
        self,
        collection: str,
        timeout: float | None = None,
        **kwargs,
    ) -> None:
        """将集合加载到内存中以供查询。

        Raises:
            MilvusError: 加载失败时抛出
        """
        try:
            await self.client.load_collection(collection_name=collection, timeout=timeout, **kwargs)
            logger.info(f"Collection '{collection}' loaded successfully.")
        except MilvusException as e:
            logger.error(
                f"MilvusException while loading collection '{collection}': {e}", exc_info=True
            )
            raise MilvusError(f"Failed to load collection '{collection}'.") from e
        except Exception as e:
            logger.error(
                f"Unexpected error while loading collection '{collection}': {e}", exc_info=True
            )
            raise MilvusError(
                f"Unexpected exception in {self.__class__.__name__}.load_collection"
            ) from e

    async def release_collection(
        self,
        collection: str,
        timeout: float | None = None,
        **kwargs,
    ) -> None:
        """将集合从内存中释放以节省资源。

        Raises:
            MilvusError: 释放失败时抛出
        """
        try:
            await self.client.release_collection(
                collection_name=collection, timeout=timeout, **kwargs
            )
            logger.info(f"Collection '{collection}' released successfully.")
        except MilvusException as e:
            logger.error(
                f"MilvusException while releasing collection '{collection}': {e}", exc_info=True
            )
            raise MilvusError(f"Failed to release collection '{collection}'.") from e
        except Exception as e:
            logger.error(
                f"Unexpected error while releasing collection '{collection}': {e}", exc_info=True
            )
            raise MilvusError(
                f"Unexpected exception in {self.__class__.__name__}.release_collection"
            ) from e

    async def create_collection(
        self,
        collection: str,
        field_model: type[BaseVectorRecord],
        timeout: float | None = None,
    ) -> bool:
        """根据 Pydantic Model 的字段定义动态创建并自动加载 Milvus 集合。

        Args:
            collection: 集合名称
            field_model: 继承自 BaseVectorRecord 的 Pydantic 模型类
            timeout: 请求超时时间（秒）

        Returns:
            bool: 创建成功返回 True

        Raises:
            MilvusError: 集合已存在或集群异常时抛出
        """
        target_db = self.db_name
        current_dbs = await self.list_dbs()
        if target_db not in current_dbs:
            await self.create_db()

        if await self.has_collection(collection, timeout=timeout):
            raise MilvusError(
                f"{self.__class__.__name__}.create_collection | "
                f"Collection '{collection}' already exists."
            )

        try:
            schema = self.client.create_schema(**field_model.schema_config)
            index_params = self.client.prepare_index_params()

            for field_name, field_info in field_model.model_fields.items():
                extra_raw = field_info.json_schema_extra
                extra: dict[str, Any] = extra_raw.copy() if isinstance(extra_raw, dict) else {}

                index_cfg: dict | None = extra.pop("index", None)
                func_cfg: dict | None = extra.pop("function", None)

                schema.add_field(field_name=field_name, **extra)

                if index_cfg:
                    index_params.add_index(field_name=field_name, **index_cfg)

                if func_cfg:
                    schema.add_function(
                        Function(
                            name=func_cfg["name"],
                            input_field_names=func_cfg["input_field_names"],
                            output_field_names=[field_name],
                            function_type=func_cfg["function_type"],
                        )
                    )

            await self.client.create_collection(
                collection_name=collection,
                schema=schema,
                index_params=index_params,
                timeout=timeout,
            )
            logger.info(
                f"Successfully created collection '{collection}' via {field_model.__name__}."
            )
            return True

        except MilvusException as e:
            if "already exists" in str(e).lower():
                logger.debug(f"Collection '{collection}' was concurrently created, skipping.")
                return True
            logger.error(f"Milvus error creating collection '{collection}'", exc_info=True)
            raise MilvusError(f"Milvus cluster failure during creating collection: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error creating collection '{collection}'", exc_info=True)
            raise MilvusError(
                f"Unexpected exception in {self.__class__.__name__}.create_collection"
            ) from e

    async def drop_collection(
        self,
        collection: str,
        timeout: float | None = None,
        **kwargs,
    ) -> None:
        """删除集合（幂等，不存在时静默跳过）。

        Raises:
            MilvusError: 权限不足或集群异常时抛出
        """
        try:
            await self.client.drop_collection(collection_name=collection, timeout=timeout, **kwargs)
            logger.info(f"Dropped collection '{collection}'.")
        except MilvusException as e:
            err_str = str(e).lower()
            if "not found" in err_str or "can't find" in err_str:
                logger.debug(f"Collection '{collection}' does not exist, skip drop.")
                return
            logger.error(f"MilvusException while dropping collection '{collection}'", exc_info=True)
            raise MilvusError(f"Failed to drop collection '{collection}'.") from e
        except Exception as e:
            logger.error(
                f"Unexpected error while dropping collection '{collection}'", exc_info=True
            )
            raise MilvusError(
                f"Unexpected exception in {self.__class__.__name__}.drop_collection"
            ) from e

    async def clear_collection(
        self,
        collection: str,
        field_model: type[BaseVectorRecord],
        timeout: float | None = None,
    ) -> bool:
        """彻底清空集合（先删除再重建，速度远快于逐行删除）。

        Args:
            collection: 集合名称
            field_model: 用于重建的 Pydantic 模型类
            timeout: 超时时间（秒）

        Returns:
            bool: 清空并重建成功返回 True

        Raises:
            MilvusError: 删除或重建失败时抛出
        """
        logger.warning(f"Clearing all data in collection '{collection}'.")
        try:
            await self.drop_collection(collection=collection, timeout=timeout)
            return await self.create_collection(
                collection=collection, field_model=field_model, timeout=timeout
            )
        except Exception as e:
            logger.error(f"Failed to clear collection '{collection}'", exc_info=True)
            raise MilvusError(f"Critical error during clearing collection '{collection}'") from e

    # 分区管理
    async def list_partitions(
        self,
        collection: str,
        timeout: float | None = None,
        **kwargs,
    ) -> list[str]:
        """列出集合下所有分区名称。

        Returns:
            list[str]: 分区名称列表，异常时返回 []
        """
        try:
            return await self.client.list_partitions(
                collection_name=collection, timeout=timeout, **kwargs
            )
        except MilvusException as e:
            logger.error(
                f"MilvusException while listing partitions for '{collection}': {e}", exc_info=True
            )
            return []
        except Exception as e:
            logger.error(
                f"Unexpected error while listing partitions for '{collection}': {e}", exc_info=True
            )
            return []

    async def create_partition(
        self,
        collection: str,
        partition: str,
        timeout: float | None = None,
        **kwargs,
    ) -> bool:
        """在集合中创建分区（幂等，已存在时直接返回 True）。

        Returns:
            bool: 创建（或已存在）成功返回 True

        Raises:
            MilvusError: 权限不足或集群异常时抛出
        """
        try:
            existing = await self.list_partitions(collection=collection, timeout=timeout)
            if partition in existing:
                logger.debug(f"Partition '{partition}' already exists in '{collection}'.")
                return True

            await self.client.create_partition(
                collection_name=collection,
                partition_name=partition,
                timeout=timeout,
                **kwargs,
            )
            logger.info(f"Created partition '{partition}' in '{collection}'.")
            return True

        except MilvusException as e:
            err_str = str(e).lower()
            if "already exists" in err_str or "duplicate" in err_str:
                logger.debug(f"Partition '{partition}' concurrently created in '{collection}'.")
                return True
            logger.error(
                f"MilvusException while creating partition '{partition}' in '{collection}': {e}",
                exc_info=True,
            )
            raise MilvusError(f"Failed to create partition '{partition}' in '{collection}'.") from e
        except Exception as e:
            logger.error(
                f"Unexpected error while creating partition '{partition}' in '{collection}': {e}",
                exc_info=True,
            )
            raise MilvusError(
                f"Unexpected exception in {self.__class__.__name__}.create_partition"
            ) from e

    async def drop_partition(
        self,
        collection: str,
        partition: str,
        timeout: float | None = None,
        **kwargs,
    ) -> None:
        """删除集合下的指定分区（幂等，不存在时静默跳过）。

        Raises:
            MilvusError: 权限不足或集群异常时抛出
        """
        try:
            await self.client.drop_partition(
                collection_name=collection,
                partition_name=partition,
                timeout=timeout,
                **kwargs,
            )
            logger.info(f"Dropped partition '{partition}' from '{collection}'.")
        except MilvusException as e:
            err_str = str(e).lower()
            if "not found" in err_str or "can't find" in err_str:
                logger.debug(
                    f"Partition '{partition}' or collection '{collection}' does not exist, skip drop."
                )
                return
            logger.error(
                f"MilvusException while dropping partition '{partition}' from '{collection}'",
                exc_info=True,
            )
            raise MilvusError(f"Failed to drop partition '{partition}'.") from e
        except Exception as e:
            logger.error(
                f"Unexpected error while dropping partition '{partition}' from '{collection}'",
                exc_info=True,
            )
            raise MilvusError(
                f"Unexpected exception in {self.__class__.__name__}.drop_partition"
            ) from e

    async def clear_partition(
        self,
        collection: str,
        partition: str,
        timeout: float | None = None,
    ) -> bool:
        """彻底清空分区（先删除再重建）。

        Returns:
            bool: 成功返回 True

        Raises:
            MilvusError: 失败时抛出
        """
        logger.warning(f"Clearing partition '{partition}' in '{collection}'.")
        try:
            await self.drop_partition(collection=collection, partition=partition, timeout=timeout)
            return await self.create_partition(
                collection=collection, partition=partition, timeout=timeout
            )
        except Exception as e:
            logger.error(
                f"Failed to clear partition '{partition}' in '{collection}'", exc_info=True
            )
            raise MilvusError(f"Critical error during clearing partition '{partition}'") from e

    # 数据写入
    async def insert(
        self,
        collection: str,
        data: list[dict[str, Any]],
        partition: str | None = None,
        timeout: float | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """向集合插入数据。

        Args:
            collection: 集合名称
            data: 数据列表，每项为字段名→值的字典
            partition: 目标分区，None 则写入默认分区
            timeout: 超时时间（秒）

        Returns:
            dict: 例如 {"insert_count": 10, "ids": [...]}

        Raises:
            MilvusError: Schema 不匹配或集群异常时抛出
        """
        if not data:
            logger.warning(f"Empty data passed to {self.__class__.__name__}.insert, skipping.")
            return {"insert_count": 0, "ids": []}

        try:
            result = await self.client.insert(
                collection_name=collection,
                data=data,
                partition_name=partition,
                timeout=timeout,
                **kwargs,
            )
            logger.debug(
                f"Inserted {result.get('insert_count', 0)} rows into '{collection}' "
                f"(partition: '{partition or 'default'}')."
            )
            return dict(result)
        except MilvusException as e:
            logger.error(f"MilvusException during insert into '{collection}': {e}", exc_info=True)
            raise MilvusError(f"Insert failed for collection '{collection}'.") from e
        except Exception as e:
            logger.error(f"Unexpected error during insert into '{collection}': {e}", exc_info=True)
            raise MilvusError(f"Unexpected exception in {self.__class__.__name__}.insert") from e

    async def upsert(
        self,
        collection: str,
        data: list[dict[str, Any]],
        partition: str | None = None,
        timeout: float | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """向集合插入或更新数据（主键存在则更新，不存在则插入）。

        Args:
            collection: 集合名称
            data: 数据列表，每项必须包含主键字段
            partition: 目标分区，None 则写入默认分区
            timeout: 超时时间（秒）

        Returns:
            dict: 例如 {"upsert_count": 10, "ids": [...]}

        Raises:
            MilvusError: Schema 不匹配或集群异常时抛出
        """
        if not data:
            logger.warning(f"Empty data passed to {self.__class__.__name__}.upsert, skipping.")
            return {"upsert_count": 0, "ids": []}

        try:
            result = await self.client.upsert(
                collection_name=collection,
                data=data,
                partition_name=partition or "",
                timeout=timeout,
                **kwargs,
            )
            logger.debug(
                f"Upserted {result.get('upsert_count', 0)} rows into '{collection}' "
                f"(partition: '{partition or 'default'}')."
            )
            return dict(result)
        except MilvusException as e:
            logger.error(f"MilvusException during upsert into '{collection}': {e}", exc_info=True)
            raise MilvusError(f"Upsert failed for collection '{collection}'.") from e
        except Exception as e:
            logger.error(f"Unexpected error during upsert into '{collection}': {e}", exc_info=True)
            raise MilvusError(f"Unexpected exception in {self.__class__.__name__}.upsert") from e

    # 数据删除
    async def delete(
        self,
        collection: str,
        filter: str,
        partition: str | None = None,
        timeout: float | None = None,
        **kwargs,
    ) -> int:
        """按布尔表达式删除满足条件的记录。

        Args:
            collection: 集合名称
            filter: 过滤表达式，例如 "id in [1,2,3]" 或 "source == 'pdf'"
            partition: 分区名称，None 则全表范围删除
            timeout: 超时时间（秒）

        Returns:
            int: 受影响的记录数

        Raises:
            MilvusError: 表达式语法错误或集群异常时抛出
        """
        if not filter or not filter.strip():
            logger.warning(f"Empty filter passed to {self.__class__.__name__}.delete, skipping.")
            return 0

        try:
            result: Any = await self.client.delete(
                collection_name=collection,
                filter=filter,
                partition_name=partition,
                timeout=timeout,
                **kwargs,
            )

            if isinstance(result, list):
                delete_count = len(result)
            elif isinstance(result, dict) or hasattr(result, "get"):
                delete_count = result.get("delete_count", 0)
            elif hasattr(result, "delete_count"):
                delete_count = getattr(result, "delete_count", 0)
            else:
                delete_count = 0

            logger.info(
                f"Deleted {delete_count} rows from '{collection}' "
                f"(partition: '{partition or 'all'}')."
            )
            return delete_count

        except MilvusException as e:
            logger.error(f"MilvusException during delete from '{collection}': {e}", exc_info=True)
            raise MilvusError(f"Delete failed for collection '{collection}'.") from e
        except Exception as e:
            logger.error(f"Unexpected error during delete from '{collection}': {e}", exc_info=True)
            raise MilvusError(f"Unexpected exception in {self.__class__.__name__}.delete") from e

    async def delete_by_ids(
        self,
        collection: str,
        ids: list[int] | list[str] | int | str,
        partition: str | None = None,
        timeout: float | None = None,
        **kwargs,
    ) -> int:
        """按主键 ID 删除指定记录。

        Args:
            collection: 集合名称
            ids: 主键 ID 或 ID 列表
            partition: 分区名称，None 则全表范围
            timeout: 超时时间（秒）

        Returns:
            int: 受影响的记录数

        Raises:
            MilvusError: 集群异常时抛出
        """
        try:
            result: Any = await self.client.delete(
                collection_name=collection,
                ids=ids,
                partition_name=partition,
                timeout=timeout,
                **kwargs,
            )

            if isinstance(result, list):
                delete_count = len(result)
            elif isinstance(result, dict) or hasattr(result, "get"):
                delete_count = result.get("delete_count", 0)
            else:
                delete_count = 0

            logger.info(f"Deleted {delete_count} rows by ID from '{collection}'.")
            return delete_count

        except MilvusException as e:
            logger.error(
                f"MilvusException during delete_by_ids from '{collection}': {e}", exc_info=True
            )
            raise MilvusError(f"Delete by IDs failed for collection '{collection}'.") from e
        except Exception as e:
            logger.error(
                f"Unexpected error during delete_by_ids from '{collection}': {e}", exc_info=True
            )
            raise MilvusError(
                f"Unexpected exception in {self.__class__.__name__}.delete_by_ids"
            ) from e

    # 数据查询
    async def get(
        self,
        collection: str,
        ids: list[int] | list[str] | int | str,
        output_fields: list[str] | None = None,
        partition_names: list[str] | None = None,
        timeout: float | None = None,
        **kwargs,
    ) -> list[VectorQueryContent]:
        """按主键 ID 精确查询记录（不涉及向量检索）。

        Args:
            collection: 集合名称
            ids: 主键 ID 或 ID 列表
            output_fields: 需要返回的字段列表，None 返回所有字段
            partition_names: 限定查询的分区列表，None 则全表
            timeout: 超时时间（秒）

        Returns:
            list[VectorQueryContent]: 查询结果列表

        Raises:
            MilvusError: 集群异常时抛出
        """
        try:
            hits = await self.client.get(
                collection_name=collection,
                ids=ids,
                output_fields=output_fields,
                partition_names=partition_names,
                timeout=timeout,
                **kwargs,
            )
            return [VectorQueryContent(record=hit) for hit in hits]
        except MilvusException as e:
            logger.error(f"MilvusException during get from '{collection}': {e}", exc_info=True)
            raise MilvusError(f"Get by IDs failed for collection '{collection}'.") from e
        except Exception as e:
            logger.error(f"Unexpected error during get from '{collection}': {e}", exc_info=True)
            raise MilvusError(f"Unexpected exception in {self.__class__.__name__}.get") from e

    async def query(
        self,
        collection: str,
        filter: str = "",
        output_fields: list[str] | None = None,
        partition_names: list[str] | None = None,
        ids: list[int] | list[str] | int | str | None = None,
        limit: int | None = None,
        timeout: float | None = None,
        **kwargs,
    ) -> list[VectorQueryContent]:
        """按标量过滤条件查询记录（结构化查询，不涉及向量检索）。

        Args:
            collection: 集合名称
            filter: 过滤表达式，例如 "category == 'news' and score > 0.8"
            output_fields: 需要返回的字段列表，None 返回所有字段
            partition_names: 限定查询的分区列表，None 则全表
            ids: 按主键 ID 查询（与 filter 互斥）
            limit: 最大返回条数，None 不限制
            timeout: 超时时间（秒）

        Returns:
            list[VectorQueryContent]: 查询结果列表

        Raises:
            MilvusError: 表达式语法错误或集群异常时抛出
        """
        if limit is not None:
            kwargs.setdefault("limit", limit)

        try:
            hits = await self.client.query(
                collection_name=collection,
                filter=filter,
                output_fields=output_fields,
                partition_names=partition_names,
                ids=ids,
                timeout=timeout,
                **kwargs,
            )
            return [VectorQueryContent(record=hit) for hit in hits]
        except MilvusException as e:
            logger.error(f"MilvusException during query on '{collection}': {e}", exc_info=True)
            raise MilvusError(f"Query failed for collection '{collection}'.") from e
        except Exception as e:
            logger.error(f"Unexpected error during query on '{collection}': {e}", exc_info=True)
            raise MilvusError(f"Unexpected exception in {self.__class__.__name__}.query") from e

    async def search(
        self,
        collection: str,
        data: list[list[float]] | list[float] | None = None,
        filter: str | None = None,
        limit: int = 10,
        output_fields: list[str] | None = None,
        partition_names: list[str] | None = None,
        anns_field: str | None = None,
        search_params: dict[str, Any] | None = None,
        ids: list[int] | list[str] | int | str | None = None,
        timeout: float | None = None,
        **kwargs,
    ) -> list[VectorSearchContent]:
        """向量相似度检索。

        Args:
            collection: 集合名称
            data: 查询向量。单向量 list[float] 或批量 list[list[float]]；与 ids 互斥
            filter: 标量前置过滤表达式，例如 "source == 'pdf'"
            limit: 返回的最相似结果数量上限
            output_fields: 随结果返回的标量字段，None 则仅返回主键和分数
            partition_names: 限定检索的分区列表，None 则全表
            anns_field: 向量字段名；Schema 中有多个向量字段时必须指定
            search_params: 索引微调参数，例如 {"metric_type": "COSINE", "params": {"ef": 64}}
            ids: 按主键 ID 检索相似项（与 data 互斥）
            timeout: 超时时间（秒）

        Returns:
            list[VectorSearchContent]: 按相似度由高到低排序的结果列表（取自第一路查询）

        Raises:
            MilvusError: 向量维度不匹配、表达式错误或集群异常时抛出
        """
        try:
            hits = await self.client.search(
                collection_name=collection,
                data=data,
                filter=filter or "",
                limit=limit,
                output_fields=output_fields,
                partition_names=partition_names,
                anns_field=anns_field,
                search_params=search_params or {},
                ids=ids,
                timeout=timeout,
                **kwargs,
            )
            return _parse_hits(hits)

        except MilvusException as e:
            logger.error(f"MilvusException during search on '{collection}': {e}", exc_info=True)
            raise MilvusError(f"Vector search failed on collection '{collection}'.") from e
        except Exception as e:
            logger.error(f"Unexpected error during search on '{collection}': {e}", exc_info=True)
            raise MilvusError(f"Unexpected exception in {self.__class__.__name__}.search") from e

    async def hybrid_search(
        self,
        collection: str,
        requests: list[AnnSearchRequest],
        ranker: BaseRanker | Function,
        limit: int = 10,
        output_fields: list[str] | None = None,
        partition_names: list[str] | None = None,
        timeout: float | None = None,
        **kwargs,
    ) -> list[VectorSearchContent]:
        """多路混合检索（稠密 + 稀疏 BM25 等），并通过 Ranker 进行重排序。

        Args:
            collection: 集合名称
            requests: AnnSearchRequest 列表，每路对应一个向量字段的检索请求
            ranker: 重排序策略，例如 RRFRanker() 或 WeightedRanker(0.7, 0.3)
            limit: 最终返回的结果数量上限
            output_fields: 随结果返回的标量字段列表
            partition_names: 限定检索的分区列表，None 则全表
            timeout: 超时时间（秒）

        Returns:
            list[VectorSearchContent]: 重排序后按分数由高到低排序的结果列表

        Raises:
            MilvusError: 参数错误或集群异常时抛出
        """
        try:
            hits = await self.client.hybrid_search(
                collection_name=collection,
                reqs=requests,
                ranker=ranker,
                limit=limit,
                output_fields=output_fields,
                partition_names=partition_names,
                timeout=timeout,
                **kwargs,
            )
            return _parse_hits(hits)

        except MilvusException as e:
            logger.error(
                f"MilvusException during hybrid_search on '{collection}': {e}", exc_info=True
            )
            raise MilvusError(f"Hybrid search failed on collection '{collection}'.") from e
        except Exception as e:
            logger.error(
                f"Unexpected error during hybrid_search on '{collection}': {e}", exc_info=True
            )
            raise MilvusError(
                f"Unexpected exception in {self.__class__.__name__}.hybrid_search"
            ) from e


class MilvusStorage:
    """同步 Milvus 存储类（与 AsyncMilvusStorage 接口一致）。"""

    def __init__(
        self,
        uri: str = "http://localhost:19530",
        user: str = "",
        password: str = "",
        db_name: str = "",
        **kwargs,
    ) -> None:
        self.db_name = db_name
        self.client = MilvusClient(uri=uri, user=user, password=password, db_name=db_name, **kwargs)

    def __enter__(self) -> "MilvusStorage":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def close(self) -> None:
        """关闭与 Milvus 的连接。"""
        self.client.close()

    # 数据库管理
    def create_db(self, timeout: float | None = None, **kwargs) -> bool:
        """创建 Milvus 数据库（幂等，已存在时直接返回 True）。"""
        target_db = self.db_name
        try:
            exist_dbs = self.list_dbs()
            if target_db in exist_dbs:
                logger.debug(f"Milvus database '{target_db}' already exists.")
                return True

            self.client.create_database(db_name=target_db, timeout=timeout, **kwargs)
            logger.info(f"Successfully created Milvus database: '{target_db}'")
            return True

        except MilvusException as e:
            if "already exists" in str(e).lower() or e.code == 65535:
                logger.debug(f"Milvus database '{target_db}' was created concurrently.")
                return True
            logger.error(f"Failed to create Milvus database '{target_db}': {e}", exc_info=True)
            return False
        except Exception as e:
            logger.error(
                f"Unexpected error while creating database '{target_db}': {e}", exc_info=True
            )
            return False

    def list_dbs(self, timeout: float | None = None, **kwargs) -> list[str]:
        """列出所有 Milvus 数据库名称。"""
        try:
            return cast(list[str], self.client.list_databases(timeout=timeout, **kwargs))
        except MilvusException as e:
            logger.error(f"MilvusException while listing databases: {e}", exc_info=True)
            return []
        except Exception as e:
            logger.error(f"Unexpected error while listing databases: {e}", exc_info=True)
            return []

    # 集合管理
    def has_collection(self, collection: str, timeout: float | None = None, **kwargs) -> bool:
        """检查集合是否存在。"""
        try:
            return bool(
                self.client.has_collection(collection_name=collection, timeout=timeout, **kwargs)
            )
        except MilvusException as e:
            logger.error(
                f"MilvusException while checking collection '{collection}': {e}", exc_info=True
            )
            return False
        except Exception as e:
            logger.error(
                f"Unexpected error while checking collection '{collection}': {e}", exc_info=True
            )
            return False

    def list_collections(self, timeout: float | None = None, **kwargs) -> list[str]:
        """列出当前数据库下的所有集合名称。"""
        try:
            return cast(list[str], self.client.list_collections(timeout=timeout, **kwargs))
        except MilvusException as e:
            logger.error(f"MilvusException while listing collections: {e}", exc_info=True)
            return []
        except Exception as e:
            logger.error(f"Unexpected error while listing collections: {e}", exc_info=True)
            return []

    def describe_collection(
        self,
        collection: str,
        timeout: float | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """获取集合的 Schema 及元数据信息。

        Raises:
            MilvusError: 集合不存在或集群异常时抛出
        """
        try:
            return cast(
                dict[str, Any],
                self.client.describe_collection(
                    collection_name=collection, timeout=timeout, **kwargs
                ),
            )
        except MilvusException as e:
            logger.error(
                f"MilvusException while describing collection '{collection}': {e}", exc_info=True
            )
            raise MilvusError(f"Failed to describe collection '{collection}'.") from e
        except Exception as e:
            logger.error(
                f"Unexpected error while describing collection '{collection}': {e}", exc_info=True
            )
            raise MilvusError(
                f"Unexpected exception in {self.__class__.__name__}.describe_collection"
            ) from e

    def get_collection_stats(
        self,
        collection: str,
        timeout: float | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """获取集合统计信息（如行数）。

        Raises:
            MilvusError: 集群异常时抛出
        """
        try:
            return cast(
                dict[str, Any],
                self.client.get_collection_stats(
                    collection_name=collection, timeout=timeout, **kwargs
                ),
            )
        except MilvusException as e:
            logger.error(
                f"MilvusException while getting stats for '{collection}': {e}", exc_info=True
            )
            raise MilvusError(f"Failed to get stats for collection '{collection}'.") from e
        except Exception as e:
            logger.error(
                f"Unexpected error while getting stats for '{collection}': {e}", exc_info=True
            )
            raise MilvusError(
                f"Unexpected exception in {self.__class__.__name__}.get_collection_stats"
            ) from e

    def get_load_state(
        self,
        collection: str,
        partition_names: list[str] | None = None,
        timeout: float | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """获取集合（或分区）的加载状态。

        Raises:
            MilvusError: 集群异常时抛出
        """
        try:
            return cast(
                dict[str, Any],
                self.client.get_load_state(
                    collection_name=collection,
                    partition_names=partition_names,
                    timeout=timeout,
                    **kwargs,
                ),
            )
        except MilvusException as e:
            logger.error(
                f"MilvusException while getting load state for '{collection}': {e}", exc_info=True
            )
            raise MilvusError(f"Failed to get load state for collection '{collection}'.") from e
        except Exception as e:
            logger.error(
                f"Unexpected error while getting load state for '{collection}': {e}", exc_info=True
            )
            raise MilvusError(
                f"Unexpected exception in {self.__class__.__name__}.get_load_state"
            ) from e

    def load_collection(
        self,
        collection: str,
        timeout: float | None = None,
        **kwargs,
    ) -> None:
        """将集合加载到内存中以供查询。

        Raises:
            MilvusError: 加载失败时抛出
        """
        try:
            self.client.load_collection(collection_name=collection, timeout=timeout, **kwargs)
            logger.info(f"Collection '{collection}' loaded successfully.")
        except MilvusException as e:
            logger.error(
                f"MilvusException while loading collection '{collection}': {e}", exc_info=True
            )
            raise MilvusError(f"Failed to load collection '{collection}'.") from e
        except Exception as e:
            logger.error(
                f"Unexpected error while loading collection '{collection}': {e}", exc_info=True
            )
            raise MilvusError(
                f"Unexpected exception in {self.__class__.__name__}.load_collection"
            ) from e

    def release_collection(
        self,
        collection: str,
        timeout: float | None = None,
        **kwargs,
    ) -> None:
        """将集合从内存中释放以节省资源。

        Raises:
            MilvusError: 释放失败时抛出
        """
        try:
            self.client.release_collection(collection_name=collection, timeout=timeout, **kwargs)
            logger.info(f"Collection '{collection}' released successfully.")
        except MilvusException as e:
            logger.error(
                f"MilvusException while releasing collection '{collection}': {e}", exc_info=True
            )
            raise MilvusError(f"Failed to release collection '{collection}'.") from e
        except Exception as e:
            logger.error(
                f"Unexpected error while releasing collection '{collection}': {e}", exc_info=True
            )
            raise MilvusError(
                f"Unexpected exception in {self.__class__.__name__}.release_collection"
            ) from e

    def create_collection(
        self,
        collection: str,
        field_model: type[BaseVectorRecord],
        timeout: float | None = None,
    ) -> bool:
        """根据 Pydantic Model 的字段定义动态创建并自动加载 Milvus 集合。

        Raises:
            MilvusError: 集合已存在或集群异常时抛出
        """
        target_db = self.db_name
        current_dbs = self.list_dbs()
        if target_db not in current_dbs:
            self.create_db()

        if self.has_collection(collection, timeout=timeout):
            raise MilvusError(
                f"{self.__class__.__name__}.create_collection | "
                f"Collection '{collection}' already exists."
            )

        try:
            schema = self.client.create_schema(**field_model.schema_config)
            index_params = self.client.prepare_index_params()

            for field_name, field_info in field_model.model_fields.items():
                extra_raw = field_info.json_schema_extra
                extra: dict[str, Any] = extra_raw.copy() if isinstance(extra_raw, dict) else {}

                index_cfg: dict | None = extra.pop("index", None)
                func_cfg: dict | None = extra.pop("function", None)

                schema.add_field(field_name=field_name, **extra)

                if index_cfg:
                    index_params.add_index(field_name=field_name, **index_cfg)

                if func_cfg:
                    schema.add_function(
                        Function(
                            name=func_cfg["name"],
                            input_field_names=func_cfg["input_field_names"],
                            output_field_names=[field_name],
                            function_type=func_cfg["function_type"],
                        )
                    )

            self.client.create_collection(
                collection_name=collection,
                schema=schema,
                index_params=index_params,
                timeout=timeout,
            )
            logger.info(
                f"Successfully created collection '{collection}' via {field_model.__name__}."
            )
            return True

        except MilvusException as e:
            if "already exists" in str(e).lower():
                logger.debug(f"Collection '{collection}' was concurrently created, skipping.")
                return True
            logger.error(f"Milvus error creating collection '{collection}'", exc_info=True)
            raise MilvusError(f"Milvus cluster failure during creating collection: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error creating collection '{collection}'", exc_info=True)
            raise MilvusError(
                f"Unexpected exception in {self.__class__.__name__}.create_collection"
            ) from e

    def drop_collection(
        self,
        collection: str,
        timeout: float | None = None,
        **kwargs,
    ) -> None:
        """删除集合（幂等，不存在时静默跳过）。

        Raises:
            MilvusError: 权限不足或集群异常时抛出
        """
        try:
            self.client.drop_collection(collection_name=collection, timeout=timeout, **kwargs)
            logger.info(f"Dropped collection '{collection}'.")
        except MilvusException as e:
            err_str = str(e).lower()
            if "not found" in err_str or "can't find" in err_str:
                logger.debug(f"Collection '{collection}' does not exist, skip drop.")
                return
            logger.error(f"MilvusException while dropping collection '{collection}'", exc_info=True)
            raise MilvusError(f"Failed to drop collection '{collection}'.") from e
        except Exception as e:
            logger.error(
                f"Unexpected error while dropping collection '{collection}'", exc_info=True
            )
            raise MilvusError(
                f"Unexpected exception in {self.__class__.__name__}.drop_collection"
            ) from e

    def clear_collection(
        self,
        collection: str,
        field_model: type[BaseVectorRecord],
        timeout: float | None = None,
    ) -> bool:
        """彻底清空集合（先删除再重建）。

        Raises:
            MilvusError: 失败时抛出
        """
        logger.warning(f"Clearing all data in collection '{collection}'.")
        try:
            self.drop_collection(collection=collection, timeout=timeout)
            return self.create_collection(
                collection=collection, field_model=field_model, timeout=timeout
            )
        except Exception as e:
            logger.error(f"Failed to clear collection '{collection}'", exc_info=True)
            raise MilvusError(f"Critical error during clearing collection '{collection}'") from e

    # 分区管理
    def list_partitions(
        self,
        collection: str,
        timeout: float | None = None,
        **kwargs,
    ) -> list[str]:
        """列出集合下所有分区名称。"""
        try:
            return cast(
                list[str],
                self.client.list_partitions(collection_name=collection, timeout=timeout, **kwargs),
            )
        except MilvusException as e:
            logger.error(
                f"MilvusException while listing partitions for '{collection}': {e}", exc_info=True
            )
            return []
        except Exception as e:
            logger.error(
                f"Unexpected error while listing partitions for '{collection}': {e}", exc_info=True
            )
            return []

    def create_partition(
        self,
        collection: str,
        partition: str,
        timeout: float | None = None,
        **kwargs,
    ) -> bool:
        """在集合中创建分区（幂等，已存在时直接返回 True）。

        Raises:
            MilvusError: 权限不足或集群异常时抛出
        """
        try:
            existing = self.list_partitions(collection=collection, timeout=timeout)
            if partition in existing:
                logger.debug(f"Partition '{partition}' already exists in '{collection}'.")
                return True

            self.client.create_partition(
                collection_name=collection,
                partition_name=partition,
                timeout=timeout,
                **kwargs,
            )
            logger.info(f"Created partition '{partition}' in '{collection}'.")
            return True

        except MilvusException as e:
            err_str = str(e).lower()
            if "already exists" in err_str or "duplicate" in err_str:
                logger.debug(f"Partition '{partition}' concurrently created in '{collection}'.")
                return True
            logger.error(
                f"MilvusException while creating partition '{partition}' in '{collection}': {e}",
                exc_info=True,
            )
            raise MilvusError(f"Failed to create partition '{partition}' in '{collection}'.") from e
        except Exception as e:
            logger.error(
                f"Unexpected error while creating partition '{partition}' in '{collection}': {e}",
                exc_info=True,
            )
            raise MilvusError(
                f"Unexpected exception in {self.__class__.__name__}.create_partition"
            ) from e

    def drop_partition(
        self,
        collection: str,
        partition: str,
        timeout: float | None = None,
        **kwargs,
    ) -> None:
        """删除集合下的指定分区（幂等，不存在时静默跳过）。

        Raises:
            MilvusError: 权限不足或集群异常时抛出
        """
        try:
            self.client.drop_partition(
                collection_name=collection,
                partition_name=partition,
                timeout=timeout,
                **kwargs,
            )
            logger.info(f"Dropped partition '{partition}' from '{collection}'.")
        except MilvusException as e:
            err_str = str(e).lower()
            if "not found" in err_str or "can't find" in err_str:
                logger.debug(
                    f"Partition '{partition}' or collection '{collection}' does not exist, skip."
                )
                return
            logger.error(
                f"MilvusException while dropping partition '{partition}' from '{collection}'",
                exc_info=True,
            )
            raise MilvusError(f"Failed to drop partition '{partition}'.") from e
        except Exception as e:
            logger.error(
                f"Unexpected error while dropping partition '{partition}' from '{collection}'",
                exc_info=True,
            )
            raise MilvusError(
                f"Unexpected exception in {self.__class__.__name__}.drop_partition"
            ) from e

    def clear_partition(
        self,
        collection: str,
        partition: str,
        timeout: float | None = None,
    ) -> bool:
        """彻底清空分区（先删除再重建）。

        Raises:
            MilvusError: 失败时抛出
        """
        logger.warning(f"Clearing partition '{partition}' in '{collection}'.")
        try:
            self.drop_partition(collection=collection, partition=partition, timeout=timeout)
            return self.create_partition(
                collection=collection, partition=partition, timeout=timeout
            )
        except Exception as e:
            logger.error(
                f"Failed to clear partition '{partition}' in '{collection}'", exc_info=True
            )
            raise MilvusError(f"Critical error during clearing partition '{partition}'") from e

    # 数据写入
    def insert(
        self,
        collection: str,
        data: list[dict[str, Any]],
        partition: str | None = None,
        timeout: float | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """向集合插入数据。

        Returns:
            dict: 例如 {"insert_count": 10, "ids": [...]}

        Raises:
            MilvusError: Schema 不匹配或集群异常时抛出
        """
        if not data:
            logger.warning(f"Empty data passed to {self.__class__.__name__}.insert, skipping.")
            return {"insert_count": 0, "ids": []}

        try:
            result = cast(
                dict[str, Any],
                self.client.insert(
                    collection_name=collection,
                    data=data,
                    partition_name=partition,
                    timeout=timeout,
                    **kwargs,
                ),
            )
            logger.debug(f"Inserted {result.get('insert_count', 0)} rows into '{collection}'.")
            return dict(result)
        except MilvusException as e:
            logger.error(f"MilvusException during insert into '{collection}': {e}", exc_info=True)
            raise MilvusError(f"Insert failed for collection '{collection}'.") from e
        except Exception as e:
            logger.error(f"Unexpected error during insert into '{collection}': {e}", exc_info=True)
            raise MilvusError(f"Unexpected exception in {self.__class__.__name__}.insert") from e

    def upsert(
        self,
        collection: str,
        data: list[dict[str, Any]],
        partition: str | None = None,
        timeout: float | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """向集合插入或更新数据（主键存在则更新，不存在则插入）。

        Returns:
            dict: 例如 {"upsert_count": 10, "ids": [...]}

        Raises:
            MilvusError: Schema 不匹配或集群异常时抛出
        """
        if not data:
            logger.warning(f"Empty data passed to {self.__class__.__name__}.upsert, skipping.")
            return {"upsert_count": 0, "ids": []}

        try:
            result = cast(
                dict[str, Any],
                self.client.upsert(
                    collection_name=collection,
                    data=data,
                    partition_name=partition or "",
                    timeout=timeout,
                    **kwargs,
                ),
            )
            logger.debug(f"Upserted {result.get('upsert_count', 0)} rows into '{collection}'.")
            return dict(result)
        except MilvusException as e:
            logger.error(f"MilvusException during upsert into '{collection}': {e}", exc_info=True)
            raise MilvusError(f"Upsert failed for collection '{collection}'.") from e
        except Exception as e:
            logger.error(f"Unexpected error during upsert into '{collection}': {e}", exc_info=True)
            raise MilvusError(f"Unexpected exception in {self.__class__.__name__}.upsert") from e

    # 数据删除
    def delete(
        self,
        collection: str,
        filter: str,
        partition: str | None = None,
        timeout: float | None = None,
        **kwargs,
    ) -> int:
        """按布尔表达式删除满足条件的记录。

        Returns:
            int: 受影响的记录数

        Raises:
            MilvusError: 表达式语法错误或集群异常时抛出
        """
        if not filter or not filter.strip():
            logger.warning(f"Empty filter passed to {self.__class__.__name__}.delete, skipping.")
            return 0

        try:
            result: Any = self.client.delete(
                collection_name=collection,
                filter=filter,
                partition_name=partition,
                timeout=timeout,
                **kwargs,
            )

            if isinstance(result, list):
                delete_count = len(result)
            elif isinstance(result, dict) or hasattr(result, "get"):
                delete_count = result.get("delete_count", 0)
            elif hasattr(result, "delete_count"):
                delete_count = getattr(result, "delete_count", 0)
            else:
                delete_count = 0

            logger.info(f"Deleted {delete_count} rows from '{collection}'.")
            return delete_count

        except MilvusException as e:
            logger.error(f"MilvusException during delete from '{collection}': {e}", exc_info=True)
            raise MilvusError(f"Delete failed for collection '{collection}'.") from e
        except Exception as e:
            logger.error(f"Unexpected error during delete from '{collection}': {e}", exc_info=True)
            raise MilvusError(f"Unexpected exception in {self.__class__.__name__}.delete") from e

    def delete_by_ids(
        self,
        collection: str,
        ids: list[int] | list[str] | int | str,
        partition: str | None = None,
        timeout: float | None = None,
        **kwargs,
    ) -> int:
        """按主键 ID 删除指定记录。

        Returns:
            int: 受影响的记录数

        Raises:
            MilvusError: 集群异常时抛出
        """
        try:
            result: Any = self.client.delete(
                collection_name=collection,
                ids=ids,
                partition_name=partition,
                timeout=timeout,
                **kwargs,
            )

            if isinstance(result, list):
                delete_count = len(result)
            elif isinstance(result, dict) or hasattr(result, "get"):
                delete_count = result.get("delete_count", 0)
            else:
                delete_count = 0

            logger.info(f"Deleted {delete_count} rows by ID from '{collection}'.")
            return delete_count

        except MilvusException as e:
            logger.error(
                f"MilvusException during delete_by_ids from '{collection}': {e}", exc_info=True
            )
            raise MilvusError(f"Delete by IDs failed for collection '{collection}'.") from e
        except Exception as e:
            logger.error(
                f"Unexpected error during delete_by_ids from '{collection}': {e}", exc_info=True
            )
            raise MilvusError(
                f"Unexpected exception in {self.__class__.__name__}.delete_by_ids"
            ) from e

    # 数据查询
    def get(
        self,
        collection: str,
        ids: list[int] | list[str] | int | str,
        output_fields: list[str] | None = None,
        partition_names: list[str] | None = None,
        timeout: float | None = None,
        **kwargs,
    ) -> list[VectorQueryContent]:
        """按主键 ID 精确查询记录。

        Raises:
            MilvusError: 集群异常时抛出
        """
        try:
            hits = self.client.get(
                collection_name=collection,
                ids=ids,
                output_fields=output_fields,
                partition_names=partition_names,
                timeout=timeout,
                **kwargs,
            )
            return [VectorQueryContent(record=hit) for hit in hits]
        except MilvusException as e:
            logger.error(f"MilvusException during get from '{collection}': {e}", exc_info=True)
            raise MilvusError(f"Get by IDs failed for collection '{collection}'.") from e
        except Exception as e:
            logger.error(f"Unexpected error during get from '{collection}': {e}", exc_info=True)
            raise MilvusError(f"Unexpected exception in {self.__class__.__name__}.get") from e

    def query(
        self,
        collection: str,
        filter: str = "",
        output_fields: list[str] | None = None,
        partition_names: list[str] | None = None,
        ids: list[int] | list[str] | int | str | None = None,
        limit: int | None = None,
        timeout: float | None = None,
        **kwargs,
    ) -> list[VectorQueryContent]:
        """按标量过滤条件查询记录。

        Raises:
            MilvusError: 表达式语法错误或集群异常时抛出
        """
        if limit is not None:
            kwargs.setdefault("limit", limit)

        try:
            hits = self.client.query(
                collection_name=collection,
                filter=filter,
                output_fields=output_fields,
                partition_names=partition_names,
                ids=ids,
                timeout=timeout,
                **kwargs,
            )
            return [VectorQueryContent(record=hit) for hit in hits]
        except MilvusException as e:
            logger.error(f"MilvusException during query on '{collection}': {e}", exc_info=True)
            raise MilvusError(f"Query failed for collection '{collection}'.") from e
        except Exception as e:
            logger.error(f"Unexpected error during query on '{collection}': {e}", exc_info=True)
            raise MilvusError(f"Unexpected exception in {self.__class__.__name__}.query") from e

    def search(
        self,
        collection: str,
        data: list[list[float]] | list[float] | None = None,
        filter: str | None = None,
        limit: int = 10,
        output_fields: list[str] | None = None,
        partition_names: list[str] | None = None,
        anns_field: str | None = None,
        search_params: dict[str, Any] | None = None,
        ids: list[int] | list[str] | int | str | None = None,
        timeout: float | None = None,
        **kwargs,
    ) -> list[VectorSearchContent]:
        """向量相似度检索。

        Args:
            collection: 集合名称
            data: 查询向量；与 ids 互斥
            filter: 标量前置过滤表达式
            limit: 返回的最相似结果数量上限
            output_fields: 随结果返回的标量字段
            partition_names: 限定检索的分区列表
            anns_field: 向量字段名（多向量字段时必须指定）
            search_params: 索引微调参数
            ids: 按主键 ID 检索相似项（与 data 互斥）
            timeout: 超时时间（秒）

        Returns:
            list[VectorSearchContent]: 按相似度由高到低排序的结果列表

        Raises:
            MilvusError: 向量维度不匹配或集群异常时抛出
        """
        try:
            hits = self.client.search(
                collection_name=collection,
                data=data,
                filter=filter or "",
                limit=limit,
                output_fields=output_fields,
                partition_names=partition_names,
                anns_field=anns_field,
                search_params=search_params or {},
                ids=ids,
                timeout=timeout,
                **kwargs,
            )
            return _parse_hits(hits)

        except MilvusException as e:
            logger.error(f"MilvusException during search on '{collection}': {e}", exc_info=True)
            raise MilvusError(f"Vector search failed on collection '{collection}'.") from e
        except Exception as e:
            logger.error(f"Unexpected error during search on '{collection}': {e}", exc_info=True)
            raise MilvusError(f"Unexpected exception in {self.__class__.__name__}.search") from e

    def search_iterator(
        self,
        collection: str,
        data: list[list[float]] | list[float],
        batch_size: int = 1000,
        filter: str | None = None,
        limit: int | None = None,
        output_fields: list[str] | None = None,
        search_params: dict[str, Any] | None = None,
        partition_names: list[str] | None = None,
        anns_field: str | None = None,
        timeout: float | None = None,
        **kwargs,
    ) -> Any:
        """创建向量检索迭代器，适合大量结果分批获取的场景。

        Args:
            collection: 集合名称
            data: 查询向量
            batch_size: 每批返回的结果数，默认 1000
            filter: 标量过滤表达式
            limit: 总返回结果上限，None 表示不限制
            output_fields: 随结果返回的字段
            search_params: 索引微调参数
            partition_names: 限定检索的分区列表
            anns_field: 向量字段名（多向量字段时必须指定）
            timeout: 超时时间（秒）

        Returns:
            SearchIterator: 可迭代的搜索迭代器对象

        Raises:
            MilvusError: 参数错误或集群异常时抛出
        """
        try:
            iterator_kwargs: dict[str, Any] = {}
            if limit is not None:
                iterator_kwargs["limit"] = limit

            return self.client.search_iterator(
                collection_name=collection,
                data=data,
                batch_size=batch_size,
                filter=filter,
                output_fields=output_fields,
                search_params=search_params,
                timeout=timeout,
                partition_names=partition_names,
                anns_field=anns_field,
                **iterator_kwargs,
                **kwargs,
            )
        except MilvusException as e:
            logger.error(
                f"MilvusException creating search_iterator for '{collection}': {e}", exc_info=True
            )
            raise MilvusError(f"Failed to create search iterator for '{collection}'.") from e
        except Exception as e:
            logger.error(
                f"Unexpected error creating search_iterator for '{collection}': {e}", exc_info=True
            )
            raise MilvusError(
                f"Unexpected exception in {self.__class__.__name__}.search_iterator"
            ) from e

    def query_iterator(
        self,
        collection: str,
        filter: str = "",
        batch_size: int = 1000,
        limit: int | None = None,
        output_fields: list[str] | None = None,
        partition_names: list[str] | None = None,
        timeout: float | None = None,
        **kwargs,
    ) -> Any:
        """创建标量查询迭代器，适合大量数据分页导出的场景。

        Args:
            collection: 集合名称
            filter: 过滤表达式
            batch_size: 每批返回的记录数，默认 1000
            limit: 总返回记录上限，None 表示不限制
            output_fields: 随结果返回的字段
            partition_names: 限定查询的分区列表
            timeout: 超时时间（秒）

        Returns:
            QueryIterator: 可迭代的查询迭代器对象

        Raises:
            MilvusError: 参数错误或集群异常时抛出
        """
        try:
            iterator_kwargs: dict[str, Any] = {}
            if limit is not None:
                iterator_kwargs["limit"] = limit

            return self.client.query_iterator(
                collection_name=collection,
                filter=filter,
                batch_size=batch_size,
                output_fields=output_fields,
                partition_names=partition_names,
                timeout=timeout,
                **iterator_kwargs,
                **kwargs,
            )
        except MilvusException as e:
            logger.error(
                f"MilvusException creating query_iterator for '{collection}': {e}", exc_info=True
            )
            raise MilvusError(f"Failed to create query iterator for '{collection}'.") from e
        except Exception as e:
            logger.error(
                f"Unexpected error creating query_iterator for '{collection}': {e}", exc_info=True
            )
            raise MilvusError(
                f"Unexpected exception in {self.__class__.__name__}.query_iterator"
            ) from e

    def hybrid_search(
        self,
        collection: str,
        requests: list[AnnSearchRequest],
        ranker: BaseRanker | Function,
        limit: int = 10,
        output_fields: list[str] | None = None,
        partition_names: list[str] | None = None,
        timeout: float | None = None,
        **kwargs,
    ) -> list[VectorSearchContent]:
        """多路混合检索（稠密 + 稀疏 BM25 等），并通过 Ranker 进行重排序。

        Args:
            collection: 集合名称
            requests: AnnSearchRequest 列表，每路对应一个向量字段的检索请求
            ranker: 重排序策略，例如 RRFRanker() 或 WeightedRanker(0.7, 0.3)
            limit: 最终返回的结果数量上限
            output_fields: 随结果返回的标量字段列表
            partition_names: 限定检索的分区列表
            timeout: 超时时间（秒）

        Returns:
            list[VectorSearchContent]: 重排序后按分数由高到低排序的结果列表

        Raises:
            MilvusError: 参数错误或集群异常时抛出
        """
        try:
            hits = self.client.hybrid_search(
                collection_name=collection,
                reqs=requests,
                ranker=ranker,
                limit=limit,
                output_fields=output_fields,
                partition_names=partition_names,
                timeout=timeout,
                **kwargs,
            )
            return _parse_hits(hits)

        except MilvusException as e:
            logger.error(
                f"MilvusException during hybrid_search on '{collection}': {e}", exc_info=True
            )
            raise MilvusError(f"Hybrid search failed on collection '{collection}'.") from e
        except Exception as e:
            logger.error(
                f"Unexpected error during hybrid_search on '{collection}': {e}", exc_info=True
            )
            raise MilvusError(
                f"Unexpected exception in {self.__class__.__name__}.hybrid_search"
            ) from e
