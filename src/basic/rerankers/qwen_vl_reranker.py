from collections.abc import Sequence
from enum import StrEnum
from typing import Any, Literal

from httpx import AsyncClient, Client
from pydantic import BaseModel, Field

from ..exceptions import RerankError
from ..loaders.base import BaseLoader
from .base import BaseReranker, RerankCandidate


class ImageDetail(StrEnum):
    AUTO = "auto"
    LOW = "low"
    HIGH = "high"


class ImageUrlParam(BaseModel):
    url: str = Field(..., description="图片URL")
    detail: ImageDetail = Field(default=ImageDetail.AUTO, description="图片解析精度")


class ImageEmbedsParam(BaseModel):
    embeds: str = Field(..., description="图片的 base64 编码字符串")


class ChatCompletionContentPartImageParam(BaseModel):
    image_url: ImageUrlParam = Field(..., description="图片相关的参数")
    type: Literal["image_url"] = Field(default="image_url", description="图片类型")


class ChatCompletionContentPartImageEmbedsParam(BaseModel):
    image_embeds: ImageEmbedsParam = Field(..., description="图片相关的参数")
    type: Literal["image_embeds"] = Field(
        default="image_embeds", description="图片类型"
    )
    uuid: str | None = Field(default=None, description="图片嵌入向量的唯一标识符")


class ChatCompletionContentPartTextParam(BaseModel):
    text: str = Field(..., description="文本内容")
    type: Literal["text"] = Field(default="text", description="文本类型")


class ChatCompletionContentPartVideoParam(BaseModel):
    video_url: ImageUrlParam = Field(..., description="视频URL")
    type: Literal["video_url"] = Field(default="video_url", description="视频类型")


class ScoreMultiModalParam(BaseModel):
    content: Sequence[
        ChatCompletionContentPartImageParam
        | ChatCompletionContentPartImageEmbedsParam
        | ChatCompletionContentPartTextParam
        | ChatCompletionContentPartVideoParam
    ] = Field(..., description="重排序的多模态内容")


class RerankRequest(BaseModel):
    use_activation: bool | None = Field(
        default=True,
        description="是否对池化层的输出应用激活函数。如果没有指定,则会使用池化层的默认设置,通常情况下默认值为 True。",
    )
    model: str | None = Field(default=None, description="模型名称")
    user: str | None = Field(default=None, description="用户ID")
    truncate_prompt_tokens: int | None = Field(
        default=None, description="截断提示词的最大 token 数,防止超出模型上下文限制"
    )
    request_id: str | None = Field(default=None, description="请求ID,用于跟踪和调试")
    priority: int | None = Field(
        default=None,
        description="请求的优先级(数值越小表示处理优先级越高;默认值:0)。如果所使用的模型不支持优先级调度,则任何非 0 的优先级都会引发错误。",
    )
    mm_processor_kwargs: dict | None = Field(
        default=None, description="多模态处理器的额外参数"
    )
    cache_salt: str | None = Field(
        default=None,
        description="如果进行了相关设置,前缀缓存将会使用所提供的字符串进行加盐处理,以防止攻击者在多用户环境中猜测提示内容。该盐值应是随机生成的,要受到第三方访问的保护,并且长度要足够长,以确保无法预测(例如,以 base64 编码的 43 个字符,相当于 256 位)。",
    )
    query: str | ScoreMultiModalParam = Field(..., description="查询文本或者多模态内容")
    documents: str | ScoreMultiModalParam | Sequence[str | ScoreMultiModalParam] = (
        Field(..., description="文档文本或者多模态内容")
    )
    # top_n: int | None = Field(default=None, description="返回的文档数量")  # 不误导下游开发


class ScoreQueriesDocumentsRequest(BaseModel):
    use_activation: bool | None = Field(
        default=True,
        description="是否对池化层的输出应用激活函数。如果没有指定,则会使用池化层的默认设置,通常情况下默认值为 True。",
    )
    model: str | None = Field(default=None, description="模型名称")
    user: str | None = Field(default=None, description="用户ID")
    truncate_prompt_tokens: int | None = Field(
        default=None, description="截断提示词的最大 token 数,防止超出模型上下文限制"
    )
    request_id: str | None = Field(default=None, description="请求ID,用于跟踪和调试")
    priority: int | None = Field(
        default=None,
        description="请求的优先级(数值越小表示处理优先级越高;默认值:0)。如果所使用的模型不支持优先级调度,则任何非 0 的优先级都会引发错误。",
    )
    mm_processor_kwargs: dict | None = Field(
        default=None, description="多模态处理器的额外参数"
    )
    cache_salt: str | None = Field(
        default=None,
        description="如果进行了相关设置,前缀缓存将会使用所提供的字符串进行加盐处理,以防止攻击者在多用户环境中猜测提示内容。该盐值应是随机生成的,要受到第三方访问的保护,并且长度要足够长,以确保无法预测(例如,以 base64 编码的 43 个字符,相当于 256 位)。",
    )
    queries: str | ScoreMultiModalParam | list[str | ScoreMultiModalParam] = Field(
        ..., description="查询文本或者多模态内容"
    )
    documents: str | ScoreMultiModalParam | Sequence[str | ScoreMultiModalParam] = (
        Field(..., description="文档文本或者多模态内容")
    )


class Qwen3VLReranker(BaseReranker[ScoreMultiModalParam]):
    def __init__(
        self,
        base_url: str,
        model_name: str,
        api_key: str,
        text_field: str = "text",
        image_field: str | None = "image_url",
        image_loader: BaseLoader | None = None,
        http_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """
        Qwen3VL 重排序模型

        Args:
            base_url:     模型服务地址
            model_name:   模型名称
            api_key:      模型服务 api_key
            text_field:   record 中文本内容所在的字段名
            image_field:  record 中图片引用所在的字段名；None 表示纯文本模式
            image_loader: 图片加载器，支持 HTTP / 本地 / MinIO 等来源；
                          None 则跳过图片（降级为纯文本打分）
            http_kwargs:  httpx 客户端参数（用于模型 API 调用），默认 timeout=None
        """
        self._base_url = base_url
        self._model_name = model_name
        self._api_key = api_key
        self._text_field = text_field
        self._image_field = image_field
        self._image_loader: BaseLoader | None = image_loader

        http_kwargs = http_kwargs or {"timeout": None}
        self._httpx_client = AsyncClient(**http_kwargs)

    # 输入构造
    async def _build_input(self, candidate: RerankCandidate) -> ScoreMultiModalParam:
        """将原始候选转换为多模态打分输入（含异步图片获取）。"""
        record = candidate.record
        text_param = ChatCompletionContentPartTextParam(
            text=record.get(self._text_field, "")
        )

        if self._image_field:
            image_url = record.get(self._image_field)
            if image_url:
                image_data_url = await self._fetch_and_convert(image_url)
                if image_data_url:
                    image_param = ChatCompletionContentPartImageParam(
                        image_url=ImageUrlParam(url=image_data_url)
                    )
                    return ScoreMultiModalParam(content=[text_param, image_param])

        return ScoreMultiModalParam(content=[text_param])

    async def _fetch_and_convert(self, image_ref: str) -> str | None:
        """通过 image_loader 加载内容并转为 base64 data URL；非图片或失败时返回 None。"""
        if not self._image_loader:
            return None
        data = await self._image_loader.load(image_ref)
        if data is None or not data.is_image:
            return None
        return data.to_data_url()

    # 校验
    def _normalize_request(self, document: ScoreMultiModalParam) -> None:
        allowed_content_types = (
            ChatCompletionContentPartTextParam,
            ChatCompletionContentPartImageParam,
        )
        url_prefixes = ("https://", "http://")
        for content in document.content:
            if not isinstance(content, allowed_content_types):
                raise RerankError(
                    f"{self.__class__.__name__} | _normalize_request: "
                    "暂不支持 ImageEmbeds / Video 类型"
                )
            if isinstance(content, ChatCompletionContentPartImageParam):
                url = content.image_url.url
                if not url.startswith(url_prefixes) and not url.startswith("data:image/"):
                    raise RerankError(
                        f"{self.__class__.__name__} | _normalize_request: "
                        f"仅支持以 {url_prefixes} 开头的在线图片或 base64 编码图片"
                    )

    # 打分
    def score(
        self,
        query: str,
        inputs: list[ScoreMultiModalParam],
        http_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ) -> list[float]:
        try:
            for doc in inputs:
                self._normalize_request(doc)

            rerank_request = RerankRequest(
                model=self._model_name,
                query=query,
                documents=inputs,
                **kwargs,
            ).model_dump(exclude_none=True)

            http_kwargs = http_kwargs or {"timeout": None}
            with Client(**http_kwargs) as client:
                rerank_response = client.post(
                    url=f"{self._base_url}/rerank",
                    json=rerank_request,
                    headers={"Authorization": f"Bearer {self._api_key}"},
                )
            rerank_response.raise_for_status()
            results = rerank_response.json()["results"]
            return [
                i["relevance_score"] for i in sorted(results, key=lambda x: x["index"])
            ]
        except RerankError:
            raise
        except Exception as e:
            raise RerankError(
                f"{self.__class__.__name__} | score 发生非预期错误: {e!s}"
            ) from e

    async def ascore(
        self,
        query: str,
        inputs: list[ScoreMultiModalParam],
        **kwargs,
    ) -> list[float]:
        try:
            for doc in inputs:
                self._normalize_request(doc)

            rerank_request = RerankRequest(
                model=self._model_name,
                query=query,
                documents=inputs,
                **kwargs,
            ).model_dump(exclude_none=True)

            rerank_response = await self._httpx_client.post(
                url=f"{self._base_url}/rerank",
                json=rerank_request,
                headers={"Authorization": f"Bearer {self._api_key}"},
            )
            rerank_response.raise_for_status()
            results = rerank_response.json()["results"]
            return [
                i["relevance_score"] for i in sorted(results, key=lambda x: x["index"])
            ]
        except RerankError:
            raise
        except Exception as e:
            raise RerankError(
                f"{self.__class__.__name__} | ascore 发生非预期错误: {e!s}"
            ) from e
