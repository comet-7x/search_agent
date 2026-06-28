import base64
from dataclasses import dataclass, field


@dataclass
class BlobData:
    """二进制内容的通用数据载体，贯穿召回、精排等各层之间传递。"""

    content: bytes
    content_type: str = field(default="application/octet-stream")

    @property
    def is_image(self) -> bool:
        return self.content_type.startswith("image/")

    @property
    def is_text(self) -> bool:
        return self.content_type.startswith("text/")

    @property
    def is_pdf(self) -> bool:
        return self.content_type == "application/pdf"

    def to_data_url(self) -> str:
        """编码为 Base64 Data URL，供模型多模态输入使用。"""
        b64 = base64.b64encode(self.content).decode()
        return f"data:{self.content_type};base64,{b64}"

    def decode(self, encoding: str = "utf-8") -> str:
        """将内容解码为字符串，用于文本类内容。"""
        return self.content.decode(encoding)

    @classmethod
    def from_data_url(cls, data_url: str) -> "BlobData":
        """从 Base64 Data URL 还原 BlobData。"""
        header, encoded = data_url.split(",", 1)
        content_type = header.split(";")[0].removeprefix("data:")
        return cls(content=base64.b64decode(encoded), content_type=content_type)
