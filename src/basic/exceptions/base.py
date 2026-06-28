class CometBaseError(Exception):
    code: int
    message: str

    def __init__(self, message: str, code: int = 8000):
        self.message = message
        self.code = code
        super().__init__(self.message)

    def __str__(self):
        return f"[{self.code}] {self.message}"


class EmbeddingError(CometBaseError):
    """嵌入模型相关异常"""

    def __init__(self, message: str):
        super().__init__(message, code=8100)


class RerankError(CometBaseError):
    """重排序模型相关异常"""

    def __init__(self, message: str):
        super().__init__(message, code=8200)


class MilvusError(CometBaseError):
    """Milvus 相关异常"""

    def __init__(self, message: str):
        super().__init__(message, code=8300)


class MinioError(CometBaseError):
    """MinIO 对象存储相关异常"""

    def __init__(self, message: str):
        super().__init__(message, code=8400)
