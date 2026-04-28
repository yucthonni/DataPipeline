from abc import ABC, abstractmethod
from typing import Any

class BaseModelWrapper(ABC):
    """
    模型包装器基类。
    所有希望通过 serving 模块提供 RESTful API 的模型都需要继承此基类并实现相关方法。
    这保证了 API 层与具体模型层（如 DDPM, 树模型, LLM 等）的解耦。
    """

    def __init__(self, model_path: str = None, **kwargs):
        """
        初始化包装器，可以提供模型路径等参数。
        
        Args:
            model_path (str, optional): 模型的加载路径。
            **kwargs: 其他模型初始化参数。
        """
        self.model_path = model_path
        self.model = None
        self.kwargs = kwargs

    @abstractmethod
    def load(self) -> None:
        """
        加载模型的具体实现。
        应该在内部初始化 `self.model`。
        """
        pass

    @abstractmethod
    def predict(self, data: Any, parameters: dict = None) -> Any:
        """
        对输入数据进行推理的具体实现。
        
        Args:
            data (Any): 输入的请求数据（来源于 Request schema，可能是 list 或 dict）。
            parameters (dict, optional): 附加参数。

        Returns:
            Any: 推理结果。可以是被 JSON 序列化的任何 Python 原生结构。
        """
        pass
