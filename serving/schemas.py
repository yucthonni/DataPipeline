from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field

class InferenceRequest(BaseModel):
    """
    通用推理请求模型。
    接受一个数据列表或字典，允许各种格式的输入。
    """
    data: Union[List[Any], Dict[str, Any]] = Field(
        ..., 
        description="模型的输入数据，支持列表（批量数据）或字典（命名特征）形式。"
    )
    parameters: Optional[Dict[str, Any]] = Field(
        default_factory=dict, 
        description="传递给推理函数的附加参数（如 max_length, temperature 等）。"
    )

class InferenceResponse(BaseModel):
    """
    通用推理响应模型。
    """
    status: str = Field("success", description="请求状态，通常为 'success' 或 'error'")
    result: Optional[Union[List[Any], Dict[str, Any], Any]] = Field(
        None, 
        description="模型的推理结果"
    )
    message: Optional[str] = Field(
        None, 
        description="额外的状态信息或错误信息"
    )
