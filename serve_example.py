import uvicorn
from typing import Any
import logging

from serving.wrapper import BaseModelWrapper
from serving.api import create_app

# 配置日志
logging.basicConfig(level=logging.INFO)

class DummyModelWrapper(BaseModelWrapper):
    """
    一个简单的示例模型包装器，用于演示如何接入 serving 模块。
    """
    def load(self) -> None:
        # 这里可以放置例如 torch.load 或 pickle.load 的逻辑
        logging.info("Initializing Dummy Model...")
        self.model = "I am a dummy model!"
    
    def predict(self, data: Any, parameters: dict = None) -> Any:
        # 这里放置真实的推理逻辑
        logging.info(f"Received data: {data}")
        logging.info(f"Received parameters: {parameters}")
        
        # 简单模拟处理
        multiplier = parameters.get("multiplier", 1) if parameters else 1
        
        if isinstance(data, list):
            result = [x * multiplier for x in data if isinstance(x, (int, float))]
        elif isinstance(data, dict):
            result = {k: v * multiplier for k, v in data.items() if isinstance(v, (int, float))}
        else:
            result = "Unsupported data format for this dummy model."
            
        return result

# 1. 初始化模型包装器
my_model = DummyModelWrapper()

# 2. 创建应用并挂载模型
app = create_app(my_model)

if __name__ == "__main__":
    # 3. 启动服务器 (使用 uvicorn)
    # 也可以在命令行通过 `uvicorn serve_example:app --reload` 启动
    logging.info("Starting server on http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
