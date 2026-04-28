# 模型服务模块 (Serving Module)

这是一个独立、模型无关的模型服务和推理模块。该模块基于 **FastAPI** 和 **Uvicorn** 构建，旨在将任何在项目中训练好（或外部加载）的模型快速挂载为 RESTful HTTP API，从而允许外部服务进行访问和推理。

## ✨ 主要特性

- **模型解耦**: 提供统一的 `BaseModelWrapper` 抽象接口。API 层和模型层完全分离，不关心具体的模型架构（如 DDPM、树模型、LLM 等）。
- **通用数据格式**: 不对输入输出做强约束。推理接口支持灵活的 List 或 Dict（JSON 格式）输入，完全由您的模型决定如何解析这些数据。
- **高性能**: 基于异步 ASGI 框架 FastAPI 构建，兼具高性能和高并发能力。
- **自带文档**: FastAPI 原生支持自动生成基于 Swagger UI 的交互式 API 接口文档，方便调试。

## 🚀 快速开始

将您的模型接入服务只需要两个简单的步骤：**实现包装器** 和 **挂载启动**。

### 1. 实现包装器

创建一个继承自 `BaseModelWrapper` 的类，并实现其 `load` 和 `predict` 抽象方法。

```python
from serving.wrapper import BaseModelWrapper

class MyRealModelWrapper(BaseModelWrapper):
    def load(self):
        """
        在服务启动时自动调用。
        在此处初始化模型权重、配置等。
        """
        print("Loading my awesome model...")
        self.model = ... # 例如 torch.load(...)
        
    def predict(self, data, parameters=None):
        """
        处理传入的推理请求。
        
        :param data: 客户端传入的数据 (List 或 Dict)。
        :param parameters: (可选) 客户端传入的其他控制参数。
        :return: 模型推理的结果，要求必须是可以 JSON 序列化的 Python 原生类型。
        """
        # 模型推理逻辑...
        result = self.model.inference(data)
        return {"predictions": result}
```

### 2. 挂载与启动

使用提供的 `create_app` 函数将您的包装器实例挂载到 FastAPI 应用上，然后使用 Uvicorn 启动。

```python
import uvicorn
from serving.api import create_app

# 初始化您的模型包装器
my_wrapper = MyRealModelWrapper()

# 创建应用并挂载模型
app = create_app(my_wrapper)

if __name__ == "__main__":
    # 启动服务器 (默认在 8000 端口)
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

> **提示**: 在项目根目录中有一个完整的演示示例 `serve_example.py`，可以直接运行体验。

## 🔌 API 接口说明

当服务启动后（默认 `http://127.0.0.1:8000`），将提供以下核心接口。您也可以在浏览器中访问 `http://127.0.0.1:8000/docs` 直接查看图形化的接口测试工具。

### GET `/health`
- **说明**: 检查服务是否存活，以及模型是否已成功加载。
- **返回示例**:
  ```json
  {
      "status": "healthy",
      "model_loaded": true
  }
  ```

### POST `/predict`
- **说明**: 执行模型推理的主接口。
- **请求体 (JSON)**:
  ```json
  {
      "data": [1.0, 2.0, 3.5], 
      "parameters": {"temperature": 0.8}
  }
  ```
  *(注：`data` 字段支持数组或对象格式，具体根据您的模型需要。`parameters` 为可选项)*
- **返回示例**:
  ```json
  {
      "status": "success",
      "result": {"predictions": [...]},
      "message": null
  }
  ```

## 📦 依赖项

本模块需要以下依赖库，它们已在项目的根级 `requirements.txt` 中被更新：
- `fastapi`
- `uvicorn`
- `pydantic`
