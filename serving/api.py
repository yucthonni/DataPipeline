from fastapi import FastAPI, HTTPException
import logging

from .schemas import InferenceRequest, InferenceResponse
from .wrapper import BaseModelWrapper

logger = logging.getLogger(__name__)

def create_app(model_wrapper: BaseModelWrapper) -> FastAPI:
    """
    创建一个挂载了给定模型的 FastAPI 实例。

    Args:
        model_wrapper (BaseModelWrapper): 已加载或将要被加载的模型包装器实例。

    Returns:
        FastAPI: 配置了 `/predict` 等路由的 FastAPI 应用实例。
    """
    app = FastAPI(
        title="Model Serving API",
        description="A model-agnostic RESTful API for model inference.",
        version="1.0.0"
    )

    # 在启动前确保模型已经加载
    @app.on_event("startup")
    async def startup_event():
        logger.info("Loading model...")
        try:
            model_wrapper.load()
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Could not initialize the model. Error: {e}")

    @app.get("/health", response_model=dict, tags=["System"])
    async def health_check():
        """健康检查接口，用于确认服务是否存活"""
        return {"status": "healthy", "model_loaded": model_wrapper.model is not None}

    @app.post("/predict", response_model=InferenceResponse, tags=["Inference"])
    async def predict(request: InferenceRequest):
        """
        统一推理接口。
        将请求中的数据和参数传递给底层的模型包装器进行处理。
        """
        try:
            # 委派给模型的 predict 方法
            result = model_wrapper.predict(
                data=request.data, 
                parameters=request.parameters
            )
            return InferenceResponse(status="success", result=result)
        
        except ValueError as e:
            # 捕获已知类型错误并返回 400
            logger.error(f"Validation error during prediction: {e}")
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            # 捕获未知错误并返回 500
            logger.error(f"Internal error during prediction: {e}")
            raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

    return app
