"""本地语音识别：SenseVoice 跑在扩展自己的隔离环境里，登记成宿主的语音转写模型。

宿主进程里只有这几行胶水：启用时登记一个带 ``transcription`` 标签的模型，
指向 ``runtime.services.asr``；宿主第一次要转写时拉起服务，空闲 30 分钟后停掉。
"""

from app.config.models_config import register_model, unregister_models

MODEL_NAME = "sensevoice-small-local"


def enable() -> None:
    register_model(
        __name__,
        {"model": MODEL_NAME, "tags": ["transcription"], "request_timeout": 120},
        service="asr",
    )


def disable() -> None:
    unregister_models(__name__)
