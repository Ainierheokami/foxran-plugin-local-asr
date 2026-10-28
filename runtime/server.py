"""SenseVoice 语音识别服务：OpenAI 兼容的 /v1/audio/transcriptions。

由宿主按需启动（端口由宿主分配），第一次识别时才加载模型（约 1–3 秒）。
返回的 text 带 SenseVoice 的语言、情绪、声音事件标记，例如
``<|zh|><|HAPPY|><|Speech|>家人们谁懂啊``，宿主会换成中文说明。
"""

from __future__ import annotations

import argparse
import io
import os
import threading
import wave

import numpy as np
import uvicorn
from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from download_model import model_dir

MODEL_NAME = "sensevoice-small-local"

_recognizer = None
_recognizer_lock = threading.Lock()


def recognizer():
    global _recognizer
    with _recognizer_lock:
        if _recognizer is None:
            import sherpa_onnx

            root = model_dir()
            _recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
                model=str(root / "model.int8.onnx"),
                tokens=str(root / "tokens.txt"),
                num_threads=max(1, min(4, (os.cpu_count() or 2) // 2)),
                use_itn=True,
                language="auto",
            )
        return _recognizer


def _read_wav(data: bytes) -> tuple[int, np.ndarray]:
    with wave.open(io.BytesIO(data)) as reader:
        rate = reader.getframerate()
        channels = reader.getnchannels()
        width = reader.getsampwidth()
        frames = reader.readframes(reader.getnframes())
    if width != 2:
        raise ValueError(f"只支持 16 位 PCM wav，收到 {width * 8} 位")
    samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return rate, samples


def transcribe(data: bytes) -> str:
    rate, samples = _read_wav(data)
    engine = recognizer()
    stream = engine.create_stream()
    stream.accept_waveform(rate, samples)
    engine.decode_stream(stream)
    result = stream.result
    tags = "".join(str(getattr(result, name, "") or "") for name in ("lang", "emotion", "event"))
    return tags + (result.text or "")


async def transcriptions(request: Request) -> JSONResponse:
    form = await request.form()
    upload = form.get("file")
    if upload is None or not hasattr(upload, "read"):
        return JSONResponse({"error": {"message": "缺少 file 字段"}}, status_code=400)
    data = await upload.read()
    try:
        text = await run_in_threadpool(transcribe, data)
    except (wave.Error, ValueError, EOFError) as exc:
        return JSONResponse({"error": {"message": f"音频无法读取：{exc}"}}, status_code=400)
    return JSONResponse({"text": text})


async def models(_request: Request) -> JSONResponse:
    return JSONResponse({"object": "list", "data": [{"id": MODEL_NAME, "object": "model", "owned_by": "local"}]})


async def health(_request: Request) -> JSONResponse:
    root = model_dir()
    ready = all((root / name).is_file() for name in ("model.int8.onnx", "tokens.txt"))
    return JSONResponse({"ok": ready, "model_loaded": _recognizer is not None}, status_code=200 if ready else 503)


app = Starlette(routes=[
    Route("/health", health),
    Route("/v1/models", models),
    Route("/v1/audio/transcriptions", transcriptions, methods=["POST"]),
])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
