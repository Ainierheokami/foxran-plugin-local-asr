"""下载 SenseVoice 模型到扩展数据目录（宿主在安装时运行一次，已下载就跳过）。

依次尝试 HF_ENDPOINT（如果设置了）、huggingface.co 和 hf-mirror.com，
哪个能连上用哪个。文件先写到 .part，下载完整后再改名，中断不会留下坏文件。
"""

from __future__ import annotations

import os
import sys
import time
import urllib.request
from pathlib import Path

REPO = "csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"
FILES = ("tokens.txt", "model.int8.onnx")
ENDPOINTS = [e for e in (os.environ.get("HF_ENDPOINT"), "https://huggingface.co", "https://hf-mirror.com") if e]


def model_dir() -> Path:
    root = Path(os.environ.get("FOXRAN_EXTENSION_DATA") or Path(__file__).resolve().parent / ".data")
    return root / "sensevoice"


def _download(url: str, target: Path) -> None:
    part = target.with_suffix(target.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "foxran-local-asr"})
    with urllib.request.urlopen(request, timeout=30) as response:
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        last_report = 0.0
        with part.open("wb") as handle:
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                handle.write(chunk)
                done += len(chunk)
                now = time.monotonic()
                if total and now - last_report > 3:
                    last_report = now
                    print(f"  {target.name}: {done / total:.0%}（{done >> 20}/{total >> 20} MB）", flush=True)
    if total and part.stat().st_size != total:
        raise OSError(f"下载不完整：{part.stat().st_size}/{total} 字节")
    part.replace(target)


def main() -> int:
    target_dir = model_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    missing = [name for name in FILES if not (target_dir / name).is_file()]
    if not missing:
        print(f"SenseVoice 模型已存在：{target_dir}", flush=True)
        return 0
    errors = []
    for endpoint in ENDPOINTS:
        try:
            for name in missing:
                if (target_dir / name).is_file():
                    continue
                url = f"{endpoint.rstrip('/')}/{REPO}/resolve/main/{name}"
                print(f"下载 {name}：{url}", flush=True)
                _download(url, target_dir / name)
            print(f"SenseVoice 模型已下载到 {target_dir}", flush=True)
            return 0
        except Exception as exc:  # 换下一个地址
            errors.append(f"{endpoint}: {exc}")
            print(f"  {endpoint} 下载失败：{exc}，换下一个地址", flush=True)
    print("模型下载失败：" + "；".join(errors), file=sys.stderr, flush=True)
    print("可以设置环境变量 HF_ENDPOINT 指定可用的镜像后，在市场页点“重建环境”重试。", file=sys.stderr, flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
