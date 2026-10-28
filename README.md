# 本地语音识别（local_asr）

Foxran 的工具扩展：在本机用 [SenseVoice-Small](https://github.com/FunAudioLLM/SenseVoice)（[sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) 的 int8 模型）识别视频里的声音，不用调云端接口。

- 支持中文、粤语、英语、日语、韩语，会带背景音乐、笑声、掌声和情绪标记，宿主会把它们换成“背景音乐”“笑声”“语气开心”这类说明。
- CPU 推理，模型约 230MB，常驻内存约 300–500MB；10 秒音频一般零点几秒到两秒出结果。

## 安装

在 Foxran 的扩展市场里点“安装”即可，宿主需要 4.9.0 以上。安装时宿主会：

1. 用 uv 给扩展建一个独立环境（`data/extensions/envs/tool-local_asr/`），里面装 sherpa-onnx 等依赖，和宿主的 Python 环境互不影响；
2. 下载模型到 `data/extensions/data/tool-local_asr/sensevoice/`（依次尝试 `HF_ENDPOINT`、huggingface.co、hf-mirror.com），卸载重装不用重新下载；
3. 启用后登记一个语音转写模型 `sensevoice-small-local`（标签 `transcription`），不用在设置页手动配。

识别服务按需启动：第一次需要听视频时拉起（首次加载模型几秒），空闲 30 分钟后自动停止。日志在 `logs/extensions/tool-local_asr-asr.log`。

国内服务器下载慢时，在设置页“扩展隔离环境”里填包镜像和 Python 下载镜像，然后在市场卡片上点“重建环境”。

## 结构

```
foxran.yaml          # runtime 块：隔离环境、模型下载、识别服务
__init__.py          # 宿主进程里的胶水：登记转写模型
runtime/
  pyproject.toml     # 隔离环境的依赖
  uv.lock            # 跨平台锁文件
  .python-version    # 隔离环境用的 Python（由 uv 下载）
  download_model.py  # 安装时运行，下载模型
  server.py          # OpenAI 兼容的 /v1/audio/transcriptions
```

## 许可

扩展代码 MIT。SenseVoice 模型遵循其[原始许可](https://github.com/FunAudioLLM/SenseVoice/blob/main/LICENSE)。
