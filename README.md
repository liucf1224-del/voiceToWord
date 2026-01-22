# voiceToWord

一个功能强大的音视频转文本工具，基于 OpenAI Whisper 模型实现高质量的语音识别，并支持标点恢复。

## 功能特点

- **多格式支持**：支持处理视频文件和音频文件
- **智能转录**：使用 OpenAI Whisper 模型进行高质量语音识别
- **分段处理**：支持长音频/视频的分段转录，提高处理效率和准确性
- **标点恢复**：集成 DistilBERT 标点恢复模型，使输出文本更加流畅自然
- **多模型选择**：提供不同大小的 Whisper 模型，平衡速度和准确性
- **跨平台兼容**：支持 Windows、macOS 和 Linux 系统
- **自动清理**：处理完成后自动清理临时文件，保持文件系统整洁

## 支持的文件格式

- **视频格式**：MP4、AVI、MOV、MKV 等常见视频格式
- **音频格式**：WAV、MP3、FLAC、AAC 等常见音频格式

## 安装说明

### 1. 克隆项目

```bash
git clone <repository-url>
cd voiceToWord
```

### 2. 安装依赖

使用 pip 安装所需依赖：

```bash
pip install -r requirements.txt
```

### 3. 安装 FFmpeg

Whisper 和 moviepy 依赖 FFmpeg 进行音频处理，请确保已安装 FFmpeg 并添加到系统路径中。

- **Windows**：从 [FFmpeg 官网](https://ffmpeg.org/download.html) 下载二进制文件，解压后将 `bin` 目录添加到系统环境变量
- **macOS**：使用 Homebrew 安装：`brew install ffmpeg`
- **Linux**：使用包管理器安装：`sudo apt install ffmpeg` (Ubuntu/Debian) 或 `sudo yum install ffmpeg` (CentOS/RHEL)

## 使用方法

### 命令行运行

在项目目录下运行：

```bash
python voice_to_sentence.py
```

### 交互流程

1. **选择输入类型**：
   - 输入 `1` 处理视频文件
   - 输入 `2` 处理音频文件
   - 直接回车默认为处理视频文件

2. **输入文件路径**：
   - 可以直接粘贴文件路径，支持带引号的路径

3. **选择 Whisper 模型**：
   - `1` - tiny：最快，准确度最低
   - `2` - base：平衡速度和准确度（推荐）
   - `3` - small：较好准确度
   - `4` - medium：高准确度
   - `5` - large：最高准确度，需要更多资源
   - 直接回车默认为 base 模型

4. **选择是否使用分段转录**：
   - 输入 `y` 或 `yes` 使用分段转录（适合长音频/视频）
   - 直接回车或输入其他内容不使用分段转录

5. **选择缓存目录**：
   - 可以使用默认缓存目录（项目目录下的 `.whisper_cache`）
   - 也可以指定自定义缓存目录

6. **选择是否使用标点恢复**：
   - 输入 `y` 或 `yes` 使用 DistilBERT 标点恢复
   - 直接回车或输入其他内容不使用标点恢复

7. **选择标点恢复模型缓存目录**：
   - 可以使用默认缓存目录（项目目录下的 `.hf_cache`）
   - 也可以指定自定义缓存目录

## 示例

### 处理视频文件

```bash
python voice_to_sentence.py
请选择输入类型:
1. 视频文件
2. 音频文件
请输入选择 (1/2，默认为1): 1
请输入文件路径: D:\videos\meeting.mp4

可用的Whisper模型:
1. tiny - 最快，准确度最低 -75m左右
2. base - 平衡速度和准确度（推荐）-138m左右
3. small - 较好准确度 -400m左右
4. medium - 高准确度
5. large - 最高准确度，需要更多资源

请选择模型 (1-5，默认为2): 2
已选择模型: base

是否使用分段转录？(适合长音频/长视频，y/N): y

是否使用本地缓存目录 D:\py_work\voiceToWord\.whisper_cache？(y/N): y

是否使用DistilBERT标点恢复？(y/N): y

是否使用DistilBERT本地缓存目录 D:\py_work\voiceToWord\.hf_cache？(y/N): y
```

### 处理音频文件

```bash
python voice_to_sentence.py
请选择输入类型:
1. 视频文件
2. 音频文件
请输入选择 (1/2，默认为1): 2
请输入文件路径: D:\audios\interview.mp3

可用的Whisper模型:
1. tiny - 最快，准确度最低
2. base - 平衡速度和准确度（推荐）
3. small - 较好准确度
4. medium - 高准确度
5. large - 最高准确度，需要更多资源

请选择模型 (1-5，默认为2): 3
已选择模型: small

是否使用分段转录？(适合长音频/长视频，y/N): n

是否使用本地缓存目录 D:\py_work\voiceToWord\.whisper_cache？(y/N): y

是否使用DistilBERT标点恢复？(y/N): y

是否使用DistilBERT本地缓存目录 D:\py_work\voiceToWord\.hf_cache？(y/N): y
```

## 输出结果

处理完成后，会在同一目录下生成与输入文件同名的 `.txt` 文件，包含转录后的文本内容。

## 核心功能说明

### 1. 音频提取 (extract_audio_from_video)

从视频文件中提取音频，转换为适合语音识别的 WAV 格式。

### 2. 音频转录 (transcribe_audio_with_whisper)

使用 Whisper 模型直接转录音频文件，适用于较短的音频。

### 3. 分段转录 (split_and_transcribe_audio)

将长音频分割成小块，逐块转录后合并结果，提高处理长音频的效率和准确性。

### 4. 标点恢复 (_apply_ml_punctuation)

使用 DistilBERT 标点恢复模型，为转录文本添加适当的标点符号，使文本更加流畅自然。

### 5. 视频处理 (process_video_to_text)

处理视频文件的主函数，包括音频提取、转录和结果保存。

### 6. 音频处理 (process_audio_to_text)

处理音频文件的主函数，包括转录和结果保存。

## 性能优化建议

1. **模型选择**：
   - 短音频（<5分钟）：推荐使用 base 模型
   - 中等长度音频（5-30分钟）：推荐使用 small 模型
   - 长音频（>30分钟）：推荐使用 medium 模型
   - 对准确度要求极高的场景：使用 large 模型

2. **硬件加速**：
   - 如果有 NVIDIA GPU，Whisper 会自动使用 CUDA 加速
   - 对于 CPU 处理，建议使用 tiny 或 base 模型以获得更快的速度

3. **分段转录**：
   - 对于长音频（>10分钟），建议使用分段转录功能
   - 分段转录可以减少内存使用，提高处理稳定性

## 常见问题

### 1. 安装依赖时出现错误

- 确保使用的是 Python 3.8 或更高版本
- 对于 Windows 用户，可能需要以管理员权限运行命令提示符
- 可以尝试使用 `pip install --upgrade pip` 升级 pip 后再安装依赖

### 2. 处理视频时出现 FFmpeg 错误

- 确保已正确安装 FFmpeg 并添加到系统路径
- 尝试使用绝对路径指定视频文件

### 3. 转录结果不准确

- 尝试使用更大的 Whisper 模型
- 对于有背景噪音的音频，可以尝试调整分段转录的参数
- 确保音频文件的采样率和质量足够好

### 4. 处理速度太慢

- 尝试使用更小的 Whisper 模型
- 如果有 GPU，确保 CUDA 已正确安装
- 对于长音频，使用分段转录功能

## 依赖项

- openai-whisper：核心语音识别模型
- torch：深度学习框架，支持 GPU 加速
- moviepy：视频处理库，用于提取音频
- pydub：音频处理库，用于分割音频
- dbpunctuator：标点恢复模型
- transformers：提供预训练模型支持
- huggingface_hub：Hugging Face 模型库接口
- hf_xet：扩展功能支持
- imageio-ffmpeg：FFmpeg 包装器


## 总结
- 总得来说可以用，这个可以支持视频或者音频转文字然后到txt，再对文字做标点恢复，这个功能还是比较完善的。
- 拓展性的话可以是及时人工和ai对话的前置组装
- 最后这个项目的性能还是比较好的，尤其是在使用GPU加速的情况下。