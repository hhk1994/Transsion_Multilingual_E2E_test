# e2e_test — 端侧 TTS 内部测试

## 文本归一化（当前已实现）

从 `trassion_test/` 下的原始句子生成 `output/normalized.txt`。

### 一键运行

```bash
cd /222042021/hehongkai/transsion_tts/Multilingual_LITs/e2e_test
chmod +x run_normalize.sh scripts/*.sh
./run_normalize.sh
```

脚本会自动：

1. 安装系统依赖（`build-essential`、`libicu-dev`，需要 sudo）
2. 编译 `bin/bn_tts`（源码来自本仓库 submodule `../Transsion_Multilingual_Text_Normalization_for_TTS`，也可用 `TN_ROOT` 覆盖）
3. 读取 `trassion_test/bn_1000_sample_sent.txt`，写出 `output/normalized.txt`

### 常用参数

```bash
# 只跑前 20 行（冒烟）
LIMIT=20 ./run_normalize.sh

# 自定义输入/输出路径
INPUT_TXT=trassion_test/bn_1000_sample_sent.txt \
OUTPUT_TXT=output/my_norm.txt \
./run_normalize.sh
```

### 依赖说明

| 依赖 | 用途 |
|------|------|
| `g++` (C++17) | 编译归一化引擎 |
| ICU4C (`libicu-dev`) | 孟加拉语数字/规则 |

无需 Python。规则文件使用 TN 仓库内的 `rules/bn.json`（编译时通过源码路径定位）。

---

## 生成 TTS Manifest（已实现）

将 `output/normalized.txt` 转为 LITs 推理所需的 `wav|text` 格式。

### 一键运行

```bash
cd /222042021/hehongkai/transsion_tts/Multilingual_LITs/e2e_test
chmod +x run_build_manifest.sh scripts/build_manifest.sh
./run_build_manifest.sh
```

输出：

| 文件 | 说明 |
|------|------|
| `output/tts_manifest.txt` | 每行 `bn_0001.wav\|归一化文本`，供 `infer_example.sh bn-en` 使用 |
| `output/manifest.index.tsv` | `utt_id` ↔ 原文行号，便于查 bad case |
| `output/tts_manifest.stats.json` | 条数统计 |

### 与归一化串联

```bash
./run_normalize.sh && ./run_build_manifest.sh
LIMIT=20 ./run_normalize.sh && LIMIT=20 ./run_build_manifest.sh
```

### Manifest 格式示例

```text
bn_0001.wav|ওহ, হ্যালো! কেমন আছো তুমি? আজ সারাদিন কী করলে?
bn_0002.wav|আরে! দারুণ তো! ...
```

---

## TTS 语音合成（已实现）

在 `e2e_test/.venv` 中安装 LITs 依赖，一键合成孟加拉语音频。

### 前置条件

1. 已有 `output/tts_manifest.txt`（先跑 `./run_build_manifest.sh`）
2. **模型权重**：`Multilingual_LITs/model_checkpoints/bn-en.ckpt` 须为真实文件（约 244MB），不能是 Git LFS 指针（134 字节）

```bash
cd ..
git lfs pull --include=model_checkpoints/bn-en.ckpt
# 或指定本地 ckpt：
CKPT_PATH=/path/to/bn-en.ckpt ./run_synthesize.sh
```

### 一键运行

```bash
cd /222042021/hehongkai/transsion_tts/Multilingual_LITs/e2e_test
chmod +x run_synthesize.sh scripts/*.sh
./run_synthesize.sh
```

首次会自动：安装 `espeak-ng`、`python3-venv`、创建 `.venv`、安装精简推理依赖（`requirements-synthesize-min.txt`，避免完整 `lits_requirements.txt` 的 protobuf 冲突）。

可将真实权重放到 `e2e_test/models/bn-en.ckpt`，或通过 `CKPT_PATH=...` 指定。

### 输出

| 路径 | 说明 |
|------|------|
| `output/wav/<INFER_ID>/` | 合成的 `bn_0001.wav` … |
| `output/wav/<INFER_ID>/meta.txt` | `wav路径\|文本` |
| `output/wav/<INFER_ID>/synthesis.log` | 推理日志（含 RTF） |
| `output/wav/latest` | 指向最近一次 run |

### 常用参数

```bash
# 冒烟：只合成 manifest 前 5 条
LIMIT=5 ./run_synthesize.sh

# 跳过 venv 重装（已装好依赖时）
SKIP_VENV=1 ./run_synthesize.sh

# 指定 run 名称
INFER_ID=bn_smoke LIMIT=10 ./run_synthesize.sh
```

### 全流程

```bash
./run_normalize.sh && ./run_build_manifest.sh && ./run_synthesize.sh
LIMIT=20 ./run_normalize.sh && LIMIT=20 ./run_build_manifest.sh && LIMIT=20 ./run_synthesize.sh
```

---

## ASR + WER（Whisper）

对已合成的 wav 做 Whisper 转写，并以 `output/normalized.txt` 为 reference 计算 WER/CER。  
默认会先用 TN 的 `bin/bn_tts` 对 ASR 假设文本做一次归一化，再参与 WER/CER 计算。

### 一键运行（默认：冒烟 5 条）

```bash
cd /222042021/hehongkai/transsion_tts/Multilingual_LITs/e2e_test
chmod +x run_asr_wer.sh scripts/*.sh
./run_asr_wer.sh
```

### 全量合成完成后

```bash
WAV_DIR=output/wav/latest SKIP_ASR_SETUP=1 ./run_asr_wer.sh
```

### 输出

| 文件 | 说明 |
|------|------|
| `output/asr/<wav_run>/wer_report.json` | 汇总 WER/CER、top bad cases |
| `output/asr/<wav_run>/wer_per_utt.csv` | 逐条 ref/hyp/wer |
| `output/asr/<wav_run>/asr_hyp.tsv` | 转写结果 |

### 常用参数

```bash
WHISPER_MODEL=medium WHISPER_DEVICE=cuda ./run_asr_wer.sh   # 省显存
WAV_DIR=output/wav/bn_smoke_5 ./run_asr_wer.sh
TN_NORMALIZE_HYP=0 ./run_asr_wer.sh                         # 关闭 ASR 假设 TN
TN_BIN=/path/to/bn_tts ./run_asr_wer.sh                     # 指定 TN 可执行文件
```

### Mozilla 孟加拉语微调 Whisper

使用 HuggingFace `mozilla-ai/whisper-large-v3-bn`：

```bash
./run_asr_wer_mozilla_bn.sh
# 输出: output/asr/bn_smoke_5_mozilla_bn/

WAV_DIR=output/wav/20260527_105012 SKIP_ASR_SETUP=1 ./run_asr_wer_mozilla_bn.sh
```

### IndicConformer 孟加拉语 ASR

使用 AI4Bharat `indic-conformer-600m-multilingual`（`bn`）：

```bash
./run_asr_wer_indicconformer_bn.sh
# 输出: output/asr/bn_smoke_5_indicconformer_bn/

# 全量
WAV_DIR=output/wav/20260527_105012 SKIP_ASR_SETUP=1 ./run_asr_wer_indicconformer_bn.sh

# 可选解码器（默认 ctc）
DECODING=rnnt ./run_asr_wer_indicconformer_bn.sh
```

> 说明：该模型在 Hugging Face 是 gated。先在模型页同意条款，然后配置 token：
>
> ```bash
> export HF_TOKEN=你的token
> ```
>
> 若仍报 401/403，请确认账号已获访问权限并重新登录后再跑。
