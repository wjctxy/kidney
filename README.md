# Kidney ULM Processing

本仓库用于运行肾脏 CEUS/ULM 后处理流程。当前推荐主线是：

```text
DICOM
-> Step00 TIC 选稳定连续帧
-> Human rapid/slow Step01-04
-> Step05 cortex/exclude mask
-> Step05 肾小球候选计数
```

`human_dcm/` 只保存当前最新一次运行的中间结果；长期保留的样本 DICOM 和 mask 放在 `all_masks/`。

## 推荐入口

把样本 DICOM 放进 `all_masks/` 或直接用 `--dicom` 登记：

```bash
.venv/bin/python step00_preprocess.py --kind human --run-full-pipeline --sample 21.0
```

或：

```bash
.venv/bin/python step00_preprocess.py --kind human --run-full-pipeline --dicom 21.0.dcm
```

样本目录约定：

```text
all_masks/
  21.0/
    21.0.dcm
    cortex_mask.npy
    cortex_mask_overlay.png
    cortex_mask.log.json
    exclude_mask.npy
    exclude_mask_overlay.png
    exclude_mask.log.json
```

如果缺 `cortex_mask.npy` 或 `exclude_mask.npy`，全流程会先跑到 Step04，用慢速密度图进入交互式 mask 绘制；保存后默认重新从 Step00 开始，用 cortex 内 TIC 选最终帧段。

## Step00

单独运行预处理：

```bash
.venv/bin/python step00_preprocess.py --kind human
.venv/bin/python step00_preprocess.py --kind human --target-frames 400
```

默认设置在 `step00_preprocess.py` 顶部：

```text
DEFAULT_TARGET_FRAMES = 200
DEFAULT_MIN_FRAMES = 200
DEFAULT_SMOOTH_WINDOW = 15
DEFAULT_BASELINE_SECONDS = 2.0
DEFAULT_MIN_ENHANCEMENT = 0.01
```

Step00 会先读取完整 DICOM 并计算 TIC(time-intensity curve)。有 cortex mask 时，TIC 是 cortex 内平均 CEUS score；没有 mask 时，TIC 退回完整 ROI 平均值。默认选择 200 帧连续稳定窗口，不再默认取前 600 帧，也不按总帧数比例取帧。

输出：

```text
human_dcm/step00_preprocess/frames.npy
human_dcm/step00_preprocess/metadata.json
human_dcm/step00_preprocess/tic_raw.npy
human_dcm/step00_preprocess/tic_selection.png
human_dcm/step00_preprocess/selected_ranges.json
human_dcm/step00_preprocess/preview_frame_000.png
```

## Human Step01-04

全流程入口会自动跑 rapid/slow 两套 profile。新样本统一使用 `step00_preprocess.py --run-full-pipeline`，
旧的 label 型入口已删除。

Rapid/slow 参数定义在 `ulm_config.py` 的 `HUMAN_PROFILES`：

```text
rapid: bandpass [1.0, 5.5] Hz, max displacement 15 px/frame, min length 5
slow:  no bandpass, max displacement 4 px/frame,  min length 10
```

Step04 主要输出：

```text
human_dcm/step04_density_metrics/human_density_total.png
human_dcm/step04_density_metrics/human_density_rapid.png
human_dcm/step04_density_metrics/human_density_slow.png
human_dcm/step04_density_metrics/human_rapid_tracks.csv
human_dcm/step04_density_metrics/human_slow_tracks.csv
human_dcm/step04_density_metrics/human_metrics.csv
human_dcm/step04_density_metrics/human_summary.txt
```

## Step05 Mask 和计数

当前 mask 绘制和计数都在 `human_step05_glomeruli_count.py`：

```bash
.venv/bin/python human_step05_glomeruli_count.py --mode cortex
.venv/bin/python human_step05_glomeruli_count.py --mode exclude
.venv/bin/python human_step05_glomeruli_count.py --mode exec
```

默认背景图是当前 Step04 输出：

```text
human_dcm/step04_density_metrics/human_density_slow.png
```

默认 mask 输出：

```text
masks/cortex_mask.npy
masks/cortex_mask_overlay.png
masks/cortex_mask.log.json
masks/exclude_mask.npy
masks/exclude_mask_overlay.png
masks/exclude_mask.log.json
```

`--mode exec` 默认读取：

```text
human_dcm/step04_density_metrics/human_slow_tracks.csv
human_dcm/step04_density_metrics/human_rapid_tracks.csv
human_dcm/step04_density_metrics/human_density_slow.png
human_dcm/step04_density_metrics/human_density_rapid.png
masks/cortex_mask.npy
masks/exclude_mask.npy
```

输出到：

```text
human_dcm/step05_glomeruli_count/
```

Step05 当前是单一候选集计数，旧版 loose/strict 双输出已移除。健康样本默认使用 `healthy_calibration`，会在 `0.04-0.30 mm` 内自动扫 DBSCAN `eps`，目标靠近 450 且优先避免低于 400。病肾/未知样本使用固定公共参数：

```bash
.venv/bin/python human_step05_glomeruli_count.py \
  --mode exec \
  --count-mode diagnostic \
  --dbscan-eps-mm <healthy_common_eps>
```

## 健康样本公共参数

全流程每跑完一个健康样本，会保存轻量 Step05 摘要：

```text
archive/healthy_step05_runs/<sample_id>_summary.json
```

多个健康样本跑完后统计公共诊断参数：

```bash
.venv/bin/python calibrate_healthy_step05_params.py
```

输出：

```text
archive/healthy_common_step05_params.json
```

## 文档索引

```text
archive/0.总览.md       当前完整流程和运行顺序
archive/2.双路设计接口.md 当前 step 之间的接口
archive/3.数据格式.md   当前文件格式和字段含义
archive/8.step05.md     Step05 mask、计数、参数和输出解释
archive/9.step00.md     Step00 TIC 选帧和样本管理
```
