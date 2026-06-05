# Kidney ULM Processing

本项目按 `archive/超声定位显微镜技术应用_整理版.md` 整理为两条 ULM 后处理链路：

- 人类链路：时域带通滤波 → 二维高斯滤波 → 正向峰值检测与追踪 → 速度分组、密度图和指标
- 小鼠链路：SVD 滤波 → 径向对称法亚像素定位 → 匈牙利追踪与重建

当前只有人类样例 `11.0.dcm`。为了保持目录结构一致，该 DICOM 会分别放在：

```text
human_dcm/11.0.dcm
mouse_dcm/11.0.dcm
```

这些目录包含大文件和生成结果，已写入 `.gitignore`。

## 统一接口

所有算法 step 之间只传结构化数据：

```text
step00_preprocess/frames.npy        float32 [T,H,W]，CEUS score/灰度、裁剪 ROI、归一化到 [0,1]
step00_preprocess/metadata.json     fps、像素尺寸、ROI、帧数等元数据
step02_gaussian_filter/*_smoothed.npy Akebia 式局部极大值搜索用 Gaussian guide
step03_track/*_detections.csv       每行一个微泡候选点
step03_track/*tracks.csv              每行一个轨迹点
step04_density_metrics/*metrics.csv   轨迹指标和密度图结果
```

PNG 只作为最终可视化结果，不作为任何下游 step 的输入。

## Step 0：预处理

```bash
python step00_preprocess.py --kind human
python step00_preprocess.py --kind mouse
```

输出：

```text
human_dcm/step00_preprocess/frames.npy
human_dcm/step00_preprocess/metadata.json
human_dcm/step00_preprocess/preview_frame_000.png
```

`step0` 会读取 DICOM 中的超声区域标签，裁掉设备界面等冗余信息。彩色伪彩 CEUS 会转成橙黄色造影增强 score；单通道 DICOM 会按灰度强度归一化。

## 人类链路

Akebia-style 双 profile 入口：

```bash
python human_run_profiles.py --label stable_200_400
```

该入口会从同一个 Step00 输入分别运行 Rapid 和 Slow：

```text
Rapid: bandpass [1, 5.5] Hz, maxLinkingDistance=15, minLength=5
Slow: no bandpass, maxLinkingDistance=4, minLength=10
```

也可以逐步运行单 profile：

```bash
python human_step01_bandpass.py
python human_step02_gaussian_filter.py
python human_step03_track.py
python human_step04_density_metrics.py
```

输出集中在 `human_dcm/` 的分 step 子目录：

```text
step01_bandpass/human_filtered.npy
step01_bandpass/signed_bandpass_frame_XXXX.png
step01_bandpass/raw_vs_signed_bandpass_frame_comparison.png
step01_bandpass/bandpass_std_projection.png
step01_bandpass/bandpass_temporal_curve_center.png
step01_bandpass/bandpass_temporal_curve_max_std.png
step01_bandpass/bandpass_stats.json
step02_gaussian_filter/human_smoothed.npy
step02_gaussian_filter/signed_smoothed_frame_XXXX.png
step02_gaussian_filter/bandpass_vs_gaussian_frame_comparison.png
step02_gaussian_filter/gaussian_stats.json
step03_track/human_detections.csv
step03_track/detections_positive_response_frame_XXXX.png
step03_track/human_tracks.csv
step03_track/tracking_summary.txt
step04_density_metrics/human_low_speed_tracks.csv
step04_density_metrics/human_high_speed_tracks.csv
step04_density_metrics/human_density_total.png
step04_density_metrics/human_density_low_speed.png
step04_density_metrics/human_density_high_speed.png
step04_density_metrics/human_density_speed_overlay.png
step04_density_metrics/human_metrics.csv
step04_density_metrics/human_summary.txt
```

## 小鼠链路

```bash
python mouse_step01_svd_filter.py
python mouse_step02_radial_symmetry_detect.py
python mouse_step03_track_reconstruct.py
```

输出集中在 `mouse_dcm/`。当前没有真实小鼠样例，因此这条链路只按接口和算法要求实现，未用小鼠数据校准参数。

## Step 5：肾小球候选计数

`human_step05_glomeruli_count.py` 使用慢速轨迹的 normalized distance 在物理坐标和 isotropic grid 上生成肾小球候选 mask，再反向筛选肾小球相关轨迹点、重建轨迹分布图，并用 DBSCAN 聚类计数。核心鼠类参数默认是 `glomerulus_radius_mm=0.05`、`iso_spacing_mm=0.02`。

当前 Step 5 loose 计数已按 CT 层扫切片约 450 个肾小球标定：`loose_inside_frac=0.0`、`dbscan_eps_mm=0.23`、`dbscan_min_samples=1`。strict 计数保留为更保守参考。

交互式制作 cortex/exclude mask：

```bash
python draw_step5_masks.py \
  --image human_dcm/step04_density_metrics/stable_200_400/human_density_slow.png \
  --output masks/cortex_mask.npy \
  --overlay masks/cortex_mask_overlay.png \
  --mode cortex

python draw_step5_masks.py \
  --image human_dcm/step04_density_metrics/stable_200_400/human_density_slow.png \
  --output masks/exclude_mask.npy \
  --overlay masks/exclude_mask_overlay.png \
  --mode exclude
```

快捷键：`Enter` 完成当前 polygon，`n` 新建下一个 polygon，`u` 撤销上一个 polygon，`s` 保存，`q` 退出。mask 会保持输入 density/image 的原始 shape，Step 5 内部会负责转换到 isotropic grid。

也可以直接使用已打包的 macOS 可执行版：

```bash
step5/app/dist/Step5MaskDrawer/Step5MaskDrawer.exe \
  --image human_dcm/step04_density_metrics/stable_200_400/human_density_slow.png \
  --output masks/cortex_mask.npy \
  --overlay masks/cortex_mask_overlay.png \
  --mode cortex
```

```bash
python human_step05_glomeruli_count.py \
  --slow-tracks human_dcm/step04_density_metrics/stable_200_400/human_slow_tracks.csv \
  --fast-tracks human_dcm/step04_density_metrics/stable_200_400/human_rapid_tracks.csv \
  --slow-density human_dcm/step04_density_metrics/stable_200_400/human_density_slow.png \
  --fast-density human_dcm/step04_density_metrics/stable_200_400/human_density_rapid.png \
  --cortex-mask masks/cortex_mask.npy \
  --exclude-mask masks/exclude_mask.npy \
  --output-dir human_dcm/step05_glomeruli_count/stable_200_400
```

关键输出包括 `final_glomeruli_*.csv`、`filtered_points_*.csv`、`glomerular_track_distribution_*.png` 和 `glomerular_track_distribution_*_on_slow_density.png`。

如果没有 `cortex_mask` 也能运行，但会在 `summary.json` 中记录误检风险 warning。

## 快速烟测

完整 DICOM 有 1802 帧。当前默认先处理 600 帧，兼顾滤波稳定性和运行时间；如需处理全部帧，使用 `--max-frames 0`。

```bash
python step00_preprocess.py --kind human
python human_step01_bandpass.py
python human_step02_gaussian_filter.py
python human_step03_track.py
python human_step04_density_metrics.py
```
