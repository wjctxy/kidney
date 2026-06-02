Read [](file:///Users/jason/Documents/Papers/kidney/human_step01_bandpass.py)

Read [](file:///Users/jason/Documents/Papers/kidney/human_step03_track.py)

Read [](file:///Users/jason/Documents/Papers/kidney/akebia/Example%20script/examples/akebia_human.m)

Read [](file:///Users/jason/Documents/Papers/kidney/akebia/Example%20script/src/compute/BlockInterp2Track.m)

Read [](file:///Users/jason/Documents/Papers/kidney/akebia/Example%20script/src/compute/ULM_localization2D_interp.m)

Read [](file:///Users/jason/Documents/Papers/kidney/human_step02_gaussian_filter.py)

## Akebia human vs 我们项目 Step1-3 对比

### 一图概览

```
Akebia (MATLAB):                    我们项目 (Python):
┌─────────────────────┐             ┌──────────────────────┐
│ .mat IQ 插值数据     │             │ step00: DICOM→npy    │
├─────────────────────┤             ├──────────────────────┤
│ butter bandpass     │ ←──对应──→ │ step01: sosfiltfilt  │
│ (可选, 仅 Rapid)    │             │ 带通 (signed 输出)   │
├─────────────────────┤             ├──────────────────────┤
│ Gaussian + 阈值     │             │ step02: Gaussian     │
│ + 局部极大值        │ ←──对应──→ │ 平滑 (保留 signed)   │
│ (合并在一函数内)    │             ├──────────────────────┤
│                     │             │ step03: 取正+峰值    │
│                     │             │ 检测+匈牙利追踪      │
├─────────────────────┤             ├──────────────────────┤
│ SimpleTracker 追踪  │ ←──对应──→ │ scipy 匈牙利追踪     │
└─────────────────────┘             └──────────────────────┘
```

### 详细差异

| 环节             | Akebia human                                              | 我们项目                                                     |
| ---------------- | --------------------------------------------------------- | ------------------------------------------------------------ |
| **数据格式**     | `.mat` 复数 IQ 插值数据                                   | `.npy` float32 灰度帧                                        |
| **带通滤波**     | MATLAB `filter`（因果滤波）                               | `scipy.sosfiltfilt`（零相位）                                |
| **带通时机**     | 跟 Gaussian+定位 合并在一个函数里；只有 Rapid 用          | 独立为 step01，始终执行                                      |
| **高斯+检测**    | 合并：`imgaussfilt` → `imregionalmax` → 固定 dB 阈值(100) | 拆分：step02 只做平滑保留 signed；step03 `np.maximum(frame,0)` 取正 + `mean+N*std` 自适应阈值 |
| **峰值检测**     | `imregionalmax`（局部极大值）                             | `peak_local_max`（带 min_distance 间距约束）                 |
| **定位精度**     | 插值数据本身有亚像素精度                                  | 当前为像素级整数坐标                                         |
| **追踪**         | SimpleTracker（LOTUS/PALA）                               | `scipy.linear_sum_assignment` 匈牙利匹配                     |
| **分块处理**     | 按 block 分批处理（parfor）                               | 逐帧串行                                                     |
| **Profile 机制** | 两套 Rapid/Slow 参数，同时运行                            | 一套参数对应一次运行                                         |

### 核心架构差异

1. **合并 vs 拆分**：Akebia 把滤波→定位→追踪写在一个 `BlockInterp2Track` 函数里；我们拆成 step01/02/03 三个独立脚本，中间结果可检查。

2. **阈值策略**：Akebia 用固定 dB 阈值(100)；我们用自适应 `mean + threshold_std * std`，更能适应不同强度数据。

3. **数据前提**：Akebia 输入是 IQ 插值数据（已有亚像素信息），我们输入是像素级灰度图，所以定位精度有别。

4. **追踪库**：Akebia 用 SimpleTracker（基于 LOTUS/PALA 的成熟 MATLAB 追踪器）；我们用 SciPy 匈牙利匹配，逻辑类似但细节实现不同。



Read [](file:///Users/jason/Documents/Papers/kidney/akebia/Example%20script/src/app/LoadData.m)

Read [](file:///Users/jason/Documents/Papers/kidney/akebia/Example%20script/src/app/CreateTrackingProfile.m)

Read [](file:///Users/jason/Documents/Papers/kidney/ulm_io.py#1-1), lines 1 to 200

Read [](file:///Users/jason/Documents/Papers/kidney/akebia/Example%20script/src/compute/ULM_localization2D_interp.m)

**核心区别**：MATLAB 用的是"插值波束合成后的数据"（已经有相位信息且空间密度高），我们的是 DICOM 最终 B 超灰度图（相位已丢失）。

### 具体差别

|              | Akebia `.mat` IQ 插值数据            | 我们 `.npy` 灰度帧                                           |
| ------------ | ------------------------------------ | ------------------------------------------------------------ |
| **数据类型** | 复数（实部+虚部），有相位            | 实数，只有亮度                                               |
| **来源**     | 临床超声机直接导出的插值波束合成数据 | 标准 DICOM cine-loop                                         |
| **预处理**   | 原始 IQ 已做空间插值，网格更密       | 波束合成→包络检测→对数压缩→扫描转换→DICOM 编码，已经过全部后处理 |
| **定位精度** | 输入本身就接近亚像素                 | 整数像素网格                                                 |
| **定位方法** | 简单 `imregionalmax` + 固定 dB 阈值  | `peak_local_max` + 自适应阈值                                |

### Step0：能不能照做 IQ 插值？

**不能。** 原因：
1. DICOM 图像已经丢失了所有**相位信息**，IQ 复数的实部/虚部不复存在
2. 扫描转换、对数压缩、RGB 编码这些都是**不可逆**的
3. 要有 IQ 数据，需要超声扫描仪在波束合成后、后处理前直接导出（通常 `.mat` 或专用格式），不是标准 DICOM 能给的

#### 替代方案（提升定位精度）

既然无法从 DICOM 恢复相位信息，可以走这些路：

1. **径向对称法**：项目里 mouse_step02_radial_symmetry_detect.py 已经实现了亚像素定位（见 mouse_step02_radial_symmetry_detect.py），可以在人类链路的 step03 里也加上
2. **高斯拟合 / 质心法**：围绕每个 `peak_local_max` 做局部 2D 高斯拟合
3. **空间超分辨率插值**：先把整帧插值到更密网格，再做检测

如果把径向对称法接入人类链路，预计定位精度会有一个量级的提升。需要我帮你加上吗？





Optimized tool selection

Read [](file:///Users/jason/Documents/Papers/kidney/akebia/Example%20script/lib/SimpleTracker/SimpleTracker)

Read [](file:///Users/jason/Documents/Papers/kidney/akebia/Example%20script/lib/SimpleTracker/SimpleTracker/simpletracker.m#1-1), lines 1 to 100

Read [](file:///Users/jason/Documents/Papers/kidney/akebia/Example%20script/lib/SimpleTracker/SimpleTracker/hungarianlinker.m#1-1), lines 1 to 100

Read [](file:///Users/jason/Documents/Papers/kidney/akebia/Example%20script/lib/SimpleTracker/SimpleTracker/hungarianlinker.m#100-100), lines 100 to 119

Read [](file:///Users/jason/Documents/Papers/kidney/akebia/Example%20script/lib/SimpleTracker/SimpleTracker/simpletracker.m#100-100), lines 100 to 300

## 我们项目匈牙利追踪对比

### 一图概览

```
Akebia SimpleTracker:                    我们项目 ulm_tracking.py:
┌──────────────────────────┐            ┌──────────────────────────┐
│ 第一遍：逐帧匈牙利匹配    │            │ 逐帧匈牙利匹配            │
│  (hungarianlinker)       │            │  (_assign → scipy)       │
├──────────────────────────┤            ├──────────────────────────┤
│ 第二遍：gap-closing      │            │ missing 计数 +            │
│  跨越1~N帧寻找未匹配点   │            │ MAX_MISSING_FRAMES 超时   │
│  (nearestneighborlinker) │            │ 终止（不等效显式桥接）    │
├──────────────────────────┤            ├──────────────────────────┤
│ 构建稀疏邻接矩阵 → track  │            │ 维护 active/finished 列表  │
└──────────────────────────┘            └──────────────────────────┘
```

### 核心差异点

| 维度             | Akebia SimpleTracker                                      | 我们项目                                                     |
| ---------------- | --------------------------------------------------------- | ------------------------------------------------------------ |
| **距离度量**     | 欧氏距离的**平方** (`sum(diff.^2)`)                       | 标准欧氏距离 (`cdist`)                                       |
| **匈牙利实现**   | Munkres（Yi Cao 独立实现）                                | SciPy `linear_sum_assignment`                                |
| **最大距离过滤** | 距离 > `max²` 设为 `Inf`，由 Munkres 内部处理             | `cost > max` 设为 `1e9`，后过滤                              |
| **Gap Closing**  | **显式两遍**：找到所有未匹配的轨迹尾/头，跨帧做最近邻桥接 | **隐式**：轨迹允许丢失 ≤ `MAX_MISSING_FRAMES`，靠下一帧匈牙利自然接上 |
| **Gap 搜索范围** | 显式扫描 `i+2` 到 `i+MaxGapClosing` 帧                    | 不主动搜索，依赖轨迹存活期内匈牙利重新匹配                   |
| **数据结构**     | 构建 `sparse` 邻接矩阵，然后 `graph` 遍历生成 track       | 维护 `active` / `finished` 轨迹列表                          |
| **最近邻备用**   | 支持切换到 `NearestNeighbor` 模式（O(n²)）                | 只用匈牙利                                                   |
| **输入格式**     | 每帧的二维坐标数组 `N×2`                                  | `frame_id` 分组的检测点字典                                  |

### 最关键的区别：Gap Closing

这是**最大差异**：

**Akebia**：匈牙利匹配之后专门跑一遍 `gap-closing`。遍历所有无匹配的轨迹终点，向前搜索 1~`MaxGapClosing` 帧内无匹配的轨迹起点，如果欧氏距离够近就用最近邻连上。这意味着**即使中间帧完全没检测到那个微泡，SimpleTracker 也能主动把断开的轨迹桥接起来**。

**我们**：不做显式桥接。轨迹丢失匹配时 `missing += 1`，如果 `missing ≤ MAX_MISSING_FRAMES` 就继续存活。下一帧匈牙利框架自然尝试匹配——但这要求中间帧**至少有一个附近的候选点被匈牙利选中**，否则轨迹就断了。如果中间帧完全没检测到该微泡（又没有被匈牙利考虑），我们的轨迹会累积 missing 计数直到超时断裂，不会被"跳帧桥接"。

### 总结

Akebia SimpleTracker 的 gap-closing 更鲁棒：微型气泡如果因噪声或遮挡短暂消失再出现，只要位移在范围内，Akebia 会主动跨帧接上。我们的实现依赖匈牙利自然延续，在缺帧场景下更容易丢轨迹。
