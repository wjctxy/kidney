请修改当前 ULM 项目的 Step 01 可视化逻辑，目标是让 PNG 可视化尽量忠实反映算法主数据中的真实数值，不再通过 abs 或 percentile normalization 改变视觉效果。

背景：
当前 Step 01 是 human DICOM 的时域带通滤波。算法主输出是：

human_dcm/step01_bandpass/human_filtered.npy

该文件保存的是 temporal_bandpass() 的原始输出，应该是 signed float32 数据，也就是允许正值和负值。bandpass 后的数据表示每个像素在目标频段内的时间动态变化，而不是普通灰度亮度图。

当前存在的问题：
1. ulm_visualization.py 中 normalize_for_display() 使用 1%–99% 百分位拉伸，会让 raw 和 bandpass 图像被单独拉满对比度，导致图像亮度和真实数值不一致。
2. save_frame_comparison() 中对 after 使用了 np.abs(after)，会把负向波动也显示成亮点，导致 bandpass 图像看起来比实际更亮、更花。
3. save_detection_preview() 中也对 frame 使用了 np.abs(frame)，这会改变 signed bandpass 数据的语义。
4. 目前 PNG 图像和 npy 主数据表现不统一，容易误导后续判断。

修改目标：
1. 去掉可视化中的 percentile normalization。
2. 去掉可视化中的 np.abs。
3. PNG 中显示的 bandpass 图应使用 signed bandpass 原始数据。
4. 图像标题和文件名要明确说明这是 signed bandpass output，不是 magnitude preview。
5. 保留 colorbar，帮助判断真实数值范围。
6. 不改变算法 step 之间的输入输出契约，仍然只通过 npy/json/csv 传递数据。
7. 不要对 human_filtered.npy 做任何 abs、normalization、clipping。

请按以下要求修改。

一、修改 ulm_visualization.py

1. 删除 normalize_for_display() 

2. 新增或替换一个函数，用于直接显示真实数值图像：

def save_frame_png_raw_scale(frame, path, title, cmap="gray", vmin=None, vmax=None):
    - 不做 np.abs
    - 不做 percentile normalization
    - 直接 plt.imshow(frame, cmap=cmap, vmin=vmin, vmax=vmax)
    - 添加 colorbar
    - 保存 PNG

3. 修改 save_frame_comparison()：
当前逻辑：

axes[0].imshow(normalize_for_display(before), cmap="gray")
axes[1].imshow(normalize_for_display(np.abs(after)), cmap="gray")

请改为：
- before 直接显示 before
- after 直接显示 after
- 不使用 normalize_for_display
- 不使用 np.abs
- raw 图建议 cmap="gray"
- bandpass 图建议 cmap="seismic" 或 "coolwarm"，因为 bandpass 是 signed 数据，正负值都需要显示
- bandpass 图的 vmin/vmax 建议使用对称范围，例如：

lim = max(abs(np.nanmin(after)), abs(np.nanmax(after)))
vmin = -lim
vmax = lim

- 每个子图都添加 colorbar
- 标题明确使用：
  "Raw frame"
  "Signed bandpass frame"

4. 修改 save_detection_preview()：
当前逻辑：

plt.imshow(normalize_for_display(np.abs(frame)), cmap="gray")

请改成：
- 不使用 np.abs
- 不使用 normalize_for_display
- 直接显示 frame
- 如果 frame 是 signed bandpass，使用 cmap="seismic" 或 "coolwarm"
- 添加 colorbar
- 检测点仍然可以叠加

5. 修改 temporal projection 相关函数：
如果已有 save_temporal_projection_comparison()，请确保：
- std projection 使用 np.std(filtered_frames, axis=0)，不要使用 np.abs(filtered_frames)
- mean projection 如果用于 signed bandpass，应使用 np.mean(filtered_frames, axis=0)，不要使用 mean(abs(...))
- 不做 percentile normalization
- 显示 signed map 时使用 diverging colormap 并添加 colorbar
- 显示 std map 时可以使用 gray，因为 std 本身非负

二、修改 human_step01_bandpass.py

1. 确保 human_filtered.npy 保存的是原始 signed bandpass 输出：

out[:, y0:y1, :] = temporal_bandpass(...)

不要做：
- np.abs
- normalize
- clip
- min-max scale

2. 修改单帧预览。
不要再保存：

np.abs(filtered[0])

也不要展示第 0 帧。请使用中间帧：

preview_idx = n_frames // 2

然后保存：

save_frame_png_raw_scale(
    filtered[preview_idx],
    output_dir / f"signed_bandpass_frame_{preview_idx:04d}.png",
    f"Signed bandpass frame {preview_idx}",
    cmap="seismic",
    vmin=-lim,
    vmax=lim,
)

其中 lim 根据该帧的最大绝对值计算。

3. 修改前后对比图。
使用：

save_frame_comparison(
    frames[preview_idx],
    filtered[preview_idx],
    output_dir / "raw_vs_signed_bandpass_frame_comparison.png",
    f"Raw frame {preview_idx}",
    f"Signed bandpass frame {preview_idx}",
)

4. 修改时间曲线图。
保留上下两个子图：
- 上图 raw intensity
- 下图 signed bandpass intensity
不要把 raw 和 bandpass 放到同一个 y 轴里。
不要对 bandpass 曲线做 abs。
不要 normalize 曲线。

5. 输出文件命名建议：

human_filtered.npy
signed_bandpass_frame_XXXX.png
raw_vs_signed_bandpass_frame_comparison.png
bandpass_temporal_curve_center.png
bandpass_temporal_curve_max_std.png
bandpass_std_projection.png
bandpass_stats.json

三、增加统计输出 bandpass_stats.json

请在 Step 01 结束后保存统计信息，帮助判断真实数值变化，而不是靠归一化 PNG 判断。

统计内容包括：
- raw mean
- raw std
- raw min
- raw max
- raw p1
- raw p50
- raw p99
- bandpass mean
- bandpass std
- bandpass min
- bandpass max
- bandpass p1
- bandpass p50
- bandpass p99
- bandpass abs_p95
- bandpass abs_p99
- fps
- lowcut
- highcut
- n_frames

注意：统计可以计算 abs_p95 / abs_p99，但不要把 abs 后的数据保存为主数据，也不要用 abs 生成默认 PNG。

四、代码风格要求

1. 所有函数加简洁 docstring。
2. 不要破坏原有 step 的 npy/json/csv 输入输出约定。
3. 不要引入新的重型依赖。
4. 修改后保证以下命令仍可运行：

python human_step01_bandpass.py

5. 修改完成后，请简要列出：
- 改了哪些文件
- 哪些地方去掉了 normalization
- 哪些地方去掉了 abs
- 新生成哪些 PNG/JSON
- human_filtered.npy 是否仍为 signed float32 数据

最终目的：
让可视化结果和主数据保持一致。也就是说，PNG 看到的 signed bandpass 图，应直接来自 human_filtered.npy 中对应帧的数据，不再通过 abs 或 percentile normalization 改变含义。