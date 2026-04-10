# K-pop 舞蹈人物走位追踪（YOLOv8 Pose + ByteTrack）

这个项目目前提供两种能力：

1. 旧版：只画出人物框（`train_0.py`）。
2. 新版：识别每位人物头部位置、进行 ID 追踪，并输出俯视走位视频与轨迹图（`formation_tracker.py`）。

---

## 环境准备

```bash
python -m venv .venv
source .venv/bin/activate
pip install ultralytics opencv-python numpy
```

---

## 运行新版走位追踪

```bash
python formation_tracker.py --video /path/to/your_dance_video.mp4
```

默认会输出：

- `annotated_tracking.mp4`：原视频 + 人物框 + 头部点 + track ID
- `topdown_tracking.mp4`：俯视动态走位视频
- `topdown_trajectories.png`：整段视频的最终轨迹图

---

## 关键参数

### 1) 调整俯视画布大小

```bash
python formation_tracker.py --video input.mp4 --topdown-size 1200x800
```

### 2) 手动提供地面四点（建议）

如果你希望俯视图更接近真实舞台平面，建议传入舞台平面的四个角点（画面中的像素点）：

```bash
python formation_tracker.py --video input.mp4 \
  --manual-corners x1 y1 x2 y2 x3 y3 x4 y4
```

顺序必须是：

1. 左上（top-left）
2. 右上（top-right）
3. 右下（bottom-right）
4. 左下（bottom-left）

> 若不提供 `--manual-corners`，程序会使用整张画面作为平面近似，俯视图是“相对坐标”的简化版本。

### 3) 调整检测与关键点阈值

```bash
python formation_tracker.py --video input.mp4 --conf-thres 0.4 --kpt-conf-thres 0.4
```

---

## 方法说明（对应你要的下一步）

1. 使用 `yolov8n-pose.pt` 获取人体关键点。
2. 从头部关键点（鼻子/眼睛/耳朵）中筛选高置信点，取平均作为“头部位置”。
3. 使用 ByteTrack 给每个人分配稳定 ID。
4. 将头部点通过单应性矩阵（homography）映射到俯视平面。
5. 累积每个 ID 的历史点，生成轨迹线，最终可视化站位变化。

---

## 旧版脚本

`train_0.py` 保留原始“只框人”的流程，可作为最基础检测脚本。
