# K-pop 舞蹈人物走位追踪（YOLOv8 Pose + ByteTrack）

这个项目目前提供两种能力：

1. 旧版：只画出人物框（`train_0.py`）。
2. 新版：识别每位人物框中心点、进行 ID 追踪，并输出俯视队形变化视频与队形快照（`formation_tracker.py`）。

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

- `annotated_tracking.mp4`：原视频 + 人物框 + 框中心点 + track ID
- `topdown_tracking.mp4`：俯视动态队形变化视频（仅显示当前位置，不画轨迹）
- `topdown_formation_snapshot.png`：最后一帧的俯视队形快照

运行时终端会持续显示进度（`[Progress] ...`），处理完成会打印 `Done.` 与输出文件路径。

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

### 3) 调整检测阈值

```bash
python formation_tracker.py --video input.mp4 --conf-thres 0.4
```

### 4) 防止点位突然跳动（推荐）

```bash
python formation_tracker.py --video input.mp4 --max-jump-px 80 --smooth-alpha 0.3
```

- `--max-jump-px`：若新点距离上一帧太远，视为异常跳点并忽略（避免跳点）。
- `--smooth-alpha`：指数平滑系数，越小越稳定，越大越灵敏；并会额外忽略左上角异常点。

### 5) 放大俯视点位

```bash
python formation_tracker.py --video input.mp4 --marker-radius 24
```

- `--marker-radius`：俯视图上人物圆点半径。

### 6) QuickTime 无法播放时（编码器相容性）

```bash
python formation_tracker.py --video input.mp4 --codec avc1
```

- 可选编码器：`avc1`、`H264`、`mp4v`、`MJPG`
- 程式会自动尝试你指定的编码器，失败会自动 fallback 到其他编码器，并在终端打印 `[Writer] ... uses codec: ...`

---

## 方法说明（对应你要的下一步）

1. 使用 `yolov8n-pose.pt` + ByteTrack 做人物追踪并获得稳定 ID。
2. 每个人使用**整个人物框的中心点**作为位置（不再使用头部关键点）。
3. 将框中心点通过单应性矩阵（homography）映射到俯视平面。
4. 对俯视坐标做异常跳点抑制 + 平滑（超大跳跃直接忽略、越界点忽略、左上角异常点保护）。
5. 俯视画面只绘制当前时刻的实心圆点与 ID，不绘制任何轨迹线。

---

## 旧版脚本

`train_0.py` 保留原始“只框人”的流程，可作为最基础检测脚本。
