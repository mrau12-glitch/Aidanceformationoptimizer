import argparse
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Track dancers by bbox center and render a top-down formation view."
    )
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument(
        "--pose-model",
        default="yolov8n-pose.pt",
        help="Ultralytics model path used for tracking (default: yolov8n-pose.pt)",
    )
    parser.add_argument(
        "--output-annotated",
        default="annotated_tracking.mp4",
        help="Output annotated tracking video path",
    )
    parser.add_argument(
        "--output-topdown-video",
        default="topdown_tracking.mp4",
        help="Output top-down tracking video path",
    )
    parser.add_argument(
        "--output-topdown-image",
        default="topdown_formation_snapshot.png",
        help="Output final top-down formation snapshot path",
    )
    parser.add_argument(
        "--topdown-size",
        default="900x600",
        help="Top-down canvas size, e.g. 900x600",
    )
    parser.add_argument(
        "--conf-thres", type=float, default=0.35, help="Detection confidence threshold"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show live windows while processing (press q to quit)",
    )
    parser.add_argument(
        "--codec",
        default="avc1",
        help="Preferred output codec (e.g. avc1, H264, mp4v, MJPG)",
    )
    parser.add_argument(
        "--manual-corners",
        nargs=8,
        type=float,
        metavar=("x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4"),
        help=(
            "Optional source plane points in video in order: top-left, top-right,"
            " bottom-right, bottom-left"
        ),
    )
    parser.add_argument(
        "--marker-radius",
        type=int,
        default=24,
        help="Circle radius for each dancer in top-down view",
    )
    parser.add_argument(
        "--max-jump-px",
        type=float,
        default=60.0,
        help="Reject raw top-down updates farther than this distance from last point",
    )
    parser.add_argument(
        "--smooth-alpha",
        type=float,
        default=0.25,
        help="EMA smoothing factor for top-down point update (0-1)",
    )
    return parser.parse_args()


def generate_colors(n: int = 200) -> List[Tuple[int, int, int]]:
    rng = np.random.default_rng(42)
    return [tuple(map(int, c)) for c in rng.integers(60, 255, size=(n, 3))]


def build_homography(
    frame_shape: Tuple[int, int, int],
    topdown_w: int,
    topdown_h: int,
    manual_corners: Optional[List[float]],
) -> np.ndarray:
    h, w = frame_shape[:2]

    if manual_corners:
        src = np.array(manual_corners, dtype=np.float32).reshape(4, 2)
    else:
        src = np.array(
            [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32
        )

    dst = np.array(
        [[0, 0], [topdown_w - 1, 0], [topdown_w - 1, topdown_h - 1], [0, topdown_h - 1]],
        dtype=np.float32,
    )
    return cv2.getPerspectiveTransform(src, dst)


def box_center(x1: float, y1: float, x2: float, y2: float) -> Tuple[float, float]:
    return (float(x1 + x2) * 0.5, float(y1 + y2) * 0.5)


def to_topdown(point_xy: Tuple[float, float], homography: np.ndarray) -> Tuple[float, float]:
    p = np.array([[[point_xy[0], point_xy[1]]]], dtype=np.float32)
    mapped = cv2.perspectiveTransform(p, homography)[0][0]
    return float(mapped[0]), float(mapped[1])


def stabilize_point(
    track_id: int,
    raw_pt: Tuple[float, float],
    last_points: Dict[int, Tuple[float, float]],
    max_jump_px: float,
    smooth_alpha: float,
    topdown_w: int,
    topdown_h: int,
) -> Optional[Tuple[int, int]]:
    raw_x, raw_y = float(raw_pt[0]), float(raw_pt[1])

    if not np.isfinite(raw_x) or not np.isfinite(raw_y):
        if track_id in last_points:
            prev_x, prev_y = last_points[track_id]
            return int(prev_x), int(prev_y)
        return None

    has_prev = track_id in last_points
    if has_prev:
        prev_x, prev_y = last_points[track_id]
    else:
        prev_x, prev_y = raw_x, raw_y

    outside_canvas = raw_x < 0 or raw_y < 0 or raw_x >= topdown_w or raw_y >= topdown_h
    if outside_canvas:
        if has_prev:
            return int(prev_x), int(prev_y)
        return None

    dx = raw_x - prev_x
    dy = raw_y - prev_y
    dist = float(np.hypot(dx, dy))

    near_top_left_corner = raw_x <= 40 and raw_y <= 40
    if near_top_left_corner and not has_prev:
        return None
    if has_prev and near_top_left_corner and dist > 5:
        return int(prev_x), int(prev_y)

    if has_prev and dist > max_jump_px:
        return int(prev_x), int(prev_y)

    alpha = min(1.0, max(0.0, smooth_alpha))
    smoothed_x = alpha * raw_x + (1.0 - alpha) * prev_x
    smoothed_y = alpha * raw_y + (1.0 - alpha) * prev_y
    last_points[track_id] = (smoothed_x, smoothed_y)
    return int(smoothed_x), int(smoothed_y)


def make_topdown_canvas(width: int, height: int) -> np.ndarray:
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = (15, 15, 15)
    return canvas


def create_video_writer(
    output_path: str,
    fps: float,
    size: Tuple[int, int],
    preferred_codec: str,
) -> cv2.VideoWriter:
    codec_candidates = [preferred_codec, "avc1", "H264", "mp4v", "MJPG"]
    tried = []
    for codec in codec_candidates:
        if codec in tried:
            continue
        tried.append(codec)
        writer = cv2.VideoWriter(
            output_path, cv2.VideoWriter_fourcc(*codec), fps, size
        )
        if writer.isOpened():
            print(f"[Writer] {output_path} uses codec: {codec}")
            return writer
        writer.release()

    raise RuntimeError(
        f"Cannot create VideoWriter for {output_path}. Tried codecs: {', '.join(tried)}"
    )


def main() -> None:
    args = parse_args()
    topdown_w, topdown_h = map(int, args.topdown_size.lower().split("x"))

    model = YOLO(args.pose_model)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_annotated = create_video_writer(
        args.output_annotated, fps, (frame_w, frame_h), args.codec
    )
    out_topdown = create_video_writer(
        args.output_topdown_video, fps, (topdown_w, topdown_h), args.codec
    )

    colors = generate_colors()
    stable_points: Dict[int, Tuple[float, float]] = {}
    homography: Optional[np.ndarray] = None
    last_topdown_frame = make_topdown_canvas(topdown_w, topdown_h)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        if homography is None:
            homography = build_homography(frame.shape, topdown_w, topdown_h, args.manual_corners)

        results = model.track(
            source=frame,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False,
            conf=args.conf_thres,
        )

        annotated = frame.copy()
        topdown_frame = make_topdown_canvas(topdown_w, topdown_h)

        if results and len(results) > 0:
            r = results[0]
            boxes = r.boxes

            if boxes is not None and boxes.id is not None:
                xyxy = boxes.xyxy.cpu().numpy()
                cls = boxes.cls.cpu().numpy().astype(int)
                ids = boxes.id.cpu().numpy().astype(int)

                for i in range(len(ids)):
                    if cls[i] != 0:
                        continue

                    tid = int(ids[i])
                    color = colors[tid % len(colors)]
                    x1, y1, x2, y2 = map(int, xyxy[i])

                    cx, cy = box_center(x1, y1, x2, y2)

                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        annotated,
                        f"ID {tid}",
                        (x1, max(20, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2,
                    )
                    cv2.circle(annotated, (int(cx), int(cy)), 6, color, -1)

                    raw_top_pt = to_topdown((cx, cy), homography)
                    top_pt = stabilize_point(
                        tid,
                        raw_top_pt,
                        stable_points,
                        args.max_jump_px,
                        args.smooth_alpha,
                        topdown_w,
                        topdown_h,
                    )
                    if top_pt is None:
                        continue

                    cv2.circle(topdown_frame, top_pt, args.marker_radius, color, -1)
                    cv2.putText(
                        topdown_frame,
                        str(tid),
                        (top_pt[0] + args.marker_radius + 4, top_pt[1] - args.marker_radius),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2,
                    )

        out_annotated.write(annotated)
        out_topdown.write(topdown_frame)
        last_topdown_frame = topdown_frame.copy()

        if total_frames > 0 and (frame_idx % 30 == 0 or frame_idx == total_frames):
            progress = (frame_idx / total_frames) * 100.0
            print(f"[Progress] {frame_idx}/{total_frames} frames ({progress:.1f}%)")
        elif total_frames <= 0 and frame_idx % 60 == 0:
            print(f"[Progress] processed {frame_idx} frames")

        if args.show:
            cv2.imshow("Annotated Tracking", annotated)
            cv2.imshow("Top-down Tracking", topdown_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.imwrite(args.output_topdown_image, last_topdown_frame)

    cap.release()
    out_annotated.release()
    out_topdown.release()
    cv2.destroyAllWindows()

    print("Done.")
    print(f"Annotated video: {args.output_annotated}")
    print(f"Top-down video: {args.output_topdown_video}")
    print(f"Top-down formation snapshot image: {args.output_topdown_image}")


if __name__ == "__main__":
    main()
