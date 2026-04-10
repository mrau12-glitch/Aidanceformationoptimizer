import argparse
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


HEAD_KP_INDEXES = [0, 1, 2, 3, 4]  # nose, left/right eye, left/right ear


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Track dancers' head positions and render a top-down formation view."
    )
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument(
        "--pose-model",
        default="yolov8n-pose.pt",
        help="Ultralytics pose model path (default: yolov8n-pose.pt)",
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
        default="topdown_trajectories.png",
        help="Output top-down trajectory image path",
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
        "--kpt-conf-thres", type=float, default=0.35, help="Keypoint confidence threshold"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show live windows while processing (press q to quit)",
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
        # fallback: full frame to top-down canvas (not true physical calibration)
        src = np.array(
            [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32
        )

    dst = np.array(
        [[0, 0], [topdown_w - 1, 0], [topdown_w - 1, topdown_h - 1], [0, topdown_h - 1]],
        dtype=np.float32,
    )
    return cv2.getPerspectiveTransform(src, dst)


def extract_head_point(
    keypoints_xy: np.ndarray,
    keypoints_conf: np.ndarray,
    conf_thres: float,
) -> Optional[Tuple[float, float]]:
    valid_points = []
    for idx in HEAD_KP_INDEXES:
        if idx < len(keypoints_conf) and keypoints_conf[idx] >= conf_thres:
            valid_points.append(keypoints_xy[idx])

    if not valid_points:
        return None

    valid_np = np.array(valid_points, dtype=np.float32)
    return float(valid_np[:, 0].mean()), float(valid_np[:, 1].mean())


def to_topdown(point_xy: Tuple[float, float], homography: np.ndarray) -> Tuple[int, int]:
    p = np.array([[[point_xy[0], point_xy[1]]]], dtype=np.float32)
    mapped = cv2.perspectiveTransform(p, homography)[0][0]
    return int(mapped[0]), int(mapped[1])


def make_topdown_canvas(width: int, height: int) -> np.ndarray:
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = (20, 20, 20)

    # light grid for readability
    for x in range(0, width, 100):
        cv2.line(canvas, (x, 0), (x, height), (45, 45, 45), 1)
    for y in range(0, height, 100):
        cv2.line(canvas, (0, y), (width, y), (45, 45, 45), 1)

    cv2.rectangle(canvas, (2, 2), (width - 3, height - 3), (80, 80, 80), 2)
    return canvas


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

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_annotated = cv2.VideoWriter(args.output_annotated, fourcc, fps, (frame_w, frame_h))
    out_topdown = cv2.VideoWriter(
        args.output_topdown_video, fourcc, fps, (topdown_w, topdown_h)
    )

    colors = generate_colors()
    tracks_topdown: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    homography: Optional[np.ndarray] = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

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
            kpts = r.keypoints

            if boxes is not None and kpts is not None and boxes.id is not None:
                xyxy = boxes.xyxy.cpu().numpy()
                cls = boxes.cls.cpu().numpy().astype(int)
                ids = boxes.id.cpu().numpy().astype(int)
                kp_xy = kpts.xy.cpu().numpy()
                kp_conf = kpts.conf.cpu().numpy()

                for i in range(len(ids)):
                    if cls[i] != 0:  # person class only
                        continue

                    tid = int(ids[i])
                    color = colors[tid % len(colors)]
                    x1, y1, x2, y2 = map(int, xyxy[i])

                    head_xy = extract_head_point(kp_xy[i], kp_conf[i], args.kpt_conf_thres)

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

                    if head_xy is None:
                        continue

                    hx, hy = int(head_xy[0]), int(head_xy[1])
                    cv2.circle(annotated, (hx, hy), 5, color, -1)

                    top_pt = to_topdown(head_xy, homography)
                    tracks_topdown[tid].append(top_pt)

                    if len(tracks_topdown[tid]) >= 2:
                        cv2.polylines(
                            topdown_frame,
                            [np.array(tracks_topdown[tid], dtype=np.int32)],
                            isClosed=False,
                            color=color,
                            thickness=2,
                        )

                    cv2.circle(topdown_frame, top_pt, 7, color, -1)
                    cv2.putText(
                        topdown_frame,
                        str(tid),
                        (top_pt[0] + 8, top_pt[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2,
                    )

        out_annotated.write(annotated)
        out_topdown.write(topdown_frame)

        if args.show:
            cv2.imshow("Annotated Tracking", annotated)
            cv2.imshow("Top-down Tracking", topdown_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    final_map = make_topdown_canvas(topdown_w, topdown_h)
    for tid, points in tracks_topdown.items():
        if not points:
            continue
        color = colors[tid % len(colors)]
        cv2.polylines(
            final_map,
            [np.array(points, dtype=np.int32)],
            isClosed=False,
            color=color,
            thickness=2,
        )
        cv2.circle(final_map, points[-1], 8, color, -1)
        cv2.putText(
            final_map,
            f"ID {tid}",
            (points[-1][0] + 8, points[-1][1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

    cv2.imwrite(args.output_topdown_image, final_map)

    cap.release()
    out_annotated.release()
    out_topdown.release()
    cv2.destroyAllWindows()

    print("Done.")
    print(f"Annotated video: {args.output_annotated}")
    print(f"Top-down video: {args.output_topdown_video}")
    print(f"Top-down trajectories image: {args.output_topdown_image}")


if __name__ == "__main__":
    main()
