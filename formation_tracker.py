import argparse
from itertools import permutations
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
    parser.add_argument("--output-annotated", default="annotated_tracking.mp4", help="Output annotated tracking video path")
    parser.add_argument("--output-topdown-video", default="topdown_tracking.mp4", help="Output top-down tracking video path")
    parser.add_argument(
        "--output-topdown-image",
        default="topdown_formation_snapshot.png",
        help="Output final top-down formation snapshot path",
    )
    parser.add_argument("--topdown-size", default="900x600", help="Top-down canvas size")
    parser.add_argument("--conf-thres", type=float, default=0.35, help="Detection confidence")
    parser.add_argument("--show", action="store_true", help="Show live windows")
    parser.add_argument("--codec", default="avc1", help="Preferred output codec")
    parser.add_argument(
        "--manual-corners",
        nargs=8,
        type=float,
        metavar=("x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4"),
        help="Optional source plane points in video: TL TR BR BL",
    )
    parser.add_argument("--marker-radius", type=int, default=24, help="Top-down marker radius")
    parser.add_argument("--max-jump-px", type=float, default=60.0, help="Top-down jump gate")
    parser.add_argument("--smooth-alpha", type=float, default=0.25, help="EMA factor")

    parser.add_argument(
        "--selection-mode",
        choices=["none", "pick"],
        default="none",
        help="none: normal tracking IDs, pick: select dancers at start and track only them",
    )
    parser.add_argument(
        "--pick-indexes",
        default="",
        help="When selection-mode=pick, comma-separated indexes from detected people, e.g. 0,1,2,3",
    )
    parser.add_argument("--max-assign-dist", type=float, default=180.0, help="Max pixel distance for assignment")
    parser.add_argument("--pick-start-sec", type=float, default=30.0, help="Pick search start second")
    parser.add_argument("--pick-search-sec", type=float, default=15.0, help="Pick search window seconds")

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
        src = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)

    dst = np.array(
        [[0, 0], [topdown_w - 1, 0], [topdown_w - 1, topdown_h - 1], [0, topdown_h - 1]], dtype=np.float32
    )
    return cv2.getPerspectiveTransform(src, dst)


def box_center(x1: float, y1: float, x2: float, y2: float) -> Tuple[float, float]:
    return (float(x1 + x2) * 0.5, float(y1 + y2) * 0.5)


def to_topdown(point_xy: Tuple[float, float], homography: np.ndarray) -> Tuple[float, float]:
    p = np.array([[[point_xy[0], point_xy[1]]]], dtype=np.float32)
    mapped = cv2.perspectiveTransform(p, homography)[0][0]
    return float(mapped[0]), float(mapped[1])


def stabilize_point(
    key: int,
    raw_pt: Tuple[float, float],
    last_points: Dict[int, Tuple[float, float]],
    max_jump_px: float,
    smooth_alpha: float,
    topdown_w: int,
    topdown_h: int,
) -> Optional[Tuple[int, int]]:
    raw_x, raw_y = float(raw_pt[0]), float(raw_pt[1])
    if not np.isfinite(raw_x) or not np.isfinite(raw_y):
        return (int(last_points[key][0]), int(last_points[key][1])) if key in last_points else None

    has_prev = key in last_points
    prev_x, prev_y = last_points[key] if has_prev else (raw_x, raw_y)

    if raw_x < 0 or raw_y < 0 or raw_x >= topdown_w or raw_y >= topdown_h:
        return (int(prev_x), int(prev_y)) if has_prev else None

    dist = float(np.hypot(raw_x - prev_x, raw_y - prev_y))
    if raw_x <= 40 and raw_y <= 40 and (not has_prev or dist > 5):
        return (int(prev_x), int(prev_y)) if has_prev else None
    if has_prev and dist > max_jump_px:
        return int(prev_x), int(prev_y)

    alpha = min(1.0, max(0.0, smooth_alpha))
    smoothed_x = alpha * raw_x + (1.0 - alpha) * prev_x
    smoothed_y = alpha * raw_y + (1.0 - alpha) * prev_y
    last_points[key] = (smoothed_x, smoothed_y)
    return int(smoothed_x), int(smoothed_y)


def make_topdown_canvas(width: int, height: int) -> np.ndarray:
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = (15, 15, 15)
    return canvas


def spread_overlapping_points(
    points: Dict[int, Tuple[int, int]],
    marker_radius: int,
    width: int,
    height: int,
) -> Dict[int, Tuple[int, int]]:
    if len(points) <= 1:
        return points

    adjusted = dict(points)
    cluster_threshold = max(6, int(marker_radius * 0.8))
    spread_radius = max(6, int(marker_radius * 0.9))

    ids = list(points.keys())
    visited = set()
    for i, slot_id in enumerate(ids):
        if slot_id in visited:
            continue
        cx, cy = points[slot_id]
        cluster = [slot_id]
        visited.add(slot_id)
        for other_id in ids[i + 1 :]:
            if other_id in visited:
                continue
            ox, oy = points[other_id]
            if np.hypot(ox - cx, oy - cy) <= cluster_threshold:
                cluster.append(other_id)
                visited.add(other_id)

        if len(cluster) <= 1:
            continue

        for idx, cid in enumerate(cluster):
            theta = 2 * np.pi * idx / len(cluster)
            nx = int(round(cx + spread_radius * np.cos(theta)))
            ny = int(round(cy + spread_radius * np.sin(theta)))
            nx = max(0, min(width - 1, nx))
            ny = max(0, min(height - 1, ny))
            adjusted[cid] = (nx, ny)

    return adjusted


def create_video_writer(output_path: str, fps: float, size: Tuple[int, int], preferred_codec: str) -> cv2.VideoWriter:
    for codec in [preferred_codec, "avc1", "H264", "mp4v", "MJPG"]:
        writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*codec), fps, size)
        if writer.isOpened():
            print(f"[Writer] {output_path} uses codec: {codec}")
            return writer
        writer.release()
    raise RuntimeError(f"Cannot create VideoWriter for {output_path}")


def parse_pick_indexes(raw: str) -> List[int]:
    if not raw.strip():
        return []
    return [int(s.strip()) for s in raw.split(",") if s.strip()]


def assign_picked_slots(
    candidates: List[Tuple[int, Tuple[float, float]]],
    slot_prev_centers: Dict[int, Tuple[float, float]],
    slot_track_ids: Dict[int, int],
    max_assign_dist: float,
) -> Dict[int, Tuple[int, Tuple[float, float]]]:
    assigned: Dict[int, Tuple[int, Tuple[float, float]]] = {}
    if not candidates or not slot_prev_centers:
        return assigned

    candidate_by_tid = {tid: center for tid, center in candidates}
    used_tids = set()

    for slot_idx, tracked_tid in slot_track_ids.items():
        if tracked_tid in candidate_by_tid and slot_idx in slot_prev_centers:
            center = candidate_by_tid[tracked_tid]
            prev = slot_prev_centers[slot_idx]
            d = float(np.hypot(center[0] - prev[0], center[1] - prev[1]))
            if d <= max_assign_dist:
                assigned[slot_idx] = (tracked_tid, center)
                used_tids.add(tracked_tid)

    unresolved_slots = [s for s in slot_prev_centers.keys() if s not in assigned]
    remaining_candidates = [(tid, c) for tid, c in candidates if tid not in used_tids]
    if not unresolved_slots or not remaining_candidates:
        return assigned

    k = min(len(unresolved_slots), len(remaining_candidates))
    best_perm = None
    best_cost = float("inf")
    slot_subset = unresolved_slots[:k]

    for perm in permutations(remaining_candidates, k):
        total_cost = 0.0
        valid = True
        for slot_idx, (_, center) in zip(slot_subset, perm):
            prev = slot_prev_centers[slot_idx]
            d = float(np.hypot(center[0] - prev[0], center[1] - prev[1]))
            if d > max_assign_dist:
                valid = False
                break
            total_cost += d
        if valid and total_cost < best_cost:
            best_cost = total_cost
            best_perm = perm

    if best_perm is not None:
        for slot_idx, (tid, center) in zip(slot_subset, best_perm):
            assigned[slot_idx] = (tid, center)

    return assigned


def detect_people_boxes(model: YOLO, frame: np.ndarray, conf: float) -> np.ndarray:
    results = model.predict(source=frame, conf=conf, verbose=False)
    if not results or len(results) == 0 or results[0].boxes is None:
        return np.empty((0, 4), dtype=np.float32)
    boxes = results[0].boxes
    cls = boxes.cls.cpu().numpy().astype(int)
    xyxy = boxes.xyxy.cpu().numpy()
    return xyxy[cls == 0]


def find_pick_frame(
    cap: cv2.VideoCapture,
    model: YOLO,
    conf_thres: float,
    start_sec: float,
    search_sec: float,
) -> Tuple[np.ndarray, np.ndarray, int]:
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start_idx = int(max(0.0, start_sec) * fps)
    if total_frames > 0:
        start_idx = min(start_idx, max(0, total_frames - 1))

    search_frames = int(max(0.0, search_sec) * fps)
    end_idx = start_idx + search_frames
    if total_frames > 0:
        end_idx = min(end_idx, total_frames - 1)

    step = max(1, int(fps // 6))
    best_frame = None
    best_boxes = np.empty((0, 4), dtype=np.float32)
    best_idx = start_idx

    for frame_idx in range(start_idx, end_idx + 1, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        boxes = detect_people_boxes(model, frame, conf_thres)
        if len(boxes) > len(best_boxes):
            best_frame, best_boxes, best_idx = frame.copy(), boxes, frame_idx
        if len(boxes) > 0:
            return frame, boxes, frame_idx

    if best_frame is not None:
        return best_frame, best_boxes, best_idx

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret0, first_frame = cap.read()
    if not ret0:
        raise RuntimeError("Cannot read first frame")
    return first_frame, detect_people_boxes(model, first_frame, conf_thres), 0


def main() -> None:
    args = parse_args()
    topdown_w, topdown_h = map(int, args.topdown_size.lower().split("x"))

    model = YOLO(args.pose_model)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    ret0, first_frame = cap.read()
    if not ret0:
        raise RuntimeError("Cannot read first frame")

    first_boxes = np.empty((0, 4), dtype=np.float32)
    if args.selection_mode == "pick":
        _, first_boxes, pick_frame_idx = find_pick_frame(
            cap=cap,
            model=model,
            conf_thres=args.conf_thres,
            start_sec=args.pick_start_sec,
            search_sec=args.pick_search_sec,
        )
        pick_sec = pick_frame_idx / (cap.get(cv2.CAP_PROP_FPS) or 30.0)
        print(
            f"[Pick] using frame at ~{pick_sec:.2f}s "
            f"(start={args.pick_start_sec:.1f}s, window={args.pick_search_sec:.1f}s)"
        )
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    picked_slot_centers: Dict[int, Tuple[float, float]] = {}
    picked_slot_track_ids: Dict[int, int] = {}
    if args.selection_mode == "pick":
        print(f"[Pick] candidate frame persons: {len(first_boxes)}")
        for i, b in enumerate(first_boxes):
            x1, y1, x2, y2 = map(int, b)
            print(f"  index={i} box=({x1},{y1},{x2},{y2})")

        indexes = parse_pick_indexes(args.pick_indexes)
        if not indexes:
            raise RuntimeError("selection-mode=pick requires --pick-indexes, e.g. --pick-indexes 0,1,2,3")
        if len(first_boxes) == 0:
            raise RuntimeError(
                "No people found in pick search window. Try increasing --pick-start-sec or --pick-search-sec."
            )

        for slot_i, det_i in enumerate(indexes, start=1):
            if 0 <= det_i < len(first_boxes):
                x1, y1, x2, y2 = first_boxes[det_i]
                picked_slot_centers[slot_i] = box_center(x1, y1, x2, y2)

        if not picked_slot_centers:
            raise RuntimeError("No valid pick indexes were provided")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_annotated = create_video_writer(args.output_annotated, fps, (frame_w, frame_h), args.codec)
    out_topdown = create_video_writer(args.output_topdown_video, fps, (topdown_w, topdown_h), args.codec)

    colors = generate_colors()
    stable_points: Dict[int, Tuple[float, float]] = {}
    homography = build_homography(first_frame.shape, topdown_w, topdown_h, args.manual_corners)
    last_topdown_frame = make_topdown_canvas(topdown_w, topdown_h)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        results = model.track(source=frame, persist=True, tracker="bytetrack.yaml", verbose=False, conf=args.conf_thres)
        annotated = frame.copy()
        topdown_frame = make_topdown_canvas(topdown_w, topdown_h)

        candidates: List[Tuple[int, Tuple[float, float], Tuple[int, int, int, int]]] = []
        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            xyxy = boxes.xyxy.cpu().numpy()
            cls = boxes.cls.cpu().numpy().astype(int)
            ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else np.arange(len(xyxy))
            for i in range(len(xyxy)):
                if cls[i] != 0:
                    continue
                x1, y1, x2, y2 = map(int, xyxy[i])
                candidates.append((int(ids[i]), box_center(x1, y1, x2, y2), (x1, y1, x2, y2)))

        if args.selection_mode == "pick" and picked_slot_centers:
            det_candidates = [(tid, center) for tid, center, _ in candidates]
            assigned = assign_picked_slots(
                det_candidates,
                picked_slot_centers,
                picked_slot_track_ids,
                args.max_assign_dist,
            )

            for slot_idx, (track_tid, center) in assigned.items():
                picked_slot_centers[slot_idx] = center
                picked_slot_track_ids[slot_idx] = track_tid

            raw_top_points: Dict[int, Tuple[int, int]] = {}
            for slot_idx, center in picked_slot_centers.items():
                color = colors[slot_idx % len(colors)]
                cv2.putText(annotated, str(slot_idx), (int(center[0]) + 6, int(center[1]) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.circle(annotated, (int(center[0]), int(center[1])), 7, color, -1)

                raw_top_pt = to_topdown(center, homography)
                top_pt = stabilize_point(slot_idx, raw_top_pt, stable_points, args.max_jump_px, args.smooth_alpha, topdown_w, topdown_h)
                if top_pt is None and slot_idx in stable_points:
                    px, py = stable_points[slot_idx]
                    top_pt = (int(px), int(py))
                if top_pt is not None:
                    raw_top_points[slot_idx] = top_pt

            display_top_points = spread_overlapping_points(raw_top_points, args.marker_radius, topdown_w, topdown_h)
            for slot_idx, top_pt in display_top_points.items():
                color = colors[slot_idx % len(colors)]
                cv2.circle(topdown_frame, top_pt, args.marker_radius, color, -1)
                cv2.putText(topdown_frame, str(slot_idx), (top_pt[0] + args.marker_radius + 4, top_pt[1] - args.marker_radius), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        else:
            for tid, center, (x1, y1, x2, y2) in candidates:
                color = colors[tid % len(colors)]
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated, f"ID {tid}", (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.circle(annotated, (int(center[0]), int(center[1])), 6, color, -1)

                raw_top_pt = to_topdown(center, homography)
                top_pt = stabilize_point(tid, raw_top_pt, stable_points, args.max_jump_px, args.smooth_alpha, topdown_w, topdown_h)
                if top_pt is None:
                    continue
                cv2.circle(topdown_frame, top_pt, args.marker_radius, color, -1)
                cv2.putText(topdown_frame, str(tid), (top_pt[0] + args.marker_radius + 4, top_pt[1] - args.marker_radius), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        out_annotated.write(annotated)
        out_topdown.write(topdown_frame)
        last_topdown_frame = topdown_frame.copy()

        if total_frames > 0 and (frame_idx % 30 == 0 or frame_idx == total_frames):
            print(f"[Progress] {frame_idx}/{total_frames} frames ({(frame_idx / total_frames) * 100.0:.1f}%)")

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
