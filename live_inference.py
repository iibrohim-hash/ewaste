"""
live_inference.py - UNet++ PCB segmentation with Intel RealSense D435I
Now with per-object depth (center of mass + masked median depth)
"""
import argparse, time
import cv2
import numpy as np
import torch
import pyrealsense2 as rs
import segmentation_models_pytorch as smp
from torchvision import transforms

CLASSES        = ["background", "board", "capacitor", "chip", "knob"]
NUM_CLASSES    = len(CLASSES)
IMG_SIZE       = 512
CONF_THRESHOLD = 0.45
ALPHA          = 0.45

CLASS_COLORS = {
    0: (0,   0,   0),
    1: (0, 200,  80),
    2: (30, 120, 255),
    3: (255,  50,  50),
    4: (200,  50, 255),
}

def load_model(weights_path, device):
    model = smp.UnetPlusPlus(
        encoder_name="efficientnet-b4",
        encoder_weights=None,
        in_channels=3,
        classes=NUM_CLASSES,
        decoder_attention_type="scse",
    )
    state = torch.load(weights_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.to(device).eval()
    print(f"[✓] Model loaded from {weights_path} on {device}")
    return model

_preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def preprocess(frame_bgr):
    rgb     = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))
    return _preprocess(resized).unsqueeze(0)

@torch.no_grad()
def run_inference(model, tensor, device):
    logits     = model(tensor.to(device))
    probs      = torch.softmax(logits, dim=1)[0]
    class_mask = probs.argmax(dim=0).cpu().numpy().astype(np.uint8)
    prob_map   = probs.max(dim=0).values.cpu().numpy()
    return class_mask, prob_map

def build_colour_mask(class_mask, prob_map, orig_h, orig_w):
    colour = np.zeros((*class_mask.shape, 3), dtype=np.uint8)
    for cls_id, bgr in CLASS_COLORS.items():
        if cls_id == 0:
            continue
        region = (class_mask == cls_id) & (prob_map >= CONF_THRESHOLD)
        colour[region] = bgr
    return cv2.resize(colour, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)


# ─────────────────────────────────────────────────────────
# DEPTH HELPERS (new)
# ─────────────────────────────────────────────────────────
def clamp_point(pt, width, height):
    """Clamp a (x, y) pixel coordinate to valid frame bounds. minAreaRect
    corners/edge-midpoints are computed geometrically and can land slightly
    outside the frame (e.g. board touching the edge of view), which crashes
    get_distance() with an out-of-range error."""
    x, y = pt
    x = min(max(int(x), 0), width - 1)
    y = min(max(int(y), 0), height - 1)
    return x, y


def masked_median_depth(depth_frame, mask_bool):
    """
    Given a boolean mask (in the depth/color-aligned pixel grid),
    return the median valid depth in meters over that region.
    Ignores 0 / invalid readings that RealSense returns for
    low-texture or out-of-range pixels.
    """
    ys, xs = np.where(mask_bool)
    if len(xs) == 0:
        return None
    # sample every Nth pixel for speed on large masks
    step = max(1, len(xs) // 400)
    xs, ys = xs[::step], ys[::step]
    depths = [depth_frame.get_distance(int(x), int(y)) for x, y in zip(xs, ys)]
    depths = [d for d in depths if d > 0]
    return float(np.median(depths)) if depths else None


def mask_centroid(mask_bool):
    """Center of mass of a boolean mask (more accurate than bbox center for odd shapes)."""
    ys, xs = np.where(mask_bool)
    if len(xs) == 0:
        return None
    return int(xs.mean()), int(ys.mean())


def mask_interior_point(mask_bool):
    """
    Point guaranteed to lie INSIDE the mask, unlike center-of-mass which
    can fall outside for concave/irregular shapes. Uses the pixel farthest
    from the mask boundary (distance transform peak). Slightly slower —
    use only if center-of-mass gives visibly wrong results for a class.
    """
    mask_u8 = mask_bool.astype(np.uint8)
    dist = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 5)
    _, _, _, max_loc = cv2.minMaxLoc(dist)  # (x, y)
    return max_loc


def get_board_geometry(mask_bool):
    """
    For a large near-rectangular region like the board: fit a rotated
    rectangle and return its center, 4 corners, and 4 edge midpoints.
    Rotated rect (not axis-aligned bbox) matters because the board may
    not be perfectly parallel to the camera frame edges.
    """
    mask_u8 = mask_bool.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 500:
        return None

    rect = cv2.minAreaRect(largest)          # ((cx,cy), (w,h), angle)
    box_pts = cv2.boxPoints(rect)             # 4 corners, float
    box_pts = box_pts.astype(np.int32)

    center = (int(rect[0][0]), int(rect[0][1]))
    corners = [tuple(p) for p in box_pts]
    edge_midpoints = [
        tuple(((box_pts[i] + box_pts[(i + 1) % 4]) / 2).astype(int))
        for i in range(4)
    ]
    return {
        "center": center,
        "corners": corners,
        "edge_midpoints": edge_midpoints,
        "angle_deg": rect[2],
    }


def draw_bboxes_with_depth(frame, class_mask, prob_map, orig_h, orig_w,
                            depth_frame, depth_intrinsics):
    """
    Same as draw_bboxes, but also looks up depth for each detected object's
    mask centroid and prints distance + deprojected 3D point on the label.
    Also computes board-specific geometry (rotated-rect center + 4 edge
    midpoints, each with depth) and returns it as board_info.
    """
    sx, sy = orig_w / IMG_SIZE, orig_h / IMG_SIZE
    objects = []  # collect results for downstream use (logging, robot arm, etc.)

    for cls_id in range(1, NUM_CLASSES):
        binary = ((class_mask == cls_id) & (prob_map >= CONF_THRESHOLD)).astype(np.uint8) * 255
        if binary.sum() == 0:
            continue
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        color = CLASS_COLORS[cls_id]

        for cnt in contours:
            if cv2.contourArea(cnt) < 150:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            x1, y1 = int(x * sx), int(y * sy)
            x2, y2 = int((x + w) * sx), int((y + h) * sy)

            roi_mask  = binary[y:y+h, x:x+w].astype(bool)
            roi_probs = prob_map[y:y+h, x:x+w]
            conf      = float(roi_probs[roi_mask].mean()) if roi_mask.any() else 0.0

            # ── Build a full-resolution mask for this single contour ──
            full_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
            scaled_cnt = (cnt.reshape(-1, 2) * [sx, sy]).astype(np.int32)
            cv2.drawContours(full_mask, [scaled_cnt], -1, 255, thickness=-1)
            full_mask_bool = full_mask.astype(bool)

            centroid = mask_centroid(full_mask_bool)
            depth_m  = masked_median_depth(depth_frame, full_mask_bool) if centroid else None

            point_3d = None
            if depth_m is not None and centroid is not None:
                point_3d = rs.rs2_deproject_pixel_to_point(
                    depth_intrinsics, [centroid[0], centroid[1]], depth_m
                )

            label_top = f"{CLASSES[cls_id]}  {conf*100:.1f}%"
            label_bot = f"{depth_m*100:.1f}cm" if depth_m is not None else "no depth"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(label_top, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
            cv2.putText(frame, label_top, (x1 + 3, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, label_bot, (x1 + 3, y2 + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)

            if centroid is not None:
                cv2.circle(frame, centroid, 4, (0, 255, 255), -1)

            objects.append({
                "class": CLASSES[cls_id],
                "confidence": conf,
                "bbox": (x1, y1, x2, y2),
                "centroid_px": centroid,
                "depth_m": depth_m,
                "point_3d_m": point_3d,   # (X, Y, Z) in camera coordinates, meters
            })

    # ── Board-specific geometry: center + 4 edge midpoints ──
    board_id = CLASSES.index("board")
    board_binary = ((class_mask == board_id) & (prob_map >= CONF_THRESHOLD))
    board_full_mask = cv2.resize(
        board_binary.astype(np.uint8), (orig_w, orig_h),
        interpolation=cv2.INTER_NEAREST
    ).astype(bool)

    board_geom = get_board_geometry(board_full_mask)
    board_info = None
    if board_geom is not None:
        # clamp geometric points (minAreaRect can extend past frame bounds)
        board_geom["center"] = clamp_point(board_geom["center"], orig_w, orig_h)
        board_geom["corners"] = [clamp_point(p, orig_w, orig_h) for p in board_geom["corners"]]
        board_geom["edge_midpoints"] = [clamp_point(p, orig_w, orig_h) for p in board_geom["edge_midpoints"]]

        # depth + 3D point at center
        center_depth = depth_frame.get_distance(*board_geom["center"])
        center_3d = (
            rs.rs2_deproject_pixel_to_point(depth_intrinsics, list(board_geom["center"]), center_depth)
            if center_depth > 0 else None
        )

        # depth at each edge midpoint - useful to detect board tilt
        edge_depths = []
        for pt in board_geom["edge_midpoints"]:
            d = depth_frame.get_distance(int(pt[0]), int(pt[1]))
            edge_depths.append(d if d > 0 else None)

        board_info = {
            "center_px": board_geom["center"],
            "center_depth_m": center_depth if center_depth > 0 else None,
            "center_3d_m": center_3d,
            "corners_px": board_geom["corners"],
            "edge_midpoints_px": board_geom["edge_midpoints"],
            "edge_depths_m": edge_depths,
            "angle_deg": board_geom["angle_deg"],
        }

        # draw the rotated rectangle outline
        cv2.polylines(frame, [np.array(board_geom["corners"])], True, (0, 255, 0), 2)
        cv2.circle(frame, board_geom["center"], 6, (0, 255, 0), -1)
        cv2.putText(frame, f"board {center_depth*100:.1f}cm" if center_depth > 0 else "board",
                    (board_geom["center"][0] + 10, board_geom["center"][1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

        for pt, d in zip(board_geom["edge_midpoints"], edge_depths):
            cv2.circle(frame, pt, 5, (0, 165, 255), -1)
            label = f"{d*100:.1f}cm" if d is not None else "n/a"
            cv2.putText(frame, label, (pt[0] + 6, pt[1] + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1, cv2.LINE_AA)

    return frame, objects, board_info


def draw_legend(frame, detected_ids):
    x0, y0 = frame.shape[1] - 210, 14
    for i, cls_id in enumerate(range(1, NUM_CLASSES)):
        active = cls_id in detected_ids
        cv2.putText(frame, CLASSES[cls_id] + (" ●" if active else ""),
                    (x0, y0 + i * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                    CLASS_COLORS[cls_id] if active else (80, 80, 80),
                    2 if active else 1, cv2.LINE_AA)
    return frame

def draw_hud(frame, fps, show_mask, show_bbox):
    h = frame.shape[0]
    cv2.putText(frame,
                f"FPS:{fps:4.1f}  Mask:{'ON' if show_mask else 'OFF'}  BBox:{'ON' if show_bbox else 'OFF'}",
                (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(frame, "Q:quit  S:save  M:mask  B:bbox",
                (10, h - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA)
    return frame

def main(weights):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = load_model(weights, device)

    # ── RealSense pipeline ──────────────────────────
    pipeline = rs.pipeline()
    config   = rs.config()
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)  # NEW
    profile = pipeline.start(config)
    print("[✓] RealSense D435I stream started")

    # Align depth to color so pixel (x,y) means the same physical point
    # in both streams. Without this, depth lookups at a segmentation
    # mask's pixel coordinates will be wrong.
    align = rs.align(rs.stream.color)  # NEW

    depth_intrinsics = (
        profile.get_stream(rs.stream.depth)
        .as_video_stream_profile()
        .get_intrinsics()
    )  # NEW - needed to convert (pixel, depth) -> real-world (X,Y,Z)

    show_mask = True
    show_bbox = True
    save_idx  = 0
    fps_timer = time.perf_counter()
    fps       = 0.0

    print("[►] Live inference running – press Q to quit")
    cv2.namedWindow("PCB Segmentation", cv2.WINDOW_NORMAL)

    try:
        while True:
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)  # NEW - align before extracting

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()  # NEW
            if not color_frame or not depth_frame:
                continue

            frame  = np.asanyarray(color_frame.get_data())
            orig_h, orig_w = frame.shape[:2]
            tensor = preprocess(frame)

            class_mask, prob_map = run_inference(model, tensor, device)

            detected_ids = {
                int(c) for c in np.unique(class_mask)
                if c != 0 and np.mean(prob_map[class_mask == c]) >= CONF_THRESHOLD
            }

            display = frame.copy()
            if show_mask:
                colour_mask = build_colour_mask(class_mask, prob_map, orig_h, orig_w)
                display = cv2.addWeighted(display, 1 - ALPHA, colour_mask, ALPHA, 0)

            objects, board_info = [], None
            if show_bbox:
                display, objects, board_info = draw_bboxes_with_depth(
                    display, class_mask, prob_map, orig_h, orig_w,
                    depth_frame, depth_intrinsics
                )

            display = draw_legend(display, detected_ids)

            now       = time.perf_counter()
            fps       = 0.9 * fps + 0.1 / max(now - fps_timer, 1e-6)
            fps_timer = now
            display   = draw_hud(display, fps, show_mask, show_bbox)

            cv2.imshow("PCB Segmentation", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                name = f"capture_{save_idx:04d}.png"
                cv2.imwrite(name, display)
                print(f"[✓] Saved {name}")
                for obj in objects:
                    print(f"    {obj['class']:10s} depth={obj['depth_m']} "
                          f"3D={obj['point_3d_m']}")
                if board_info:
                    print(f"    board center depth={board_info['center_depth_m']} "
                          f"3D={board_info['center_3d_m']} "
                          f"edge_depths={board_info['edge_depths_m']} "
                          f"angle={board_info['angle_deg']:.1f}deg")
                save_idx += 1
            elif key == ord("m"):
                show_mask = not show_mask
            elif key == ord("b"):
                show_bbox = not show_bbox

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("[✓] Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="unetpp_efficientnetb4_best.pth")
    args = parser.parse_args()
    main(args.weights)
