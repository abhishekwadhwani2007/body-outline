import os
import sys
import warnings
import cv2
import numpy as np
import tensorflow as tf
from scipy.ndimage import gaussian_filter1d
from dotenv import load_dotenv

load_dotenv()

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pose_landmark_full.tflite")

# Six glow layers stacked from wide+faint to thin+bright.
# This gives that neon halo look without any external shader.
GLOW_LAYERS = [
    (24, 0.055, (150, 235, 190)),
    (17, 0.095, (95, 235, 155)),
    (11, 0.165, (65, 235, 125)),
    (6,  0.310, (45, 220, 105)),
    (3,  0.620, (35, 235, 115)),
    (1,  0.860, (185, 245, 205)),
]

# (landmark A index, landmark B index, stroke width as fraction of shoulder width)
BODY_SEGMENTS = [
    (11, 12, 0.35),  # shoulders
    (11, 23, 0.22),  # left side of torso
    (12, 24, 0.22),  # right side of torso
    (11, 13, 0.27),  # left upper arm
    (13, 15, 0.23),  # left forearm
    (12, 14, 0.27),  # right upper arm
    (14, 16, 0.23),  # right forearm
    (23, 25, 0.18),  # left thigh
    (25, 27, 0.14),  # left shin
    (24, 26, 0.18),  # right thigh
    (26, 28, 0.14),  # right shin
]

# Joint caps — circles drawn at each landmark to round off segment ends
# (landmark index, radius as fraction of shoulder width)
BODY_JOINTS = [
    (0,  0.14),  # head
    (11, 0.13),  # left shoulder
    (12, 0.13),  # right shoulder
    (13, 0.13),  # left elbow
    (14, 0.13),  # right elbow
    (15, 0.22),  # left wrist — larger to cover a hand or held object
    (16, 0.22),  # right wrist
    (23, 0.12),  # left hip
    (24, 0.12),  # right hip
    (25, 0.10),  # left knee
    (26, 0.10),  # right knee
    (27, 0.08),  # left ankle
    (28, 0.08),  # right ankle
]

# Outer knuckle landmarks — used to fill in the hand area near each wrist
HAND_LANDMARKS = [(17, 0.10), (18, 0.10), (19, 0.10), (20, 0.10)]


def run_pose(img_bgr):
    H, W = img_bgr.shape[:2]
    interp = tf.lite.Interpreter(model_path=MODEL_PATH)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    sz = inp["shape"][1]

    rgb = cv2.cvtColor(cv2.resize(img_bgr, (sz, sz)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    interp.set_tensor(inp["index"], rgb[np.newaxis])
    interp.invoke()

    outputs = interp.get_output_details()
    raw = interp.get_tensor(outputs[0]["index"])[0]
    score = float(interp.get_tensor(outputs[1]["index"])[0][0])
    seg_raw = interp.get_tensor(outputs[2]["index"])[0, :, :, 0]

    lms = raw.reshape(39, 5).copy()
    lms[:, 0] = lms[:, 0] / sz * W
    lms[:, 1] = lms[:, 1] / sz * H

    # Sigmoid converts raw logits to probabilities
    seg_prob = 1.0 / (1.0 + np.exp(-np.clip(seg_raw, -50, 50)))
    seg_mask = cv2.resize(seg_prob, (W, H), interpolation=cv2.INTER_LINEAR)
    return lms, score, seg_mask


def get_body_scale(lms):
    """Shoulder width (or hip width scaled up) — used to size all strokes proportionally."""
    l_sh, r_sh = lms[11], lms[12]
    l_hp, r_hp = lms[23], lms[24]
    if l_sh[3] > 0.3 and r_sh[3] > 0.3:
        d = np.linalg.norm(l_sh[:2] - r_sh[:2])
        if d > 20:
            return d
    if l_hp[3] > 0.3 and r_hp[3] > 0.3:
        d = np.linalg.norm(l_hp[:2] - r_hp[:2])
        if d > 20:
            return d * 1.3
    return 100.0


def build_skeleton_mask(H, W, lms, held_object=None):
    scale = get_body_scale(lms)
    mask = np.zeros((H, W), dtype=np.uint8)

    for a, b, w_ratio in BODY_SEGMENTS:
        if lms[a][3] < 0.25 or lms[b][3] < 0.25:
            continue
        cv2.line(mask, tuple(lms[a][:2].astype(int)), tuple(lms[b][:2].astype(int)),
                 255, thickness=max(4, int(scale * w_ratio)), lineType=cv2.LINE_AA)

    for idx, r_ratio in BODY_JOINTS:
        if lms[idx][3] < 0.25:
            continue
        cv2.circle(mask, tuple(lms[idx][:2].astype(int)),
                   max(3, int(scale * r_ratio)), 255, -1, cv2.LINE_AA)

    for idx, r_ratio in HAND_LANDMARKS:
        if lms[idx][3] < 0.20:
            continue
        cv2.circle(mask, tuple(lms[idx][:2].astype(int)),
                   max(3, int(scale * r_ratio)), 255, -1, cv2.LINE_AA)
        wrist = lms[15 if idx in (17, 19) else 16]
        if wrist[3] > 0.2:
            cv2.line(mask, tuple(wrist[:2].astype(int)), tuple(lms[idx][:2].astype(int)),
                     255, thickness=max(3, int(scale * 0.08)), lineType=cv2.LINE_AA)

    # Head — ellipse from ears, fallback circle from nose
    l_ear, r_ear, nose = lms[7], lms[8], lms[0]
    if l_ear[3] > 0.3 and r_ear[3] > 0.3:
        cx = int((l_ear[0] + r_ear[0]) / 2)
        cy = int((l_ear[1] + r_ear[1]) / 2)
        span = abs(l_ear[0] - r_ear[0])
        cv2.ellipse(mask, (cx, cy), (int(span * 0.42), int(span * 0.52)),
                    0, 0, 360, 255, -1, cv2.LINE_AA)
    elif nose[3] > 0.3:
        cv2.circle(mask, tuple(nose[:2].astype(int)),
                   max(6, int(scale * 0.18)), 255, -1)

    if held_object is not None:
        cx, cy, rx, ry = held_object
        cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 255, -1, cv2.LINE_AA)

    # Feet — ankle + heel + toe points connected
    for ankle_i, heel_i, toe_i in [(27, 29, 31), (28, 30, 32)]:
        if lms[ankle_i][3] < 0.25:
            continue
        cv2.circle(mask, tuple(lms[ankle_i][:2].astype(int)),
                   max(4, int(scale * 0.12)), 255, -1)
        for fi in [heel_i, toe_i]:
            if lms[fi][3] > 0.2:
                cv2.circle(mask, tuple(lms[fi][:2].astype(int)),
                           max(3, int(scale * 0.08)), 255, -1)
                cv2.line(mask, tuple(lms[ankle_i][:2].astype(int)),
                         tuple(lms[fi][:2].astype(int)),
                         255, thickness=max(3, int(scale * 0.10)), lineType=cv2.LINE_AA)
    return mask


def detect_held_object(img_bgr, lms, scale):
    """
    Tries to detect something held in front of the wrists (like a ball or prop).
    Looks for a dark, saturated blob in the region between/around the wrists.
    Returns (cx, cy, rx, ry) if found, else None.
    """
    H, W = img_bgr.shape[:2]
    lw, rw = lms[15], lms[16]

    if lw[3] < 0.15 and rw[3] < 0.15:
        return None

    if lw[3] > 0.15 and rw[3] > 0.15:
        mid = (lw[:2] + rw[:2]) * 0.5
        wrist_dist = np.linalg.norm(lw[:2] - rw[:2])
    else:
        mid = lw[:2] if lw[3] > rw[3] else rw[:2]
        wrist_dist = scale * 0.6

    if wrist_dist > scale * 1.2:
        return None

    l_sh, r_sh = lms[11], lms[12]
    torso_cx = (l_sh[0] + r_sh[0]) * 0.5 if (l_sh[3] > 0.2 and r_sh[3] > 0.2) else W * 0.5
    side = -1 if mid[0] < torso_cx else 1

    x1 = max(0, int(mid[0] - scale * (4.0 if side < 0 else 0.8)))
    x2 = min(W, int(mid[0] + scale * (0.8 if side < 0 else 4.0)))
    y1 = max(0, int(mid[1] - scale * 1.8))
    y2 = min(H, int(mid[1] + scale * 1.8))
    if x2 <= x1 or y2 <= y1:
        return None

    roi = img_bgr[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    dark = ((gray < 96) & (hsv[:, :, 1] > 12)).astype(np.uint8) * 255
    k = max(5, int(scale * 0.08)) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, kernel, iterations=2)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, kernel, iterations=1)

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(dark, 8)
    best, best_score = None, -1.0
    for lbl in range(1, count):
        x, y, w, h, area = stats[lbl]
        if not (scale**2 * 0.25 < area < scale**2 * 3.5):
            continue
        gcx, gcy = x1 + centroids[lbl][0], y1 + centroids[lbl][1]
        if (gcx - mid[0]) * side <= -scale * 0.25:
            continue
        if abs(gcy - mid[1]) >= scale * 1.1:
            continue
        sc = area * (1.65 - min(1.0, abs((w / max(1, h)) - 1.35) / 1.35)) - abs(gcx - mid[0]) * scale * 0.6
        if sc > best_score:
            best_score, best = sc, lbl

    if best is None:
        return None

    x, y, w, h, _ = stats[best]
    cx, cy = centroids[best]
    rx = min(max(int(w * 0.55), int(scale * 0.48)), int(scale * 0.72))
    ry = min(max(int(h * 0.62), int(scale * 0.50)), int(scale * 0.70))
    return (int(x1 + cx + side * scale * 0.03), int(y1 + cy), rx, ry)


def refine_with_grabcut(img_bgr, coarse_mask, floor_y=None, force_fg=None, iterations=3):
    """
    GrabCut sharpens the mask edges against the actual image pixels.
    We downscale to 800px max before running it — GrabCut is slow at full resolution.
    """
    H, W = img_bgr.shape[:2]
    sc = min(800 / max(H, W), 1.0)
    sh, sw = int(H * sc), int(W * sc)
    img_s = cv2.resize(img_bgr, (sw, sh))
    mask_s = cv2.resize(coarse_mask, (sw, sh), interpolation=cv2.INTER_NEAREST)

    ke = max(3, int(min(sh, sw) * 0.02)) | 1
    kd = max(3, int(min(sh, sw) * 0.03)) | 1
    fg_core = cv2.erode(mask_s, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ke, ke)), iterations=2)
    fg_wide = cv2.dilate(mask_s, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kd, kd)), iterations=2)

    gc = np.full(mask_s.shape, cv2.GC_BGD, dtype=np.uint8)
    gc[fg_wide > 127] = cv2.GC_PR_FGD
    gc[mask_s > 127] = cv2.GC_PR_FGD
    gc[fg_core > 127] = cv2.GC_FGD

    if force_fg is not None:
        fs = cv2.resize(force_fg, (sw, sh), interpolation=cv2.INTER_NEAREST)
        gc[fs > 127] = cv2.GC_FGD

    if floor_y is not None:
        fy = min(int(floor_y * sc), sh - 1)
        gc[fy:][mask_s[fy:] <= 127] = cv2.GC_BGD

    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(img_s, gc, None, bgd, fgd, iterations, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return coarse_mask

    refined = np.where((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    if sc < 1.0:
        refined = (cv2.resize(refined, (W, H), interpolation=cv2.INTER_LINEAR) > 127).astype(np.uint8) * 255
    return refined


def smooth_head_region(binary_mask, lms, scale):
    """
    Hair and loose strands often create noisy, spiky edges at the top of the head.
    This cleans up just that area using morphological closing + connected component filtering,
    without touching anything below the shoulders.
    """
    nose, l_ear, r_ear = lms[0], lms[7], lms[8]
    l_sh, r_sh = lms[11], lms[12]
    if any(p[3] < 0.20 for p in [nose, l_ear, r_ear, l_sh, r_sh]):
        return binary_mask

    H, W = binary_mask.shape[:2]
    head_mid = (l_ear[:2] + r_ear[:2]) * 0.5
    face_dir = -1 if nose[0] < head_mid[0] else 1
    back_ext = head_mid + np.array([-face_dir, 0]) * scale * 1.85

    x1 = max(0, int(min(nose[0], back_ext[0]) - scale * 0.22))
    x2 = min(W, int(max(nose[0], back_ext[0]) + scale * 0.22))
    y1 = max(0, int(min(nose[1], l_ear[1], r_ear[1]) - scale * 1.05))
    y2 = min(H, int(min(l_sh[1], r_sh[1]) + scale * 0.10))
    if x2 <= x1 or y2 <= y1:
        return binary_mask

    roi = binary_mask[y1:y2, x1:x2].copy()
    k_close = max(3, int(scale * 0.040)) | 1
    k_blur = max(5, int(scale * 0.065)) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close))
    smooth = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel)
    smooth = cv2.GaussianBlur(smooth, (k_blur, k_blur), 0)
    smooth = (smooth > 112).astype(np.uint8) * 255

    # Only accept smoothed regions that actually overlap the head ellipse
    seed = np.zeros_like(roi)
    cv2.ellipse(seed,
                (int(head_mid[0] - x1), int(head_mid[1] - y1)),
                (max(10, int(scale * 0.42)), max(12, int(scale * 0.58))),
                0, 0, 360, 255, -1, cv2.LINE_AA)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(smooth, 8)
    keep = np.zeros_like(roi)
    for lbl in range(1, count):
        if stats[lbl, cv2.CC_STAT_AREA] > scale**2 * 0.08 and np.any(seed[labels == lbl]):
            keep[labels == lbl] = 255

    if np.any(keep):
        roi[keep > 0] = 255
        result = binary_mask.copy()
        result[y1:y2, x1:x2] = roi
        return result
    return binary_mask


def smooth_contour(contour, sigma=6.0, num_points=600):
    """Resample contour points to be evenly spaced, then apply a Gaussian blur along the curve."""
    pts = contour.reshape(-1, 2).astype(np.float64)
    if len(pts) < 5:
        return contour
    diffs = np.diff(pts, axis=0)
    cum_len = np.concatenate(([0], np.cumsum(np.hypot(diffs[:, 0], diffs[:, 1]))))
    if cum_len[-1] < 1:
        return contour
    t = np.linspace(0, cum_len[-1], max(num_points, len(pts)), endpoint=False)
    xs = gaussian_filter1d(np.interp(t, cum_len, pts[:, 0]), sigma=sigma, mode="wrap")
    ys = gaussian_filter1d(np.interp(t, cum_len, pts[:, 1]), sigma=sigma, mode="wrap")
    return np.stack([xs, ys], axis=1).reshape(-1, 1, 2).astype(np.int32)


def extract_contours(binary_mask):
    H, W = binary_mask.shape[:2]
    k = max(3, int(max(H, W) * 0.002)) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    blur_k = max(5, int(max(H, W) * 0.008)) | 1
    mask = (cv2.GaussianBlur(mask, (blur_k, blur_k), 0) > 127).astype(np.uint8) * 255

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if not contours or hierarchy is None:
        return None

    h = hierarchy[0]
    outer = max((i for i in range(len(contours)) if h[i][3] == -1),
                key=lambda i: cv2.contourArea(contours[i]), default=-1)
    if outer < 0:
        return None

    result = [smooth_contour(contours[outer], sigma=6.0, num_points=600)]

    # Include significant holes in the silhouette (e.g. gap between arm and torso)
    child = h[outer][2]
    while child != -1:
        if cv2.contourArea(contours[child]) > H * W * 0.001:
            result.append(smooth_contour(contours[child], sigma=4.2, num_points=300))
        child = h[child][0]

    return result


def draw_glow(canvas, contours, base_dim):
    scale = base_dim / 800.0
    for raw_thick, alpha, color in GLOW_LAYERS:
        overlay = canvas.copy()
        cv2.drawContours(overlay, contours, -1, color,
                         thickness=max(1, int(raw_thick * scale)), lineType=cv2.LINE_AA)
        cv2.addWeighted(overlay, alpha, canvas, 1.0 - alpha, 0, canvas)
    return canvas


def restore_face(canvas, original, binary_mask, lms, scale):
    """
    The glow stroke covers the face area since it traces the full silhouette edge.
    This blends original face pixels back inside the head region so the person is visible.
    """
    nose, l_ear, r_ear = lms[0], lms[7], lms[8]
    l_sh, r_sh = lms[11], lms[12]
    if any(p[3] < 0.20 for p in [nose, l_ear, r_ear, l_sh, r_sh]):
        return

    H, W = binary_mask.shape[:2]
    pts = np.array([nose[:2], l_ear[:2], r_ear[:2], l_sh[:2], r_sh[:2]])
    x1 = max(0, int(pts[:, 0].min() - scale * 0.45))
    x2 = min(W, int(pts[:, 0].max() + scale * 1.55))
    y1 = max(0, int(pts[:, 1].min() - scale * 1.05))
    y2 = min(H, int(min(l_sh[1], r_sh[1]) + scale * 0.04))
    if x2 <= x1 or y2 <= y1:
        return

    roi_mask = binary_mask[y1:y2, x1:x2]
    dist = cv2.distanceTransform((roi_mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
    alpha = np.clip((dist - scale * 0.10) / (scale * 0.12), 0, 1).astype(np.float32)
    alpha = cv2.GaussianBlur(alpha, ((max(5, int(scale * 0.05)) | 1,) * 2), 0)

    canvas[y1:y2, x1:x2] = (
        canvas[y1:y2, x1:x2] * (1 - alpha[..., None]) +
        original[y1:y2, x1:x2] * alpha[..., None]
    ).astype(np.uint8)


def process(input_path, output_path):
    if not os.path.exists(MODEL_PATH):
        print(f"Error: model not found at {MODEL_PATH}")
        sys.exit(1)

    img = cv2.imread(input_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {input_path}")

    H, W = img.shape[:2]
    print(f"[1/4] Loaded {os.path.basename(input_path)} ({W}x{H})")

    print("[2/4] Running pose detection...")
    lms, score, seg_mask = run_pose(img)
    print(f"      Confidence: {score:.3f}")

    print("[3/4] Building body mask and drawing outline...")
    scale = get_body_scale(lms)
    obj = detect_held_object(img, lms, scale)
    skel = build_skeleton_mask(H, W, lms, obj)

    coarse = np.maximum((seg_mask > 0.5).astype(np.uint8) * 255, skel)
    k = max(3, int(max(H, W) * 0.003)) | 1
    coarse = cv2.morphologyEx(coarse, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))

    # Arms get cut off by GrabCut when they're thin — force them to stay as foreground
    force_fg = np.zeros((H, W), dtype=np.uint8)
    arm_thick = max(6, int(scale * 0.30))
    for sh_i, el_i, wr_i in [(11, 13, 15), (12, 14, 16)]:
        sh, el, wr = lms[sh_i], lms[el_i], lms[wr_i]
        if sh[3] > 0.20 and el[3] > 0.20:
            cv2.line(force_fg, tuple(sh[:2].astype(int)), tuple(el[:2].astype(int)), 255, arm_thick, cv2.LINE_AA)
        if el[3] > 0.20 and wr[3] > 0.20:
            cv2.line(force_fg, tuple(el[:2].astype(int)), tuple(wr[:2].astype(int)), 255, arm_thick, cv2.LINE_AA)
    if obj:
        cv2.ellipse(force_fg, (obj[0], obj[1]), (obj[2], obj[3]), 0, 0, 360, 255, -1, cv2.LINE_AA)

    foot_ys = [int(lms[fi][1]) for fi in [29, 30, 31, 32, 27, 28] if lms[fi][3] > 0.15]
    floor_y = (max(foot_ys) + int(H * 0.025)) if foot_ys else None

    binary = refine_with_grabcut(img, coarse, floor_y=floor_y, force_fg=force_fg)
    binary = smooth_head_region(binary, lms, scale)

    contours = extract_contours(binary)
    if contours is None:
        print("Warning: no body contour found — saving original image.")
        cv2.imwrite(output_path, img)
        return

    canvas = img.copy()
    draw_glow(canvas, contours, max(H, W))
    restore_face(canvas, img, binary, lms, scale)

    print(f"[4/4] Saving -> {output_path}")
    cv2.imwrite(output_path, canvas)
    print("Done!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Draw a neon body outline over a person in an image.")
    parser.add_argument("input",  nargs="?", default=os.getenv("INPUT_IMAGE", "testImage.png"),
                        help="Input image path (or set INPUT_IMAGE in .env)")
    parser.add_argument("output", nargs="?", default=os.getenv("OUTPUT_IMAGE"),
                        help="Output image path (or set OUTPUT_IMAGE in .env)")
    args = parser.parse_args()
    out = args.output or args.input.rsplit(".", 1)[0] + "_overlay.png"
    process(args.input, out)
