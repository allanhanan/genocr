import cv2
import numpy as np
import onnxruntime as ort
import pyclipper
import math
from pathlib import Path

# ====================================================================
# CONFIGURATION
# ====================================================================

# --- Detection Parameters (from your working script) ---
MAX_SIDE = 960
DET_THRESH = 0.3
BOX_THRESH = 0.6
UNCLIP_RATIO = 1.5
MIN_AREA = 80

# --- Recognition Parameters ---
REC_H, REC_W = 48, 320
REC_BATCH_SIZE = 16

# ====================================================================
# DETECTION & GEOMETRY (Your working code - unchanged)
# ====================================================================

def resize_keep_ratio(img, max_side=MAX_SIDE):
    h, w = img.shape[:2]
    scale = min(max_side / max(h, w), 1.0) if max(h, w) > max_side else (max_side / max(h, w))
    nh, nw = int(h * scale), int(w * scale)
    nh = nh + (32 - nh % 32) % 32
    nw = nw + (32 - nw % 32) % 32
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return resized, scale

def normalize_imagenet(img):
    x = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std = np.array([0.229, 0.224, 0.225], np.float32)
    x = (x - mean) / std
    return x.transpose(2, 0, 1)[None, ...].astype(np.float32)

def order_points_clockwise(pts):
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    tl = np.argmin(s)
    br = np.argmax(s)
    tr = np.argmin(diff)
    bl = np.argmax(diff)
    return np.array([pts[tl], pts[tr], pts[br], pts[bl]], dtype=np.float32)

def quad_from_contour(contour):
    rect = cv2.minAreaRect(contour)
    return cv2.boxPoints(rect).astype(np.float32)

def unclip(box, ratio=UNCLIP_RATIO):
    area = cv2.contourArea(box)
    length = cv2.arcLength(box, True)
    if length == 0: return box
    distance = int(area * ratio / length)
    pco = pyclipper.PyclipperOffset()
    pco.AddPath(box.astype(np.int32), pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    out = pco.Execute(distance)
    if not out: return box
    out = np.array(out[0]).astype(np.float32)
    rect = cv2.minAreaRect(out)
    return cv2.boxPoints(rect).astype(np.float32)

def box_score(prob_map, box):
    h, w = prob_map.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [box.astype(np.int32)], 1)
    return cv2.mean(prob_map, mask)[0]

class PPOCRv5DetONNX:
    def __init__(self, det_onnx_path, use_cuda=False):
        providers = ['CPUExecutionProvider']
        if use_cuda: providers.insert(0, 'CUDAExecutionProvider')
        self.sess = ort.InferenceSession(str(det_onnx_path), providers=providers)
        self.input_name = self.sess.get_inputs()[0].name

    def detect(self, bgr_image):
        orig_h, orig_w = bgr_image.shape[:2]
        resized, scale = resize_keep_ratio(bgr_image, MAX_SIDE)
        inp = normalize_imagenet(resized)
        prob = self.sess.run(None, {self.input_name: inp})[0][0, 0]
        bitmap = (prob > DET_THRESH).astype(np.uint8)

        contours, _ = cv2.findContours(bitmap, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for cnt in contours:
            if len(cnt) < 4: continue
            box = quad_from_contour(cnt)
            if box_score(prob, box) < BOX_THRESH: continue
            box = unclip(box)
            box /= scale
            if cv2.contourArea(box.astype(np.int32)) < MIN_AREA: continue
            box[:, 0] = np.clip(box[:, 0], 0, orig_w - 1)
            box[:, 1] = np.clip(box[:, 1], 0, orig_h - 1)
            boxes.append(box.astype(np.int32))
        return boxes

# ====================================================================
# RECOGNITION PIPELINE (With official fix for wide regions)
# ====================================================================

def warp_quad(bgr_image, quad):
    points = quad.astype(np.float32)
    w = int(max(np.linalg.norm(points[0] - points[1]), np.linalg.norm(points[2] - points[3])))
    h = int(max(np.linalg.norm(points[0] - points[3]), np.linalg.norm(points[1] - points[2])))
    if w <= 0 or h <= 0: return None
    
    dst_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    M = cv2.getPerspectiveTransform(points, dst_pts)
    crop = cv2.warpPerspective(bgr_image, M, (w, h), borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_CUBIC)
    
    if crop.shape[0] > 0 and crop.shape[1] > 0 and (crop.shape[0] * 1.0 / crop.shape[1] >= 1.5):
        crop = np.rot90(crop)
    return crop

def rec_preprocess(img):
    img_h, img_w, _ = img.shape
    ratio = img_w / float(img_h)
    
    resized_w = int(math.ceil(REC_H * ratio))
    
    resized_image = cv2.resize(img, (resized_w, REC_H), interpolation=cv2.INTER_LINEAR)
    norm_img = resized_image.astype(np.float32) / 255.0

    padding_img = np.zeros((REC_H, REC_W, 3), dtype=np.float32)
    width_to_copy = min(resized_w, REC_W)
    padding_img[:REC_H, :width_to_copy, :] = norm_img[:, :width_to_copy, :]
    
    return padding_img.transpose(2, 0, 1)

def split_wide_crop(crop, chunk_width=REC_W, overlap=64):
    h, w, _ = crop.shape
    # If the image is not wider than the chunk width, no need to split.
    if w <= chunk_width:
        return [crop]
    
    chunks = []
    # Use a larger overlap to ensure context is not lost at the edges.
    for start in range(0, w, chunk_width - overlap):
        end = min(start + chunk_width, w)
        chunks.append(crop[:, start:end, :])
        if end == w:
            break
    return chunks

def decode_recognition(preds, character_dict):
    """CTC decoding with verbose debug logging for inspection."""

    # Remove batch dimension if present
    if preds.ndim == 3:
        preds = preds[0]

    preds_idx = preds.argmax(axis=1)

    # DEBUG: Print summary info
    print(f"DEBUG Python: time={len(preds_idx)}, num_chars={preds.shape[1]}")
    print("DEBUG Python: First 10 timesteps detailed:")
    for t in range(min(10, len(preds))):
        top3 = preds[t].argsort()[-3:][::-1]
        print(f"  t={t}: top3=[", end="")
        for i, idx in enumerate(top3):
            prob = preds[t][idx]
            char_str = (
                "blank" if idx == 0
                else character_dict[idx - 1] if 0 < idx <= len(character_dict)
                else "OOB"
            )
            print(f"({idx}:{char_str}:{prob:.3f})", end="")
            if i < len(top3) - 1:
                print(", ", end="")
        print("]")

    print("DEBUG Python: Raw sequence:", preds_idx.tolist()[:20])

    # Actual decoding logic
    text = ""
    last_idx = -1
    for idx in preds_idx:
        shifted_idx = idx - 1
        if idx != 0 and idx != last_idx and 0 <= shifted_idx < len(character_dict):
            char = character_dict[shifted_idx]
            if char == '<EOS>':
                char = ' '
            text += char
        last_idx = idx

    return text.strip()


class PPOCRv5RecONNX:
    def __init__(self, rec_onnx_path, dict_path, use_cuda=False):
        providers = ['CPUExecutionProvider']
        if use_cuda: providers.insert(0, 'CUDAExecutionProvider')
        self.sess = ort.InferenceSession(str(rec_onnx_path), providers=providers)
        self.input_name = self.sess.get_inputs()[0].name
        with open(dict_path, 'r', encoding='utf-8') as f:
            # Use rstrip() to handle different line endings and keep space entries
            self.charset = ['blank'] + [line.rstrip() for line in f if line.rstrip()] + ['<EOS>']
        print(f"✅ Loaded recognition model with {len(self.charset)} characters")

    def recognize_batch(self, crops):
        if not crops:
            return []
            
        preprocessed_chunks = [rec_preprocess(crop) for crop in crops if crop is not None]
        
        if not preprocessed_chunks: return []

        batch_input = np.stack(preprocessed_chunks, axis=0)
        batch_output = self.sess.run(None, {self.input_name: batch_input})[0]
        
        return [decode_recognition(output, self.charset) for output in batch_output]

# ====================================================================
# MAIN TEST SCRIPT
# ====================================================================

def test_ocr():
    project_root = Path(__file__).resolve().parents[2]
    models = project_root / "models"
    det_onnx = models / "ocr_detection_multilingual" / "inference.onnx"
    rec_onnx = models / "ocr_recognition_multilingual" / "inference.onnx"
    dict_path = models / "ocr_recognition_multilingual" / "ppocrv5_dict.txt"
    image_file = project_root / "tests/test_assets/sample_document.png"

    for f in [det_onnx, rec_onnx, dict_path, image_file]:
        assert f.exists(), f"File not found: {f}"

    det = PPOCRv5DetONNX(det_onnx, use_cuda=False)
    rec = PPOCRv5RecONNX(rec_onnx, dict_path, use_cuda=False)
    img = cv2.imread(str(image_file))

    boxes = det.detect(img)
    print(f"Detected {len(boxes)} regions.")

    vis_boxes = img.copy()
    for b in boxes:
        cv2.polylines(vis_boxes, [b.astype(np.int32)], True, (0, 255, 0), 2)
    cv2.imwrite(str(image_file.parent / "detected_vis.png"), vis_boxes)
    print(f"Saved visualization: {image_file.parent / 'detected_vis.png'}")

    final_texts = []
    for i, box in enumerate(boxes):
        crop = warp_quad(img, box)
        if crop is None:
            final_texts.append("[Invalid Crop]")
            continue
        
        chunks = split_wide_crop(crop)
        
        recognized_texts = rec.recognize_batch(chunks)
        
        # A simple but effective join for overlapping chunks
        full_text = ' '.join(t for t in recognized_texts if t).strip()
        final_texts.append(full_text)
        
    print("\n--- OCR Results ---")
    for i, text in enumerate(final_texts):
        print(f"Region {i+1}: {text}")

if __name__ == "__main__":
    test_ocr()
