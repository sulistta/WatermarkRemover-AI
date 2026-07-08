import sys
import click
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw

# Monkey-patch: cached_download was removed in huggingface_hub 0.24, add compatibility shim
import huggingface_hub
if not hasattr(huggingface_hub, 'cached_download'):
    huggingface_hub.cached_download = huggingface_hub.hf_hub_download

from transformers import AutoProcessor, Florence2ForConditionalGeneration
import torch
from torch.nn import Module
import tqdm
from loguru import logger
from enum import Enum
import os
import tempfile
import shutil
import subprocess

try:
    from cv2.typing import MatLike
except ImportError:
    MatLike = np.ndarray


def download_lama_model():
    """Download LaMA model using iopaint."""
    logger.info("Downloading LaMA model... (this may take a few minutes)")
    print("Downloading LaMA model (~196MB)... Please wait.")

    result = subprocess.run(
        [sys.executable, "-m", "iopaint", "download", "--model", "lama"],
        capture_output=False,  # Show download progress
        text=True
    )

    if result.returncode != 0:
        logger.error("Failed to download LaMA model")
        return False

    logger.info("LaMA model downloaded successfully")
    print("LaMA model downloaded!")
    return True


def load_lama_model(device):
    """Load LaMA model, downloading if necessary."""
    from iopaint.model_manager import ModelManager

    try:
        return ModelManager(name="lama", device=device)
    except NotImplementedError as e:
        if "Unsupported model: lama" in str(e):
            print("LaMA model not available, attempting to download...")
            if download_lama_model():
                # Re-import to refresh model registry
                import importlib
                import iopaint.model
                importlib.reload(iopaint.model)
                # Try again
                return ModelManager(name="lama", device=device)
            else:
                raise RuntimeError("Failed to download LaMA model. Please run manually: python\\python.exe -m iopaint download --model lama")
        raise

class TaskType(str, Enum):
    OPEN_VOCAB_DETECTION = "<OPEN_VOCABULARY_DETECTION>"
    """Detect bounding box for objects and OCR text"""

def identify(task_prompt: TaskType, image: MatLike, text_input: str, model: Florence2ForConditionalGeneration, processor: AutoProcessor, device: str):
    if not isinstance(task_prompt, TaskType):
        raise ValueError(f"task_prompt must be a TaskType, but {task_prompt} is of type {type(task_prompt)}")

    prompt = task_prompt.value if text_input is None else task_prompt.value + text_input
    inputs = processor(text=prompt, images=image, return_tensors="pt")

    # Cast tensors to match the model's dtype and move to the correct device
    inputs = {k: v.to(device).to(model.dtype) if v.is_floating_point() else v.to(device)
              for k, v in inputs.items()}

    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        do_sample=False,
        num_beams=1,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    return processor.post_process_generation(
        generated_text, task=task_prompt.value, image_size=(image.width, image.height)
    )

def get_watermark_mask(image: MatLike, model: Florence2ForConditionalGeneration, processor: AutoProcessor, device: str, max_bbox_percent: float, detection_prompt: str = "watermark"):
    """
    Detect watermarks and create a mask for inpainting.

    Args:
        image: PIL Image
        model: Florence-2 model
        processor: Florence-2 processor
        device: cuda or cpu
        max_bbox_percent: Maximum bbox size as percentage of image
        detection_prompt: Text prompt for detection (e.g. "watermark", "watermark Sora logo", "Getty Images")
    """
    task_prompt = TaskType.OPEN_VOCAB_DETECTION
    parsed_answer = identify(task_prompt, image, detection_prompt, model, processor, device)

    mask = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(mask)

    detection_key = "<OPEN_VOCABULARY_DETECTION>"
    if detection_key in parsed_answer and "bboxes" in parsed_answer[detection_key]:
        image_area = image.width * image.height
        for bbox in parsed_answer[detection_key]["bboxes"]:
            x1, y1, x2, y2 = map(int, bbox)
            bbox_area = (x2 - x1) * (y2 - y1)
            if (bbox_area / image_area) * 100 <= max_bbox_percent:
                draw.rectangle([x1, y1, x2, y2], fill=255)
            else:
                logger.warning(f"Skipping large bounding box: {bbox} covering {bbox_area / image_area:.2%} of the image")

    return mask


def get_heygen_local_mask(image: Image.Image):
    """
    Build a pixel-level mask for repeated translucent HeyGen watermarks.

    HeyGen marks are usually pale, low-saturation text/logo strokes over the
    image. A local-brightness residual catches those strokes better than the
    generic Florence bounding-box detector, and avoids masking whole face/torso
    rectangles before LaMA inpainting.
    """
    rgb = np.array(image.convert("RGB"))
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    background = cv2.GaussianBlur(gray, (0, 0), sigmaX=11, sigmaY=11)
    bright_residual = cv2.subtract(gray, background)

    local_candidate = (
        (bright_residual > 7) &
        (saturation < 95) &
        (value > 95)
    ).astype(np.uint8) * 255

    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(local_candidate, cv2.MORPH_OPEN, kernel_open)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    filtered = np.zeros_like(mask)
    image_area = mask.shape[0] * mask.shape[1]
    max_component_area = max(3000, int(image_area * 0.018))

    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]
        if area < 5 or area > max_component_area:
            continue
        if w > image.width * 0.75 or h > image.height * 0.35:
            continue
        filtered[labels == label] = 255

    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    filtered = cv2.dilate(filtered, kernel_dilate, iterations=1)
    return Image.fromarray(filtered)


def _extract_heygen_template(image: Image.Image):
    rgb = np.array(image.convert("RGB"))
    height, width = rgb.shape[:2]

    x0 = int(width * 0.132)
    y0 = int(height * 0.094)
    template_width = max(80, int(width * 0.264))
    template_height = max(60, int(height * 0.113))

    if x0 + template_width > width or y0 + template_height > height:
        return None

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    background = cv2.GaussianBlur(gray, (0, 0), sigmaX=11, sigmaY=11)
    residual = cv2.subtract(gray, background)

    stroke = (
        (residual > 4) &
        (hsv[:, :, 1] < 145) &
        (hsv[:, :, 2] > 60)
    ).astype(np.uint8) * 255
    stroke = cv2.morphologyEx(stroke, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)))

    template_mask = stroke[y0:y0 + template_height, x0:x0 + template_width]
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(template_mask, 8)
    filtered = np.zeros_like(template_mask)
    max_area = max(800, int(template_width * template_height * 0.25))

    for label in range(1, num_labels):
        _, _, _, _, area = stats[label]
        if 5 <= area <= max_area:
            filtered[labels == label] = 255

    if np.count_nonzero(filtered) < template_width * template_height * 0.015:
        return None

    gray_float = gray.astype(np.float32)
    background_float = cv2.GaussianBlur(gray_float, (0, 0), sigmaX=17, sigmaY=17)
    alpha = np.clip(
        np.maximum(gray_float - background_float, 0) / np.maximum(255 - background_float, 1),
        0,
        0.45
    )
    template_alpha = alpha[y0:y0 + template_height, x0:x0 + template_width] * (filtered > 0)
    template_alpha = cv2.GaussianBlur(template_alpha, (0, 0), sigmaX=1.0, sigmaY=1.0)
    template_alpha = np.where(template_alpha > 0.015, template_alpha, 0)

    template_mask = cv2.dilate(filtered, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
    return {
        "x0": x0,
        "y0": y0,
        "step_x": max(1, int(width * 0.389)),
        "step_y": max(1, int(height * 0.219)),
        "mask": template_mask,
        "alpha": template_alpha,
    }


def build_heygen_template_maps(image: Image.Image, alpha_gain: float = 1.5):
    template = _extract_heygen_template(image)
    if template is None:
        mask = np.array(get_heygen_local_mask(image))
        alpha = (cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (0, 0), sigmaX=1.2, sigmaY=1.2) * 0.18)
        return Image.fromarray(mask), alpha

    width, height = image.size
    template_mask = template["mask"]
    template_alpha = template["alpha"] * alpha_gain
    template_height, template_width = template_mask.shape
    full_mask = np.zeros((height, width), np.uint8)
    full_alpha = np.zeros((height, width), np.float32)

    for y in range(template["y0"] - template["step_y"], height + template["step_y"], template["step_y"]):
        for x in range(template["x0"] - (2 * template["step_x"]), width + template["step_x"], template["step_x"]):
            vx1, vy1 = max(0, x), max(0, y)
            vx2, vy2 = min(width, x + template_width), min(height, y + template_height)
            if vx2 <= vx1 or vy2 <= vy1:
                continue
            if (vx2 - vx1) * (vy2 - vy1) < template_width * template_height * 0.18:
                continue

            sx1, sy1 = vx1 - x, vy1 - y
            sx2, sy2 = sx1 + (vx2 - vx1), sy1 + (vy2 - vy1)
            full_mask[vy1:vy2, vx1:vx2] = np.maximum(
                full_mask[vy1:vy2, vx1:vx2],
                template_mask[sy1:sy2, sx1:sx2]
            )
            full_alpha[vy1:vy2, vx1:vx2] = np.maximum(
                full_alpha[vy1:vy2, vx1:vx2],
                template_alpha[sy1:sy2, sx1:sx2]
            )

    full_alpha = np.clip(full_alpha, 0, 0.42)
    return Image.fromarray(full_mask), full_alpha


def get_heygen_mask(image: Image.Image):
    mask, _ = build_heygen_template_maps(image)
    return mask


def remove_heygen_watermark(image: Image.Image, alpha_map: np.ndarray = None, mask_image: Image.Image = None):
    if alpha_map is None:
        mask, alpha_map = build_heygen_template_maps(image)
    elif mask_image is not None:
        mask = mask_image
    else:
        mask = Image.fromarray((alpha_map > 0).astype(np.uint8) * 255)

    rgb = np.array(image.convert("RGB")).astype(np.float32)
    height, _ = rgb.shape[:2]
    alpha = np.clip(alpha_map, 0, 0.42)
    deblended = (rgb - 255 * alpha[:, :, None]) / np.maximum(1 - alpha[:, :, None], 0.2)
    deblended = np.clip(deblended, 0, 255).astype(np.uint8)

    mask_array = np.maximum(
        np.array(mask.convert("L")),
        (alpha > 0.003).astype(np.uint8) * 255
    )
    mask_array = cv2.dilate(
        mask_array,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1
    )
    if np.count_nonzero(mask_array) == 0:
        return Image.fromarray(deblended)

    telea = cv2.inpaint(
        cv2.cvtColor(deblended, cv2.COLOR_RGB2BGR),
        mask_array,
        2,
        cv2.INPAINT_TELEA
    )
    telea = cv2.cvtColor(telea, cv2.COLOR_BGR2RGB).astype(np.float32)
    repair_alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=0.8, sigmaY=0.8)
    hsv = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    y_coords = np.arange(height)[:, None]
    low_risk_surface = (
        (saturation < 55) |
        ((value > 175) & (saturation < 95)) |
        ((y_coords < height * 0.25) & (saturation < 125))
    ).astype(np.float32)
    low_risk_surface = cv2.GaussianBlur(low_risk_surface, (0, 0), sigmaX=1.2, sigmaY=1.2)
    mask_feather = cv2.GaussianBlur((mask_array > 0).astype(np.float32), (0, 0), sigmaX=0.8, sigmaY=0.8)
    repair_alpha = np.maximum(repair_alpha, mask_feather * low_risk_surface * 0.10)
    blend_weight = np.clip(repair_alpha / 0.22, 0, 1) ** 1.25
    clean = (
        deblended.astype(np.float32) * (1 - blend_weight[:, :, None]) +
        telea * blend_weight[:, :, None]
    )
    return Image.fromarray(np.clip(clean, 0, 255).astype(np.uint8))


def get_mask(image: Image.Image, mask_mode: str, model: Florence2ForConditionalGeneration = None, processor: AutoProcessor = None, device: str = "cpu", max_bbox_percent: float = 10.0, detection_prompt: str = "watermark", heygen_mask: Image.Image = None):
    if mask_mode == "heygen":
        return heygen_mask if heygen_mask is not None else get_heygen_mask(image)
    return get_watermark_mask(image, model, processor, device, max_bbox_percent, detection_prompt)


def detect_only(image: MatLike, model: Florence2ForConditionalGeneration, processor: AutoProcessor, device: str, max_bbox_percent: float, detection_prompt: str = "watermark"):
    """
    Detect watermarks and return bounding boxes WITHOUT creating mask or inpainting.
    Used for preview mode to show what would be detected.

    Returns:
        list of dicts with bbox info: [{"bbox": [x1,y1,x2,y2], "area_percent": float, "accepted": bool}, ...]
    """
    task_prompt = TaskType.OPEN_VOCAB_DETECTION
    parsed_answer = identify(task_prompt, image, detection_prompt, model, processor, device)

    results = []
    detection_key = "<OPEN_VOCABULARY_DETECTION>"

    if detection_key in parsed_answer and "bboxes" in parsed_answer[detection_key]:
        image_area = image.width * image.height
        for bbox in parsed_answer[detection_key]["bboxes"]:
            x1, y1, x2, y2 = map(int, bbox)
            bbox_area = (x2 - x1) * (y2 - y1)
            area_percent = (bbox_area / image_area) * 100
            accepted = area_percent <= max_bbox_percent

            results.append({
                "bbox": [x1, y1, x2, y2],
                "area_percent": round(area_percent, 2),
                "accepted": accepted
            })

    return results


def preview_heygen_mask(image: Image.Image):
    mask = get_heygen_mask(image)
    mask_array = np.array(mask)
    overlay = image.convert("RGBA")
    red = Image.new("RGBA", overlay.size, (255, 0, 80, 105))
    clear = Image.new("RGBA", overlay.size, (0, 0, 0, 0))
    overlay.alpha_composite(Image.composite(red, clear, mask))

    covered_pixels = int(np.count_nonzero(mask_array))
    area_percent = round((covered_pixels / mask_array.size) * 100, 2) if mask_array.size else 0
    detections = []
    if covered_pixels:
        detections.append({
            "bbox": [0, 0, image.width, image.height],
            "area_percent": area_percent,
            "accepted": True,
            "mode": "heygen"
        })

    return overlay.convert("RGB"), detections


def process_image_with_lama(image: MatLike, mask: MatLike, model_manager):
    from iopaint.schema import HDStrategy, LDMSampler, InpaintRequest as Config

    config = Config(
        ldm_steps=50,
        ldm_sampler=LDMSampler.ddim,
        hd_strategy=HDStrategy.CROP,
        hd_strategy_crop_margin=64,
        hd_strategy_crop_trigger_size=800,
        hd_strategy_resize_limit=1600,
    )
    result = model_manager(image, mask, config)

    if result.dtype in [np.float64, np.float32]:
        result = np.clip(result, 0, 255).astype(np.uint8)

    return result

def make_region_transparent(image: Image.Image, mask: Image.Image):
    image = image.convert("RGBA")
    mask = mask.convert("L")
    transparent_image = Image.new("RGBA", image.size)
    for x in range(image.width):
        for y in range(image.height):
            if mask.getpixel((x, y)) > 0:
                transparent_image.putpixel((x, y), (0, 0, 0, 0))
            else:
                transparent_image.putpixel((x, y), image.getpixel((x, y)))
    return transparent_image

def is_video_file(file_path):
    """Check if the file is a video based on its extension"""
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm']
    return Path(file_path).suffix.lower() in video_extensions

def process_video(input_path, output_path, florence_model, florence_processor, model_manager, device, transparent, max_bbox_percent, force_format, detection_prompt="watermark", mask_mode="florence", progress_offset=0, progress_scale=100):
    """Process a video file by extracting frames, removing watermarks, and reconstructing the video"""
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        logger.error(f"Error opening video file: {input_path}")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Determine output format
    if force_format:
        output_format = force_format.upper()
    else:
        output_format = "MP4"  # Default to MP4 for videos
    
    # Create output video file
    output_path = Path(output_path)
    if output_path.is_dir():
        output_file = output_path / f"{input_path.stem}_no_watermark.{output_format.lower()}"
    else:
        output_file = output_path.with_suffix(f".{output_format.lower()}")
    
    # Create a temporary file for the video without audio
    temp_dir = tempfile.mkdtemp()
    temp_video_path = Path(temp_dir) / f"temp_no_audio.{output_format.lower()}"
    
    # Set codec based on output format
    if output_format.upper() == "MP4":
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    elif output_format.upper() == "AVI":
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
    else:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Default to MP4
    
    out = cv2.VideoWriter(str(temp_video_path), fourcc, fps, (width, height))

    heygen_mask = None
    heygen_alpha = None
    if mask_mode == "heygen":
        cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
        ret, reference_frame = cap.read()
        if ret:
            reference_image = Image.fromarray(cv2.cvtColor(reference_frame, cv2.COLOR_BGR2RGB))
            heygen_mask, heygen_alpha = build_heygen_template_maps(reference_image)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # Process each frame
    with tqdm.tqdm(total=total_frames, desc="Processing video frames") as pbar:
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert frame to PIL Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            # Get watermark mask
            mask_image = get_mask(pil_image, mask_mode, florence_model, florence_processor, device, max_bbox_percent, detection_prompt, heygen_mask)
            
            # Process frame
            if transparent:
                # For video, we can't use transparency, so we'll fill with a color or background
                result_image = make_region_transparent(pil_image, mask_image)
                # Convert RGBA to RGB by filling transparent areas with white
                background = Image.new("RGB", result_image.size, (255, 255, 255))
                background.paste(result_image, mask=result_image.split()[3])
                result_image = background
            elif mask_mode == "heygen":
                result_image = remove_heygen_watermark(pil_image, heygen_alpha, heygen_mask)
            else:
                lama_result = process_image_with_lama(np.array(pil_image), np.array(mask_image), model_manager)
                result_image = Image.fromarray(cv2.cvtColor(lama_result, cv2.COLOR_BGR2RGB))
            
            # Convert back to OpenCV format and write to output video
            frame_result = cv2.cvtColor(np.array(result_image), cv2.COLOR_RGB2BGR)
            out.write(frame_result)
            
            # Update progress
            frame_count += 1
            pbar.update(1)
            local_progress = frame_count / total_frames
            progress = int(progress_offset + local_progress * progress_scale)
            print(f"Processing frame {frame_count}/{total_frames}, overall_progress:{progress}%")
    
    # Release resources
    cap.release()
    out.release()
    
    # Combine processed video with original audio using FFmpeg
    try:
        logger.info("Merging processed video with original audio...")
        
        # Check if FFmpeg is available
        try:
            subprocess.check_output(["ffmpeg", "-version"], stderr=subprocess.STDOUT)
        except (subprocess.SubprocessError, FileNotFoundError):
            logger.warning("FFmpeg is not available. Video will be produced without audio.")
            shutil.copy(str(temp_video_path), str(output_file))
        else:
            # Use FFmpeg to combine processed video with original audio
            ffmpeg_cmd = [
                "ffmpeg", "-y",
                "-i", str(temp_video_path),  # Processed video without audio
                "-i", str(input_path),       # Original video with audio
                "-c:v", "libx264",
                "-preset", "medium",
                "-crf", "18",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                "-c:a", "aac",               # Encode audio as AAC for better compatibility
                "-map", "0:v:0",             # Use video track from first file (processed video)
                "-map", "1:a:0",             # Use audio track from second file (original video)
                "-shortest",                  # End when the shortest track ends
                str(output_file)
            ]

            # Execute FFmpeg
            subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            logger.info("Audio/video merge completed successfully!")
    except Exception as e:
        logger.error(f"Error during audio/video merge: {str(e)}")
        # In case of error, use video without audio
        shutil.copy(str(temp_video_path), str(output_file))
    finally:
        # Clean up temporary files
        try:
            os.remove(str(temp_video_path))
            os.rmdir(temp_dir)
        except:
            pass
    
    final_progress = progress_offset + progress_scale
    logger.info(f"input_path:{input_path}, output_path:{output_file}, overall_progress:{final_progress}")
    return output_file


def process_video_two_pass(input_path, output_path, florence_model, florence_processor, model_manager, device, transparent, max_bbox_percent, force_format, detection_prompt="watermark", detection_skip=1, fade_in_sec=0.0, fade_out_sec=0.0, mask_mode="florence", progress_offset=0, progress_scale=100):
    """
    Two-pass video processing with frame skip detection and fade in/out handling.

    Pass 1: Detect watermarks every N frames (sparse detection)
    Pass 2: Apply inpainting to all frames using interpolated masks

    This is more efficient for videos where watermarks don't change rapidly,
    and handles fade in/out watermarks by extending the mask temporally.
    """
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        logger.error(f"Error opening video file: {input_path}")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Convert seconds to frames
    fade_in_frames = int(fade_in_sec * fps)
    fade_out_frames = int(fade_out_sec * fps)

    logger.info(f"Two-pass processing: {total_frames} frames, skip={detection_skip}, fade_in={fade_in_frames}f, fade_out={fade_out_frames}f")

    # Determine output format
    if force_format:
        output_format = force_format.upper()
    else:
        output_format = "MP4"

    # Create output video file
    output_path = Path(output_path)
    if output_path.is_dir():
        output_file = output_path / f"{input_path.stem}_no_watermark.{output_format.lower()}"
    else:
        output_file = output_path.with_suffix(f".{output_format.lower()}")

    # ========== PASS 1: DETECTION (sparse) ==========
    logger.info("Pass 1: Detecting watermarks...")
    detections = {}  # frame_idx -> [bbox, bbox, ...]
    detection_frames = list(range(0, total_frames, detection_skip))

    with tqdm.tqdm(total=len(detection_frames), desc="Pass 1: Detection") as pbar:
        for frame_idx in detection_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            bboxes = detect_only(pil_image, florence_model, florence_processor, device, max_bbox_percent, detection_prompt)

            if bboxes:
                accepted_bboxes = [b["bbox"] for b in bboxes if b["accepted"]]
                if accepted_bboxes:
                    detections[frame_idx] = accepted_bboxes

            pbar.update(1)
            local_progress = (pbar.n / len(detection_frames)) * 0.5  # Pass 1 = 0-50% local
            progress = int(progress_offset + local_progress * progress_scale)
            print(f"Pass 1: frame {frame_idx}/{total_frames}, overall_progress:{progress}%")

    logger.info(f"Pass 1 complete: found watermarks in {len(detections)} detection points")

    # ========== TIMELINE EXPANSION ==========
    # Create frame->bbox mapping with fade in/out expansion
    frame_masks = {}  # frame_idx -> [bbox, ...]

    for det_frame, bboxes in detections.items():
        # Expand backwards (fade in) - watermark might be fading in before detection
        start_frame = max(0, det_frame - fade_in_frames)
        # Expand forwards (fade out) - continue masking after detection
        # Also include frames until next detection point
        end_frame = min(total_frames, det_frame + detection_skip + fade_out_frames)

        for f in range(start_frame, end_frame):
            if f not in frame_masks:
                frame_masks[f] = []
            # Add bboxes, avoiding duplicates
            for bbox in bboxes:
                if bbox not in frame_masks[f]:
                    frame_masks[f].append(bbox)

    logger.info(f"Timeline expanded: {len(frame_masks)} frames will have inpainting applied")

    # ========== PASS 2: INPAINTING ==========
    logger.info("Pass 2: Applying inpainting...")

    # Create temporary file for video without audio
    temp_dir = tempfile.mkdtemp()
    temp_video_path = Path(temp_dir) / f"temp_no_audio.{output_format.lower()}"

    # Set codec
    if output_format.upper() == "MP4":
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    elif output_format.upper() == "AVI":
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
    else:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out = cv2.VideoWriter(str(temp_video_path), fourcc, fps, (width, height))

    # Reset video to beginning
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    with tqdm.tqdm(total=total_frames, desc="Pass 2: Inpainting") as pbar:
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx in frame_masks:
                # This frame needs inpainting
                pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                # Create mask from bboxes
                mask = Image.new("L", pil_image.size, 0)
                draw = ImageDraw.Draw(mask)
                for bbox in frame_masks[frame_idx]:
                    x1, y1, x2, y2 = bbox
                    draw.rectangle([x1, y1, x2, y2], fill=255)

                # Apply inpainting or transparency
                if transparent:
                    result_image = make_region_transparent(pil_image, mask)
                    background = Image.new("RGB", result_image.size, (255, 255, 255))
                    background.paste(result_image, mask=result_image.split()[3])
                    result_image = background
                else:
                    lama_result = process_image_with_lama(np.array(pil_image), np.array(mask), model_manager)
                    result_image = Image.fromarray(cv2.cvtColor(lama_result, cv2.COLOR_BGR2RGB))

                frame_result = cv2.cvtColor(np.array(result_image), cv2.COLOR_RGB2BGR)
            else:
                # No watermark detected for this frame, copy original
                frame_result = frame

            out.write(frame_result)
            frame_idx += 1
            pbar.update(1)
            local_progress = 0.5 + (frame_idx / total_frames) * 0.5  # Pass 2 = 50-100% local
            progress = int(progress_offset + local_progress * progress_scale)
            print(f"Pass 2: frame {frame_idx}/{total_frames}, overall_progress:{progress}%")

    cap.release()
    out.release()

    # ========== MERGE WITH AUDIO ==========
    try:
        logger.info("Merging processed video with original audio...")
        try:
            subprocess.check_output(["ffmpeg", "-version"], stderr=subprocess.STDOUT)
        except (subprocess.SubprocessError, FileNotFoundError):
            logger.warning("FFmpeg is not available. Video will be produced without audio.")
            shutil.copy(str(temp_video_path), str(output_file))
        else:
            ffmpeg_cmd = [
                "ffmpeg", "-y",
                "-i", str(temp_video_path),
                "-i", str(input_path),
                "-c:v", "libx264",
                "-preset", "medium",
                "-crf", "18",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                "-c:a", "aac",
                "-map", "0:v:0",
                "-map", "1:a:0",
                "-shortest",
                str(output_file)
            ]
            subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            logger.info("Audio/video merge completed successfully!")
    except Exception as e:
        logger.error(f"Error during audio/video merge: {str(e)}")
        shutil.copy(str(temp_video_path), str(output_file))
    finally:
        try:
            os.remove(str(temp_video_path))
            os.rmdir(temp_dir)
        except:
            pass

    final_progress = progress_offset + progress_scale
    logger.info(f"input_path:{input_path}, output_path:{output_file}, overall_progress:{final_progress}")
    return output_file


def handle_one(image_path: Path, output_path: Path, florence_model, florence_processor, model_manager, device, transparent, max_bbox_percent, force_format, overwrite, detection_prompt="watermark", detection_skip=1, fade_in=0.0, fade_out=0.0, mask_mode="florence", progress_offset=0, progress_scale=100):
    # SAFETY: Never overwrite the input file
    if image_path.resolve() == output_path.resolve():
        logger.error(f"Cannot overwrite input file: {image_path}. Choose a different output path.")
        print(f"ERROR: Cannot overwrite input file! Choose a different output folder.")
        return

    if output_path.exists() and not overwrite:
        logger.info(f"Skipping existing file: {output_path}")
        return

    # Check if it's a video file
    if is_video_file(image_path):
        # Use two-pass if detection_skip > 1 or fade handling is needed
        use_two_pass = detection_skip > 1 or fade_in > 0 or fade_out > 0
        if use_two_pass:
            if mask_mode == "heygen":
                logger.warning("HeyGen mask mode processes videos frame-by-frame; detection skip and fade settings are ignored.")
                return process_video(image_path, output_path, florence_model, florence_processor, model_manager, device, transparent, max_bbox_percent, force_format, detection_prompt, mask_mode, progress_offset, progress_scale)
            return process_video_two_pass(image_path, output_path, florence_model, florence_processor, model_manager, device, transparent, max_bbox_percent, force_format, detection_prompt, detection_skip, fade_in, fade_out, mask_mode, progress_offset, progress_scale)
        else:
            return process_video(image_path, output_path, florence_model, florence_processor, model_manager, device, transparent, max_bbox_percent, force_format, detection_prompt, mask_mode, progress_offset, progress_scale)

    # Process image
    image = Image.open(image_path).convert("RGB")
    heygen_mask = None
    heygen_alpha = None
    if mask_mode == "heygen":
        heygen_mask, heygen_alpha = build_heygen_template_maps(image)
    mask_image = get_mask(image, mask_mode, florence_model, florence_processor, device, max_bbox_percent, detection_prompt, heygen_mask)

    if transparent:
        result_image = make_region_transparent(image, mask_image)
    elif mask_mode == "heygen":
        result_image = remove_heygen_watermark(image, heygen_alpha, heygen_mask)
    else:
        lama_result = process_image_with_lama(np.array(image), np.array(mask_image), model_manager)
        result_image = Image.fromarray(cv2.cvtColor(lama_result, cv2.COLOR_BGR2RGB))

    # Determine output format
    if force_format:
        output_format = force_format.upper()
    elif transparent:
        output_format = "PNG"
    else:
        output_format = image_path.suffix[1:].upper()
        if output_format not in ["PNG", "WEBP", "JPG"]:
            output_format = "PNG"
    
    # Map JPG to JPEG for PIL compatibility
    if output_format == "JPG":
        output_format = "JPEG"

    if transparent and output_format == "JPG":
        logger.warning("Transparency detected. Defaulting to PNG for transparency support.")
        output_format = "PNG"

    new_output_path = output_path.with_suffix(f".{output_format.lower()}")
    result_image.save(new_output_path, format=output_format)
    # Report progress for this image (end of range)
    final_progress = progress_offset + progress_scale
    print(f"input_path:{image_path}, output_path:{new_output_path}, overall_progress:{final_progress}%")
    return new_output_path

@click.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path(), required=False, default=None)
@click.option("--preview", is_flag=True, help="Preview mode: detect watermarks and output JSON with base64 image (no processing).")
@click.option("--overwrite", is_flag=True, help="Overwrite existing files in bulk mode.")
@click.option("--transparent", is_flag=True, help="Make watermark regions transparent instead of removing.")
@click.option("--max-bbox-percent", default=10.0, help="Maximum percentage of the image that a bounding box can cover.")
@click.option("--force-format", type=click.Choice(["PNG", "WEBP", "JPG", "MP4", "AVI"], case_sensitive=False), default=None, help="Force output format. Defaults to input format.")
@click.option("--detection-prompt", default="watermark", help="Text prompt for watermark detection (e.g. 'watermark', 'watermark Sora logo', 'Getty Images').")
@click.option("--mask-mode", type=click.Choice(["florence", "heygen"], case_sensitive=False), default="florence", help="Mask generation mode. 'heygen' uses an OpenCV mask for repeated translucent HeyGen watermarks.")
@click.option("--detection-skip", default=1, type=int, help="Detect watermarks every N frames for videos (1-10). Higher = faster but may miss brief watermarks.")
@click.option("--fade-in", default=0.0, type=float, help="Extend mask backwards by N seconds to handle fade-in watermarks.")
@click.option("--fade-out", default=0.0, type=float, help="Extend mask forwards by N seconds to handle fade-out watermarks.")
def main(input_path: str, output_path: str, preview: bool, overwrite: bool, transparent: bool, max_bbox_percent: float, force_format: str, detection_prompt: str, mask_mode: str, detection_skip: int, fade_in: float, fade_out: float):
    # Input validation
    if detection_skip < 1 or detection_skip > 10:
        logger.warning(f"detection_skip must be 1-10, got {detection_skip}. Using 1.")
        detection_skip = max(1, min(10, detection_skip))
    if fade_in < 0:
        fade_in = 0
    if fade_out < 0:
        fade_out = 0

    input_path = Path(input_path)
    mask_mode = mask_mode.lower()
    if mask_mode not in {"florence", "heygen"}:
        logger.warning(f"Unknown mask mode '{mask_mode}'. Using florence.")
        mask_mode = "florence"

    # ========== PREVIEW MODE ==========
    if preview:
        import json
        import base64
        from io import BytesIO
        import random

        # Get sample image from input
        if input_path.is_dir():
            # Get a random image from directory
            images = list(input_path.glob("*.[jp][pn]g")) + list(input_path.glob("*.webp"))
            videos = list(input_path.glob("*.mp4")) + list(input_path.glob("*.avi")) + list(input_path.glob("*.mov"))
            files = images + videos
            if not files:
                print(json.dumps({"error": "No supported files found in directory"}))
                return
            sample_path = random.choice(files)
        else:
            sample_path = input_path

        # Load image (extract frame if video)
        if is_video_file(sample_path):
            cap = cv2.VideoCapture(str(sample_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            # Get frame from middle of video
            cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                print(json.dumps({"error": f"Could not read frame from video: {sample_path}"}))
                return
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            source_type = "video"
            source_frame = total_frames // 2
        else:
            pil_image = Image.open(sample_path).convert("RGB")
            source_type = "image"
            source_frame = None

        if mask_mode == "heygen":
            pil_image, detections = preview_heygen_mask(pil_image)
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"

            # Force no dtype for CUDA (intentional default)
            # Apply float32 for CPU (compatibility)
            model_dtype = torch.float32 if device == "cpu" else None

            florence_model = Florence2ForConditionalGeneration.from_pretrained(
                "florence-community/Florence-2-large",
                torch_dtype=model_dtype).to(device).eval()
            florence_processor = AutoProcessor.from_pretrained("florence-community/Florence-2-large")

            # Run detection
            detections = detect_only(pil_image, florence_model, florence_processor, device, max_bbox_percent, detection_prompt)

            # Draw bounding boxes on image
            draw = ImageDraw.Draw(pil_image)
            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                color = (0, 255, 0) if det["accepted"] else (255, 0, 0)  # Green if accepted, red if rejected
                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                # Draw label
                label = f"{det['area_percent']:.1f}%"
                draw.text((x1, y1 - 15), label, fill=color)

        # Convert to base64
        buffer = BytesIO()
        pil_image.save(buffer, format="PNG")
        img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        # Output JSON result
        result = {
            "image": img_base64,  # Just base64, GUI adds prefix
            "detections": detections,
            "source": str(sample_path),
            "source_type": source_type,
            "source_frame": source_frame,
            "prompt_used": detection_prompt,
            "max_bbox_percent": max_bbox_percent,
            "mask_mode": mask_mode
        }
        print(json.dumps(result))
        return

    # ========== NORMAL PROCESSING MODE ==========
    output_path = Path(output_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Force no dtype for CUDA (intentional default)
    # Apply float32 for CPU (compatibility)
    model_dtype = torch.float32 if device == "cpu" else None

    if mask_mode == "heygen":
        florence_model = None
        florence_processor = None
        logger.info("HeyGen mask mode enabled; skipping Florence-2 model load")
    else:
        florence_model = Florence2ForConditionalGeneration.from_pretrained(
            "florence-community/Florence-2-large",
            torch_dtype=model_dtype).to(device).eval()
        florence_processor = AutoProcessor.from_pretrained("florence-community/Florence-2-large")
        logger.info("Florence-2 Model loaded")

    if not transparent and mask_mode != "heygen":
        model_manager = load_lama_model(device)
        logger.info("LaMa model loaded")
    else:
        model_manager = None

    if input_path.is_dir():
        if not output_path.exists():
            output_path.mkdir(parents=True)

        # Include video files in the search
        images = list(input_path.glob("*.[jp][pn]g")) + list(input_path.glob("*.webp"))
        videos = list(input_path.glob("*.mp4")) + list(input_path.glob("*.avi")) + list(input_path.glob("*.mov")) + list(input_path.glob("*.mkv"))
        files = images + videos
        total_files = len(files)

        for idx, file_path in enumerate(tqdm.tqdm(files, desc="Processing files")):
            output_file = output_path / file_path.name
            # Calculate progress range for this file
            progress_offset = int(idx / total_files * 100)
            progress_scale = int(100 / total_files)
            handle_one(file_path, output_file, florence_model, florence_processor, model_manager, device, transparent, max_bbox_percent, force_format, overwrite, detection_prompt, detection_skip, fade_in, fade_out, mask_mode, progress_offset, progress_scale)
    else:
        # Single file mode - if output is a directory, construct file path
        if output_path.is_dir():
            output_file = output_path / input_path.name
        else:
            output_file = output_path

        # Ensure video output has proper extension
        if is_video_file(input_path) and output_file.suffix.lower() not in ['.mp4', '.avi', '.mov', '.mkv']:
            if force_format and force_format.upper() in ["MP4", "AVI"]:
                output_file = output_file.with_suffix(f".{force_format.lower()}")
            else:
                output_file = output_file.with_suffix(".mp4")  # Default to mp4

        handle_one(input_path, output_file, florence_model, florence_processor, model_manager, device, transparent, max_bbox_percent, force_format, overwrite, detection_prompt, detection_skip, fade_in, fade_out, mask_mode)
        print(f"input_path:{input_path}, output_path:{output_file}, overall_progress:100")

if __name__ == "__main__":
    main()
