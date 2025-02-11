import os
import cv2
import glob
import uuid
import base64
import numpy as np
import requests
from PIL import Image, ImageFilter
from rembg import remove
from io import BytesIO
import sys
import re
import openai
from vision_agent.tools import *
from typing import *
#import replicate
from pillow_heif import register_heif_opener
register_heif_opener()
import vision_agent as va
from vision_agent.tools import register_tool
from vision_agent.tools import load_image, florence2_object_detection, overlay_bounding_boxes, save_image, \
    qwen2_vl_images_vqa, florence2_sam2_instance_segmentation
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
import anthropic

# Optional: If you use openai or anthropic calls, import them accordingly
# from anthropic import Anthropic

# -------------------------------------------------------------------
# Clean Up Temp Files
# -------------------------------------------------------------------
def cleanup_temp_files(save_directory="uploaded_image"):
    # Remove composite images
    for i in range(1, 4):
        temp_comp = f"composite_option_{i}.png"
        if os.path.exists(temp_comp):
            os.remove(temp_comp)
    # Remove known output images
    for file in ['output.png', 'sh_output.png', 'temp_shoe.png', 'temp_mask.png', 'debug_mask_gray.png']:
        if os.path.exists(file):
            os.remove(file)
    # Clean the uploaded images directory
    if os.path.exists(save_directory):
        for file in os.listdir(save_directory):
            file_path = os.path.join(save_directory, file)
            if os.path.isfile(file_path):
                try:
                    os.remove(file_path)
                except Exception:
                    pass

# -------------------------------------------------------------------
# Object Detection (Florence2 style or Qwen2; placeholder code)
# -------------------------------------------------------------------
def detect_objects(image_path):
    # 1. Load the image
    image = load_image(image_path)

    # 2. Detect objects
    detections = florence2_object_detection("detect the one major object in the scene", image)
    # Update label for detected objects to "foreground"
    for detection in detections:
        detection['label'] = "foreground"

    # 3. Draw bounding boxes on the image
    image_with_boxes = overlay_bounding_boxes(image, detections)

    # 4. Generate two different captions
    # Caption for Claude input (more technical and detailed)
    claude_prompt = "Describe this product in detail, focusing on its physical attributes, materials, design elements, and any visible technical specifications. Include color information, patterns, textures, and overall appearance. Be objective and thorough."

    # Caption for final product AD (marketing focused)
    ad_prompt = "Create a succinct product AD by describing this product and highlighting its features, color patterns, quality, and any unique benefits it offers. Focus on how the product stands out, its appearance, and why someone would want to buy it. Avoid starting with phrases like 'The image depicts' or similar phrases."

    claude_caption = qwen2_vl_images_vqa(claude_prompt, [image])
    ad_caption = qwen2_vl_images_vqa(ad_prompt, [image])

    # 5. Save the resulting image
    output_path = "output_image_with_boxes.jpg"
    save_image(image_with_boxes, output_path)

    # 6. Return the results with both captions
    return {
        "output_image_path": output_path,
        "claude_caption": claude_caption,
        "ad_caption": ad_caption
    }


# -------------------------------------------------------------------
# Segment & Save Objects (SAM-based or other)
# -------------------------------------------------------------------
def segment_and_save_objects(image_path, prompt):
    # Load the image
    image = load_image(image_path)

    # Use florence2_sam2_instance_segmentation to segment the objects
    segmentation_results = florence2_sam2_instance_segmentation(prompt, image)

    # Check if any objects were detected
    if segmentation_results and len(segmentation_results) > 0:
        # Get image dimensions
        height, width = image.shape[:2]

        # Filter out objects whose masks touch the boundaries of the image
        valid_objects = []
        for detected_object in segmentation_results:
            mask = detected_object['mask']
            # Check if the object is touching the boundary
            if not (
                    np.any(mask[0, :]) or  # Top boundary
                    np.any(mask[-1, :]) or  # Bottom boundary
                    np.any(mask[:, 0]) or  # Left boundary
                    np.any(mask[:, -1])  # Right boundary
            ):
                valid_objects.append(detected_object)

        # Check if there are any valid objects left
        if not valid_objects:
            print(f"No valid objects detected for prompt: {prompt}.")
            return None

        # Sort the valid objects by mask area (descending order)
        valid_objects = sorted(valid_objects, key=lambda obj: np.sum(obj['mask']), reverse=True)

        # Define a threshold to filter smaller objects (relative to the largest valid object's mask area)
        largest_mask_area = np.sum(valid_objects[0]['mask'])
        area_threshold = 0.4  # Keep objects with mask areas at least 70% of the largest mask area

        # Initialize a combined mask for filtered objects
        combined_mask = np.zeros((height, width), dtype=np.uint8)

        # Iterate over valid objects and filter them based on the area threshold
        for detected_object in valid_objects:
            mask = detected_object['mask']
            mask_area = np.sum(mask)
            if mask_area >= area_threshold * largest_mask_area:
                # Add the object's mask to the combined mask
                combined_mask = np.logical_or(combined_mask, mask)

        # Apply the combined mask to the original image
        segmented_objects = image * combined_mask[:, :, np.newaxis]

        # Create a white background
        white_background = np.ones_like(image) * 255

        # Blend the segmented objects with the white background
        final_image = np.where(combined_mask[:, :, np.newaxis] == 1, segmented_objects, white_background)

        # Generate a filename for the segmented objects
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        segmented_filename = f"{base_name}_{prompt.replace(' ', '_')}_segmented.png"

        # Save the segmented objects
        save_image(final_image.astype(np.uint8), segmented_filename)

        return segmented_filename
    else:
        print(f"No {prompt} detected in the image.")
        return None



# -------------------------------------------------------------------
# Image Enhancement
# -------------------------------------------------------------------
def unsharp_mask(image, sigma=1.0, strength=1.5):
    # Gaussian blur
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    # Weighted difference
    sharpened = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
    return sharpened

def auto_enhance_image_simple(img_bgr: np.ndarray) -> np.ndarray:
    """
    Simple brightness/contrast + mild unsharp on the object region (white is background).
    """
    if img_bgr is None:
        raise ValueError("Input image is None.")

    # 1) Separate foreground from white BG
    lower_white = np.array([200, 200, 200], np.uint8)
    upper_white = np.array([255, 255, 255], np.uint8)
    mask_bg = cv2.inRange(img_bgr, lower_white, upper_white)
    fg_mask = cv2.bitwise_not(mask_bg)

    # 2) Stats on the foreground
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    fg_pixels = gray[fg_mask > 0]
    if len(fg_pixels) == 0:
        return img_bgr
    mean_val = np.mean(fg_pixels)
    std_val = np.std(fg_pixels)

    target_mean = 128.0
    target_std = 64.0

    beta = target_mean - mean_val
    alpha = 1.0
    if std_val > 1e-5:
        alpha = target_std / std_val

    alpha = np.clip(alpha, 0.9, 1.1)
    beta = np.clip(beta, -20, 20)

    # 3) Apply brightness/contrast
    enhanced = cv2.convertScaleAbs(img_bgr, alpha=alpha, beta=beta)

    # 4) Mild unsharp if needed
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap_std = lap[fg_mask > 0].std()
    if lap_std < 10.0:
        blurred = cv2.GaussianBlur(enhanced.astype(np.float32), (3, 3), 0)
        amount = 0.5
        sharpened_float = cv2.addWeighted(enhanced.astype(np.float32), 1+amount, blurred, -amount, 0)
        sharpened = np.clip(sharpened_float, 0, 255).astype(np.uint8)
        # preserve original background
        bg_mask_3c = cv2.merge([mask_bg]*3)
        final_img = np.where(bg_mask_3c == 255, img_bgr, sharpened)
    else:
        # If no need to sharpen
        final_img = enhanced

    return final_img

# -------------------------------------------------------------------
# Background Selection / Generation
# -------------------------------------------------------------------
class BackgroundSelector:
    def encode_image(self, image_path: str):
        with open(image_path, 'rb') as f:
            data = f.read()
        ext = image_path.lower().split('.')[-1]
        media_types = {
            'jpg': 'image/jpeg',
            'jpeg': 'image/jpeg',
            'png': 'image/png',
            'gif': 'image/gif',
            'webp': 'image/webp'
        }
        mt = media_types.get(ext, 'image/jpeg')
        encoded = base64.b64encode(data).decode('utf-8')
        return encoded, mt

    def select_best_background_openai(self, item, image_directory):
        # Prepare the system prompt – similar to your original prompt.
        system_prompt = (
            f"Rate how well each background complements the product with caption: '{item}' "
            "considering:\n"
            "1. Score higher for backgrounds that contain unmatching colors with the foreground.\n"
            "For each image, consider a diverse set of combinations with an open mind and "
            "respond with ONLY a number between 1-20, one per line."
        )

        # Find all supported images in the provided directory.
        supported_formats = ('*.jpg', '*.jpeg', '*.png', '*.gif', '*.webp')
        background_gallery = []
        for fmt in supported_formats:
            background_gallery.extend(glob.glob(os.path.join(image_directory, fmt)))

        if not background_gallery:
            raise ValueError(f"No supported images found in directory: {image_directory}")

        batch_size = 3
        scores = []

        for i in range(0, len(background_gallery), batch_size):
            batch = background_gallery[i:i + batch_size]

            # Build the user message by starting with a header
            user_message = "Rate each background (one score per line):\n\n"
            # Append each image's details to the user message.
            for img_path in batch:
                try:
                    encoded_image, media_type = self.encode_image(img_path)
                    # Including the entire encoded image might be too long; consider sending only a portion,
                    # or a placeholder indicating the image is attached.
                    snippet = encoded_image[:100]  # first 100 characters
                    user_message += f"Image ({os.path.basename(img_path)} - {media_type}): {snippet}...\n\n"
                except Exception as e:
                    print(f"Error encoding image {img_path}: {str(e)}")
                    continue

            try:
                # Call the OpenAI ChatCompletion API (using GPT-4 here; change model if needed)
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message}
                    ],
                    max_tokens=8000
                )
                # Extract the assistant's reply text.
                reply = response['choices'][0]['message']['content'].strip()

                # Split the reply into lines and try to convert each line to a float.
                found_scores = []
                for line in reply.splitlines():
                    stripped = line.strip()
                    # Remove any extraneous punctuation (if any) and check if it's numeric.
                    if stripped.replace('.', '', 1).isdigit():
                        found_scores.append(float(stripped))

                if len(found_scores) == len(batch):
                    for score, img_path in zip(found_scores, batch):
                        scores.append((score, img_path))
                else:
                    print(
                        f"Skipping batch due to mismatched scores: got {len(found_scores)} scores for {len(batch)} images"
                    )

            except Exception as e:
                print(f"Error processing batch: {str(e)}")
                continue

        # After scoring all batches, print the results.
        print("\nBackground Scores:")
        print("-----------------")
        for score, img_path in sorted(scores, key=lambda x: x[0], reverse=True):
            print(f"Score: {score:>5.1f} | Background: {os.path.basename(img_path)}")
        print("-----------------")

        if not scores:
            print("Warning: Using fallback background selection")
            return background_gallery[:3]

        # Return the top 3 backgrounds
        top_images = sorted(scores, key=lambda x: x[0], reverse=True)[:3]
        return [img_path for _, img_path in top_images]

    def generate_background_images(self, image_directory):
        """
        Calls OpenAI's image generation to produce 3 backgrounds.
        Make sure you have OPENAI_API_KEY set.
        """
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        if not OPENAI_API_KEY:
            print("OPENAI_API_KEY not found. Returning empty.")
            return []

        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        }
        endpoint = "https://api.openai.com/v1/images/generations"

        prompt = (
            "Create a clean, professional product background in a uniform silver or gray color "
            "with a very subtle, soft texture—no distinct patterns, no text, and no objects. Keep it well-lit, "
            "minimal, and slightly matte, suitable for showcasing items in e-commerce. "
            "Make sure it fully fills the frame—no corners or edges visible."
        )

        payload = {
            "prompt": prompt,
            "n": 3,
            "size": "1024x1024",
            "response_format": "url"
        }

        try:
            response = requests.post(endpoint, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            outputs = [entry["url"] for entry in data.get("data", [])]
        except Exception as e:
            print("Error generating images with OpenAI:", e)
            return []

        # Store them in a subdir
        out_dir = os.path.join(image_directory, "generated_backgrounds")
        os.makedirs(out_dir, exist_ok=True)
        paths = []
        for i, url in enumerate(outputs):
            try:
                r = requests.get(url)
                r.raise_for_status()
                fname = f"generated_bg_{uuid.uuid4().hex[:8]}_{i}.png"
                fpath = os.path.join(out_dir, fname)
                with open(fpath, "wb") as f:
                    f.write(r.content)
                paths.append(fpath)
            except Exception as e:
                print("Error downloading generated image:", e)
                continue

        return paths

# -------------------------------------------------------------------
# Synthetic Shadow + Composite
# -------------------------------------------------------------------
def create_composite_with_shadow(
    bg_path: str,
    fg_path: str,
    output_path: str,
    shadow_offset=(15, 15),
    shadow_blur=20,
    shadow_opacity=120
):
    """
    1. Remove background from the foreground image if not already RGBA.
    2. Create a shadow from the alpha mask.
    3. Paste shadow onto background, then paste foreground on top.
    4. Save final composite to `output_path`.
    """
    # Load background
    bg = Image.open(bg_path).convert("RGBA")
    bg_w, bg_h = bg.size

    # Load foreground, remove BG
    fg_img_raw = Image.open(fg_path).convert("RGBA")
    # Ensure we remove any leftover backgrounds
    fg_img_clean = remove(fg_img_raw)
    fg_w, fg_h = fg_img_clean.size

    # We'll resize the background to match the foreground's scale
    # or vice versa. Usually you'd want to keep the BG bigger.
    # Let's handle it so the final composite is the same size as the BG.
    # If foreground is bigger than BG, we scale down the foreground.
    if fg_w > bg_w or fg_h > bg_h:
        scale = min(bg_w / fg_w, bg_h / fg_h)
        new_w = int(fg_w * scale)
        new_h = int(fg_h * scale)
        fg_img_clean = fg_img_clean.resize((new_w, new_h), Image.Resampling.LANCZOS)
        fg_w, fg_h = new_w, new_h

    # Optionally, you can keep the FG size fixed and resize the BG:
    # But for now, we proceed with the approach above.

    # Create a new composite base with the same size as BG
    composite = Image.new("RGBA", (bg_w, bg_h), (0, 0, 0, 0))

    # Paste the background
    composite.paste(bg, (0, 0))

    # Build the shadow
    # 1) Extract alpha channel from the foreground
    alpha = fg_img_clean.split()[3]  # single-channel
    # 2) Create a black shape from alpha
    shadow = Image.new("RGBA", (fg_w, fg_h), color=(0, 0, 0, 0))
    # Place black in the RGB channels where alpha is > 0
    # We can do this by placing a black image masked by alpha
    black_img = Image.new("RGBA", (fg_w, fg_h), color=(0, 0, 0, shadow_opacity))
    shadow.paste(black_img, mask=alpha)

    # 3) Blur the shadow
    shadow = shadow.filter(ImageFilter.GaussianBlur(shadow_blur))

    # 4) Determine position to place the FG (centered, or your own logic)
    # For demonstration: center them
    cx_bg, cy_bg = bg_w // 2, bg_h // 2
    cx_fg, cy_fg = fg_w // 2, fg_h // 2
    offset_x = cx_bg - cx_fg
    offset_y = cy_bg - cy_fg

    # Then shift the shadow by shadow_offset
    shadow_x = offset_x + shadow_offset[0]
    shadow_y = offset_y + shadow_offset[1]

    # Paste shadow first
    composite.alpha_composite(shadow, (shadow_x, shadow_y))

    # Paste the foreground on top
    composite.alpha_composite(fg_img_clean, (offset_x, offset_y))

    # Save
    composite.convert("RGB").save(output_path, format="PNG", quality=95)
    return output_path
