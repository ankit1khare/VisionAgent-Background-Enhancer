import os
import cv2
import sys
import time
import glob
import base64
import requests
import numpy as np
import streamlit as st
from pathlib import Path
from PIL import Image
from rembg import remove
from dotenv import load_dotenv
from pillow_heif import register_heif_opener
register_heif_opener()
from typing import List, Any
import openai
import vision_agent as va
from vision_agent.tools import *
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from anthropic import Anthropic

# ---- Local Imports (make sure vision_tools.py is in the same directory) ----
from vision_tools import (
    detect_objects,
    segment_and_save_objects,
    unsharp_mask,
    auto_enhance_image_simple,
    BackgroundSelector,
    create_composite_with_shadow,
    cleanup_temp_files
)

# ------------------------------------------
# Streamlit Configuration & Basic Setup
# ------------------------------------------
st.set_page_config(
    page_title="Vision Agent with Natural Shadows",
    layout="centered"
)

load_dotenv()
os.environ["OPENAI_API_KEY"] = "YOUR OPENAI KEY"

col1, col2 = st.columns(2)

# Define Global Paths
FILE = Path(__file__).resolve()
ROOT = FILE.parent
if ROOT not in sys.path:
    sys.path.append(str(ROOT))
ROOT = ROOT.relative_to(Path.cwd())

# Define Directories
IMAGES_DIR = ROOT / 'images'
BG_DIR = ROOT / 'backgrounds'
OUTPUT_PATH = ROOT / 'output.png'

# Create Directories if needed
SAVE_DIRECTORY = "uploaded_image"
SAMPLE_DIRECTORY = "sample_images"
os.makedirs(SAVE_DIRECTORY, exist_ok=True)
os.makedirs(SAMPLE_DIRECTORY, exist_ok=True)

MAX_DIMENSION = 1920

# Session State Initialization
def initialize_session_state():
    if "temp_directory" not in st.session_state:
        st.session_state.temp_directory = "temp_files"
        os.makedirs(st.session_state.temp_directory, exist_ok=True)

    default_states = {
        "processing": False,
        "selected_bg_path": None,
        "top_bg_paths": None,
        "caption": None,
        "sam_dimensions": None,
        "final_caption": None,
        "preview_images_data": None,
        "previous_upload": None,
        "previous_sample": None,
        "upload_time": None,
        "current_image_path": None,
        "image_type": None
    }

    for key, default_value in default_states.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

initialize_session_state()

# ------------------------------------------
# Helper Functions
# ------------------------------------------
def handle_new_upload(uploaded_file):
    # Reset all states
    states_to_reset = [
        "processing",
        "selected_bg_path",
        "top_bg_paths",
        "caption",
        "sam_dimensions",
        "final_caption",
        "preview_images_data"
    ]
    for state in states_to_reset:
        st.session_state[state] = None

    # Update upload metadata
    st.session_state.upload_time = time.time()
    st.session_state.previous_upload = uploaded_file.name if uploaded_file else None
    st.session_state.image_type = uploaded_file.type if uploaded_file else None

    # Clean up old files
    cleanup_temp_files(SAVE_DIRECTORY)
    return True

# ------------------------------------------
# Sidebar: Upload & Sample Images
# ------------------------------------------
st.sidebar.title("Configurations")
source_image = st.sidebar.file_uploader(
    "Choose an Image ...",
    type=("jpg", "png", "jpeg", "bmp", "webp", "HEIC")
)

save_path = None
if source_image:
    # Ensure we keep original extension
    ext = source_image.type.split('/')[-1]
    if ext.lower() in ["heic", "heif"]:
        ext = "jpg"  # Convert HEIC to JPG
    filename = "uploaded_image." + ext
    save_path = os.path.join(SAVE_DIRECTORY, filename)

# If a new file is uploaded, handle it
if source_image and (source_image.name != st.session_state.previous_upload):
    if handle_new_upload(source_image):
        st.rerun()

# Save the uploaded image if it exists
if source_image:
    # Read into PIL
    image = Image.open(source_image).convert("RGB")
    image_np = np.array(image)
    h, w = image_np.shape[:2]
    if max(h, w) > MAX_DIMENSION:
        scale = MAX_DIMENSION / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        image_np = cv2.resize(image_np, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Save the (possibly resized) image
    is_success, buffer = cv2.imencode(f".{ext}", cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
    if is_success:
        with open(save_path, "wb") as f:
            f.write(buffer)
    st.sidebar.success(f"Uploaded Image Saved Successfully")

# Sample Images in the Sidebar
st.sidebar.subheader("Or Choose a Sample Image")
sample_images = [f"image{i}.jpg" for i in range(1, 7)]
sample_images_paths = [str(IMAGES_DIR / img) for img in sample_images]
sample_image = st.sidebar.selectbox("Select a Sample Image", sample_images, index=0)
sampleImage_path = str(IMAGES_DIR / sample_image)

# Display sample if no user upload
if source_image is None:
    st.sidebar.image(sampleImage_path, caption=f"Sample: {sample_image}", use_container_width=True)
    selected_image_path = sampleImage_path
else:
    st.sidebar.image(save_path, caption="Uploaded Image", use_container_width=True)
    selected_image_path = save_path

# Start Action Button
start_action = st.sidebar.button("Start Action")

# ------------------------------------------
# Core Logic Flow
# ------------------------------------------
if start_action and not st.session_state.processing:
    st.session_state.processing = True

    # Step 1: Detect Objects
    with st.spinner("Detecting Objects..."):
        results = detect_objects(selected_image_path)
        detection_output = results["output_image_path"]
        st.session_state.caption = results["claude_caption"]
        st.session_state.final_caption = results["ad_caption"]

    st.markdown("<h3 style='text-align: center;'>Object Detection</h3>", unsafe_allow_html=True)
    st.image(detection_output, caption="Detected Object(s)", use_container_width=True)
    time.sleep(3)
    st.empty()

    # Step 2: Segmentation (SAM)
    with st.spinner("Segmenting Main Object..."):
        seg_result = segment_and_save_objects(
            selected_image_path,
            prompt='detect the one major object in the scene and avoid background clutter objects like napkins, hairs, tissues, etc.'
        )
        if seg_result:
            sam_img = Image.open(seg_result)
            st.session_state.sam_dimensions = (sam_img.width, sam_img.height)
            st.image(sam_img, caption="Segmented Main Object", use_container_width=True)
        else:
            st.warning("No valid objects found for segmentation!")
    time.sleep(3)
    st.empty()

    # Step 3: Enhance Foreground
    with st.spinner("Enhancing Foreground..."):
        # Convert segmented to BGR
        seg_bgr = cv2.imread(seg_result)
        # Step 3a: Sharpen
        sharpened = unsharp_mask(seg_bgr)
        # Step 3b: Auto Enhance
        final_foreground = auto_enhance_image_simple(sharpened)
        sh_output_path = "sh_output.png"
        cv2.imwrite(sh_output_path, final_foreground)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<h4 style='text-align: center;'>Original Segmented</h4>", unsafe_allow_html=True)
            st.image(seg_result, caption="Original", use_container_width=True)
        with col2:
            st.markdown("<h4 style='text-align: center;'>Enhanced Foreground</h4>", unsafe_allow_html=True)
            st.image(sh_output_path, caption="Enhanced", use_container_width=True)

    time.sleep(3)
    st.empty()

    # Step 4: Choose/Generate Backgrounds
    with st.spinner("Selecting / Generating Backgrounds..."):
        bs = BackgroundSelector()
        # Example: Generate new backgrounds (or you can do "select_best_background_openai")
        top_bg_paths = bs.select_best_background_openai(st.session_state.caption, str(BG_DIR))
        # But let's just call generate_background_images to get 3 new backgrounds:
        # top_bg_paths = bs.generate_background_images(str(BG_DIR))
        st.session_state.top_bg_paths = top_bg_paths

    st.session_state.processing = False

# ------------------------------------------
# After Processing: Show 3 Composites
# ------------------------------------------
if not st.session_state.processing and st.session_state.top_bg_paths:
    st.markdown("<h3 style='text-align: center;'>Applying Top 3 Backgrounds with Shadows</h3>", unsafe_allow_html=True)

    # If we haven't built the 3 preview images yet, do so
    if st.session_state.preview_images_data is None:
        preview_images = []
        for idx, bg_path in enumerate(st.session_state.top_bg_paths):
            try:
                # Composite with synthetic shadow
                composite_path = f"composite_option_{idx+1}.png"
                create_composite_with_shadow(
                    bg_path,
                    "sh_output.png",  # The enhanced foreground
                    output_path=composite_path,
                    shadow_offset=(30, 30),  # or tweak to your preference
                    shadow_blur=25,          # bigger => softer shadow
                    shadow_opacity=120       # 0-255 (0 = fully transparent, 255 = fully opaque)
                )
                preview_images.append(composite_path)
            except Exception as e:
                print(f"Error compositing image {idx+1}: {str(e)}")
                continue

        st.session_state.preview_images_data = preview_images

    # Display the 3 composite previews
    c1, c2, c3 = st.columns(3)
    with c1:
        st.image(st.session_state.preview_images_data[0], caption="Option 1", use_container_width=True)
    with c2:
        st.image(st.session_state.preview_images_data[1], caption="Option 2", use_container_width=True)
    with c3:
        st.image(st.session_state.preview_images_data[2], caption="Option 3", use_container_width=True)

    # Let user pick one
    st.session_state.selected_bg_path = st.selectbox(
        "Choose Background Option",
        st.session_state.top_bg_paths,
        format_func=lambda x: f"Option {st.session_state.top_bg_paths.index(x) + 1}"
    )

    if st.session_state.selected_bg_path:
        selected_idx = st.session_state.top_bg_paths.index(st.session_state.selected_bg_path)
        st.markdown("<h3 style='text-align: center;'>Final Selected Background</h3>", unsafe_allow_html=True)
        st.image(st.session_state.preview_images_data[selected_idx], caption="Buy Now!", use_container_width=True)

        # Styled Product Description
        if st.session_state.final_caption:
            st.markdown(f"""
            <div style='background-color: #f8f9fa; padding: 20px; border-radius: 10px; margin: 20px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                <h4 style='color: #1f2937; margin-bottom: 10px; font-size: 1.2em;'>About this item</h4>
                <p style='color: #4b5563; line-height: 1.6; font-size: 1.1em; margin-bottom: 15px;'>{st.session_state.final_caption}</p>
            </div>
            """, unsafe_allow_html=True)

# ------------------------------------------
# Reset if sample changes
# ------------------------------------------
if "previous_sample" not in st.session_state:
    st.session_state.previous_sample = sample_image

if st.session_state.previous_sample != sample_image:
    # Clear states
    st.session_state.processing = False
    st.session_state.selected_bg_path = None
    st.session_state.top_bg_paths = None
    st.session_state.caption = None
    st.session_state.sam_dimensions = None
    st.session_state.final_caption = None
    st.session_state.preview_images_data = None
    st.session_state.previous_upload = None
    # Cleanup
    cleanup_temp_files(SAVE_DIRECTORY)
    # Update
    st.session_state.previous_sample = sample_image
    st.rerun()
