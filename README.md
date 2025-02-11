# VisionAgent-Background-Enhancer
A Streamlit-based app that transforms cluttered item photos into polished images using VisionAgent for object detection, semantic segmentation, and synthetic shadow generation.

This repository demonstrates how to build this **Streamlit app** that uses [VisionAgent]([https://github.com/your-org/vision-agent](https://github.com/landing-ai/vision-agent)) to:

1. **Upload** or **select** a sample image.
2. **Run object detection** and **semantic segmentation** to isolate the main foreground object.
3. **Automatically remove the background** and **enhance** the object (brightness, contrast, slight sharpening).
4. **Select or generate** new background images.
5. **Composite** the foreground onto each background, adding a **synthetic shadow** for a polished, natural look.

The result is an easy-to-use interactive app that helps you create professional product shots or general photo compositing without extensive manual editing. VisionAgent’s specialized segmentation/inpainting tasks handle the visuals, while Streamlit provides the front-end experience.

---

## Contents

1. [Prerequisites](#prerequisites)
2. [Repository Structure](#repository-structure)
3. [Setup & Installation](#setup--installation)
4. [Environment Variables](#environment-variables)
5. [Running the App](#running-the-app)
6. [Usage Walkthrough](#usage-walkthrough)
7. [Customization Tips](#customization-tips)
8. [Known Limitations](#known-limitations)
9. [License](#license)

---

## Prerequisites

- **Python** 3.8 or higher
- **Pip** for installing dependencies
- A valid **OpenAI API Key**/ **Claude API Key**/ **Gemini API Key** (for text/image generation calls, optional but recommended)
- Basic familiarity with **Streamlit** (for local UI) and **VisionAgent** (for CV tasks)

---

## Repository Structure

```plaintext
.
├── app.py
├── vision_tools.py
├── requirements.txt
├── images/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ... Sample images
├── backgrounds/
│   ├── bg1.jpg
│   ├── bg2.jpg
│   └── ... Default background images
├── uploaded_image/         # Temporary directory for user uploads
├── sample_images/          # Temp directory for selected sample images
└── README.md               # This file
```

- **app.py**  
  Main Streamlit application that orchestrates file upload, object detection, segmentation, background selection, and final compositing.
- **vision_tools.py**  
  A helper module containing various CV utility functions (e.g., detection, segmentation, shadow generation) that integrate with VisionAgent.

---

## Setup & Installation

1. **Clone** this repository:
   ```bash
   git clone https://github.com/your-username/visionagent-natshadows.git
   cd visionagent-natshadows
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   This installs `streamlit`, `opencv-python`, `pillow`, `rembg`, `requests`, `langchain`, `anthropic`, etc.

3. **Install or reference** [VisionAgent](https://github.com/your-org/vision-agent) if it’s not already installed.  
   Make sure you can `import vision_agent as va` successfully.

---

## Environment Variables

This application uses the **OpenAI API** for:
- Generating short textual prompts/captions (marketing copy or background ranking).
- Optionally generating background images (`generate_background_images`).

You must set `OPENAI_API_KEY` in your environment, for example:
```bash
export OPENAI_API_KEY="sk-xxxxxxxxxxxxxxxxxx"
```

Or place it in a `.env` file that `dotenv` can load.

---

## Running the App

After installing all dependencies and setting your environment variables, run:

```bash
streamlit run app.py
```

- Open your web browser to the URL displayed in the console (typically `http://localhost:8501`).
- You’ll see the **Streamlit** interface for uploading images or choosing a sample.

---

## Usage Walkthrough

1. **Upload an Image**  
   - On the left sidebar, click **“Browse files”** to upload your own product/photo.  
   - Or pick one of the **sample images** from the dropdown.

2. **Click “Start Action”**  
   - The app will run **object detection** to find the main object in your image.  
   - It will then **segment** the object, removing background clutter.

3. **Preview Enhanced Foreground**  
   - The app applies a simple auto-enhancement process (brightness, contrast, slight unsharp mask).

4. **Choose or Generate Backgrounds**  
   - By default, the app attempts to **select** the best backgrounds from the `backgrounds/` folder via an OpenAI call.  
   - You can also switch to generating brand-new backgrounds if you uncomment the relevant code (`generate_background_images`).

5. **Composite & Select Final**  
   - Three composited previews are shown, each with a synthetic shadow behind the foreground.  
   - Select your favorite background from the dropdown.  
   - A final “Buy Now!” or “About this item” **marketing caption** is displayed if the prompt generation was successful.

6. **Save / Download**  
   - You can right-click on any composite preview to save the final image.

---

## Customization Tips

- **Background Generation**:  
  - To generate backgrounds using OpenAI’s image-generation API, uncomment the code in `app.py` where `generate_background_images` is called (instead of `select_best_background_openai`).
- **Shadow Parameters** (`create_composite_with_shadow`):  
  - Tweak `shadow_offset`, `shadow_blur`, and `shadow_opacity` to get different shadow intensities or directions.
- **Segmentation Logic** (`segment_and_save_objects`):  
  - Adjust the prompt or boundary checks to handle multiple objects or different thresholds.
- **Captions**:  
  - Switch from **QWen** to **Claude** or **GPT** in `detect_objects` if desired, or refine the prompt strings for more specialized marketing copy.

---

For any questions or issues, please open an [Issue]([https://github.com/your-username/visionagent-natshadows/issues](https://github.com/ankit1khare/VisionAgent-Background-Enhancer/issues)).
