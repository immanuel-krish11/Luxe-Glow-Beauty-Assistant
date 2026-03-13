# Skin Care and Outfit Recommendation App 🌟

A comprehensive, AI-powered Flask web application that provides in-depth facial skin analysis, natural remedy prescriptions, and personalized outfit color recommendations based on your unique skin tone and features.

---

## 🚀 Features

- **Skin Type Detection:** Uses a custom trained EfficientNet-B0 model to accurately classify skin type (`dry`, `normal`, or `oily`).
- **Acne Severity Prediction:** Uses a ResNet-50 model to evaluate the severity of acne from the uploaded image.
- **Demographics Analysis:** Leverages `DeepFace` to estimate age, gender, and race.
- **Advanced Skin Metrics:** Integrates with the **Roboflow API**(because of unavailability of datasets) to detect dark circles, eyebags, pigmentation, spots, and wrinkles.
- **Natural Remedy Prescriptions:** Utilizes the **Google Gemini Pro** (`gemini-3-flash-preview`) API to prescribe 5 highly tailored, homemade, and natural skincare remedies based on the complete multi-model facial diagnosis.
- **Outfit Recommendation Pipeline:** Uses the `Segformer` (face-parsing) model to determine your Monk Skin Tone (MST 1-10) and suggests 3 harmonious top and bottom wear color combinations (with hex codes) tailored to your complexion.
- **Blazing Fast Concurrency:** Leverages Python's `ThreadPoolExecutor` to run all heavy machine learning models concurrently, ensuring low latency and a smooth user experience.

---

## 🎥 Demo Video

> **Note:** Watch the demo video below to see the complete pipeline in action—from uploading a face image to getting the dynamic skin analysis and custom outfit recommendations.

<!-- 
🚨 IMPORTANT: Replace the link below with the actual URL/path to your demo video! 
You can use an MP4 file path or a YouTube/Vimeo embed link.
-->

[![Demo Video Placeholder](https://img.youtube.com/vi/YOUR_VIDEO_ID/0.jpg)](https://www.youtube.com/watch?v=YOUR_VIDEO_ID)

*(If you have a local video file, you can embed it like this: `<video src="static/demo_video.mp4" controls="controls" style="max-width: 100%;">Your browser does not support the video tag.</video>`)*

---

## 🛠️ Technology Stack

- **Backend Framework:** Flask
- **Machine Learning & AI:** 
  - PyTorch (for custom `EfficientNet` and `ResNet` models)
  - HuggingFace Transformers (`Segformer` for face-parsing)
  - DeepFace (for demographics)
- **External APIs:**
  - Google GenAI / Gemini (for natural remedy extraction)
  - Roboflow Inference API (for advanced skin condition mapping)
- **Frontend:** HTML, CSS, JavaScript (Jinja2 Templates)

---

## 📂 Project Structure

```text
skin-care-mk2/
│
├── app.py                      # Main entry point for the Flask web server and routing
├── mainFunctions.py            # Async executor to handle parallel model inferences
├── demo.py                     # Experimental playground script
├── otherSkinPred.py            # Roboflow API integration for eyebags, dark circles, etc.
│
├── models/                     # Custom trained PyTorch models (.pth weights)
│   ├── acne_resnet50_best.pth
│   └── skin_type_efficientnetb0_acc88.pth
│
├── modelOutputs/               # Local ML inference scripts
│   ├── faceDemographics.py     # Wraps DeepFace for Age/Gender/Race
│   ├── skinAcne.py             # Inference script for Acne Severity
│   ├── skinToneMonk.py         # Extracts Monk Skin Tone using Segformer
│   └── skinType.py             # Instantiates and runs EfficientNet for skin type
│
├── recommend/                  # Recommendation logic
│   ├── outfit.py               # Generates outfit colors based on MST
│   └── prescribe.py            # Calls Gemini to generate natural remedies
│   └── outfitSource/           # CSV databases containing hex color rules mapped to MST
│
├── custom-images/              # Secure directory for handling user uploads temporarily
├── static/                     # CSS, JS, and static assets
├── templates/                  # HTML templates for the UI pages (index, results, outfits, etc.)
└── .env                        # Environment variables (Google API keys, etc.)
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository
Ensure you are in the project root directory (`skin-care-mk2`).

### 2. Create and Activate a Virtual Environment
It is highly recommended to use a virtual environment due to the heavy ML dependencies.
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
Make sure you install the required packages. *(You may need to install specific PyTorch binaries matching your system architecture).*
```bash
pip install flask werkzeug torch torchvision pandas numpy deepface transformers inference_sdk google-genai python-dotenv
```

### 4. Environment Variables
Create a `.env` file in the root directory and add your API keys:
```env
GEMINI_API_KEY=your_google_gemini_api_key_here
ROBOFLOW_API_KEY=your_roboflow_api_key_here
```
> *Note: The Roboflow API key is currently hardcoded in `otherSkinPred.py`. For security, consider moving it to the `.env` file!*

---

## 🚀 How to Run

1. Simply run the flask application from your terminal:
   ```bash
   python app.py
   ```
2. Open your web browser and navigate to:
   ```
   http://127.0.0.1:5000
   ```
3. Upload an image of a face and explore the generated diagnostics and recommendations!

---

## 🧠 ML Inference Pipeline Under The Hood

When an image is uploaded via the `/results` endpoint:
1. The image is securely saved to `custom-images/`.
2. `mainFunctions.py` fires off `ThreadPoolExecutor`.
3. In **parallel**: 
   - `predict_skin_type` classifies the skin baseline.
   - `AcneSeverityPredictor` evaluates acne stages.
   - `ageGenderRace` runs DeepFace analytics.
   - `other_predictions` hits the Roboflow endpoint.
4. The aggregated JSON diagnosis is passed to `prescribe_remedy`.
5. The Gemini API analyzes the technical JSON string and generates 5 natural, tailored home remedies.
6. The compiled context is passed to `results.html` for rendering.

For outfit recommendations, `/outfitresult` determines the user's specific **Monk Skin Tone (MST 1-10)** using a semantic segmentation face-parser, and matches it against curated `.csv` databases to provide stunning clothing color combinations.
