from flask import Flask, render_template, request, send_from_directory
import os
import torch
from werkzeug.utils import secure_filename
from modelOutputs.skinType import  load_trained_model# to load the skin type model
from modelOutputs.skinAcne import AcneSeverityPredictor # to load the acne model
from mainFunctions import begin_face_analyze, recommend_outfit, prescription # for inference
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

import time # loading and inference




# mandatory lines of code
# ----------------------------------------------IMPORTANT TO LOAD MODELS-------------------------------------------- 
# loading models for face analyzing
SAVED_ACNE_MODEL_PATH = "models/acne_resnet50_best.pth"
SAVED_TYPE_MODEL_PATH = "models/skin_type_efficientnetb0_acc88.pth"

predictor_acne = AcneSeverityPredictor(model_path=SAVED_ACNE_MODEL_PATH)
predictor_type = load_trained_model(SAVED_TYPE_MODEL_PATH)


# loading model for skin tone and outfit recommendation part
device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
model_name_skinTone = "jonathandinu/face-parsing"
seg_processor_skinTone = SegformerImageProcessor.from_pretrained(model_name_skinTone)
seg_model_skin_tone = SegformerForSemanticSegmentation.from_pretrained(model_name_skinTone).to(device)
seg_model_skin_tone.eval()

# ------------------------------------------------------------------------------------------------------------------


app = Flask(__name__)

UPLOAD_FOLDER = "custom-images"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/menu")
def menu():
    return render_template("menu.html") #"Main Menu Page"

@app.route("/face-analysis")
def face():
    return render_template("analysis.html")

@app.route("/outfit")
def outfit():
    return render_template("outfit.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/results", methods=["POST"])
def results():

    if "image" not in request.files:
        return "No file uploaded"

    file = request.files["image"]

    if file.filename == "":
        return "No file selected"

    # filename = secure_filename(file.filename) # this is for original filename

    filename = str(int(time.time())) + "_" + secure_filename(file.filename)

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)

    file.save(filepath)

    # Later you can send filepath to your ML model
    try:

        analysis_result = begin_face_analyze(filepath, predictor_type= predictor_type, predictor_acne= predictor_acne)
        
        prescribe_result = prescription(analysis_result)

    except Exception as e:
        print("Error in face analysis : ", e)
        return render_template("error.html")
    
    return render_template("results.html",
                           acne = analysis_result["acne"],
                           age = analysis_result["age"],
                           gender = analysis_result["gender"],
                           race = analysis_result["race"],
                           type = analysis_result["type"],
                           other = analysis_result["other"],
                           prescribe = prescribe_result,
                           image_filename = filename
                           )


@app.route("/outfitresult", methods=["POST"])
def outfitresult():

    if "image" not in request.files:
        return "No file uploaded"

    file = request.files["image"]

    if file.filename == "":
        return "No file selected"

    # filename = secure_filename(file.filename) # this is for original filename

    filename = str(int(time.time())) + "_" + secure_filename(file.filename)

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)

    file.save(filepath)
    
    try: 
        # result is in this format - [[], [], []]
        outfits_name, outfits_hex = recommend_outfit(seg_model_skin_tone, seg_processor_skinTone, filepath)

    except Exception as e:
        print("Error in Outfit Recommendation : ", e)
        return render_template("error.html")

    return render_template("outfitresult.html",
                           image_filename = filename,
                           outfits_name = outfits_name,
                           outfits_hex = outfits_hex # result is in this format - [[], [], []]
                           )

if __name__ == "__main__":
    app.run(debug=False) # debug = False when  actually using!!!
