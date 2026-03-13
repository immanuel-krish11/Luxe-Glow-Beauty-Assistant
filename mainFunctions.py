# i am thinking that all the function calls will be here and i could convert it for API
# this module calls all the function for face analyzing and returns all results

from modelOutputs.skinType import predict_skin_type # to predict skin-type -> Dry/Normal/Oily 
from recommend.outfit import outfit_redommendation # to get 3 outfit recommendation 
from otherSkinPred import other_predictions # to get more analyzed results like acne, dark circles, eyebags, pigmentation, spots and wrinkles
from modelOutputs.faceDemographics import ageGenderRace # to predict age, gender and race
from recommend.prescribe import prescribe_remedy # to get homemade natural remedies
from concurrent.futures import ThreadPoolExecutor # to run all of them parallely
import json # to convert str object to dictionary

# https://aistudio.google.com/usage?timeRange=last-90-days -> to check the usage of the api


def begin_face_analyze(img_path, predictor_type, predictor_acne):

    # calling all functions for analyzing facial details
    with ThreadPoolExecutor() as executor:
        future_acne = executor.submit(predictor_acne.predict, img_path)
        future_age = executor.submit(ageGenderRace, img_path)
        future_type = executor.submit(predict_skin_type, predictor_type, img_path)
        future_other = executor.submit(other_predictions, img_path)

        prediction_acne, confidence_acne = future_acne.result()
        age, gender, race = future_age.result()
        prediction_type, confidence_type = future_type.result()
        prediction_other = future_other.result()
    print(confidence_acne)
    print(confidence_type)

    result = {
        "acne" : prediction_acne,
        "age" : age,
        "gender" : gender,
        "race" : race,
        "type" : prediction_type,
        "other": prediction_other,
    }
    return result



def recommend_outfit(seg_model, seg_processor, img_path):

    outfits_name, outfits_hex = outfit_redommendation(seg_model, seg_processor, img_path)

    return outfits_name, outfits_hex

def prescription(diagnosis):
    result_text = ""
    try: 
        response = prescribe_remedy(diagnosis)
        response = json.loads(response)
        print("response type - ", type(response))
        for i, value in enumerate(response.values(), start=1):
            result_text += f"{i}. {value}\n"
    except Exception as e:
        result_text = "This process is not available at this moment! Sorry for the inconvenience caused..."
        print("Error for prescription block : ", e)
    return result_text

