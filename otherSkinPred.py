# this will predict things like - acne, dark circles, eyebags, pigmentation, spots and wrinkles
# this performs an api call to roboflow which give different outputs; out of which we need only class predicted and confidence
# link to the roboflow -> https://universe.roboflow.com/buyume-ahuro/dark_circle/model/1


from inference_sdk import InferenceHTTPClient

def other_predictions(img_path):
    CLIENT = InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key="GkIOUKGcKbmZvh8O8Ml0"
    )
    result = CLIENT.infer(img_path, model_id="dark_circle/1")
    # conf = round(result['predictions'][0]["confidence"] * 100, 2)
    # predicted_class = result['predictions'][0]["class"]
    # return predicted_class, conf
    predictions = {}
    for i in result['predictions']:
        conf = round(i["confidence"] * 100, 2)
        predictions[i["class"]] = conf
    return predictions


# sample usage
# img_path = "/Users/krishprakash/Desktop/skin-care-mk2/custom-images/custom_image1.jpg"
# # print(result)
# # predicted_class, conf = other_predictions(img_path)
# # print(f"Predicted Class: {predicted_class} - Confidence: {conf} ")
# print(other_predictions(img_path))
