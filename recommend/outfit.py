# this module will take img-path and will return some recommendations of colours
# the colours will be for outfit -> outfit recommendation

# the source contains data in this format -
# color_name_for_top_wear, hexadecimal_for_top, color_name_for_bottom_wear, hexadecimal_for_bottom


import csv
import random
from modelOutputs.skinToneMonk import predict_skin_tone # to predict skin-tone -> MST1 - MST10


def fetch_outfit(source_file):
    with open(source_file, newline='') as csvfile:
        reader = list(csv.reader(csvfile))  # read all rows as lists

    # Pick any 3 random rows
    outfits = random.sample(reader, 3)
    outfits_name = []
    outfits_hex = []

    # Result is in the format [[], [], []]

    for i in outfits:
        outfits_hex.append([i[1], i[3]])
        outfits_name.append([i[0], i[2]])

    return outfits_name, outfits_hex #returns the name list and the hexadecimal of those colours list


def outfit_redommendation(seg_model, seg_processor, img_path):
    monk_label, confidence = predict_skin_tone(seg_model, seg_processor, img_path)
    monk_label = monk_label.split()
    monk_label = monk_label[-1] # seperating the tone with metadata: converting Monk Skin Tone (MST) 10 -> 10

    outfit_source = f"recommend/outfitSource/MST{monk_label}.csv"
    
    outfits_name, outfits_hex = fetch_outfit(outfit_source)

    return outfits_name, outfits_hex 
