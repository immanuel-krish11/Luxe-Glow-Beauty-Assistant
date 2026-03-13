# import this file and call the ageGenderRace() function to get AGE, GENDER AND RACE of a person.
# the function returns age, gender and race

# here, we use deepface library which comes with pre built weights to detect the demographics(but is somewhat heavy weight)


from deepface import DeepFace

def ageGenderRace(img_path):
        
    result = DeepFace.analyze(
        img_path = img_path,
        actions = ['age', 'gender', 'race'] # can also add emotion but there's no need for it
    )

    age = result[0]['age']
    gender = result[0]['dominant_gender']
    race = result[0]['dominant_race']

    return age, gender, race


# sample usage
# img_path = "custom-images/custom_image1.jpg"
# age, gender, race = ageGenderRace(img_path)

# print("age = ", age)
# print("gender = ", gender)
# print("race = ", race) 
