



import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image




# Load pre-trained models
def load_models(model_path_1, model_path_2):
    model_1 = models.resnet50(pretrained=False)
    num_ftrs = model_1.fc.in_features
    model_1.fc = nn.Sequential(
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 5),
        nn.Sigmoid()
    )

    model_2 = models.resnext50_32x4d(pretrained=False)
    num_ftrs = model_2.fc.in_features
    model_2.fc = nn.Sequential(
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 5),
        nn.Sigmoid()
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    model_1 = model_1.to(device)
    model_1.load_state_dict(torch.load(model_path_1))
    model_1.eval()

    model_2 = model_2.to(device)
    model_2.load_state_dict(torch.load(model_path_2))
    model_2.eval()

    return model_1, model_2

# Ensemble prediction function
def ensemble_prediction(model_1, model_2, inputs):
    outputs_1 = model_1(inputs)
    outputs_2 = model_2(inputs)
    ensemble_outputs = (outputs_1 + outputs_2) / 2
    return ensemble_outputs






# Load image from file
def load_image_from_file(file, transform=None):
    image = Image.open(file).convert('RGB')
    if transform:
        image = transform(image)
    return image




def generate_feedback(predicted_classes):
    feedback = []
    for prediction in predicted_classes:
        feedback_for_prediction = []
        labels = prediction.split(", ")
        for label in labels:
            if label == "Artefact":
                feedback_for_prediction.append("There are unwanted artifacts in the image. Please check the probe placement and scanning technique to minimize artifacts.")
            elif label == "Incorrect_Gain":
                feedback_for_prediction.append("The gain settings are not properly adjusted. Please adjust the gain to ensure the image is neither over- nor under-exposed.")
            elif label == "Incorrect_Position":
                feedback_for_prediction.append("The probe or patient's positioning is not optimal. Make sure to correctly position the probe for a clear and accurate image of the kidney.")
            elif label == "Optimal":
                feedback_for_prediction.append("You are doing good! The image quality is optimal.")
            elif label == "Wrong":
                feedback_for_prediction.append("The captured image is not of the intended kidney or an entirely different organ. Please ensure the correct scanning technique and probe placement.")
            else:
                feedback_for_prediction.append("I am sorry, something went Wrong")
        feedback.append(feedback_for_prediction)
    return feedback








""" 
def generate_feedback(predicted_classes):
    feedback = []
    for prediction in predicted_classes:
        feedback_for_prediction = []
        labels = prediction.split(", ")
        for label in labels:
            if "Optimal" in label:
                feedback_for_prediction.append("You are doing good!")
            else:
                feedback_for_prediction.append("You need some improvements.")
        feedback.append(feedback_for_prediction)
    return feedback
 """


