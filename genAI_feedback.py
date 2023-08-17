


import openai
import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())


openai.api_type = os.getenv('api_type')
openai.api_base = os.getenv('api_base')
openai.api_version = os.getenv('api_version')
openai.api_key  = os.getenv('OPENAI_API_KEY')



def generate_feedback(predicted_classes):
    feedback = []

    label_information = '''


1. Artefact: This label refers to ultrasound images where unwanted artifacts are present, affecting the overall image quality.
 Artifacts can result from various factors, such as acoustic shadows, reverberations, or other distortions. In the context of kidney ultrasound images, these artifacts may hinder proper assessment of the kidney's structure and function.

 
2. Incorrect Gain: The "Incorrect_Gain" label is assigned to ultrasound images where the gain settings are not properly adjusted.
 Gain refers to the amplification of the returning ultrasound signals, which affects the brightness of the image. When the gain is either too high or too low, it can lead to over- or under-exposure of the image, making it difficult to accurately evaluate the kidney's condition.

3. Incorrect Position: Images labeled as "Incorrect_Position" are those where the ultrasound probe or patient's positioning is not 
optimal for capturing a clear and accurate image of the kidney. Proper positioning is crucial for obtaining high-quality ultrasound images, as it ensures that the ultrasound beam is appropriately aligned with the target anatomy. Inaccurate positioning may result in poor visualization of the kidney or the inclusion of irrelevant structures in the image.


4. Optimal: The "Optimal" label is given to ultrasound images that exhibit high quality and are free from any significant artifacts, 
positioning errors, or gain issues. These images provide a clear and accurate representation of the kidney, allowing for precise assessment of its structure and function. Such images are ideal for medical professionals to use when evaluating a patient's kidney health.

5. Wrong: The "Wrong" label is assigned to ultrasound images where the captured anatomy is not the intended kidney or the image
 is of an entirely different organ or body part. This may occur if the ultrasound probe is placed incorrectly or if the operator is not familiar with the correct scanning technique for kidney imaging. Images with this label are not suitable for kidney quality evaluation, as they do not accurately represent the target organ.

'''

    for prediction in predicted_classes:
        feedback_for_prediction = []
        labels = prediction.split(", ")

        prompt = f"{label_information}\nBased on the above information, please provide feedback for an ultrasound image with the following characteristics: "
        prompt += ", ".join(labels)

        response = openai.Completion.create(
            engine="chatgpt-4",
            prompt=prompt,
            temperature=0.7,
            max_tokens=100,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            n=1,
            stop=None,
        )

        feedback_text = response.choices[0].text.strip()
        feedback_for_prediction.append(feedback_text)
        feedback.append(feedback_for_prediction)

    return feedback







predicted_classes = ["Artefact", "Incorrect_Gain", "Optimal"]
feedback = generate_feedback(predicted_classes)

for i, fb in enumerate(feedback):
    print(f"Feedback for prediction {i + 1}:")
    print(fb[0])
    print()