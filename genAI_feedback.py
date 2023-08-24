







import openai
import os
import requests
import json
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())


openai.api_type = os.getenv('api_type')
openai.api_base = os.getenv('api_base')
openai.api_version = os.getenv('api_version')
openai.api_key = os.getenv('OPENAI_API_KEY')




def get_completion(prompt, engine="chatgpt"):
    messages = [{"role": "user", "content": prompt}]

    headers = {
        "Content-Type": "application/json",
        "api-key": openai.api_key,
        "Cache-Control": "no-cache"
    }

    payload = {
        "model": engine,
        "messages": messages,
        "max_tokens": 8000,
        "temperature": 0.7,
    }

    response = requests.post('https://api.hku.hk/openai/deployments/chatgpt/chat/completions?api-version=2023-03-15-preview', headers=headers, data=json.dumps(payload))
    response_data = response.json()
    return response_data['choices'][0]['message']['content']




def generate_feedback(predicted_classes):
    feedback = []
    label_information = '''

1. Artefact:  There are artefacts on the image. These could be due to acoustic shadowing
from the ribs (having the kidneys in the upper retroperitoneum) or ring-down 
artefacts from gas bubbles in the colon. If you are interested in reading more on 
ultrasound artefacts, please refer to explanation on Moodle MBBS Clinical Skills
Platform under Radiological, Ultrasound Section I 2.3. 
You can avoid these arfefacts from the ribs by asking the subject to take a deep 
inspiration, which will lower the diaphragm and hence the position of the kidney away 
from the ribcage, or by positioning the probe in between the ribs to avoid the artefact. 
To overcome ring-down artefacts, having the subject fasted will help or gently applying
pressure to displace the gas away from the area (but sometimes even with these efforts,
nothing work!). 

2. Incorrect Gain/Inappropriate gain:  The image is either too "bright" or too "dark". 
The Gain controls the brightness of the image (see Moodle MBBS Clinical Skills Platform 
under Radiological, Ultrasound Section I 3.5). Another good reference for explanation can 
be found on:

 https://123sonography.com/blog/ultrasound-101-part-5-gain-and-time-gain-compensation#:~:text=What%20is%20gain%3F,much%20each%20echo%20is%20amplified.

3. Incorrect Position/ Suboptimal positioning: The organ of interest, i.e. kidney, 
is not correctly positioned in the captured imaged. It is not centrally placed or 
incompletely imaged (cut-off, cropped). Having the subject in decubitus position helps 
to get good access to image the kidney as explained in the video on Moodle MBBS Clinical
Skills Platform under Radiological, Ultrasound Section IIC 7.1. If this is not possible, 
then try positioning the probe anteriorly and posteriorly to find an ideal view of the
kidney maybe required as every subject is different. 

4. Optimal: Well done/good work for obtaining optimal image quality of the kidney. 

5. Wrong: The image acquired is of the wrong organ. Please refer to Moodle MBBS Clinical 
Skills Platform under Radiological, Ultrasound Section IIC 7.1, where there is a video
demonstrating how to perform and acquire image of the kidney using ultrasound:

https://moodle.hku.hk/mod/page/view.php?id=2748898

'''
    for prediction in predicted_classes:
        feedback_for_prediction = []
        labels = prediction.split(", ")
        prompt = f"{label_information}\n\nBased on the above information, please provide feedback for an ultrasound image with the following characteristics: {', '.join(labels)}.Also check if url link is provided for particular.If link is there, provide the link along with generated for that particular label.\n Please provide a creative response while maintaining an academic tone and factual information."

        response_text = get_completion(prompt)
        feedback_text = response_text.strip()
        feedback_for_prediction.append(feedback_text)
        feedback.append(feedback_for_prediction)

    return feedback



############################ EXAMPLE    #############################


""" 

predicted_classes = ["Artefact", "Incorrect Gain", "Optimal"]
feedback = generate_feedback(predicted_classes)

for i, fb in enumerate(feedback):
    print(f"Feedback for prediction {i + 1}:")
    print(fb[0])
    print()
 """





