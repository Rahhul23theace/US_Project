





from flask import Flask, request, redirect, url_for, render_template, flash
from werkzeug.utils import secure_filename
from flask import send_from_directory
from builtins import zip
import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from util import load_models, ensemble_prediction, load_image_from_file
from genAI_feedback import generate_feedback





app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png', 'gif'}
app.jinja_env.globals.update(zip=zip)

app.secret_key = 'super secret key'
app.config['SESSION_TYPE'] = 'filesystem'





def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']





@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'files[]' not in request.files:
            flash('No file part')
            return redirect(request.url)
        files = request.files.getlist('files[]')
        if not files or files[0].filename == '':
            flash('No selected file')
            return redirect(request.url)

        name = request.form.get('name')
        scholar_id = request.form.get('scholar_id')

        filenames = []
        predicted_classes = []



        for i, file in enumerate(files):
            if file and allowed_file(file.filename):
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(file_path)


                data_transforms = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                ])
                image = load_image_from_file(file, transform=data_transforms)

                model_1, model_2 = load_models("resnet50.pt", "resnext50_32x4d.pt")
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                inputs = image.unsqueeze(0).to(device)
                outputs = ensemble_prediction(model_1, model_2, inputs)

                class_labels = ['Artefact', 'Incorrect_Gain', 'Incorrect_Position', 'Optimal', 'Wrong']

                predicted_labels = (outputs > 0.5).squeeze().detach().cpu().numpy().astype(int)
                if sum(predicted_labels) == 0:
                    top_prob_index = outputs.squeeze().detach().cpu().numpy().argmax()
                    predicted_label = f"{class_labels[top_prob_index]}"
                else:
                    predicted_label = ", ".join([class_labels[i] for i in range(len(predicted_labels)) if predicted_labels[i] == 1])

                predicted_classes.append(predicted_label)

                predicted_label = predicted_label.replace(" ", "_")
                filename = secure_filename(f"{name}_{scholar_id}_{predicted_label}_{file.filename}")
                os.rename(file_path, os.path.join(app.config['UPLOAD_FOLDER'], filename))
                filenames.append(filename)

        feedback_messages = generate_feedback(predicted_classes)
        file_urls = [url_for('uploaded_file', filename=f) for f in filenames]

        return render_template('index.html', predicted_classes=predicted_classes, feedback_messages=feedback_messages, file_urls=file_urls, feedback_data=zip(files, feedback_messages))

    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)



if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000)





""" 














import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from flask import Flask, request, redirect, url_for, render_template, flash
from werkzeug.utils import secure_filename
from flask import send_from_directory
from builtins import zip


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
            if "Optimal" in label:
                feedback_for_prediction.append("You are doing good!")
            else:
                feedback_for_prediction.append("You need some improvements.")
        feedback.append(feedback_for_prediction)
    return feedback


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png', 'gif'}
app.jinja_env.globals.update(zip=zip)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET', 'POST'])
def upload_file():

    if request.method == 'POST':
        if 'files[]' not in request.files:
            flash('No file part')
            return redirect(request.url)
        files = request.files.getlist('files[]')
        if not files or files[0].filename == '':
            flash('No selected file')
            return redirect(request.url)

        name = request.form.get('name')
        scholar_id = request.form.get('scholar_id')

        filenames = []
        predicted_classes = []

        for i, file in enumerate(files):
            if file and allowed_file(file.filename):
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(file_path)

                data_transforms = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                ])
                image = load_image_from_file(file, transform=data_transforms)

                model_1, model_2 = load_models("E:/Kidney_US/US_data/flask/resnet50.pt", "E:/Kidney_US/US_data/flask/resnext50_32x4d.pt")
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                inputs = image.unsqueeze(0).to(device)
                outputs = ensemble_prediction(model_1, model_2, inputs)

                class_labels = ['Artefact', 'Incorrect_Gain', 'Incorrect_Position', 'Optimal', 'Wrong']

                predicted_labels = (outputs > 0.5).squeeze().detach().cpu().numpy().astype(int)

                if sum(predicted_labels) == 0:
                    top_two_prob_indices = outputs.squeeze().detach().cpu().numpy().argsort()[-2:][::-1]
                    predicted_label = f"Top 2 {class_labels[top_two_prob_indices[0]]} {round(outputs.squeeze()[top_two_prob_indices[0]].item(), 3)} {class_labels[top_two_prob_indices[1]]} {round(outputs.squeeze()[top_two_prob_indices[1]].item(), 3)}"
                else:
                    predicted_label = ", ".join([class_labels[i] for i in range(len(predicted_labels)) if predicted_labels[i] == 1])
                    predicted_classes.append(predicted_label)

                predicted_label = predicted_label.replace(" ", "_")
                filename = secure_filename(f"{name}_{scholar_id}_{predicted_label}_{file.filename}")
                os.rename(file_path, os.path.join(app.config['UPLOAD_FOLDER'], filename))
                filenames.append(filename)

        feedback_messages = generate_feedback(predicted_classes)
        file_urls = [url_for('uploaded_file', filename=f) for f in filenames]

        return render_template('index.html', predicted_classes=predicted_classes, feedback_messages=feedback_messages, file_urls=file_urls, feedback_data=zip(files, feedback_messages))
 
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.secret_key = 'super secret key'
    app.config['SESSION_TYPE'] = 'filesystem'
    app.run(debug=True)


 """