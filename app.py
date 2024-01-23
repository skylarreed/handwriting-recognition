import os

import base64
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, flash, redirect, url_for, render_template
from keras import backend as K
from keras.optimizers import Adam

UPLOAD_FOLDER = 'uploads/'
model = tf.keras.models.load_model('model')

model.summary()
model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])
print(model.get_weights())


def preprocess(img):
    (h, w) = img.shape

    final_img = np.ones([64, 256]) * 255  # blank white image

    # crop
    if w > 256:
        img = img[:, :256]

    if h > 64:
        img = img[:64, :]

    final_img[:h, :w] = img
    return cv2.rotate(final_img, cv2.ROTATE_90_CLOCKWISE)


alphabet = u"ABCDEFGHIJKLMNOPQRSTUVWXYZ-' "
max_str_len = 24  # max length of input labels
num_of_characters = len(alphabet) + 1  # +1 for ctc pseudo blank
num_of_timestamps = 64  # max length of predicted labels


def num_to_label(num):
    ret = ""
    for ch in num:
        if ch == -1:  # CTC Blank
            break
        else:
            ret += alphabet[ch]
    return ret


def make_prediction(img):
    img_predictions = []

    image = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    image = preprocess(image)
    image = image / 255
    img_predictions.append(image)
    img_predictions = np.array(img_predictions).reshape(-1, 256, 64, 1)

    prediction = model.predict(img_predictions)

    out = K.get_value(K.ctc_decode(prediction, input_length=np.ones(prediction.shape[0]) * prediction.shape[1],
                                   greedy=True)[0][0])

    return num_to_label(out[0])


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        print(request.files)
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']

        print(file)
        print(file.filename)
        if file.filename == '':
            flash('No file selected')
            return redirect(request.url)

        if file:
            print("here")
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
            return redirect(f'/predict/{file.filename}')
    return redirect(url_for('index'))


@app.route('/')
def index():  # put application's code here
    return render_template('index.html')


@app.route('/predict/<filename>', methods=['GET'])
def predict(filename):
    # Build the full file path
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    # Read the image file
    image_data = open(file_path, 'rb').read()

    # Encode image data to base64
    encoded_image_data = base64.b64encode(image_data).decode('utf-8')

    # Make predictions using HandwritingRecognizer
    predicted_text = make_prediction(file_path)

    # Pass encoded image data and prediction to the template
    return render_template('predict.html', prediction=predicted_text, encoded_image_data=encoded_image_data)


if __name__ == '__main__':
    app.run()
