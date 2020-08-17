import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from cv2 import cv2
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
# import tensorflow


app = Flask(__name__)

model = load_model("model_digit.h5")
print("Loaded model from disk")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    # int_features = [int(x) for x in request.form.values()]
    # final_features = [np.array(int_features)]
	# file = request.files['image']
	# if not file: return render_template('index.html', label="No file")
    file = request.files['image']
    # image1 = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    image1 = plt.imread(file)
    image1 = image1[:,:,0]

    image1 = cv2.resize(image1, (28, 28))
    image1 = np.invert(image1)



    image1 = image1.astype('float32')
    image1 = image1.reshape(1, 28, 28, 1)
    image1 /= 255

    prediction = model.predict(image1)

    output = np.argmax(prediction)

    return render_template('index.html', prediction_text='The entered number is: {}'.format(output))

# @app.route('/predict_api',methods=['POST'])
# def predict_api():
#     '''
#     For direct API calls trought request
#     '''
#     data = request.get_json(force=True)
#     prediction = model.predict([np.array(list(data.values()))])

#     output = prediction[0]
#     return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)