import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from cv2 import cv2
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import imutils
from PIL import Image
import tensorflow as tf

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
    user_test = image1
    col = Image.open(file)
    gray = col.convert('L')
    bw = gray.point(lambda x: 0 if x<150 else 255, '1')
    bw.save("bw_image.jpg")
    # bw
    img_array = cv2.imread("bw_image.jpg", cv2.IMREAD_GRAYSCALE)
    img_array = cv2.bitwise_not(img_array)
    # print(img_array.size)
    # plt.imshow(img_array, cmap = plt.cm.binary)
    # plt.show()
    img_size = 28
    new_array = cv2.resize(img_array, (img_size,img_size), interpolation=0)
    # plt.imshow(new_array, cmap = plt.cm.binary)
    # plt.show()
    user_test = tf.keras.utils.normalize(new_array, axis = 1)
    user_test = user_test.reshape(1, 28, 28, 1)
    image1=user_test
    # image1 = cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)
    # # image1 = cv2.fastNlMeansDenoising(image1, None,15,7,21)
    # image1 = cv2.GaussianBlur(image1, (5,5), 0)
    # # image1 = cv2.adaptiveThreshold(image1,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,17,5)
   
    # kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (2, 2))
    # image1 = cv2.morphologyEx(image1, cv2.MORPH_CLOSE, kernel)
    # # image1 = image1[:,:,0]

    # image1 = cv2.resize(image1, (28, 28), interpolation = cv2.INTER_AREA)
    # image1 = np.invert(image1)



    # image1 = image1.astype('float32')
    # image1 = image1.reshape(1, 28, 28, 1)
    # image1 /= 255

    prediction = model.predict(image1)

    output = np.argmax(prediction)

    return render_template('index.html', prediction_text='The entered number is: {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)