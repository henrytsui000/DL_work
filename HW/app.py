from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import os
import time
app = Flask(__name__)
import tensorflow as tf
import os
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

#set data_path
data_path = "D:\machine-learning\car_bybicle"

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

model_dir = os.path.join(data_path, 'model-logs')

model_path = model_dir + '/{}-last-model.h5'.format('baic_model')

model = tf.keras.models.load_model("D:\machine-learning\car_bybicle\\baic_model-last-model.h5")

def gogo():
    img_path = 'D:\DL_work\database\pic_num.jpg'
    img = tf.keras.preprocessing.image.load_img( img_path, target_size=(224, 224) )
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = np.array([img])
    img = preprocess_input(img)
    y_pred = model.predict(img)
    y_pred_class = y_pred.argmax(-1)
    class_label = {0 : 'Motorcycle', 1 : 'car', 2 : 'Bicycle'}
    print('模型的預測結果為{0}, 類別為{1}'.format(y_pred, class_label[y_pred_class[0]]))
    return class_label[y_pred_class[0]]

@app.route("/")
def index():
    return render_template('index.html')

PEOPLE_FOLDER = os.path.join('static', 'pic')
app.config['UPLOAD_FOLDER'] = PEOPLE_FOLDER
@app.route('/uploader', methods=['GET', 'POST'])
def uploader():
    if request.method == 'POST':
        input = request.files['file']
        input.save(os.path.join("D:\DL_work\database", secure_filename('pic_num.jpg')))
        s = gogo()
        full_filename = os.path.join(app.config['UPLOAD_FOLDER'], f'{s}.png')
        return render_template('ans.html', answer = s, fuck = full_filename)

@app.route('/goback')
def goback(): 
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

