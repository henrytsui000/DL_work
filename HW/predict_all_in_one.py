def gogo():
    import tensorflow as tf
    import os
    import numpy as np
    from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

    #set data_path
    data_path = "D:\machine-learning\car_bybicle"

    #set image path
    img_path = 'D:\DL_work\database\pic_num.jpg'

    #讀取影像並縮放為模型input的大小
    img = tf.keras.preprocessing.image.load_img( img_path, target_size=(224, 224) )
    img = tf.keras.preprocessing.image.img_to_array(img)


    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    # img = np.expand_dims(img, axis=0)
    img = np.array([img])


    #把影像進行對應的前處理
    img = preprocess_input(img)

    #讀取模型
    model_dir = os.path.join(data_path, 'model-logs')

    model_path = model_dir + '/{}-last-model.h5'.format('baic_model')

    model = tf.keras.models.load_model("D:\machine-learning\car_bybicle\\baic_model-last-model.h5")

    #進行預測
    y_pred = model.predict(img)
    y_pred_class = y_pred.argmax(-1)
    class_label = {0 : 'Motorcycle', 1 : 'car', 2 : 'Bicycle'}

    print('模型的預測結果為{0}, 類別為{1}'.format(y_pred, class_label[y_pred_class[0]]))
    return class_label[y_pred_class[0]]