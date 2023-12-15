import os
import numpy as np
import matplotlib.pyplot as plt

from keras.applications.vgg16 import VGG16
from keras.applications.resnet import ResNet50
from keras.models import Model
from keras.models import model_from_json
from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
import keras.losses


import pandas as pd
import cv2
from glob import glob
from PIL import Image
import time

from threading import Thread
import json
import math

motion_kind:list
with open('motion.json','r') as f:
    motion_kind=list(json.load(f).keys())
#motion_kind=['arm_left','arm_straight','arm_up','run','walk']
models:dict={}
historys:dict={}

test_idx=11
scale_num=12

def resize_image(image_matrix):
    output_image=np.resize(image_matrix,(16*scale_num,3*scale_num,3))

    return output_image


def read_scale_dataset():
    train_data = pd.read_csv("motion_train.csv")
    test_data = pd.read_csv("motion_test.csv")
    
    # visualizing class label frequency in the input data
    train_data['label'].value_counts().plot.bar(color='cyan')
    X_train = train_data.drop('label', axis=1).values
    
    X_train_arm=X_train.T[0:144].T
    X_train_leg=X_train.T[144:288].T
    
   
    X_train_arm = X_train_arm.reshape((X_train_arm.shape[0], 16, 3, 3))
    temp_new=[]
    for x in X_train_arm:
        x=resize_image(x)
        temp_new.append(x)
        
    X_train_arm=np.array(temp_new)
        
    X_train_arm = X_train_arm.astype('float32')
    X_train_arm = X_train_arm/255.0
    
    X_train_leg = X_train_leg.reshape((X_train_leg.shape[0], 16, 3, 3))   

    temp_new=[]
    for x in X_train_leg:
        x=resize_image(x)
        temp_new.append(x)
        
    X_train_leg=np.array(temp_new)


    X_train_leg = X_train_leg.astype('float32')
    X_train_leg = X_train_leg/255.0
    
    
    Y_train = train_data['label'].values
    
    
    X_test = test_data.drop('label', axis=1).values
    
    X_test_arm=X_test.T[0:144].T
    X_test_leg=X_test.T[144:288].T

    
    X_test_arm = X_test_arm.reshape((X_test_arm.shape[0], 16, 3, 3))

    temp_new=[]
    for x in X_test_arm:
        x=resize_image(x)
        temp_new.append(x)
        
    X_test_arm=np.array(temp_new)



    X_test_arm = X_test_arm.astype('float32')
    X_test_arm = X_test_arm/255.0
    
    X_test_leg = X_test_leg.reshape((X_test_leg.shape[0], 16, 3, 3))


    temp_new=[]
    for x in X_test_leg:
        x=resize_image(x)
        temp_new.append(x)
        
    X_test_leg=np.array(temp_new)

    X_test_leg = X_test_leg.astype('float32')
    X_test_leg = X_test_leg/255.0
    
    
    
    Y_test = test_data['label'].values
    
    return X_train_arm,X_train_leg, Y_train, X_test_arm,X_test_leg, Y_test


def load_model(include_top=True):
    model=VGG16(input_shape=(16*scale_num,3*scale_num,3),weights=None,include_top=include_top)
    return model

def get_img_size_model(model):
    """Returns image size for image processing to be used in the model
    Args:
        model: Keras model instance 
    Returns:
        img_size_model: Tuple of integers, image size
    """
    img_size_model = (16, 3)
        
    return img_size_model

def get_layername_feature_extraction(model):
    """ Return the name of last layer for feature extraction   
    Args:
        model: Keras model instance
    Returns:
        layername_feature_extraction: String, name of the layer for feature extraction
    """
    layername_feature_extraction = 'fc2'
    
    return layername_feature_extraction

def get_layers_list(model):
    """Get a list of layers from a model
    Args:
        model: Keras model instance
    Returns:
        layers_list: List of string of layername
    """
    layers_list = []
    for i in range(len(model.layers)):
        layer = model.layers[i]        
        layers_list.append(layer.name)
        
    return layers_list

def get_feature_vector(model):
    """ Get a feature vector extraction from an image by using a keras model instance
    Args:
        model: Keras model instance used to do the classification.
        img_path: String to the image path which will be predicted
    Returns:
        feature_vect: List of visual feature from the input image
    """
    
    # Creation of a new keras model instance without the last layer
    layername_feature_extraction = get_layername_feature_extraction(model)
    
    model_feature_vect = Model(inputs=model.input, outputs=model.get_layer(layername_feature_extraction).output)
    
    model_feature_vect.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model_feature_vect


#팔 다리 총 두번 실행하도록
def evaluate_model(X_train, Y_train, X_test, Y_test):
    # model fitting
    model = get_feature_vector(load_model())
    
    history = model.fit(X_train, Y_train, epochs=100, batch_size=14)
    print('\n\n')
    model.summary()
    print('\n\n')
    model.evaluate(X_test, Y_test)
    
    return model, history


def runLearn(X_train,Y_train,X_test,Y_test,part):
    model, history = evaluate_model(X_train, Y_train, X_test, Y_test)
    models[part]=model
    historys[part]=history
    
def predict(model,history,X_test,Y_test,part):
    Y_pred = model.predict(X_test)
    print(part)
    for i in range(len(Y_test)):
        print("\n",i,"\nActual label: ", motion_kind[Y_test[i]])
        print("Predicted label: ", motion_kind[np.argmax(Y_pred[i])])
    #learning_curve(history)
    

X_train_arm,X_train_leg, Y_train, X_test_arm,X_test_leg, Y_test = read_scale_dataset()


# Checking authenticity of data
print('Class Label', motion_kind[Y_train[test_idx]])
#plt.imshow(X_train_arm[440].reshape(16,3,3))
#plt.imshow(X_train_leg[440].reshape(16,3,3))


arm_learner=Thread(target=runLearn(X_train=X_train_arm,Y_train=Y_train,X_test=X_test_arm,Y_test=Y_test,part='arm'))
leg_learner=Thread(target=runLearn(X_train=X_train_leg,Y_train=Y_train,X_test=X_test_leg,Y_test=Y_test,part='leg'))
arm_learner.start()
leg_learner.start()

arm_learner.join()
leg_learner.join()


predict(models['arm'],historys['arm'],X_test_arm,Y_test,part='arm')
predict(models['leg'],historys['leg'],X_test_leg,Y_test,part='leg')

