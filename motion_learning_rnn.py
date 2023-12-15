import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import keras.losses
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, AveragePooling2D, Flatten, SimpleRNN, Activation
from keras.losses import CosineSimilarity
from keras.applications.vgg16 import VGG16
from keras.applications.resnet import ResNet50
from keras import optimizers
from keras.wrappers.scikit_learn import KerasClassifier
import os
from tqdm import tqdm
from PIL import Image
from threading import Thread
import json

motion_kind:list
with open('motion.json','r') as f:
    motion_kind=list(json.load(f).keys())
#motion_kind=['arm_left','arm_straight','arm_up','run','walk']
models:dict={}
historys:dict={}

test_idx=11

def read_scale_dataset():
    train_data = pd.read_csv("motion_train.csv")
    test_data = pd.read_csv("motion_test.csv")
    
    # visualizing class label frequency in the input data
    train_data['label'].value_counts().plot.bar(color='cyan')
    X_train = train_data.drop('label', axis=1).values
    
    X_train_arm=X_train.T[0:144].T
    X_train_leg=X_train.T[144:288].T
    
   
    X_train_arm = X_train_arm.reshape((X_train_arm.shape[0], 16, 3, 3))   
    X_train_arm = X_train_arm.astype('float32')
    X_train_arm = X_train_arm/255.0 
    
    X_train_leg = X_train_leg.reshape((X_train_leg.shape[0], 16, 3, 3))   
    X_train_leg = X_train_leg.astype('float32')
    X_train_leg = X_train_leg/255.0
    
    
    Y_train = train_data['label'].values
    
    
    X_test = test_data.drop('label', axis=1).values
    
    X_test_arm=X_test.T[0:144].T
    X_test_leg=X_test.T[144:288].T

    
    X_test_arm = X_test_arm.reshape((X_test_arm.shape[0], 16, 3, 3))
    X_test_arm = X_test_arm.astype('float32')
    X_test_arm = X_test_arm/255.0
    
    X_test_leg = X_test_leg.reshape((X_test_leg.shape[0], 16, 3, 3))
    X_test_leg = X_test_leg.astype('float32')
    X_test_leg = X_test_leg/255.0
    
    
    
    Y_test = test_data['label'].values
    
    return X_train_arm,X_train_leg, Y_train, X_test_arm,X_test_leg, Y_test


#input_shape : 16,3,3
def create_model():
    model = Sequential([
        SimpleRNN(48,input_shape=(16,3,3),return_sequences=True),
        SimpleRNN(48,return_sequences=False),
        Flatten(),
        Dense(56,activation='relu'),
        Dense(len(motion_kind),activation='softmax')
    ])
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    return model


#팔 다리 총 두번 실행하도록
def evaluate_model(X_train, Y_train, X_test, Y_test):
    # model fitting
    model = create_model()
    history = model.fit(X_train, Y_train, epochs=100, batch_size=14)
    print('\n\n')
    model.summary()
    print('\n\n')
    model.evaluate(X_test, Y_test)
    
    return model, history


def learning_curve(hist):
    print('\n')
    plt.subplot(3, 1, 1)
    plt.title('Classification Accuracy')
    plt.plot(hist.history['accuracy'], color='blue', label='train')
    plt.subplot(3, 1, 3)
    plt.title('Cross Entropy Loss')
    plt.plot(hist.history['loss'], color='red', label='train')
    plt.show()
    
    
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

