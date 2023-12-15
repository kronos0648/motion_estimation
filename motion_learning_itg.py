import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, AveragePooling2D, Flatten
import os
from tqdm import tqdm
from PIL import Image


motion_kind=['arm_left','arm_straight','run','walk']
test_idx=11

def read_scale_dataset():
    train_data = pd.read_csv("motion_train.csv")
    test_data = pd.read_csv("motion_test.csv")
    
    # visualizing class label frequency in the input data
    train_data['label'].value_counts().plot.bar(color='cyan')
    
    X_train = train_data.drop('label', axis=1).values
    X_train = X_train.reshape((X_train.shape[0], 16, 6, 3))   
    X_train = X_train.astype('float32')
    X_train = X_train/255.0
    
    Y_train = train_data['label'].values
    
    X_test = test_data.drop('label', axis=1).values
    X_test = X_test.reshape((X_test.shape[0], 16, 6, 3))
    X_test = X_test.astype('float32')
    X_test = X_test/255.0
    
    Y_test = test_data['label'].values
    
    return X_train, Y_train, X_test, Y_test


def create_model():
    model = Sequential([
        Conv2D(56, kernel_size=(4,3), kernel_initializer='he_uniform', input_shape=(16,6,3), strides=(2,3), activation='relu'),
        AveragePooling2D((3,1)),
        Flatten(),
        Dense(112, kernel_initializer='he_uniform', activation='relu'),
        Dense(4, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

def evaluate_model(X_train, Y_train, X_test, Y_test):
    # model fitting
    model = create_model()
    history = model.fit(X_train, Y_train, epochs=20, batch_size=14)
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
    
    
    
X_train, Y_train, X_test, Y_test = read_scale_dataset()

# Checking authenticity of data
print('Class Label', motion_kind[Y_train[test_idx]])
plt.imshow(X_train[440].reshape(16,6,3))

model, history = evaluate_model(X_train, Y_train, X_test, Y_test)

# predicting values
Y_pred = model.predict(X_test)

# manually checking authenticity of model in terms of predictions
print("\nActual label: ", motion_kind[Y_test[test_idx]])
print("Predicted label: ", motion_kind[np.argmax(Y_pred[test_idx])])
print("The corresponding Image: ")
plt.imshow(X_test[test_idx].reshape(16,6,3))

learning_curve(history)