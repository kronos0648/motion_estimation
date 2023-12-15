import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten

def read_scale_dataset():
    train_data = pd.read_csv("mnist/mnist_train.csv", header=None)
    test_data = pd.read_csv("mnist/mnist_test.csv", header=None)
    
    # visualizing class label frequency in the input data
    train_data[0].value_counts().plot.bar(color='cyan')
    
    X_train = train_data.drop(0, axis=1).values
    X_train = X_train.reshape((X_train.shape[0], 28, 28, 1))
    X_train = X_train.astype('float32')
    X_train = X_train/255.0
    
    Y_train = train_data[0].values
    
    X_test = test_data.drop(0, axis=1).values
    X_test = X_test.reshape((X_test.shape[0], 28, 28, 1))
    X_test = X_test.astype('float32')
    X_test = X_test/255.0
    
    Y_test = test_data[0].values
    
    return X_train, Y_train, X_test, Y_test


def create_model():
    model = Sequential([
        Conv2D(32, (3,3), kernel_initializer='he_uniform', input_shape=(28,28,1), activation='relu'),
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(128, kernel_initializer='he_uniform', activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

def evaluate_model(X_train, Y_train, X_test, Y_test):
    # model fitting
    model = create_model()
    history = model.fit(X_train, Y_train, epochs=20, batch_size=32)
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
print('Class Label', Y_train[440])
plt.imshow(X_train[440].reshape(28,28))

model, history = evaluate_model(X_train, Y_train, X_test, Y_test)

# predicting values
Y_pred = model.predict(X_test)

# manually checking authenticity of model in terms of predictions
print("\nActual label: ", Y_test[786])
print("Predicted label: ", np.argmax(Y_pred[786]))
print("The corresponding Image: ")
plt.imshow(X_test[786].reshape(28,28))

learning_curve(history)