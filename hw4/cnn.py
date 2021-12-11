import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Input, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback
import matplotlib.pyplot as plt
import sys

class MyCallback(Callback):
    def __init__(self, Xtest, Ytest):
        self.Xtest = Xtest
        self.Ytest = Ytest
        self.test_accs = []

    def on_epoch_end(self, epoch, logs={}):
        loss, acc = self.model.evaluate(self.Xtest, self.Ytest, verbose=0)
        self.test_accs.append(acc)
        print('- test_loss: {} - test_accuracy: {}\n'.format(loss, acc))


def main():
    print("Running cnn.py file")
    #print("Check the output in cnn-output.txt file")
    #print("Program is running:")
    '''
    backup_stdout = sys.stdout
    backup_stderr = sys.stderr
    outfile = None
    
    try:
        outfile = open('cnn-output.txt', 'w')
        sys.stdout = outfile
        sys.stderr= outfile
    except:
        print("Could not open cnn-output.txt file")
    
    '''
    
    traindf = pd.read_csv('data/fashion-mnist_train.csv')
    testdf = pd.read_csv('data/fashion-mnist_test.csv')
    
    
    traindf.head()
    
    
    
    print(traindf.shape)
    
    
    
    traindata = traindf.values
    testdata = testdf.values
    
    
    
    Xtrain = traindata[:traindata.shape[0]//2,1:]
    Ytrain = traindata[:traindata.shape[0]//2,0]
    Xtest = testdata[:testdata.shape[0]//10,1:]
    Ytest = testdata[:testdata.shape[0]//10,0]
    
    
    
    onehot_encoder1 = OneHotEncoder(sparse=False)
    onehot_encoder2 = OneHotEncoder(sparse=False)
    
    Ytrain_onehot = onehot_encoder1.fit_transform(Ytrain.reshape(-1, 1))
    Ytest_onehot = onehot_encoder2.fit_transform(Ytest.reshape(-1, 1))
    
    
    Ytrain_onehot.shape, Ytest_onehot.shape
    
    
    
    print(Xtrain.shape, Ytrain.shape, Xtest.shape, Ytest.shape)
    
    
    
    Xtrain = Xtrain.reshape(-1, 28, 28,1)
    Xtest = Xtest.reshape(-1, 28, 28, 1)
    
    
    
    print(Xtrain.shape, Ytrain.shape, Xtest.shape, Ytest.shape)
    
    
    input_frame = Input(shape=Xtrain[0].shape)
    x = Conv2D(8, (5, 5), strides=(2,2), padding="same", activation='relu')(input_frame)
    x = Conv2D(16, (3, 3), strides=(2,2), padding="same", activation='relu')(x)
    x = Conv2D(32, (3, 3), strides=(2,2), padding="same", activation='relu')(x)
    x = Conv2D(32, (3, 3), strides=(2,2), padding="same", activation='relu')(x)
    x = AveragePooling2D((2,2))(x)
    x = Flatten()(x)
    output  = Dense(10, activation='softmax')(x)
    
    model = Model(input_frame, output)
    
    
    
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    myCallback = MyCallback(Xtest, Ytest_onehot)
    model.summary()

    #return
    
    history = model.fit(Xtrain, Ytrain_onehot, batch_size=32, epochs=10, verbose=1, callbacks=[myCallback])
    
    epochs = [i for i in range(1,11)]
    plt.figure()
    plt.title("Training and Test accuracies vs Epochs (Fashion-MNIST with CNN)")
    plt.plot(epochs,history.history['accuracy'])
    plt.plot(epochs,myCallback.test_accs)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    
    plt.legend(["train_accuracy", "test_accuracy"])
    plt.savefig('cnn-graph.png')
    plt.show()
    print("The figure is saved as cnn-graph.png")
    
    
    ''' 
    try:
        sys.stdout = backup_stdout
        sys.stderr = backup_stderr
        outfile.close()
    except:
        print("Could not close cnn-output.txt file")
    
    '''




if __name__ == "__main__":
    main()



