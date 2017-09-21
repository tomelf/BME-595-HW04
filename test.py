from my_img2num import MyImg2Num
from nn_img2num import NnImg2Num
from mnist import MNIST
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import time

def main():
    print("Load MNIST")
    mndata = MNIST('./python-mnist/data')
    train_data, train_label = mndata.load_training()
    train_data = np.array(train_data)
    train_data = train_data.reshape(train_data.shape[0], int(np.sqrt(train_data.shape[1])), int(np.sqrt(train_data.shape[1])))
    train_label = np.array(train_label)
    test_data, test_label = mndata.load_testing()
    test_data = np.array(test_data)
    test_data = test_data.reshape(test_data.shape[0], int(np.sqrt(test_data.shape[1])), int(np.sqrt(test_data.shape[1])))
    test_label = np.array(test_label)

    max_epoch = 5
    models = [MyImg2Num, NnImg2Num]

    epochs = range(1, max_epoch+1)
    for model in models:
        print("Start {0} model".format(type(model()).__name__))
        training_errors = []
        testing_errors = []
        training_time = []
        for epoch in epochs:
            print("== Start training for {0:d} epochs".format(epoch))
            start_time = time.time()
            my = model()
            for i in range(epoch):
                my.train()
                print("-- Finish epoch {0:d}".format(i+1))
            print("Done!")
            print("== Start testing on training and testing set")
            training_time.append(time.time()-start_time)
            my_pred = my.forward(train_data)
            training_errors.append(1-accuracy_score(train_label, my_pred))
            my_pred = my.forward(test_data)
            testing_errors.append(1-accuracy_score(test_label, my_pred))
            print("Done!")
            print
        plt.title(type(model()).__name__)
        plt.xlabel("epochs")
        plt.ylabel("error")
        plt.plot(epochs, training_errors, color="blue", label="training_error")
        plt.plot(epochs, testing_errors, color="red", label="testing_error")
        plt.legend(loc='upper right')
        plt.xticks(epochs)
        plt.savefig("{0}_error_ep-{1:d}.png".format(type(model()).__name__, max_epoch))
        plt.clf()

        plt.title(type(model()).__name__)
        plt.xlabel("epochs")
        plt.ylabel("seconds")
        plt.plot(epochs, training_time, color="blue", label="training_time")
        plt.legend(loc='upper right')
        plt.xticks(epochs)
        plt.savefig("{0}_speed_ep-{1:d}.png".format(type(model()).__name__, max_epoch))
        plt.clf()

if __name__ == "__main__":
    main()
