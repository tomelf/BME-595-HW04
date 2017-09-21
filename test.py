from my_img2num import MyImg2Num
from nn_img2num import NnImg2Num
from mnist import MNIST
from sklearn.metrics import accuracy_score
import numpy as np

def main():
    print("Init MyImg2Num")
    my = MyImg2Num()
    my.train()

    print("Init NnImg2Num")
    nn = NnImg2Num()
    nn.train()

    print("Load MNIST")
    mndata = MNIST('./python-mnist/data')

    print("==== Start testing (one sample)====")
    test_data, test_label = mndata.load_testing()
    test_label = oneHot(test_label)
    print("expected label", test_label[0])
    td = np.array(test_data[0])
    print("MyImg2Num.forward", my.forward(td))
    print("NnImg2Num.forward", nn.forward(td))

    # print("==== Start testing (MNIST testing set)====")
    # test_data, test_label = mndata.load_testing()
    # test_data = np.array(test_data)
    # test_data = test_data.reshape(test_data.shape[0], int(np.sqrt(test_data.shape[1])), int(np.sqrt(test_data.shape[1])))
    # test_label = np.array(test_label)
    # print "test_data.shape", test_data.shape
    # print "test_label.shape", test_label.shape
    # my_pred = oneHotToNumber(my.forward(test_data).numpy())
    # nn_pred = oneHotToNumber(nn.forward(test_data).numpy())
    # print "my_pred.shape", my_pred.shape
    # print "nn_pred.shape", nn_pred.shape
    #
    # print "accuracy_score(test_label, my_pred)", accuracy_score(test_label, my_pred)
    # print "accuracy_score(test_label, nn_pred)", accuracy_score(test_label, nn_pred)


def oneHot(label):
    ret = []
    max_value = max(label)
    min_value = min(label)
    for l in label:
        ret.append([1 if i==l else 0 for i in range(max_value-min_value+1)])
    return ret

def oneHotToNumber(one_hot_arr):
    return np.argmax(one_hot_arr, 1)

if __name__ == "__main__":
    main()
