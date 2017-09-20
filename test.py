from my_img2num import MyImg2Num
from nn_img2num import NnImg2Num
from mnist import MNIST
import numpy as np

def main():
    print("Init MyImg2Num")
    my = MyImg2Num()
    my.train()

    print("Init NnImg2Num")
    nn = NnImg2Num()
    nn.train()

    print("==== Start testing ====")
    print("Load MNIST")
    mndata = MNIST('./python-mnist/data')
    test_data, test_label = mndata.load_testing()
    test_label = oneHot(test_label)
    print("expected label", test_label[0])
    td = np.array(test_data[0]).reshape(28, 28)
    print("MyImg2Num.forward", my.forward(td))
    print("NnImg2Num.forward", nn.forward(td))

def oneHot(label):
    ret = []
    max_value = max(label)
    min_value = min(label)
    for l in label:
        ret.append([1 if i==l else 0 for i in range(max_value-min_value+1)])
    return ret

if __name__ == "__main__":
    main()
