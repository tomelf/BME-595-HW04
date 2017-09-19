from my_img2num import MyImg2Num
from mnist import MNIST
import torch

def main():
    m = MyImg2Num()
    m.train()

    # Load MNIST
    mndata = MNIST('./python-mnist/data')
    test_data, test_label = mndata.load_testing()
    print test_label[0]
    test_label = oneHot(test_label)
    print test_label[0]
    print m.forward(test_data[0])

def oneHot(label):
    ret = []
    max_value = max(label)
    min_value = min(label)
    for l in label:
        ret.append([1 if i==l else 0 for i in range(max_value-min_value+1)])
    return ret

if __name__ == "__main__":
    main()
