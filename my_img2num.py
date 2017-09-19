from NeuralNetwork import NeuralNetwork
from mnist import MNIST
import torch
import numpy as np

class MyImg2Num(NeuralNetwork):
    def __init__(self):
        super(MyImg2Num, self).__init__()
        # Load MNIST
        mndata = MNIST('./python-mnist/data')
        self.train_data, self.train_label = mndata.load_training()
        # oneHot encoding
        label = []
        max_value = max(self.train_label)
        min_value = min(self.train_label)
        for l in self.train_label:
            label.append([1 if i==l else 0 for i in range(max_value-min_value+1)])
        self.train_label = label
        in_layer = len(self.train_data[0])
        out_layer = len(self.train_label[0])
        self.train_data = torch.ByteTensor(self.train_data)
        self.train_label = torch.ByteTensor(self.train_label)
        # Intialize NeuralNetwork
        self.build(in_layer, in_layer/2, out_layer*2, out_layer)
        self.eta=0.2

    def forward(self, img):
        img = torch.ByteTensor(img)
        img = img.view(img.size()[0]*img.size()[1]) if len(img.size()) == 2 else img
        output = super(MyImg2Num, self).forward(img)
        return np.argmax(output.numpy())

    def train(self):
        print(type(self).__name__, "Start training")
        for i in range(len(self.train_data)):
            if (i+1) % 1000 == 0:
                print(type(self).__name__, "{0:d} images were processed ...".format(i+1))
            if (i+1) >= 3000:
                break
            td = self.train_data[i]
            tl = self.train_label[i]
            pred_label = self.forward(td)
            super(MyImg2Num, self).backward(tl)
            super(MyImg2Num, self).updateParams(self.eta)
        print(type(self).__name__, "Finish training")
