from NeuralNetwork import NeuralNetwork
from mnist import MNIST
import torch
import numpy as np

class MyImg2Num(NeuralNetwork):
    def __init__(self):
        super(MyImg2Num, self).__init__()
        # Load MNIST
        print("Load MNIST")
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
        # Intialize NeuralNetwork
        self.build(in_layer, in_layer/2, out_layer*2, out_layer)

    def forward(self, img):
        img = torch.ByteTensor(img)
        img = img.view(img.size()[0]*img.size()[1]) if len(img.size()) == 2 else img
        output = super(MyImg2Num, self).forward(img)
        return np.argmax(output.numpy())

    def train(self):
        print("Start training")
        train_data = torch.ByteTensor(self.train_data)
        train_label = torch.ByteTensor(self.train_label)
        for i in range(len(train_data)):
            if (i+1) % 1000 == 0:
                print("{0:d} images were processed ...".format(i+1))
            # if (i+1) >= 5000:
            #     break
            data = train_data[i]
            label = train_label[i]
            output = self.forward(data)
            super(MyImg2Num, self).backward(label)
            super(MyImg2Num, self).updateParams(eta=0.2)
