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
        super(MyImg2Num, self).build(in_layer, in_layer/2, out_layer)
        self.eta=0.2

    def forward(self, img):
        img = torch.ByteTensor(img)
        if len(img.size()) == 3:
            img = img.view(img.size()[0], img.size()[1]*img.size()[2])
        elif len(img.size()) == 2:
            img = img.view(img.size()[0]*img.size()[1])
        output = super(MyImg2Num, self).forward(img)
        # return output
        return np.argmax(output.numpy())

    def train(self):
        batch_size = 32
        epoch = 1

        print(type(self).__name__, "Start training")
        for ep in range(epoch):
            print(type(self).__name__, "Start epoch {0:d}".format(ep+1))

            current_index = 0
            num_train_data = self.train_data.size()[0]
            i = 1
            while current_index < num_train_data:
                if current_index >= (1000*i):
                    # print(type(self).__name__, "{0:d} images were processed ...".format(current_index))
                    i += 1
                td = self.train_data[current_index:current_index+batch_size]
                tl = self.train_label[current_index:current_index+batch_size]
                pred_label = super(MyImg2Num, self).forward(td)
                super(MyImg2Num, self).backward(tl)
                super(MyImg2Num, self).updateParams(self.eta)
                current_index += td.size()[0]
            # print(type(self).__name__, "{0:d} images were processed ...".format(current_index))

        print(type(self).__name__, "Finish training")
