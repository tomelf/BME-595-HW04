from mnist import MNIST
from torch.autograd import Variable
import torch
import numpy as np

class NnImg2Num(object):
    def __init__(self):
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
        # Intialize NeuralNetwork
        self.train_data = torch.FloatTensor(self.train_data)
        self.train_label = torch.FloatTensor(self.train_label)
        self.model = torch.nn.Sequential(
          torch.nn.Linear(in_layer, in_layer/2),
          torch.nn.Sigmoid(),
          torch.nn.Linear(in_layer/2, out_layer*2),
          torch.nn.Sigmoid(),
          torch.nn.Linear(out_layer*2, out_layer),
          torch.nn.Sigmoid()
        )
        self.loss_function = torch.nn.MSELoss()
        eta = 0.2
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=eta)

    def forward(self, img):
        return np.argmax(self.model(Variable(torch.FloatTensor(img))).data.numpy())

    def train(self):
        print(type(self).__name__, "Start training")
        for i in range(self.train_data.size()[0]):
            if (i+1) % 1000 == 0:
                print(type(self).__name__, "{0:d} images were processed ...".format(i+1))
            if (i+1) >= 3000:
                break
            td = Variable(self.train_data[i])
            tl = Variable(self.train_label[i])
            self.optimizer.zero_grad()
            pred_label = self.model(td)
            loss = self.loss_function(pred_label, tl)
            loss.backward()
            self.optimizer.step()
        print(type(self).__name__, "Finish training")
