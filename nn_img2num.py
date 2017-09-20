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
          torch.nn.Linear(in_layer/2, out_layer),
          torch.nn.Sigmoid(),
        )
        self.loss_function = torch.nn.MSELoss()
        eta = 0.2
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=eta)

    def forward(self, img):
        img = torch.FloatTensor(img)
        img = img.view(img.size()[0]*img.size()[1]) if len(img.size()) == 2 else img
        return np.argmax(self.model(Variable(torch.FloatTensor(img))).data.numpy())

    def train(self):
        batch_size = 32

        current_index = 0
        num_train_data = self.train_data.size()[0]
        i = 1
        print(type(self).__name__, "Start training")
        while current_index < num_train_data:
            if current_index >= (1000*i):
                print(type(self).__name__, "{0:d} images were processed ...".format(current_index))
                i += 1
            td = Variable(self.train_data[current_index:current_index+batch_size])
            tl = Variable(self.train_label[current_index:current_index+batch_size])
            self.optimizer.zero_grad()
            pred_label = self.model(td)
            loss = self.loss_function(pred_label, tl)
            loss.backward()
            self.optimizer.step()
            current_index += td.size()[0]
        print(type(self).__name__, "{0:d} images were processed ...".format(current_index))
        print(type(self).__name__, "Finish training")
