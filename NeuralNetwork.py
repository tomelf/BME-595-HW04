import torch
import math

class NeuralNetwork(object):
    def __init__(self):
        self.eta = 0.5
        self.Theta = []
        self.dE_dTheta = []
        self.a = []
        self.a_hat = []
        self.delta = []

    def build(self, in_layer, *h_arr):
        for i in range(len(h_arr)):
            s = in_layer if i==0 else h_arr[i-1]
            e = h_arr[i]
            th = torch.normal(means=torch.zeros(e, s+1), std=torch.zeros(e, s+1).fill_(1/math.sqrt(e))).float()
            self.Theta.append(th)

    def getLayer(self, layer):
        return self.Theta[layer-1].transpose(0,1)

    def forward(self, inputs):
        x = inputs.float()
        self.a = []
        self.a_hat = []

        x = x.view(x.size()[0], 1) if len(x.size()) == 1 else x.transpose(0,1)
        self.a.append(x) # a(1) = x
        bias = torch.ones(1, self.a[-1].size()[1]).float()
        self.a_hat.append(torch.cat((bias, self.a[-1]), 0)) # a_hat(1) = [bias, a(1)]

        for idx, th in enumerate(self.Theta):
            z = th.mm(self.a_hat[idx]) # z(l+1)  = theta(l) * a_hat(l)
            self.a.append(torch.sigmoid(z)) # a(l+1) = activate(z(l+1))
            bias = torch.ones(1, self.a[-1].size()[1]).float()
            self.a_hat.append(torch.cat((bias, self.a[-1]), 0))

        return self.a[-1].transpose(0,1)

    def backward(self, targets):
        targets = targets.float()
        targets = targets.view(targets.size()[0], 0) if len(targets.size()) == 1 else targets.transpose(0,1)
        bias = torch.zeros(1, targets.size()[1]).float()
        targets = torch.cat((bias, targets), 0)

        self.dE_dTheta = []
        self.delta = []

        for i in reversed(range(len(self.a_hat))):
            if i == len(self.a_hat)-1:
                # delta(L) = (a_hat(L)-y) * (a_hat(L) * (1-a_hat(L)))
                self.delta.insert(0, (self.a_hat[i]-targets) * (self.a_hat[i]*(1-self.a_hat[i])))
            else:
                th = self.Theta[i]
                last_d = self.delta[0][1:]
                # delta(l) = (theta(l)^T x delta(l+1)) * (a_hat(l) * (1-a_hat(l)))
                d = (th.transpose(0,1).mm(last_d)) * (self.a_hat[i] * (1-self.a_hat[i]))
                # dE_dTheta(l) = a(l) x (delta(l+1)^T)
                dd = self.a_hat[i].mm(last_d.transpose(0, 1)).transpose(0, 1)
                self.delta.insert(0, d)
                self.dE_dTheta.insert(0, dd)

    def updateParams(self, eta):
        self.eta = eta
        batch_size = self.a[0].size()[1]
        for i in range(len(self.Theta)):
            self.Theta[i] -= (self.eta * self.dE_dTheta[i] / batch_size)
