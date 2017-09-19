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
        return self.Theta[layer-1].view(self.Theta[layer-1].size()[1], self.Theta[layer-1].size()[0])

    def forward(self, input):
        input = input.float()
        self.dE_dTheta = []
        self.a = []
        self.a_hat = []
        self.delta = []

        input = input.view(1, input.size()[0]) if len(input.size()) == 1 else input
        input = input.view(input.size()[1], input.size()[0])

        self.a.append(input.clone()) # a(1) = x
        bias = torch.ones(1, input.size()[1]).float()
        self.a_hat.append(torch.cat((bias, input), 0)) # a_hat(1) = [bias, a(1)]

        for idx, th in enumerate(self.Theta):
            z = th.mm(self.a_hat[idx]) # z(l+1)  = theta(l) * a_hat(l)
            self.a.append(torch.sigmoid(z)) # a(l+1) = activate(z(l+1))
            bias = torch.ones(1, self.a[idx].size()[1]).float()
            self.a_hat.append(torch.cat((bias, self.a[idx+1]), 0)) # a_hat(l) = [bias, a(l)]

        return self.a[-1].view(self.a[-1].size()[1], self.a[-1].size()[0])

    def backward(self, target):
        target = target.float()
        target = target.view(1, target.size()[0]) if len(target.size()) == 1 else target
        target = target.view(target.size()[1], target.size()[0])

        bias = torch.zeros(1, target.size()[1]).float()
        target = torch.cat((bias, target), 0)

        for i in reversed(range(len(self.a))):
            if i == len(self.a_hat)-1:
                # delta(L) = (a_hat(L)-y) * (a_hat(L) * (1-a_hat(L)))
                self.delta.insert(0, (self.a_hat[i]-target) * (self.a_hat[i]*(1-self.a_hat[i])))
            else:
                th = self.Theta[i]
                last_delta = self.delta[0][1:]
                # delta(l) = (theta(l)^T x delta(l+1)) * (a_hat(l) * (1-a_hat(l)))
                delta = (th.transpose(0,1).mm(last_delta)) * (self.a_hat[i] * (1-self.a_hat[i]))
                # dE_dTheta(l) = a(l) x (delta(l+1)^T)
                dE_dTheta = self.a_hat[i].mm(last_delta.transpose(0, 1)).transpose(0, 1)
                self.delta.insert(0, delta)
                self.dE_dTheta.insert(0, dE_dTheta)

    def updateParams(self, eta):
        self.eta = eta
        for i in range(len(self.Theta)):
            th = self.Theta[i]
            dE_dTheta = self.dE_dTheta[i]
            self.Theta[i] = th - self.eta * dE_dTheta
