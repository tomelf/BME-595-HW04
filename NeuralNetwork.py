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

    def forward(self, inputs):
        inputs = inputs.float()
        self.dE_dTheta = []
        self.a = []
        self.a_hat = []
        self.delta = []

        inputs = inputs.view(1, inputs.size()[0]) if len(inputs.size()) == 1 else inputs
        for input in inputs:
            a = []
            a_hat = []

            input = input.view(input.size()[0], 1) if len(input.size()) == 1 else input
            a.append(input.clone()) # a(1) = x
            bias = torch.ones(1, input.size()[1]).float()
            a_hat.append(torch.cat((bias, input), 0)) # a_hat(1) = [bias, a(1)]

            for idx, th in enumerate(self.Theta):
                z = th.mm(a_hat[idx]) # z(l+1)  = theta(l) * a_hat(l)
                a.append(torch.sigmoid(z)) # a(l+1) = activate(z(l+1))
                bias = torch.ones(1, a[idx].size()[1]).float()
                a_hat.append(torch.cat((bias, a[idx+1]), 0)) # a_hat(l) = [bias, a(l)]

            self.a.append(a)
            self.a_hat.append(a_hat)

        if len(self.a) > 1:
            return torch.cat(([a[-1].view(a[-1].size()[1], a[-1].size()[0]) for a in self.a]))
        else:
            return a[-1].view(a[-1].size()[1], a[-1].size()[0])

    def backward(self, targets):
        targets = targets.view(1, targets.size()[0]) if len(targets.size()) == 1 else targets
        self.dE_dTheta = []
        self.delta = []
        for idx, target in enumerate(targets):
            dE_dTheta = []
            delta = []
            a = self.a[idx]
            a_hat = self.a_hat[idx]

            target = target.float()
            target = target.view(target.size()[0], 1) if len(target.size()) == 1 else target
            bias = torch.zeros(1, target.size()[1]).float()
            target = torch.cat((bias, target), 0)

            for i in reversed(range(len(a))):
                if i == len(a_hat)-1:
                    # delta(L) = (a_hat(L)-y) * (a_hat(L) * (1-a_hat(L)))
                    delta.insert(0, (a_hat[i]-target) * (a_hat[i]*(1-a_hat[i])))
                else:
                    th = self.Theta[i]
                    last_d = delta[0][1:]
                    # delta(l) = (theta(l)^T x delta(l+1)) * (a_hat(l) * (1-a_hat(l)))
                    d = (th.transpose(0,1).mm(last_d)) * (a_hat[i] * (1-a_hat[i]))
                    # dE_dTheta(l) = a(l) x (delta(l+1)^T)
                    dd = a_hat[i].mm(last_d.transpose(0, 1)).transpose(0, 1)
                    delta.insert(0, d)
                    dE_dTheta.insert(0, dd)

            self.dE_dTheta.append(dE_dTheta)
            self.delta.append(delta)

    def updateParams(self, eta):
        self.eta = eta
        m = len(self.dE_dTheta)
        for i in range(len(self.Theta)):
            th = self.Theta[i]
            sum_dE_dTheta = None
            for idx, dd in enumerate(self.dE_dTheta):
                if idx == 0:
                    sum_dE_dTheta = dd[i].clone()
                else:
                    sum_dE_dTheta += dd[i]
            self.Theta[i] = th - self.eta * sum_dE_dTheta / m
