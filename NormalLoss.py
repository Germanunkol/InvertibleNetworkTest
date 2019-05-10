from torch import Tensor
import torch
import math
import numpy as np

x = Tensor(np.arange(-4, 4, 0.05)).cuda()  # Abtastpunkte zur Fehlerbestimmung, evtl. die Samples als Abtastpunkte nutzen?

def gauss(x, scale=Tensor([1]).cuda()):
    """ Calculate the gauss function. Mean = 1
    :param x: input values
    :type x: Tensor
    :param scale: scale^0.5
    :type scale: Tensor
    :rtype: Tensor
    """
    pi2 = Tensor([math.pi*2]).cuda()
    e = Tensor([math.e]).cuda()
    half = Tensor([-0.5]).cuda()
    t = 1 / (pi2 * scale.pow(2)).pow(0.5) * e.pow(half/scale.pow(2) * x.pow(2))
    return t

scale=Tensor([0.3989422917366028]).cuda()

def loss(samples):
    """ Calculate a loss for samples that should be normal distributed.
    :param samples: 
    :return: 
    """
    return (gauss(x)-gauss(torch.t(x.expand((len(samples), len(x))))-samples,
                           scale=scale).sum(dim=1)/len(samples)).abs().mean()  # scale=gauss(0)


if __name__ == '__main__':  # train net to norm-distribute samples
    import matplotlib.pyplot as plt
    data = Tensor(np.arange(-4, 4, 8 / 99)).cuda()
    
    plt.hist(data.detach().cpu().numpy())
    plt.show()

    H = 99
    net = torch.nn.Sequential(
        torch.nn.Linear(99, H),
        torch.nn.ReLU(),
        torch.nn.Linear(H, H),
        torch.nn.ReLU(),
        torch.nn.Linear(H, H),
        torch.nn.ReLU(),
        torch.nn.Linear(H, 99),
    ).cuda()

    # takes in a module and applies the specified weight initialization
    def weights_init_uniform(m):
        classname = m.__class__.__name__
        # for every Linear layer in a model..
        if classname.find('Linear') != -1:
            # apply a uniform distribution to the weights and a bias=0
            m.weight.data.uniform_(0.0, 1.0)
            m.bias.data.fill_(0)

    net.apply(weights_init_uniform)


    opt = torch.optim.SGD(net.parameters(), lr=1e-6)

    for i in range(1000):
        opt.zero_grad()
        out = net(data)
        l = loss(out)
        print(l.item())
        l.backward()
        opt.step()
        if i % 100 == 0:
            plt.hist(out.detach().cpu().numpy())
            plt.show()


