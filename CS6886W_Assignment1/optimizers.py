import torch.optim as optim
def create_optimizer(name, params, lr):
    n=name.lower()
    if n=='sgd': return optim.SGD(params,lr=lr,momentum=.9)
    if n=='nesterov': return optim.SGD(params,lr=lr,momentum=.9,nesterov=True)
    if n=='adam': return optim.Adam(params,lr=lr)
    if n=='adagrad': return optim.Adagrad(params,lr=lr)
    if n=='rmsprop': return optim.RMSprop(params,lr=lr,momentum=.9)
    if n=='nadam': return optim.NAdam(params,lr=lr)
    raise ValueError('Unknown optimizer: '+name)
