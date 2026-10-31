from .relu import ReLUActivation
from .sigmoid import SigmoidActivation
from .tanh import TanhActivation
from .silu import SiLUActivation
from .gelu import GELUActivation

def get_activation(name):
    table = {'relu': ReLUActivation, 'sigmoid': SigmoidActivation, 'tanh': TanhActivation,
             'silu': SiLUActivation, 'gelu': GELUActivation}
    return table[name.lower()]()
