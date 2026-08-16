import random
from engine import Value


class Neuron:

    def __init__(self, n_inputs):
        self.weights = [Value(random.uniform(-1, 1)) for _ in range(n_inputs)]
        self.bias = Value(random.uniform(-1,1))

    def __call__(self,x):
        # w * x +b
        activation = sum((wi * xi for wi, xi in zip(self.weights, x)), self.bias)
        output = activation.tanh()

        return output

    def parameters(self):
        return self.weights + [self.bias]


class Layer:
    def __init__(self, n_inputs, n_outputs):
        self.neurons = [Neuron(n_inputs) for _ in range(n_outputs)]

    def __call__(self, x):
        outputs = [neuron(x) for neuron in self.neurons]
        return outputs[0] if len(outputs) == 1 else outputs

    def parameters(self):
        return [param for neuron in self.neurons for param in neuron.parameters()]


class MLP:
    """Building a MLP to test Micrograd"""

    def __init__(self, n_inputs, n_outputs):
        size = [n_inputs] + n_outputs
        self.layers = [Layer(size[i], size[i+1]) for i in range(len(n_outputs))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [param for layer in self.layers for param in layer.parameters()]
