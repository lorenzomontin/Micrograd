# Micrograd

An autograd engine built from scratch by following
[Andrej Karpathy's micrograd](https://github.com/karpathy/micrograd) tutorial.

This whole project is learning and understanding exactly how backpropagation works under the hood, without relying on PyTorch or
any other autograd framework to do the math for me.

<img width="1133" height="680" alt="image" src="https://github.com/user-attachments/assets/d3652623-744b-4406-90fd-a9c50cca4c92" />

## What's in here

- **`engine.py`**: the `Value` class: a scalar wrapper that tracks its value,
  its gradient, and the mini computational graph of operations that produced
  it. Every operator (`+`, `*`, `**`, `tanh`, `exp`, ...) builds a small
  `_backward()` closure that knows how to propagate gradient to its inputs
  using the chain rule. Calling `.backward()` on the output node performs a
  topological sort of the graph and calls each node's local `_backward()` in
  reverse order, giving every node in the graph a `.grad` value.
- **`nn.py`**: a tiny neural network library (`Neuron`, `Layer`, `MLP`) built
  entirely on top of `Value`, with no other numerical dependencies.
- **`draw.py`**: a `graphviz`-based visualizer (`draw_dot`) for inspecting
  the computational graph, including each node's forward value and
  backward-pass gradient.
- **`demo.ipynb`**: the full walkthrough: manual (by-hand) backprop on a
  single neuron, verification against `.backward()`, a cross-check against
  PyTorch's autograd, and training a small MLP on a toy binary classification
  dataset.


## How it works

For any operation `out = f(a, b, ...)`, the derivative of `out` with respect
to each input is computed and used to accumulate gradient via the chain rule:

```
input.grad += (∂out/∂input) * out.grad
```

For example, `Value.__mul__`:

```python
def __mul__(self, other):
    out = Value(self.data * other.data, (self, other), '*')

    def _backward():
        self.grad += other.data * out.grad
        other.grad += self.data * out.grad
    out._backward = _backward

    return out
```

Gradients are *accumulated* (`+=`, not `=`) because a single `Value` can feed
into multiple downstream operations and overwriting would silently break the
multivariate chain rule for any node reused more than once in the graph.

## Verifying correctness

The engine's gradients are checked against PyTorch's `autograd` on the same
computation (same inputs, weights, and bias) in `demo.ipynb`. This was the most useful sanity check while
debugging the engine during development.

## Setup

```bash
pip install -r requirements.txt
```

`graphviz` also requires the system binary (used by `draw_dot` to render
graph images), not just the Python package:

```bash
# macOS
brew install graphviz

# Debian/Ubuntu
sudo apt install graphviz
```

## Usage

```python
from engine import Value

a = Value(2.0)
b = Value(-3.0)
c = a * b + 1
c.backward()

print(a.grad, b.grad)  # gradients of c w.r.t. a and b
```

```python
from nn import MLP

model = MLP(3, [4, 4, 1])  # 3 inputs -> two hidden layers of 4 -> 1 output
out = model([2.0, 3.0, -1.0])
```

## What's next

- Extending the engine with new functions (`log`, `sigmoid`, etc.)


## Acknowledgements

This project follows [Andrej Karpathy's micrograd](https://github.com/karpathy/micrograd)
and the accompanying ["The spelled-out intro to neural networks and backpropagation"](https://www.youtube.com/watch?v=VMj-3S1tku0)
video. All credit for the design and pedagogy goes to him. This repo is my
own implementation, written while following along, for learning purposes.
