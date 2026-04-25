# 06 — Deep Learning Intro (PyTorch)

Dataset: MNIST Handwritten Digits (via `torchvision`)  
Model: Feedforward neural network (784 -> 256 -> 128 -> 10)  
Framework: PyTorch 2.0+

| Concept | Where shown |
|---|---|
| Tensors and DataLoaders | Loading and batching MNIST |
| `nn.Module` architecture | `MNISTNet` class definition |
| Forward pass + backprop | Training loop |
| Loss and optimizer | `CrossEntropyLoss`, `Adam` |
| Train vs eval mode | `model.train()` / `model.eval()` |
| Training curves | Loss and accuracy over 10 epochs |

This feedforward network forms the foundation for more complex architectures
(CNNs, transformers) in future projects.
