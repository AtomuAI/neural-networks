# BewusstseinLabs Neural Networks Library - C++

## This is currently legacy c++ code that will be updated in the future.

The BewusstseinLabs Neural Networks Library is a collection of C++ classes that provide various layers for building neural networks.

## Core

The library is built upon the following core classes:

- [Shape](./docs/core/shape.md): A class that represents a set of dimensions.
- [Tensor](./docs/core/tensor.md): A multi-dimensional vector class that represents a volume of data in a neural network.

## Layers

The library provides the following layers:

- [Base Layer](./docs/layer/base.md): The `BaseLayer` class that serves as the foundation for all other layers.
- [Node Layer](./docs/layer/node.md): The `NodeLayer` class that represents a volume of nodes in a neural network.
- [Bias Layer](./docs/layer/bias.md): The `BiasLayer` class that applies bias to a `NodeLayer`.
- [Dense Layer](./docs/layer/dense.md): The `DenseLayer` class that applies weighted connections between two `NodeLayer`'s.
- [Convolution Layer](./docs/layer/convolution.md): The `ConvolutionLayer` class that applies a filter to a `NodeLayer`.
- [Pooling Layer](./docs/layer/pooling.md): The `PoolingLayer` class that performs pooling operations on a `NodeLayer`.
- [Activation Layer](./docs/layer/activation.md): The `ActivationLayer` class that applies an activation function to a `NodeLayer`'s nodes.
- [Normalization Layer](./docs/layer/normalization.md): The `NormalizationLayer` class that performs normalization operations on a `NodeLayer`.
- [Dropout Layer](./docs/layer/dropout.md): The `DropoutLayer` class that randomly sets a fraction of a `NodeLayer`'s nodes to zero during training.
- [Softmax Layer](./docs/layer/softmax.md): The `SoftmaxLayer` class that applies the softmax function to a `NodeLayer`.
- [Cost Layer](./docs/layer/cost.md): The `CostLayer` class that calculates the cost or loss of a `NodeLayer` with respect to target values.

Each layer is implemented as a separate class and provides specific functionality for a particular operation in a neural network.

## Documentation

For detailed documentation on each layer and its usage, please refer to the individual documentation files linked above.