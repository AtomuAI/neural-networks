# BaseLayer Class Documentation

[Return to `README`](/README.md)

The `BaseLayer` class is a part of the `nn` namespace and is used to represent a base layer in a neural network.

## References

- [`Shape`](/docs/core/shape.md)
- [`Tensor`](/docs/core/tensor.md)

## Members

- `const LayerType layer_type`: A constant representing the type of the layer.
- `TrainingMode training_mode`: A variable representing the training mode of the layer.
- `Counter<u64> time_step`: A counter representing the time step of the layer.

## Constructors
```
BaseLayer( const LayerType layer_type, const TrainingMode training_mode )
```
Where `default: layer_type = LayerType::base_layer` & `default: training_mode = TrainingMode::off`
- Constructs a `BaseLayer` with the specified layer type and training mode.

## Destructors

```
~BaseLayer()
```
- Destructs a `BaseLayer`.

## Methods

### Get Layer Type:
```
LayerType get_layer_type() const
```
- Returns the type of the `BaseLayer`.

### Get Timer:
```
const Counter<u64>& get_timer() const
```
- Returns the timer of the `BaseLayer`.

### Get Time Step:
```
u64 get_time_step() const
```
- Returns the time step of the `BaseLayer`.

### Tick Time Step:
```
void tick_time_step()
```
- Increments the time step of the `BaseLayer`.

### Reset Time Step:
```
void reset_time_step()
```
- Resets the time step of the `BaseLayer`.

### Reshape:
```
template <typename T, u8 N>
void reshape( const Shape<N>& shape, Tensor<T, N>& parameters, Tensor<T, N>& jacobian, Tensor<T, N>& momentum, Tensor<T, N>& velocity )
```
- Reshapes the `parameters`, `jacobian`, `momentum`, and `velocity` tensors with the specified shape.

```
template <typename T, u8 N>
void reshape( const Shape<N>& shape, Tensor<T, N>& nodes, Tensor<T, N>& delta )
```
- Reshapes the `nodes` and `delta` tensors with the specified shape.

### Resize:
```
template <typename T, u8 N>
void resize( const Shape<N>& shape, Tensor<T, N>& parameters, Tensor<T, N>& jacobian, Tensor<T, N>& momentum, Tensor<T, N>& velocity )
```
- Resizes the `parameters`, `jacobian`, `momentum`, and `velocity` tensors with the specified shape.

```
template <typename T, u8 N>
void resize( const Shape<N>& shape, Tensor<T, N>& nodes, Tensor<T, N>& delta )
```
- Reshapes the `nodes` and `delta` tensors with the specified shape.

### Allocate Training Memory:
```
template <typename T, u8 N>
void allocate_training_memory( const Tensor<T, N>& parameters, Tensor<T, N>& jacobian, Tensor<T, N>& momentum, Tensor<T, N>& velocity )
```
- Allocates the parameters, jacobian, momentum, and velocity tensors for training.

### Gradient Decent Normal:
```
template <typename T, u8 N>
void gradient_decent_normal( T& parameter, T& jacobian, const Dim batch_size, const StepSize step_size )
```
- Performs gradient descent on the `parameter` with the specified jacobian, batch size, and step size.

```
template <typename T, u8 N>
void gradient_decent_normal( Tensor<T, N>& parameters, Tensor<T, N>& jacobian, const Dim batch_size, const StepSize step_size )
```
- Performs gradient descent on the `parameters` with the specified jacobian, batch size, and step size.


### Gradient Decent Momentum:
```
template <typename T, u8 N>
void gradient_decent_momentum( T& parameter, T& jacobian, T& momentum, const Dim batch_size, const StepSize step_size, const Beta momentum_step_size)
```
- Performs gradient descent on the `parameter` with the specified jacobian, momentum, batch size, step size, and momentum step size.

```
template <typename T, u8 N>
void gradient_decent_momentum( Tensor<T, N>& parameters, Tensor<T, N>& jacobian, Tensor<T, N>& momentum, const Dim batch_size, const StepSize step_size, const Beta momentum_step_size)
```
- Performs gradient descent on the `parameters` with the specified jacobian, momentum, batch size, step size, and momentum step size.


### Gradient Decent Adam:
```
template <typename T, u8 N>
void gradient_decent_adam( T& parameter, T& jacobian, T& momentum, T& velocity, const Dim batch_size, const StepSize step_size, const Beta beta1, const Beta beta2, const Epsilon epsilon )
```
- Performs gradient descent on the `parameter` with the specified jacobian, momentum, velocity, batch size, step size, beta1, beta2, and epsilon.

```
template <typename T, u8 N>
void gradient_decent_adam( Tensor<T, N>& parameters, Tensor<T, N>& jacobian, Tensor<T, N>& momentum, Tensor<T, N>& velocity, const Dim batch_size, const StepSize step_size, const Beta beta1, const Beta beta2, const Epsilon epsilon )
```
- Performs gradient descent on the `parameters` with the specified jacobian, momentum, velocity, batch size, step size, beta1, beta2, and epsilon.

[Return to `README`](/README.md)