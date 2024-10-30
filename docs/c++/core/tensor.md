# Tensor Class Documentation

[Return to `README`](/README.md)

The `Tensor` class is a part of the `nn` namespace and is a template class that represents a tensor of a specified dimensionality and type.

## References

- [`Shape`](/docs/core/shape.md)

## Diagram

![image](/docs/core/tensor/tensor.jpg)

## Template Parameters

- `T`: The data type of the tensor.
- `u8 N`: The dimensionality of the tensor.

## Template

- `Tensor<typename T, u8 N>`

## Aliases

- `Tensor1D<T>` as `Tensor<T, 1>`
- `Tensor2D<T>` as `Tensor<T, 2>`
- `Tensor3D<T>` as `Tensor<T, 3>`
- `Tensor4D<T>` as `Tensor<T, 4>`
- `Tensor5D<T>` as `Tensor<T, 5>`

## Members

- `static const u8 dimensionality`: A static constant representing the dimensionality of the tensor.
- `Shape<N> shape`: The shape of the tensor. [Shape](../shape.md)
- `std::vector<T> data`: The data of the tensor stored in a vector.

## Constructors
```
Tensor()
```
- Constructs a `Tensor` with default shape and data.

```
explicit Tensor( const Shape<N> shape )
```
- Constructs a `Tensor` with the specified shape and default data.

```
Tensor( const Shape<N> shape, T scalar )
```
- Constructs a `Tensor` with the specified shape and fills it with the specified scalar.

```
Tensor( const Shape<N> shape, const std::vector<T>& data )
```
- Constructs a `Tensor` with the specified shape and data.

## Destructors

```
~Tensor()
```
- Destructs a `Tensor`.

## Methods

### Get Shape:
```
const Shape<N>& get_shape() const
```
- Returns the shape of the `Tensor`.

### Get Size:
```
Size get_size() const
```
- Returns the size of the `Tensor`.

### Get Vector:
```
const std::vector<typename std::conditional<is_bool<T>::value, u8, T>::type>& get_vector() const
```
- Returns the data of the `Tensor` as a vector.

### Get Pointer:
```
const T* const get_ptr() const
```
- Returns a pointer to the data of the `Tensor`.

### Reshape:
```
void reshape( const Shape<N>& shape )
```
- Reshapes the `Tensor`.

### Resize:
```
void resize( const Shape<N>& shape )
```
- Resizes the `Tensor`.

### Fill:
```
void fill( const T value )
```
- Fills the `Tensor` with the specified value.

### Set:
```
void set( std::vector<T> data )
```
- Sets the data of the `Tensor` with the specified vector.

### Zero:
```
void zero()
```
- Sets all elements of the `Tensor` to zero.

### Randomize:
```
void randomize( const T min, const T max )
```
- Randomizes the elements of the `Tensor` within the specified range.

## Operators

### Get Element ND Index:
```
inline T operator[]( const DimND<N>& indices ) const
```
- Returns the value of the element at the specified indices.

### Get Element Linear Index:
```
inline T operator[]( const Idx index ) const
```
- Returns the value of the element at the specified index.

### Get Element Reference ND Index:
```
inline T& operator[]( const DimND<N>& indices )
```
- Returns a reference to the element at the specified indices.

### Get Element Reference Linear Index:
```
inline T& operator[]( const Idx index )
```
- Returns a reference to the element at the specified index.

### Assign:
```
inline Tensor<T, N>& operator=( const Tensor<T, N>& other )
```
- Assigns the value of `other` to the current `Tensor`.

### Add:
```
template <typename U> inline Tensor<T, N> operator+( const Tensor<U, N>& other ) const
```
- Returns a new `Tensor` that is the result of element-wise addition of the current `Tensor` and `other`.

### Subtract:
```
template <typename U> inline Tensor<T, N> operator-( const Tensor<U, N>& other ) const
```
- Returns a new `Tensor` that is the result of element-wise subtraction of `other` from the current `Tensor`.

### Multiply:
```
template <typename U> inline Tensor<T, N> operator*( const Tensor<U, N>& other ) const
```
- Returns a new `Tensor` that is the result of element-wise multiplication of the current `Tensor` and `other`.

### Divide:
```
template <typename U> inline Tensor<T, N> operator/( const Tensor<U, N>& other ) const
```
- Returns a new `Tensor` that is the result of element-wise division of the current `Tensor` by `other`.

### Add Assign:
```
template <typename U> inline Tensor<T, N>& operator+=( const Tensor<U, N>& other )
```
- Performs an element-wise addition of `other` to the current `Tensor` and assigns the result to the current `Tensor`.

### Subtract Assign:
```
template <typename U> inline Tensor<T, N>& operator-=( const Tensor<U, N>& other )
```
- Performs an element-wise subtraction of `other` from the current `Tensor` and assigns the result to the current `Tensor`.

### Multiply Assign:
```
template <typename U> inline Tensor<T, N>& operator*=( const Tensor<U, N>& other )
```
- Performs an element-wise multiplication of the current `Tensor` and `other` and assigns the result to the current `Tensor`.

### Divide Assign:
```
template <typename U> inline Tensor<T, N>& operator/=( const Tensor<U, N>& other )
```
- Performs an element-wise division of the current `Tensor` by `other` and assigns the result to the current `Tensor`.

### Scalar Addition:
```
template <typename U> inline Tensor<T, N> operator+( const U& scalar ) const
```
- Returns a new `Tensor` that is the result of adding `scalar` to each element of the current `Tensor`.

### Scalar Subtraction:
```
template <typename U> inline Tensor<T, N> operator-( const U& scalar ) const
```
- Returns a new `Tensor` that is the result of subtracting `scalar` from each element of the current `Tensor`.

### Scalar Multiplication:
```
template <typename U> inline Tensor<T, N> operator*( const U& scalar ) const
```
- Returns a new `Tensor` that is the result of multiplying each element of the current `Tensor` by `scalar`.

### Scalar Division:
```
template <typename U> inline Tensor<T, N> operator/( const U& scalar ) const
```
- Returns a new `Tensor` that is the result of dividing each element of the current `Tensor` by `scalar`.

### Scalar Addition Assign:
```
template <typename U> inline Tensor<T, N>& operator+=( const U& scalar )
```
- Adds `scalar` to each element of the current `Tensor` and assigns the result to the current `Tensor`.

### Scalar Subtraction Assign:
```
template <typename U> inline Tensor<T, N>& operator-=( const U& scalar )
```
- Subtracts `scalar` from each element of the current `Tensor` and assigns the result to the current `Tensor`.

### Scalar Multiplication Assign:
```
template <typename U> inline Tensor<T, N>& operator*=( const U& scalar )
```
- Multiplies each element of the current `Tensor` by `scalar` and assigns the result to the current `Tensor`.

### Scalar Division Assign:
```
template <typename U> inline Tensor<T, N>& operator/=( const U& scalar )
```
- Divides each element of the current `Tensor` by `scalar` and assigns the result to the current `Tensor`.

### Equality:
```
template <typename U> inline bool operator==( const Tensor<U, N>& other ) const
```
- Returns `true` if the current `Tensor` is equal to `other`.

### Less Than:
```
template <typename U> inline bool operator<( const Tensor<U, N>& other ) const
```
- Returns `true` if the current `Tensor` is less than `other`.

### Greater Than:
```
template <typename U> inline bool operator>( const Tensor<U, N>& other ) const
```
- Returns `true` if the current `Tensor` is greater than `other`.

### Less Than or Equal To:
```
template <typename U> inline bool operator<=( const Tensor<U, N>& other ) const
```
- Returns `true` if the current `Tensor` is less than or equal to `other`.

### Greater Than or Equal To:
```
template <typename U> inline bool operator>=( const Tensor<U, N>& other ) const
```
- Returns `true` if the current `Tensor` is greater than or equal to `other`.

### Reduce Add:
```
inline T reduce_add()
```
- Returns the sum of all elements in the `Tensor`.

### Reduce Subtract:
```
inline T reduce_subtract()
```
- Returns the result of subtracting all elements in the `Tensor` from the first element.

### Reduce Multiply:
```
inline T reduce_multiply()
```
- Returns the product of all elements in the `Tensor`.

### Reduce Divide:
```
inline T reduce_divide()
```
- Returns the result of dividing the first element in the `Tensor` by all other elements.

[Return to `README`](/README.md)