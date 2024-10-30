# Shape Class Documentation

[Return to `README`](/README.md)

The `Shape` class is a part of the `nn` namespace and is a template class used to represent the shape of a tensor in a neural network. The template parameter `N` represents the dimensionality of the shape.

## Template Parameters

- `u8 N`: The dimensionality of the shape.

## Template

- `Shape<u8 N>`

## Aliases

- `Shape1D` as `Shape<1>`
- `Shape2D` as `Shape<2>`
- `Shape3D` as `Shape<3>`
- `Shape4D` as `Shape<4>`
- `Shape5D` as `Shape<5>`

## Members

- `static const u8 dimensionality`: A static constant representing the dimensionality of the shape.
- `std::array<Dim, N> dim`: An array representing the dimensions of the shape.
- `Size size`: The total volume of the shape.

## Constructors

```
explicit Shape( const DimND<N>& shape );
```
- Constructs a ND `Shape` with the specified dimensions.

```
explicit Shape( const Dim width = 0 )
```
- Constructs a 1D `Shape` with the specified width.

```
explicit Shape( const Dim width = 0, const Dim height = 1 )
```
- Constructs a 2D `Shape` with the specified width and height.

```
explicit Shape( const Dim width = 0, const Dim height = 1, const Dim depth = 1 )
```
- Constructs a 3D `Shape` with the specified width, height, and depth.

```
explicit Shape( const Dim width = 0, const Dim height = 1, const Dim depth = 1, const Dim channels = 1 )
```
- Constructs a 4D `Shape` with the specified width, height, depth, and channels.

```
explicit Shape( const Dim width = 0, const Dim height = 1, const Dim depth = 1, const Dim channels = 1, const Dim batches = 1 )
```
- Constructs a 5D `Shape` with the specified width, height, depth, channels, and batches.

## Destructors

```
~Shape()
```
- Destructs a `Shape`.

## Methods

### Resize:
```
void resize( const Dim width )
```
- Resizes the 1D `Shape` to the specified width.

```
void resize( const Dim width, const Dim height )
```
- Resizes the 2D `Shape` to the specified width and height.

```
void resize( const Dim width, const Dim height, const Dim depth )
```
- Resizes the 3D `Shape` to the specified width, height, and depth.

```
void resize( const Dim width, const Dim height, const Dim depth, const Dim channels )
```
- Resizes the 4D `Shape` to the specified width, height, depth, and channels.

```
void resize( const Dim width, const Dim height, const Dim depth, const Dim channels, const Dim batches )
```
- Resizes the 5D `Shape` to the specified width, height, depth, channels, and batches.

### Reshape:
```
void reshape( const Dim width, const Dim height )
```
- Reshapes the 2D `Shape` to the specified width and height.

```
void reshape( const Dim width, const Dim height, const Dim depth )
```
- Reshapes the 3D `Shape` to the specified width, height, and depth.

```
void reshape( const Dim width, const Dim height, const Dim depth, const Dim channels )
```
- Reshapes the 4D `Shape` to the specified width, height, depth, and channels.

```
void reshape( const Dim width, const Dim height, const Dim depth, const Dim channels, const Dim batches )
```
- Reshapes the 5D `Shape` to the specified width, height, depth, channels, and batches.

### Equal:
```
inline bool equal( const Dim width )
```
- Checks if the 1D `Shape` is equal to the specified width.

```
inline bool equal( const Dim width, const Dim height )
```
- Checks if the 2D `Shape` is equal to the specified width and height.

```
inline bool equal( const Dim width, const Dim height, const Dim depth )
```
- Checks if the 3D `Shape` is equal to the specified width, height, and depth.

```
inline bool equal( const Dim width, const Dim height, const Dim depth, const Dim channels )
```
- Checks if the 4D `Shape` is equal to the specified width, height, depth, and channels.

```
inline bool equal( const Dim width, const Dim height, const Dim depth, const Dim channels, const Dim batches )
```
- Checks if the 5D `Shape` is equal to the specified width, height, depth, channels, and batches.

### Width Index:
```
inline Idx width_index( const Idx height_index, const Dim x )
```
- Returns the width index for the specified height index and x-coordinate.

### Height Index:
```
inline Idx height_index( const Idx depth_index, const Dim y )
```
- Returns the height index for the specified depth index and y-coordinate.

### Depth Index:
```
inline Idx depth_index( const Idx channel_index, const Dim z )
```
- Returns the depth index for the specified channel index and z-coordinate.

### Channel Index:
```
inline Idx channel_index( const Idx batch_index, const Dim c )
```
- Returns the channel index for the specified batch index and c-coordinate.

### Batch Index:
```
inline Idx batch_index( const Dim b )
```
- Returns the batch index for the specified b-coordinate.

### Index:
```
inline Idx index( const Dim x, const Dim y )
```
- Returns the linear index for the specified x and y coordinates.

```
inline Idx index( const Dim x, const Dim y, const Dim z )
```
- Returns the linear index for the specified x, y, and z coordinates.

```
inline Idx index( const Dim x, const Dim y, const Dim z, const Dim c )
```
- Returns the linear index for the specified x, y, z, and c coordinates.

```
inline Idx index( const Dim x, const Dim y, const Dim z, const Dim c, const Dim b )
```
- Returns the linear index for the specified x, y, z, c, and b coordinates.

### Within Width:
```
inline bool within_width( const Dim x )
```
- Checks if the specified x-coordinate is within the width of the `Shape`.

### Within Height:
```
inline bool within_height( const Dim y )
```
- Checks if the specified y-coordinate is within the height of the `Shape`.

### Within Depth:
```
inline bool within_depth( const Dim z )
```
- Checks if the specified z-coordinate is within the depth of the `Shape`.

### Within Channels:
```
inline bool within_channels( const Dim c )
```
- Checks if the specified c-coordinate is within the channels of the `Shape`.

### Within Batches:
```
inline bool within_batches( const Dim b )
```
- Checks if the specified b-coordinate is within the batches of the `Shape`.

### Width:
```
inline Dim width()
```
- Returns the width of the `Shape`.

### Height:
```
inline Dim height()
```
- Returns the height of the `Shape`.

### Depth:
```
inline Dim depth()
```
- Returns the depth of the `Shape`.

### Channels:
```
inline Dim channels()
```
- Returns the channels of the `Shape`.

### Batches:
```
inline Dim batches()
```
- Returns the batches of the `Shape`.

### Volume:
```
inline Size volume()
```
- Returns the volume of the `Shape`.

### Distance:
```
inline Size distance( const u8 begin, const u8 end )
```
- Returns the distance between the specified begin and end coordinates.

## Operators

### Assign:
```
inline Shape<N>& operator=( const Shape<N>& other )
```
- Assigns the value of `other` to the current `Shape`.

### Equal:
```
inline bool operator==( const Shape<N>& other ) const
```
- Checks if the current `Shape` is equal to `other`.

### Not Equal:
```
inline bool operator!=( const Shape<N>& other ) const
```
- Checks if the current `Shape` is not equal to `other`.

### Less Than:
```
inline bool operator<( const Shape<N>& other ) const
```
- Checks if the current `Shape` is less than `other`.

### Greater Than:
```
inline bool operator>( const Shape<N>& other ) const
```
- Checks if the current `Shape` is greater than `other`.

### Less Than or Equal To:
```
inline bool operator<=( const Shape<N>& other ) const
```
- Checks if the current `Shape` is less than or equal to `other`.

### Greater Than or Equal To:
```
inline bool operator>=( const Shape<N>& other ) const
```
- Checks if the current `Shape` is greater than or equal to `other`.

[Return to `README`](/README.md)