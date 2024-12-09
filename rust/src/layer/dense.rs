// Copyright 2024 Bewusstsein Labs

use std::{
    fmt::Debug,
    ops::{ Add, AddAssign, Deref, DerefMut, Mul }
};

use memory::{ Memory, MemoryTraits, MemoryType, stack::Stack, heap::Heap };
use linear_algebra::{ tensor::{ Tensor, TensorTraits, contract }, shape::Shape };
use crate::layer::{
    Layer,
    node::NodeLayer
};

#[derive( Clone, Default, Debug )]
pub struct DenseLayer<T, M>
where
    T: 'static + Default + Debug + Clone + Copy + PartialEq,
    M: MemoryType,
    Memory<T, M>: MemoryTraits<Type = T>
{
    weights: Tensor<T, 2, M>,
}

impl<T, M> Deref for DenseLayer<T, M>
where
    T: 'static + Default + Debug + Clone + Copy + PartialEq,
    M: MemoryType,
    Memory<T, M>: MemoryTraits<Type = T>
{
    type Target = Tensor<T, 2, M>;

    fn deref( &self ) -> &Self::Target {
        &self.weights
    }
}

impl<T, M> DerefMut for DenseLayer<T, M>
where
    T: 'static + Default + Debug + Clone + Copy + PartialEq,
    M: MemoryType,
    Memory<T, M>: MemoryTraits<Type = T>
{
    fn deref_mut( &mut self ) -> &mut Self::Target {
        &mut self.weights
    }
}

impl<T, M> Layer<T, 2, M> for DenseLayer<T, M>
where
    T: 'static + Default + Debug + Clone + Copy + PartialEq + Add<Output = T> + AddAssign + Mul<Output = T>,
    M: MemoryType,
    Memory<T, M>: MemoryTraits<Type = T>,
    Tensor<T, 2, M>: TensorTraits<T, 2, M>
{
    fn new( shape: Shape<2> ) -> Self {
        Self {
            weights: Tensor::<T, 2, M>::new( shape ),
        }
    }

    fn take( shape: Shape<2>, memory: <Memory<T, M> as MemoryTraits>::Take ) -> Self {
        Self {
            weights: Tensor::<T, 2, M>::take( shape, memory ),
        }
    }

    fn inference<O, P>( &self, input: &NodeLayer<T, O>, output: &mut NodeLayer<T, P> )
    where
        T: 'static + Default + Debug + Clone + Copy + PartialEq + Add<Output = T> + AddAssign + Mul<Output = T>,
        O: MemoryType,
        P: MemoryType,
        Memory<T, O>: MemoryTraits<Type = T>,
        Memory<T, P>: MemoryTraits<Type = T>,
        Tensor<T, 5, O>: TensorTraits<T, 5, O>,
        Tensor<T, 5, P>: TensorTraits<T, 5, P>
    {
        contract( &self.weights, & **input, &mut **output, &[1], &[0] );
    }

    fn backprop<O, P>( &self, _: &NodeLayer<T, P>, input: &mut NodeLayer<T, O>, output_delta: &Tensor<T, 5, P>, input_delta: &mut Tensor<T, 5, O>, grad: &mut Tensor<T, 2, M> )
    where
        T: 'static + Default + Debug + Clone + Copy + PartialEq + Add<Output = T> + AddAssign + Mul<Output = T>,
        O: MemoryType,
        P: MemoryType,
        Memory<T, O>: MemoryTraits<Type = T>,
        Memory<T, P>: MemoryTraits<Type = T>,
        Tensor<T, 5, O>: TensorTraits<T, 5, O>,
        Tensor<T, 5, P>: TensorTraits<T, 5, P>
    {
        contract( output_delta, & **input, grad, &[1], &[0] );
        contract( &self.weights, output_delta, input_delta, &[1], &[0] );
    }

    fn grad_descent( &mut self, grad: &Tensor<T, 2, M> ) {
        for ( i, weight ) in self.weights.iter_mut().enumerate() {
            *weight += grad[ i ];
        }
    }

    fn grad_descent_momentum( &mut self, grad: &Tensor<T, 2, M>, momentum: &mut Tensor<T, 2, M> ) {
        let _ = grad;
        let _ = momentum;
    }

    fn grad_descent_adam( &mut self, grad: &Tensor<T, 2, M>, momentum: &mut Tensor<T, 2, M>, velocity: &mut Tensor<T, 2, M> ) {
        let _ = grad;
        let _ = momentum;
        let _ = velocity;
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn inference_test() {
        use memory::{ Memory, stack::Stack };
        use linear_algebra::{ tensor::{ Tensor, TensorTraits }, shape::Shape };
        use crate::layer::{
            Layer,
            node::NodeLayer
        };
        use super::DenseLayer;

        let input = NodeLayer::<f32, Stack<4>>::take(
            [ 4, 1, 1, 1, 1 ].into(),
            [
                1.0, 1.0, 1.0, 1.0
            ],
        );
        let mut output = NodeLayer::<f32, Stack<4>>::take(
            [ 4, 1, 1, 1, 1 ].into(),
            [
                0.0, 0.0, 0.0, 0.0
            ],
        );
        let dense = DenseLayer::<f32, Stack<16>>::take(
            [ 4, 4 ].into(),
            [
                2.0, 2.0, 2.0, 2.0,
                2.0, 2.0, 2.0, 2.0,
                2.0, 2.0, 2.0, 2.0,
                2.0, 2.0, 2.0, 2.0
            ],
        );

        println!( "Before:");
        println!( "{:?}", input );
        println!( "{:?}", output );

        dense.inference( &input, &mut output );

        println!( "After:");
        println!( "{:?}", input );
        println!( "{:?}", output );
    }
}
