// Copyright 2024 Bewusstsein Labs

use std::{
    fmt::Debug,
    ops::{ Deref, DerefMut }
};

use memory::{ Memory, MemoryTraits, MemoryType, stack::Stack, heap::Heap };
use linear_algebra::{ tensor::{ Tensor, TensorTraits }, shape::Shape };
use crate::layer::{
    Layer,
    node::NodeLayer
};

#[derive( Clone, Default, Debug )]
pub struct NormalizationLayer<T, M>
where
    T: 'static + Default + Debug + Clone + Copy + PartialEq,
    M: MemoryType,
    Memory<T, M>: MemoryTraits<Type = T>
{
    params: Tensor<T, 4, M>,
}

impl<T, M> Deref for NormalizationLayer<T, M>
where
    T: 'static + Default + Debug + Clone + Copy + PartialEq,
    M: MemoryType,
    Memory<T, M>: MemoryTraits<Type = T>
{
    type Target = Tensor<T, 4, M>;

    fn deref( &self ) -> &Self::Target {
        &self.params
    }
}

impl<T, M> DerefMut for NormalizationLayer<T, M>
where
    T: 'static + Default + Debug + Clone + Copy + PartialEq,
    M: MemoryType,
    Memory<T, M>: MemoryTraits<Type = T>
{
    fn deref_mut( &mut self ) -> &mut Self::Target {
        &mut self.params
    }
}

impl<T, M> Layer<T, 4, M> for NormalizationLayer<T, M>
where
    T: 'static + Default + Debug + Clone + Copy + PartialEq,
    M: MemoryType,
    Memory<T, M>: MemoryTraits<Type = T>,
    Tensor<T, 4, M>: TensorTraits<T, 4, M>
{
    fn new( shape: Shape<4>, memory: <Memory<T, M> as MemoryTraits>::New ) -> Self {
        Self {
            params: Tensor::<T, 4, M>::new( shape, memory ),
        }
    }

    fn inference<O, P>( &self, input: &NodeLayer<T, O>, output: &mut NodeLayer<T, P> )
    where
        O: MemoryType, P: MemoryType, Memory<T, O>: MemoryTraits<Type = T>, Memory<T, P>: MemoryTraits<Type = T>
    {
        let _ = input;
        let _ = output;
    }

    fn backprop<O, P>( &self, output: &NodeLayer<T, P>, input: &mut NodeLayer<T, O>, output_delta: &Tensor<T, 5, P>, input_delta: &mut Tensor<T, 5, O>, grad: &mut Tensor<T, 4, M> )
    where
    O: MemoryType, P: MemoryType, Memory<T, O>: MemoryTraits<Type = T>, Memory<T, P>: MemoryTraits<Type = T>
    {
        let _ = output;
        let _ = input;
        let _ = output_delta;
        let _ = input_delta;
        let _ = grad;
    }

    fn iter( &self ) -> impl Iterator<Item = &T> {
        self.params.iter()
    }
    fn iter_mut( &mut self ) -> impl Iterator<Item = &mut T> {
        self.params.iter_mut()
    }
}
