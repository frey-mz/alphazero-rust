use burn::prelude::*;

pub trait NeuralNet<const ACTION_SIZE: usize> : Clone{
    type State;
    type BurnBackend : Backend;

    fn predict(&self, perspective: &Self::State)->(Tensor<Self::BurnBackend, 1, Float>, f64);
    fn train(&mut self, training_data: Vec<&(Self::State, Tensor<Self::BurnBackend, 1>, f64)>);
}
