use burn::prelude::*;

pub trait NeuralNet<const ACTION_SIZE: usize> : Clone{
    type State;
    type BurnBackend : Backend;

    fn predict(&self, perspective: &Self::State, device: &<Self::BurnBackend as Backend>::Device)->(Tensor<Self::BurnBackend, 1, Float>, Tensor<Self::BurnBackend, 1, Float>);
    fn train(&self, training_data: Vec<&(Self::State, Tensor<Self::BurnBackend, 1>, f64)>, device: &<Self::BurnBackend as Backend>::Device)->Self;
}