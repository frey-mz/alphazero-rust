use burn::prelude::*;
#[derive(Clone, Copy)]
pub enum SimulationResult{
    Win,
    Loss,
    Draw
}
impl SimulationResult{
    pub fn to_value(&self)->f64{
        match self{
            Self::Win=>1.0,
            Self::Loss=>-1.0,
            Self::Draw=>0.0,
        }
    }
    pub fn inverse(&self)->Self{
        match self{
            Self::Win=>Self::Loss,
            Self::Loss=>Self::Win,
            Self::Draw=>Self::Draw,
        }
    }
}



pub trait Environment<const ACTION_SIZE: usize>{
    type State;
    type BurnBackend: Backend;

    fn get_new_state(&self)->(Self::State, Self::State);
    fn get_next_state(&self, state: &Self::State, action: usize)->Self::State;
    fn get_valid_actions(&self, state: &Self::State)->Tensor<Self::BurnBackend, 1, Bool>;
    fn get_simulation_result(&self, state: &Self::State)->Option<SimulationResult>;
    fn get_symmetrical_policies(&self, state: &Self::State, policy: &Tensor<Self::BurnBackend, 1, Float>)->Vec<(Self::State, Tensor<Self::BurnBackend, 1, Float>)>;
}