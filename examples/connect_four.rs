use std::sync::Arc;

use alphazero_rust::coach::{learn, CoachOptions};
use alphazero_rust::environment::{Environment, SimulationResult};
use alphazero_rust::mcts::SearchOptions;
use alphazero_rust::net::NeuralNet;
use burn::backend::wgpu::WgpuDevice;
use burn::backend::Wgpu;
use burn::prelude::*;
use burn::tensor::activation::log_softmax;
use burn::tensor::cast::ToElement;
use burn::nn::{
        conv::Conv2d,
        Linear, LinearConfig, Relu,
    };
use nn::{BatchNorm, BatchNormConfig, PaddingConfig2d};

fn main() {
    let coach_options = Arc::new(CoachOptions{
        temp_threshold: 100,
        episodes: 100,
        iterations: 10,
        max_iterations_for_training: 50,
        update_threshold: 0.55,
    });
    let search_options = Arc::new(SearchOptions{
        cpuct: 1.25,
        rollout_count: 100,
    });

    let agent = Connect4Net::new();

    let env = Connect4;

    learn(coach_options, search_options, agent, &env);
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]

enum CellType {
    Empty,
    Enemy,
    Hero,
}
impl Default for CellType {
    fn default() -> Self {
        Self::Empty
    }
}
impl CellType {
    fn invert(&self) -> Self {
        match self {
            Self::Empty => Self::Empty,
            Self::Enemy => Self::Hero,
            Self::Hero => Self::Enemy,
        }
    }
}

#[derive(Clone, PartialEq, Eq, Default)]
struct Board([[CellType; 7]; 6]);

impl Board {
    fn invert(&self) -> Self {
        let mut new_board = Self::default();

        for y in 0..6 {
            for x in 0..7 {
                new_board.0[y][x] = self.0[y][x].invert();
            }
        }

        new_board
    }
}

impl std::hash::Hash for Board {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        for y in 0..6 {
            for x in 0..7 {
                self.0[y][x].hash(state);
            }
        }
    }
}

#[derive(Clone, Hash, PartialEq, Eq)]
struct Connect4State {
    board: Board,
    turn: bool,
}

struct Connect4;


impl Environment<7> for Connect4 {
    type State = Connect4State;
    type BurnBackend = Wgpu;
    fn get_new_state(&self) -> (Self::State, Self::State) {
        (
            Connect4State {
                board: Default::default(),
                turn: true,
            },
            Connect4State {
                board: Default::default(),
                turn: false,
            },
        )
    }
    fn get_next_state(&self, state: &Self::State, action: usize) -> Self::State {
        let mut new_state = state.clone();

        for y in 0..6 {
            if new_state.board.0[y][action] != CellType::Empty {
                new_state.board.0[y][action] = if new_state.turn {
                    CellType::Hero
                } else {
                    CellType::Enemy
                };
                break;
            }
        }

        new_state
    }
    fn get_valid_actions(&self, state: &Self::State) -> Tensor<Self::BurnBackend, 1, Bool> {
        let mut valids = [false; 7];

        for action in 0..7 {
            if state.board.0[5][action] == CellType::Empty {
                valids[action] = true;
            }
        }

        Tensor::from_data(valids, &Default::default())
    }
    fn get_simulation_result(&self, state: &Self::State) -> Option<SimulationResult> {
        if !self
            .get_valid_actions(state)
            .into_data()
            .iter::<bool>()
            .any(|x| x)
        {
            return Some(SimulationResult::Draw);
        }

        for row in 0..3 {
            for column in 0..7 {
                'outer: for player in [CellType::Enemy, CellType::Hero] {
                    for delta in 0..4 {
                        if state.board.0[row + delta][column] != player {
                            break 'outer;
                        }
                    }
                    if player == CellType::Hero {
                        return Some(SimulationResult::Win);
                    } else {
                        return Some(SimulationResult::Loss);
                    }
                }
            }
        }
        for row in 0..6 {
            for column in 0..4 {
                'outer: for player in [CellType::Enemy, CellType::Hero] {
                    for delta in 0..4 {
                        if state.board.0[row][column + delta] != player {
                            break 'outer;
                        }
                    }
                    if player == CellType::Hero {
                        return Some(SimulationResult::Win);
                    } else {
                        return Some(SimulationResult::Loss);
                    }
                }
            }
        }

        for row in 0..3 {
            for column in 0..4 {
                'outer: for player in [CellType::Enemy, CellType::Hero] {
                    for delta in 0..4 {
                        if state.board.0[row + delta][column + delta] != player {
                            break 'outer;
                        }
                    }
                    if player == CellType::Hero {
                        return Some(SimulationResult::Win);
                    } else {
                        return Some(SimulationResult::Loss);
                    }
                }
            }
        }

        for row in 3..6 {
            for column in 0..4 {
                'outer: for player in [CellType::Enemy, CellType::Hero] {
                    for delta in 0..4 {
                        if state.board.0[row - delta][column + delta] != player {
                            break 'outer;
                        }
                    }
                    if player == CellType::Hero {
                        return Some(SimulationResult::Win);
                    } else {
                        return Some(SimulationResult::Loss);
                    }
                }
            }
        }

        None
    }
    fn get_symmetrical_policies(
        &self,
        state: &Self::State,
        policy: &Tensor<Self::BurnBackend, 1, Float>,
    ) -> Vec<(Self::State, Tensor<Self::BurnBackend, 1, Float>)> {
        let mut syms = vec![];

        syms.push((state.clone(), policy.clone()));

        let mut new_state = state.clone();

        for row in 0..6 {
            new_state.board.0[row].reverse();
        }

        let mut new_policy = [0.0; 7];

        for (index, policy) in policy.clone().into_data().iter::<f64>().enumerate() {
            new_policy[6 - index] = policy;
        }
        let new_policy = Tensor::from_floats(new_policy, &Default::default());

        syms.push((new_state, new_policy));

        syms
    }
}

#[derive(Clone, Module, Debug)]
struct Connect4Net {
    conv: ConvBlock<Wgpu>,
    outblock: OutBlock<Wgpu>,
    resblocks: [ResBlock<Wgpu>;5]
}

impl NeuralNet<7> for Connect4Net {
    type BurnBackend = Wgpu;
    type State = Connect4State;

    fn predict(&self, perspective: &Self::State) -> (Tensor<Self::BurnBackend, 1, Float>, f64) {
        let mut data = [[[0.0;6];7];2];

        let board = if perspective.turn{
            &perspective.board
        }else{
            &perspective.board.invert()
        };

        for row in 0..6{
            for col in 0..7{
                match board.0[row][col]{
                    CellType::Empty => {},
                    CellType::Enemy => {data[1][col][row] = 1.0;},
                    CellType::Hero => {data[0][col][row] = 1.0;},
                }
            }
        }

        let tensor = Tensor::<Wgpu, 3>::from_data(data, &WgpuDevice::BestAvailable);

        assert!(tensor.shape() == Shape::new([2,7,6]));

        let (policy, value) = self.forward(tensor);
        let (policy, value) = (policy, value.into_scalar().to_f64());
        (policy, value)
    }

    fn train(&mut self, training_data: Vec<&(Self::State, Tensor<Self::BurnBackend, 1>, f64)>) {
        todo!()
    }
}

impl Connect4Net {
    fn new() -> Self {
        let device = &WgpuDevice::BestAvailable;
        Self {
            conv: ConvBlock::new([2, 128], [3,3], device),
            outblock: OutBlock::new(device),
            resblocks: std::array::from_fn(|_|ResBlock::new(device)),
        }
    }
    fn forward(&self, input: Tensor<Wgpu, 3>)-> (Tensor<Wgpu, 1>, Tensor<Wgpu, 1>){
        let input = input.reshape([1, 2, 7, 6]).detach();

        let mut input = self.conv.forward(input);
        for block in self.resblocks.iter(){
            input = block.forward(input);
        }
        self.outblock.forward(input)

    }
}



#[derive(Module, Debug)]
struct ResBlock<B: Backend> {
    conv:  ConvBlock<B>,
    conv2: ConvBlock<B>,

}
impl<B: Backend> ResBlock<B> {
    fn new(device: &Device<B>) -> Self {

        Self {
            conv: ConvBlock::new([128, 128], [3,3], device),
            conv2: ConvBlock::new([128, 128], [3, 3], device),
        }
    }

    fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let residual = input.clone();
        let input = self.conv.forward(input);
        let input = self.conv2.forward_with_residual(input, residual);

        input
    }
}



#[derive(Module, Debug)]
pub struct ConvBlock<B: Backend> {
    conv: Conv2d<B>,
    norm: BatchNorm<B, 2>,
    activation: Relu,
}

impl<B: Backend> ConvBlock<B> {
    pub fn new(channels: [usize; 2], kernel_size: [usize; 2], device: &B::Device) -> Self {
        let conv = nn::conv::Conv2dConfig::new(channels, kernel_size)
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);
        let norm = BatchNormConfig::new(channels[1]).init(device);

        Self {
            conv,
            norm,
            activation: nn::Relu::new(),
        }
    }

    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.conv.forward(input);
        let x = self.norm.forward(x);

        self.activation.forward(x)
    }
    pub fn forward_with_residual(&self, input: Tensor<B, 4>, residual: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.conv.forward(input);
        let x = self.norm.forward(x);
        let x = x.add(residual);

        self.activation.forward(x)
    }
}


#[derive(Module, Debug)]
struct OutBlock<B: Backend> {
    conv: ConvBlock<B>,
    conv2: ConvBlock<B>,
    fc: Linear<B>,
    fc2: Linear<B>,
    fc3: Linear<B>,

}
impl<B: Backend> OutBlock<B> {
    fn new(device: &B::Device) -> Self {

        Self {
            conv: ConvBlock::new([2, 128], [3, 3], device),
            conv2: ConvBlock::new([32, 128], [3, 3], device),
            fc: LinearConfig::new(6*7*32, 7).init(device),
            fc2: LinearConfig::new(2*6*7, 32).init(device),
            fc3: LinearConfig::new(32, 1).init(device),

        }
    }

    fn forward(&self, input: Tensor<B, 4>) -> (Tensor<B, 1>, Tensor<B, 1>) {
        let value = self.conv.forward(input.clone());

        let value = value.reshape([2*6*7_i32]);
        let value = self.fc2.forward(value);
        let value = Relu::new().forward(value);


        let value = self.fc3.forward(value);
        let value = value.tanh();

        let policy = self.conv2.forward(input);

        let policy = policy.reshape([6 * 7 * 32_i32]);
        let policy = self.fc.forward(policy);

        let policy = log_softmax(policy, 0).exp();

        (policy, value)
    }
}
