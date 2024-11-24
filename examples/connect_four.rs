use alphazero_rust::arena::grouped_simulations_against_random;
use alphazero_rust::coach::{learn, CoachOptions};
use alphazero_rust::environment::{Environment, SimulationResult};
use alphazero_rust::mcts::{MCTSState, SearchOptions};
use alphazero_rust::net::NeuralNet;
use burn::backend::wgpu::{self, JitBackend, WgpuDevice, WgpuRuntime};
use burn::backend::{Autodiff, Wgpu};
use burn::data::dataloader::batcher::Batcher;
use burn::data::dataloader::DataLoaderBuilder;
use burn::data::dataset::Dataset;
use burn::module::AutodiffModule;
use burn::nn::{conv::Conv2d, Linear, LinearConfig, Relu};
use burn::optim::{AdamConfig, GradientsParams, Optimizer, SgdConfig};
use burn::prelude::*;
use burn::record::CompactRecorder;
use burn::tensor::activation::{log_softmax, softmax};
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::cast::ToElement;
use burn::tensor::loss::cross_entropy_with_logits;
use burn::train::metric::{AccuracyMetric, LossMetric};
use burn::train::ClassificationOutput;
use burn::train::TrainOutput;
use burn::train::ValidStep;
use burn::train::{LearnerBuilder, TrainStep};
use nn::loss::{CrossEntropyLoss, CrossEntropyLossConfig, MseLoss};
use nn::{BatchNorm, BatchNormConfig, PaddingConfig2d};
use rand::seq::SliceRandom;
use rand::thread_rng;
use std::fmt::{Display, Write};
use std::sync::Arc;
fn main() {
     
    let coach_options = Arc::new(CoachOptions {
        temp_threshold: 100,
        episodes: 10,
        iterations: 10,
        max_iterations_for_training: 50,
        update_threshold: 0.55,
    });
    let search_options = Arc::new(SearchOptions {
        cpuct: 1.25,
        rollout_count: 100,
    });

    let device = WgpuDevice::BestAvailable;
    let agent = Connect4Random;

    let env = Connect4;

    let (prospective_agent_wins, random_wins, _draws) = grouped_simulations_against_random(coach_options.episodes, search_options.clone(), 0.0, &agent, &device, &env);
    println!("{}% win against random", ((prospective_agent_wins as f64)/(prospective_agent_wins as f64 + random_wins as f64) * 100.0).round());

//    learn(coach_options, search_options, agent, &device, &env);
 /* 
    let search_options = Arc::new(SearchOptions {
        cpuct: 1.25,
        rollout_count: 200,
    });

    let device = WgpuDevice::BestAvailable;
    let agent = Connect4Random;
    let env = Connect4;

    use CellType::*;

    let mut state = Connect4State{
        turn: true,
        board: Board([
            [Empty, Empty, Empty, Empty, Empty, Empty, Empty],
            [Empty, Empty, Empty, Empty, Empty, Empty, Empty],
            [Empty, Empty, Empty, Empty, Empty, Empty, Empty],
            [Empty, Empty, Empty, Empty, Empty, Empty, Empty],
            [Empty, Empty, Empty, Empty, Empty, Empty, Empty],
            [Empty, Empty, Empty, Empty, Empty, Empty, Empty],

        ])
    };

    for i in 0..{
        let mut search = alphazero_rust::mcts::MCTS::new(search_options.clone(), &env);
        let policy = search.get_action_probability(&state, &agent, &device, 0.0);
        let policy : Vec<_> = policy
        .into_data()
        .iter::<f64>()
        .enumerate().collect();

        let best_move = policy.choose_weighted(&mut thread_rng(), |x|x.1).unwrap().0;

        search.write_graph(&state, &format!("dots/{i}.dot"));

        state = env.get_next_state(&state, best_move);
        state = state.to_active_perspective();

        println!("{state}");

        if env.get_simulation_result(&state).is_some(){
            break;
        }


    }
*/


}

#[derive(Clone)]
struct Connect4Random;


impl NeuralNet<7> for Connect4Random{
    type BurnBackend = Autodiff<JitBackend<WgpuRuntime, f32, i32>>;
    type State = Connect4State;
    
    fn predict(&self, perspective: &Self::State, device: &<Self::BurnBackend as Backend>::Device)->(Tensor<Self::BurnBackend, 1, Float>, Tensor<Self::BurnBackend, 1, Float>) {
        let weights : [f64; 7] = [1.0; 7];
        (Tensor::from_floats(weights, device), Tensor::from_floats([0.0], device))
    }
    
    fn train(&self, training_data: Vec<&(Self::State, Tensor<Self::BurnBackend, 1>, f64)>, device: &<Self::BurnBackend as Backend>::Device)->Self {
        self.clone()
    }

}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]

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

#[derive(Clone, PartialEq, Eq, Default, Debug)]
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

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct Connect4State {
    board: Board,
    turn: bool,
}

impl MCTSState for Connect4State{
    fn to_active_perspective(&self)->Self {
        if self.is_action_perspective(){
            return self.clone()
        }
        let board = self.board.invert();
        Self { board, turn: true }
    }

    fn is_action_perspective(&self)->bool {
        self.turn
    }
}

impl Display for Connect4State {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.turn {
            f.write_str("YOUR TURN \n")?;
        } else {
            f.write_str("OPP TURN \n")?;
        }
        for y in (0..6).rev() {
            for x in 0..7 {
                match self.board.0[y][x] {
                    CellType::Empty => f.write_char('_')?,
                    CellType::Enemy => f.write_char('O')?,
                    CellType::Hero => f.write_char('X')?,
                };
            }
            f.write_char('\n')?;
        }
        Ok(())
    }
}

struct Connect4;

impl Environment<7> for Connect4 {
    type State = Connect4State;
    type BurnBackend = Autodiff<JitBackend<WgpuRuntime, f32, i32>>;
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
            if new_state.board.0[y][action] == CellType::Empty {
                new_state.board.0[y][action] = if new_state.turn {
                    CellType::Hero
                } else {
                    CellType::Enemy
                };
                break;
            }
        }

        new_state.turn = !new_state.turn;

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
                            continue 'outer;
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
                            continue 'outer;
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
                            continue 'outer;
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
                            continue 'outer;
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

        let new_policy = policy.clone().flip([0]);

        syms.push((new_state, new_policy));

        syms
    }
}

#[derive(Config)]
pub struct TrainingConfig {
    #[config(default = 10)]
    pub num_epochs: usize,
    #[config(default = 64)]
    pub batch_size: usize,
    #[config(default = 8)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 1e-4)]
    pub lr: f64,
    pub optimizer: SgdConfig,
}

impl<B: Backend + AutodiffBackend> NeuralNet<7> for Connect4Net<B> {
    type BurnBackend = B;
    type State = Connect4State;

    fn predict(
        &self,
        perspective: &Self::State,
        device: &B::Device,
    ) -> (
        Tensor<Self::BurnBackend, 1, Float>,
        Tensor<Self::BurnBackend, 1, Float>,
    ) {
        assert!(perspective.is_action_perspective());
        let mut data = [[[0.0; 6]; 7]; 2];

        let board = if perspective.turn {
            &perspective.board
        } else {
            &perspective.board.invert()
        };

        for row in 0..6 {
            for col in 0..7 {
                match board.0[row][col] {
                    CellType::Empty => {}
                    CellType::Enemy => {
                        data[1][col][row] = 1.0;
                    }
                    CellType::Hero => {
                        data[0][col][row] = 1.0;
                    }
                }
            }
        }

        let tensor = Tensor::<B, 3>::from_data(data, device);

        assert!(tensor.shape() == Shape::new([2, 7, 6]));

        let (policy, value) = self.forward(tensor);

        (policy, value)
    }

    fn train(
        &self,
        training_data: Vec<&(Self::State, Tensor<Self::BurnBackend, 1>, f64)>,
        device: &B::Device,
    ) -> Self {
        let dataset: DataSet<(Connect4State, [f64; 7], f64), 7> = DataSet {
            training_data: training_data
                .iter()
                .map(|(state, policy, value)| {
                    //let policy = log_softmax(policy.clone(), 0);
                    let mut p = [0.0; 7];

                    for (i, v) in policy.to_data().iter::<f64>().enumerate() {
                        p[i] = v;
                    }

                    (state.clone(), p, *value)
                })
                .collect(),
        };

        let config_optimizer = SgdConfig::new();
        let config = TrainingConfig::new(config_optimizer);

        B::seed(config.seed);

        // Create the model and optimizer.
        let mut optim = config.optimizer.init();

        // Create the batcher.
        let batcher_train = Connect4Batcher::<B>::new(device.clone());

        // Create the dataloaders.
        let dataloader_train = DataLoaderBuilder::new(batcher_train)
            .batch_size(config.batch_size)
            .shuffle(config.seed)
            .num_workers(config.num_workers)
            .build(dataset);

        let mut model = self.clone();

        for _epoch in 1..config.num_epochs + 1 {
            for (_iteration, batch) in dataloader_train.iter().enumerate() {
                let outputs: (Vec<_>, Vec<_>) = batch
                    .states
                    .iter()
                    .map(|x| model.predict(x, device))
                    .unzip(); //collect predictions and seperate them into a list of policy estimates and a list of value estimates

                let (policies, values) = outputs;
                let policies = Tensor::cat(policies, 0).reshape([batch.batch_size, 7]);
                //reshape policies -> [batch size, target size]

                let target_policies =
                    Tensor::cat(batch.target_policies, 0).reshape([batch.batch_size, 7]);
                //reshape target policies -> [batch size, target size]

                let policy_loss = cross_entropy_with_logits(policies, target_policies);
                //calculate loss

                let values = Tensor::cat(values, 0).reshape([batch.batch_size, 1]);
                //reshape values -> [batch size, target size]
                let target_values =
                    Tensor::cat(batch.target_values, 0).reshape([batch.batch_size, 1]);
                //reshape target targets -> [batch size, target size]


                let value_loss =
                    MseLoss::new().forward(values, target_values, nn::loss::Reduction::Auto);
                //calculate value loss
                
                let loss = policy_loss.add(value_loss);
                //combine loss

                let loss = loss.backward();
                let grads = GradientsParams::from_grads(loss, &model);

                model = optim.step(config.lr, model, grads);

            }
        }
        model
    }
}

struct DataSet<State, const ACTION_SIZE: usize> {
    training_data: Vec<State>,
}

impl<State: std::marker::Sync + std::marker::Send + Clone, const ACTION_SIZE: usize> Dataset<State>
    for DataSet<State, ACTION_SIZE>
{
    fn get(&self, index: usize) -> Option<State> {
        self.training_data.get(index).cloned()
    }

    fn len(&self) -> usize {
        self.training_data.len()
    }
}


#[derive(Module, Debug)]
struct Connect4Net<B: Backend> {
    conv: ConvBlock<B>,
    conv2: ConvBlock<B>,
    fc: Linear<B>,
    fc2: Linear<B>,
    fc3: Linear<B>,
}

impl<B: Backend> Connect4Net<B> {
    fn new(device: &Device<B>) -> Self {
        Self {
            conv: ConvBlock::new([2, 128], [3, 3], device),
            conv2: ConvBlock::new([32, 128], [3, 3], device),
            fc: LinearConfig::new(6 * 7 * 32, 7).init(device),
            fc2: LinearConfig::new(2 * 6 * 7, 32).init(device),
            fc3: LinearConfig::new(32, 1).init(device),
        }
    }
    fn forward(&self, input: Tensor<B, 3>) -> (Tensor<B, 1>, Tensor<B, 1>) {
        let input = input.reshape([1, 2, 7, 6]).detach();
        let value = self.conv.forward(input.clone());

        let value = value.reshape([2 * 6 * 7_i32]);
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
/* 
#[derive(Module, Debug)]
struct ResBlock<B: Backend> {
    conv: ConvBlock<B>,
    conv2: ConvBlock<B>,
}
impl<B: Backend> ResBlock<B> {
    fn new(device: &Device<B>) -> Self {
        Self {
            conv: ConvBlock::new([128, 128], [3, 3], device),
            conv2: ConvBlock::new([128, 128], [3, 3], device),
        }
    }

    fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let residual = input.clone();
        let input = self.conv.forward(input);
        let input = self.conv2.forward_with_residual(input, residual);

        input
    }
}*/
 
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
    pub fn forward_with_residual(
        &self,
        input: Tensor<B, 4>,
        residual: Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        let x = self.conv.forward(input);
        let x = self.norm.forward(x);
        let x = x.add(residual);

        self.activation.forward(x)
    }
}
/* 
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
            fc: LinearConfig::new(6 * 7 * 32, 7).init(device),
            fc2: LinearConfig::new(2 * 6 * 7, 32).init(device),
            fc3: LinearConfig::new(32, 1).init(device),
        }
    }

    fn forward(&self, input: Tensor<B, 4>) -> (Tensor<B, 1>, Tensor<B, 1>) {
        let value = self.conv.forward(input.clone());

        let value = value.reshape([2 * 6 * 7_i32]);
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
*/
#[derive(Clone)]
pub struct Connect4Batcher<B: Backend> {
    device: B::Device,
}

impl<B: Backend> Connect4Batcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

#[derive(Clone, Debug)]
pub struct Connect4Batch<B: Backend> {
    pub states: Vec<Connect4State>,
    pub target_policies: Vec<Tensor<B, 1>>,
    pub target_values: Vec<Tensor<B, 1>>,
    pub batch_size: usize,
}

impl<B: Backend> Batcher<(Connect4State, [f64; 7], f64), Connect4Batch<B>> for Connect4Batcher<B> {
    fn batch(&self, items: Vec<(Connect4State, [f64; 7], f64)>) -> Connect4Batch<B> {
        let batch_size = items.len();
        let mut policies = vec![];
        let mut values = vec![];
        let mut states = vec![];

        for (state, policy, value) in items {
            states.push(state);
            policies.push(Tensor::from_data(policy, &self.device));
            values.push(Tensor::from_data([value], &self.device));
        }

        Connect4Batch {
            batch_size,
            states,
            target_policies: policies,
            target_values: values,
        }
    }
}
