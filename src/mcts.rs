use std::collections::HashMap;
use std::sync::Arc;

const EPS: f64 = 1e-8;

use burn::{prelude::*, tensor::cast::ToElement};
use rand::seq::SliceRandom;

use crate::{
    environment::{Environment, SimulationResult},
    net::NeuralNet,
};
pub struct MCTS<'a, const ACTION_SIZE: usize, State, BurnBackend: Backend> {
    arena: HashMap<State, NodeType<ACTION_SIZE, BurnBackend>>,
    env: &'a dyn Environment<ACTION_SIZE, State = State, BurnBackend = BurnBackend>,
    options: Arc<SearchOptions>,
}

pub struct SearchOptions {
    pub cpuct: f64,
    pub rollout_count: usize,
}

enum NodeType<const ACTION_SIZE: usize, BurnBackend: Backend> {
    Terminal(SimulationResult),
    Branch(Node<ACTION_SIZE, BurnBackend>),
}

struct Node<const ACTION_SIZE: usize, BurnBackend: Backend> {
    policy: Tensor<BurnBackend, 1, Float>,
    edges: [Option<(usize, Option<f64>)>; ACTION_SIZE],

    visits: usize,
    //children: [(usize, Rc<RefCell<NodeType<ACTION_SIZE>>>); ACTION_SIZE]
}

impl<'a, const ACTION_SIZE: usize, State: Clone + std::hash::Hash + Eq, BurnBackend: Backend>
    MCTS<'a, ACTION_SIZE, State, BurnBackend>
{
    pub fn new(
        options: Arc<SearchOptions>,
        env: &'a dyn Environment<ACTION_SIZE, State = State, BurnBackend = BurnBackend>,
    ) -> Self {
        Self {
            arena: HashMap::new(),
            env,
            options,
        }
    }


    pub fn get_action_probability(&mut self, state: &State, net: &impl NeuralNet<ACTION_SIZE, State = State, BurnBackend = BurnBackend>, temperature: f64)-> Tensor<BurnBackend, 1>{
        for _ in 0..self.options.rollout_count{
            println!("calling rollout");
            self.search(state, net);
        }
        println!("done rollout");


        let Some(NodeType::Branch(node)) = self.arena.get(state) else {unreachable!()};

        let counts : Vec<_> = node.edges.iter().map(|x|x.map(|x|x.0)).map(|x|x.unwrap_or(0)).collect();

        if temperature == 0.0{
            let max_counts = *counts.iter().max().unwrap();
            let best_counts : Vec<_> = counts
            .iter()
            .enumerate()
            .filter(|(_, x)|x == &&max_counts)
            .map(|(index, _)|index)
            .collect();
            
            let best_count = *best_counts.choose(&mut rand::thread_rng()).unwrap();

            let mut calls : [f64; ACTION_SIZE] = [0.0; ACTION_SIZE];
            calls[best_count] = 1.0;

            return Tensor::from_floats(calls, &Default::default())
        }

        let mut calls : [f64; ACTION_SIZE] = [0.0; ACTION_SIZE];

        for i in 0..ACTION_SIZE{
            calls[i] = (counts[i] as f64).powf(1.0 / temperature);
        }

        let calls_sum : f64 = calls.iter().sum();

        for i in 0..ACTION_SIZE{
            calls[i] = calls[i]/calls_sum;
        }

        Tensor::from_floats(calls, &Default::default())

    }
    
    pub fn search(
        &mut self,
        state: &State,
        net: &impl NeuralNet<ACTION_SIZE, State = State, BurnBackend = BurnBackend>,
    ) -> f64 {

        let node = match self.arena.get(state) {
            Some(NodeType::Terminal(game_result)) => return game_result.to_value(),
            None => {
                if let Some (game_result) = self.env.get_simulation_result(state){
                    self.arena.insert(state.clone(), NodeType::Terminal(game_result));
                    return game_result.to_value();
                }

                let (policy, score) = net.predict(state);
                let valid_actions = self.env.get_valid_actions(state);

                let mut policy = policy.mul(valid_actions.clone().float());

                let policy_sum = policy.clone().sum().into_scalar();

                if policy_sum.to_f32() > 0.0 {
                    policy = policy.div_scalar(policy_sum);
                } else {
                    eprintln!("All valid moves were masked, doing a workaround.");
                    policy = policy.add_scalar(policy_sum);
                    let policy_sum = policy.clone().sum().into_scalar();
                    policy = policy.div_scalar(policy_sum);
                }
                let mut edges = [None; ACTION_SIZE];

                for (i, valid) in valid_actions.into_data().iter().enumerate() {
                    if valid {
                        edges[i] = Some((0, None));
                    }
                }

                self.arena.insert(
                    state.clone(),
                    NodeType::Branch(Node {
                        visits: 0,
                        policy,
                        edges,
                    }),
                );
                return score;
            }
            Some(NodeType::Branch(node)) => node,
        };

        let mut cur_best = f64::NEG_INFINITY;
        let mut best_action = 0;

        for (i, (edge, policy)) in node
            .edges
            .iter()
            .zip(node.policy.to_data().iter::<f64>())
            .enumerate()
        {
            if let Some(edge) = edge {
                let edge_visits = edge.0 as f64;
                let visits = node.visits as f64;
                let exp = match edge.1{
                    Some(q_value) => {
                         q_value + self.options.cpuct * policy * visits.sqrt() / (1.0 + edge_visits)
                    }
                    None => self.options.cpuct * policy * (visits + EPS).sqrt(),
                };
                if exp > cur_best {
                    cur_best = exp;
                    best_action = i;
                }
            }
        }

        let next_state = self.env.get_next_state(&state, best_action);

        let score = self.search(&next_state, net);

        let Some(NodeType::Branch(node)) = self.arena.get_mut(state) else{
            unreachable!()
        };

        node.visits += 1;

        match &mut node.edges[best_action]{
            Some((edge_visits, q_value)) => {
                match q_value{
                    Some(q_value) => {
                        *q_value = (*edge_visits as f64 * *q_value + score) / (*edge_visits + 1) as f64;
                        *edge_visits += 1;
                    },
                    None => {
                        *q_value = Some(score);
                        *edge_visits = 1;
                    },
                }
            },
            None => unreachable!(),
        }


        score
    }
}
