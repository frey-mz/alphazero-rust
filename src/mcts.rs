use std::ops::Neg;
use std::sync::Arc;
use std::{collections::HashMap, fmt::Display};

const EPS: f64 = 1e-8;

use burn::{prelude::*, tensor::cast::ToElement};
use petgraph::graph::{DiGraph, NodeIndex};
use rand::seq::SliceRandom;

use crate::{
    environment::{Environment, SimulationResult},
    net::NeuralNet,
};

pub trait MCTSState {
    fn to_active_perspective(&self) -> Self;
    fn is_action_perspective(&self) -> bool;
}
pub struct MCTS<'a, const ACTION_SIZE: usize, State, BurnBackend: Backend> {
    arena: HashMap<State, NodeType<ACTION_SIZE>>,
    env: &'a dyn Environment<ACTION_SIZE, State = State, BurnBackend = BurnBackend>,
    options: Arc<SearchOptions>,
}

pub struct SearchOptions {
    pub cpuct: f64,
    pub rollout_count: usize,
}

enum NodeType<const ACTION_SIZE: usize> {
    Terminal(SimulationResult),
    Branch(Node<ACTION_SIZE>),
}

struct Node<const ACTION_SIZE: usize> {
    policy: [f64; ACTION_SIZE],
    valids: [bool; ACTION_SIZE],
    edges: [MCTSEdge; ACTION_SIZE], //visits, value
    visits: usize,
}

impl<
        'a,
        const ACTION_SIZE: usize,
        State: Clone + std::hash::Hash + Eq + Display + MCTSState,
        BurnBackend: Backend,
    > MCTS<'a, ACTION_SIZE, State, BurnBackend>
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

    pub fn get_action_probability(
        &mut self,
        state: &State,
        net: &impl NeuralNet<ACTION_SIZE, State = State, BurnBackend = BurnBackend>,
        device: &Device<BurnBackend>,
        temperature: f64,
    ) -> Tensor<BurnBackend, 1> {
        for _ in 0..self.options.rollout_count {
            self.search(state, net, device);
        }

        let Some(NodeType::Branch(node)) = self.arena.get(state) else {
            unreachable!()
        };

        if temperature == 0.0 {
            let max_counts = node.edges.iter().map(|x| x.visits).max().unwrap();
            let best_counts: Vec<_> = node
                .edges
                .iter()
                .map(|x| x.visits)
                .enumerate()
                .filter(|(_, x)| x == &max_counts)
                .map(|(index, _)| index)
                .collect();

            let best_count = *best_counts.choose(&mut rand::thread_rng()).unwrap();

            let mut calls: [f64; ACTION_SIZE] = [0.0; ACTION_SIZE];
            calls[best_count] = 1.0;

            return Tensor::from_floats(calls, &Default::default());
        }

        let mut calls: [f64; ACTION_SIZE] = [0.0; ACTION_SIZE];

        for i in 0..ACTION_SIZE {
            calls[i] = (node.edges[i].visits as f64).powf(1.0 / temperature);
        }

        let calls_sum: f64 = calls.iter().sum();

        for i in 0..ACTION_SIZE {
            calls[i] = calls[i] / calls_sum;
        }

        Tensor::from_floats(calls, &Default::default())
    }

    fn search_and_backprop(
        &mut self,
        state: &State,
        action: usize,
        net: &impl NeuralNet<ACTION_SIZE, State = State, BurnBackend = BurnBackend>,
        device: &Device<BurnBackend>,
    ) -> f64 {
        let next_state = self.env.get_next_state(&state, action);
        let val = self.search(&next_state, net, device);
        let Some(NodeType::Branch(node)) = self.arena.get_mut(&state) else {
            panic!()
        };

        node.edges[action].q_value =
            (node.edges[action].visits as f64 * node.edges[action].q_value + val)
                / (node.edges[action].visits as f64 + 1.0);

        node.edges[action].visits += 1;
        node.visits += 1;
        val
    }

    pub fn search(
        &mut self,
        state: &State,
        net: &impl NeuralNet<ACTION_SIZE, State = State, BurnBackend = BurnBackend>,
        device: &Device<BurnBackend>,
    ) -> f64 {
        //println!("{}", state);

        match self.arena.get(state) {
            None => {
                //
                if let Some(simulation_result) = self.env.get_simulation_result(state) {
                    self.arena
                        .insert(state.clone(), NodeType::Terminal(simulation_result));
                    return simulation_result.to_value();
                }

                let is_active = state.is_action_perspective();

                let (policy_tensor, score) = net.predict(&state.to_active_perspective(), device);
                let mut score = score.into_scalar().to_f64();

                if !is_active {
                    score = score.neg();
                }

                let valid_actions = self.env.get_valid_actions(state);

                let mut policy_tensor = policy_tensor.mul(valid_actions.clone().float());

                let policy_tensor_sum = policy_tensor.clone().sum().into_scalar();

                if policy_tensor_sum.to_f32() > 0.0 {
                    policy_tensor = policy_tensor.div_scalar(policy_tensor_sum);
                } else {
                    eprintln!("All valid moves were masked, doing a workaround.");
                    policy_tensor = valid_actions.clone().float();
                    let policy_tensor_sum = policy_tensor.clone().sum().into_scalar();
                    policy_tensor = policy_tensor.div_scalar(policy_tensor_sum);
                }
                let mut valids = [false; ACTION_SIZE];

                for (index, valid) in valid_actions.into_data().iter::<bool>().enumerate() {
                    valids[index] = valid
                }
                let mut policy = [0.0; ACTION_SIZE];

                for (index, q) in policy_tensor.into_data().iter::<f64>().enumerate() {
                    policy[index] = q
                }

                self.arena.insert(
                    state.clone(),
                    NodeType::Branch(Node {
                        policy,
                        valids,
                        edges: [MCTSEdge::default(); ACTION_SIZE],
                        visits: 0,
                    }),
                );

                score
            }
            Some(NodeType::Terminal(simulation_result)) => simulation_result.to_value(),
            Some(NodeType::Branch(node)) => {
                let mut cur_best = f64::NEG_INFINITY;
                let mut best_action = 0;

                for action in 0..ACTION_SIZE {
                    if !node.valids[action] {
                        continue;
                    }

                    let visits = node.visits as f64;
                    let policy = node.policy[action];
                    let exp = if node.edges[action].visits != 0 {
                        node.edges[action].q_value
                            + self.options.cpuct * policy * visits.sqrt() / (1.0 + node.edges[action].visits as f64)
                    } else {
                        self.options.cpuct * policy * (visits + EPS).sqrt()
                    };
                    if exp > cur_best {
                        cur_best = exp;
                        best_action = action;
                    }
                }
                self.search_and_backprop(state, best_action, net, device)
            }
        }
    }

    pub fn write_graph(&self, state: &State, path: &str) {
        let mut graph = DiGraph::new();
        self.create_petgraph_node(state, &mut graph);
        let dot = petgraph::dot::Dot::new(&graph);

        println!("graph genned, total states in hash: {}", self.arena.len());

        std::fs::write(path, dot.to_string()).expect("Unable to write file");
    }

    pub fn create_petgraph_node(
        &self,
        state: &State,
        graph: &mut DiGraph<State, MCTSEdge>,
    ) -> Option<NodeIndex> {
        //let node = graph.add_node(state.clone());
        match self.arena.get(state) {
            None => None,
            Some(NodeType::Terminal(_simulation_result)) => {
                println!("added terminal state");

                let index = graph.add_node(state.clone());

                Some(index)
            }
            Some(NodeType::Branch(node)) => {
                println!("added branch state");

                let index = graph.add_node(state.clone());

                for action in 0..ACTION_SIZE {
                    let new_state = self.env.get_next_state(state, action);
                    if let Some(new_index) = self.create_petgraph_node(&new_state, graph) {
                        graph.add_edge(index, new_index, node.edges[action]);
                    }
                }

                Some(index)
            }
        }
    }
}
#[derive(Clone, Copy, Default)]
pub struct MCTSEdge {
    visits: usize,
    q_value: f64,
}

impl Display for MCTSEdge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!(
            "visits: {} q_value: {}",
            self.visits, self.q_value
        ))?;
        Ok(())
    }
}
