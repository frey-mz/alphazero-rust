use std::sync::Arc;

use burn::prelude::*;

use crate::{
    environment::{Environment, SimulationResult},
    mcts::{SearchOptions, MCTS},
    net::NeuralNet,
};

pub fn simulate<
    'a,
    const ACTION_SIZE: usize,
    State: Clone + std::hash::Hash + Eq,
    BurnBackend: Backend,
>(
    search_options: Arc<SearchOptions>,
    temperature: f64,
    p1: &impl NeuralNet<ACTION_SIZE, State = State, BurnBackend = BurnBackend>,
    p2: &impl NeuralNet<ACTION_SIZE, State = State, BurnBackend = BurnBackend>,
    env: &impl Environment<ACTION_SIZE, State = State, BurnBackend = BurnBackend>,
) -> SimulationResult{
    let (mut s1, mut s2) = env.get_new_state();
    loop {
        let mut search = MCTS::new(search_options.clone(), env);
        let prob = search.get_action_probability(&s1, p1, temperature);
        let best_move = prob
            .into_data()
            .iter::<f64>()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap()
            .0;

        s1 = env.get_next_state(&s1, best_move);

        if let Some(result) = env.get_simulation_result(&s1){
            return result;
        }

        s2 = env.get_next_state(&s2, best_move);

        if let Some(result) = env.get_simulation_result(&s2){
            return result.inverse();
        }

        let mut search = MCTS::new(search_options.clone(), env);
        let prob = search.get_action_probability(&s2, p2, temperature);
        let best_move = prob
            .into_data()
            .iter::<f64>()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap()
            .0;

        s1 = env.get_next_state(&s1, best_move);

        if let Some(result) = env.get_simulation_result(&s1){
            return result;
        }

        s2 = env.get_next_state(&s2, best_move);


        if let Some(result) = env.get_simulation_result(&s2){
            return result.inverse();
        }
    }
}


pub fn grouped_simulations<
    'a,
    const ACTION_SIZE: usize,
    State: Clone + std::hash::Hash + Eq,
    BurnBackend: Backend,
>(
    num_simulations: usize,
    search_options: Arc<SearchOptions>,
    temperature: f64,
    p1: &impl NeuralNet<ACTION_SIZE, State = State, BurnBackend = BurnBackend>,
    p2: &impl NeuralNet<ACTION_SIZE, State = State, BurnBackend = BurnBackend>,
    env: &impl Environment<ACTION_SIZE, State = State, BurnBackend = BurnBackend>,
) -> (usize, usize, usize){
    let mut p1_wins = 0;
    let mut p2_wins = 0;
    let mut draws = 0;

    for _ in 0..num_simulations/2{
        match simulate(search_options.clone(), temperature, p1, p2, env){
            SimulationResult::Win => p1_wins+=1,
            SimulationResult::Loss => p2_wins+=1,
            SimulationResult::Draw => draws+=1,
        }
    }

    for _ in num_simulations/2..num_simulations{
        match simulate(search_options.clone(), temperature, p2, p1, env){
            SimulationResult::Win => p2_wins+=1,
            SimulationResult::Loss => p1_wins+=1,
            SimulationResult::Draw => draws+=1,
        }
    }

    (p1_wins, p2_wins, draws)
}
