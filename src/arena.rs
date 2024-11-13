use std::{fmt::Display, io::Write, sync::Arc};

use burn::prelude::*;
use rand::{seq::SliceRandom, thread_rng};

use crate::{
    environment::{Environment, SimulationResult},
    mcts::{MCTSState, SearchOptions, MCTS},
    net::NeuralNet,
};

pub fn simulate<
    'a,
    const ACTION_SIZE: usize,
    State: Clone + std::hash::Hash + Eq + Display + MCTSState,
    BurnBackend: Backend,
>(
    search_options: Arc<SearchOptions>,
    temperature: f64,
    p1: &impl NeuralNet<ACTION_SIZE, State = State, BurnBackend = BurnBackend>,
    p2: &impl NeuralNet<ACTION_SIZE, State = State, BurnBackend = BurnBackend>,
    device: &Device<BurnBackend>,
    env: &impl Environment<ACTION_SIZE, State = State, BurnBackend = BurnBackend>,
) -> SimulationResult {
    let (mut s1, mut s2) = env.get_new_state();
    loop {
        let mut search = MCTS::new(search_options.clone(), env);
        let prob = search.get_action_probability(&s1, p1, device, temperature);
        let best_move = prob
            .into_data()
            .iter::<f64>()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap()
            .0;

        s1 = env.get_next_state(&s1, best_move);
        //println!("{s1}");

        if let Some(result) = env.get_simulation_result(&s1) {
            return result;
        }

        s2 = env.get_next_state(&s2, best_move);

        if let Some(result) = env.get_simulation_result(&s2) {
            return result.inverse();
        }

        let mut search = MCTS::new(search_options.clone(), env);
        let prob = search.get_action_probability(&s2, p2, device, temperature);
        let best_move = prob
            .into_data()
            .iter::<f64>()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap()
            .0;

        s1 = env.get_next_state(&s1, best_move);
        //println!("{s1}");

        if let Some(result) = env.get_simulation_result(&s1) {
            return result;
        }

        s2 = env.get_next_state(&s2, best_move);

        if let Some(result) = env.get_simulation_result(&s2) {
            return result.inverse();
        }
    }
}

pub fn grouped_simulations<
    'a,
    const ACTION_SIZE: usize,
    State: Clone + std::hash::Hash + Eq + Display + MCTSState,
    BurnBackend: Backend,
>(
    num_simulations: usize,
    search_options: Arc<SearchOptions>,
    temperature: f64,
    p1: &impl NeuralNet<ACTION_SIZE, State = State, BurnBackend = BurnBackend>,
    p2: &impl NeuralNet<ACTION_SIZE, State = State, BurnBackend = BurnBackend>,
    device: &Device<BurnBackend>,
    env: &impl Environment<ACTION_SIZE, State = State, BurnBackend = BurnBackend>,
) -> (usize, usize, usize) {
    let mut p1_wins = 0;
    let mut p2_wins = 0;
    let mut draws = 0;

    println!("starting simulations");

    for _ in 0..num_simulations / 2 {
        let sim = simulate(search_options.clone(), temperature, p1, p2, device, env);
        match sim {
            SimulationResult::Win => p1_wins += 1,
            SimulationResult::Loss => p2_wins += 1,
            SimulationResult::Draw => draws += 1,
        }

        print!(" {:?}", sim);
        std::io::stdout().flush().unwrap();
    }

    for _ in num_simulations / 2..num_simulations {
        let sim = simulate(search_options.clone(), temperature, p2, p1, device, env);
        match sim {
            SimulationResult::Win => p2_wins += 1,
            SimulationResult::Loss => p1_wins += 1,
            SimulationResult::Draw => draws += 1,
        }
        print!(" {:?}", sim.inverse());
        std::io::stdout().flush().unwrap();
    }
    println!();
    println!("sims done {p1_wins} wins, {p2_wins} losses, {draws} draws");

    (p1_wins, p2_wins, draws)
}

pub fn simulate_against_random<
    'a,
    const ACTION_SIZE: usize,
    State: Clone + std::hash::Hash + Eq + Display + MCTSState,
    BurnBackend: Backend,
>(
    search_options: Arc<SearchOptions>,
    temperature: f64,
    p1: &impl NeuralNet<ACTION_SIZE, State = State, BurnBackend = BurnBackend>,
    device: &Device<BurnBackend>,
    env: &impl Environment<ACTION_SIZE, State = State, BurnBackend = BurnBackend>,
) -> SimulationResult {
    let (mut s1, mut s2) = env.get_new_state();
    loop {
        let mut search = MCTS::new(search_options.clone(), env);
        let prob = search.get_action_probability(&s1, p1, device, temperature);
        let best_move = prob
            .into_data()
            .iter::<f64>()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap()
            .0;

        s1 = env.get_next_state(&s1, best_move);
        //println!("{s1}");

        if let Some(result) = env.get_simulation_result(&s1) {
            return result;
        }

        s2 = env.get_next_state(&s2, best_move);

        if let Some(result) = env.get_simulation_result(&s2) {
            return result.inverse();
        }

        let random_actions: Vec<_> = env
            .get_valid_actions(&s2)
            .into_data()
            .iter::<bool>()
            .enumerate()
            .filter(|x| x.1)
            .map(|x| x.0)
            .collect();

        let best_move = *random_actions.choose(&mut thread_rng()).unwrap();

        s1 = env.get_next_state(&s1, best_move);

        if let Some(result) = env.get_simulation_result(&s1) {
            return result;
        }

        s2 = env.get_next_state(&s2, best_move);

        if let Some(result) = env.get_simulation_result(&s2) {
            return result.inverse();
        }
    }
}

pub fn grouped_simulations_against_random<
    'a,
    const ACTION_SIZE: usize,
    State: Clone + std::hash::Hash + Eq + Display + MCTSState,
    BurnBackend: Backend,
>(
    num_simulations: usize,
    search_options: Arc<SearchOptions>,
    temperature: f64,
    p1: &impl NeuralNet<ACTION_SIZE, State = State, BurnBackend = BurnBackend>,
    device: &Device<BurnBackend>,
    env: &impl Environment<ACTION_SIZE, State = State, BurnBackend = BurnBackend>,
) -> (usize, usize, usize) {
    let mut p1_wins = 0;
    let mut p2_wins = 0;
    let mut draws = 0;

    println!("starting simulations");

    for _ in 0..num_simulations / 2 {
        let sim = simulate_against_random(search_options.clone(), temperature, p1, device, env);
        match sim {
            SimulationResult::Win => p1_wins += 1,
            SimulationResult::Loss => p2_wins += 1,
            SimulationResult::Draw => draws += 1,
        }

        print!(" {:?}", sim);
        std::io::stdout().flush().unwrap();
    }

    for _ in num_simulations / 2..num_simulations {
        let sim = simulate_against_random(search_options.clone(), temperature, p1, device, env);
        match sim {
            SimulationResult::Win => p2_wins += 1,
            SimulationResult::Loss => p1_wins += 1,
            SimulationResult::Draw => draws += 1,
        }
        print!(" {:?}", sim.inverse());
        std::io::stdout().flush().unwrap();
    }
    println!("sims done {p1_wins}, {p2_wins}, {draws}");

    (p1_wins, p2_wins, draws)
}
