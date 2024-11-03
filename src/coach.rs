use crate::{
    arena::grouped_simulations, environment::{Environment, SimulationResult}, mcts::{SearchOptions, MCTS}, net::NeuralNet
};
use std::{collections::VecDeque, sync::Arc};

use burn::prelude::*;

pub struct CoachOptions {
    pub temp_threshold: usize,
    pub episodes: usize,
    pub iterations: usize,
    pub max_iterations_for_training: usize,
    pub update_threshold: f64,
}

pub fn execute_episode<
    'a,
    const ACTION_SIZE: usize,
    State: Clone + std::hash::Hash + Eq,
    BurnBackend: Backend,
>(
    coach_options: Arc<CoachOptions>,
    search_options: Arc<SearchOptions>,
    p1: &impl NeuralNet<ACTION_SIZE, State = State, BurnBackend = BurnBackend>,
    p2: &impl NeuralNet<ACTION_SIZE, State = State, BurnBackend = BurnBackend>,
    env: &impl Environment<ACTION_SIZE, State = State, BurnBackend = BurnBackend>,
) -> Vec<(State, Tensor<BurnBackend, 1, Float>, f64)>{
    let mut p1_training_examples = vec![];
    let mut p2_training_examples = vec![];

    let (mut s1, mut s2) = env.get_new_state();

    let mut episode_steps = 0;
    loop {
        episode_steps += 1;
        let temperature = if episode_steps < coach_options.temp_threshold {
            1.0
        } else {
            0.0
        };
        println!("getting move 1");
        let mut search = MCTS::new(search_options.clone(), env);
        let prob = search.get_action_probability(&s1, p1, temperature);
        let syms = env.get_symmetrical_policies(&s1, &prob);

        let best_move = prob
            .into_data()
            .iter::<f64>()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap()
            .0;

        for sym in syms {
            p1_training_examples.push(sym);
        }

        s1 = env.get_next_state(&s1, best_move);

        if let Some(result) = env.get_simulation_result(&s1) {
            return combine_training_examples(p1_training_examples, p2_training_examples, result)
        }

        s2 = env.get_next_state(&s2, best_move);

        if let Some(result) = env.get_simulation_result(&s2) {
            return combine_training_examples(p2_training_examples, p1_training_examples, result)
        }

        let mut search = MCTS::new(search_options.clone(), env);
        let prob = search.get_action_probability(&s2, p2, temperature);
        let syms = env.get_symmetrical_policies(&s2, &prob);

        let best_move = prob
            .into_data()
            .iter::<f64>()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap()
            .0;

        for sym in syms {
            p2_training_examples.push(sym);
        }

        s1 = env.get_next_state(&s1, best_move);

        if let Some(result) = env.get_simulation_result(&s1) {
            return combine_training_examples(p1_training_examples, p2_training_examples, result)
        }

        s2 = env.get_next_state(&s2, best_move);

        if let Some(result) = env.get_simulation_result(&s2) {
            return combine_training_examples(p2_training_examples, p1_training_examples, result)
        }

        println!("step!");
    }
}

fn combine_training_examples<BurnBackend: Backend, State>(
    p1_sims: Vec<(State, Tensor<BurnBackend, 1, Float>)>,
    p2_sims: Vec<(State, Tensor<BurnBackend, 1, Float>)>,
    result: SimulationResult,
) -> Vec<(State, Tensor<BurnBackend, 1, Float>, f64)>{
    let p1_result = result.to_value();
    let p1_iter = p1_sims.into_iter().map(|(a, b)|(a, b, p1_result));
    let p2_result = -result.to_value();
    let p2_iter = p2_sims.into_iter().map(|(a, b)|(a, b, p2_result));

    p1_iter.chain(p2_iter).collect()
}



pub fn learn<
    'a,
    const ACTION_SIZE: usize,
    State: Clone + std::hash::Hash + Eq,
    BurnBackend: Backend,
>(
    coach_options: Arc<CoachOptions>,
    search_options: Arc<SearchOptions>,
    mut agent: impl NeuralNet<ACTION_SIZE, State = State, BurnBackend = BurnBackend>,
    env: &impl Environment<ACTION_SIZE, State = State, BurnBackend = BurnBackend>,
)->impl NeuralNet<ACTION_SIZE, State = State, BurnBackend = BurnBackend>{
    let mut total_training_examples: VecDeque<Vec<(State, Tensor<BurnBackend, 1>, f64)>> = VecDeque::new();

    for _iterations in 0..coach_options.iterations{
        let training_examples: Vec<(State, Tensor<BurnBackend, 1>, f64)> = execute_episode(coach_options.clone(), search_options.clone(), &agent, &agent, env);
        println!("episode executed~");
        total_training_examples.push_back(training_examples);

        if total_training_examples.len() > coach_options.max_iterations_for_training{
            total_training_examples.pop_front();
        }

        let training_data: Vec<_> = total_training_examples.iter().flatten().collect();

        let mut prospective_agent = agent.clone();
        prospective_agent.train(training_data);
        
        let (prospective_agent_wins, agent_wins, _draws) = grouped_simulations(coach_options.episodes, search_options.clone(), 0.0, &prospective_agent, &agent, env);

        let non_draws = agent_wins as f64 + prospective_agent_wins as f64;

        if (non_draws!=0.0) && prospective_agent_wins as f64 / non_draws >= coach_options.update_threshold{
            agent = prospective_agent;
        }
    }
    agent
}