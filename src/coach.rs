use crate::{
    arena::{grouped_simulations, grouped_simulations_against_random}, environment::{Environment, SimulationResult}, mcts::{MCTSState, SearchOptions, MCTS}, net::NeuralNet
};
use std::{collections::VecDeque, fmt::Display, io::Write, sync::Arc};

use burn::{prelude::*, tensor::backend::AutodiffBackend};

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
    State: Clone + std::hash::Hash + Eq + Display + MCTSState,
    BurnBackend: Backend,
>(
    coach_options: Arc<CoachOptions>,
    search_options: Arc<SearchOptions>,
    p1: &impl NeuralNet<ACTION_SIZE, State = State, BurnBackend = BurnBackend>,
    p2: &impl NeuralNet<ACTION_SIZE, State = State, BurnBackend = BurnBackend>,
    device: &Device<BurnBackend>,
    env: &impl Environment<ACTION_SIZE, State = State, BurnBackend = BurnBackend>,
) -> (Vec<(State, Tensor<BurnBackend, 1, Float>, f64)>, SimulationResult){
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
        let mut search = MCTS::new(search_options.clone(), env);
        let prob = search.get_action_probability(&s1, p1, device, temperature);

        assert!(s1.is_action_perspective());

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
            return (combine_training_examples(p1_training_examples, p2_training_examples, result), result)
        }

        s2 = env.get_next_state(&s2, best_move);

        if let Some(result) = env.get_simulation_result(&s2) {
            return (combine_training_examples(p2_training_examples, p1_training_examples, result), result.inverse())
        }

        let mut search = MCTS::new(search_options.clone(), env);
        let prob = search.get_action_probability(&s2, p2, device, temperature);

        assert!(s2.is_action_perspective());

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

        //println!("{s1}");

        if let Some(result) = env.get_simulation_result(&s1) {
            return (combine_training_examples(p1_training_examples, p2_training_examples, result), result)
        }

        s2 = env.get_next_state(&s2, best_move);

        if let Some(result) = env.get_simulation_result(&s2) {
            return (combine_training_examples(p2_training_examples, p1_training_examples, result), result.inverse())
        }
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
    State: Clone + std::hash::Hash + Eq + Display + MCTSState + Send,
    BurnBackend: Backend + AutodiffBackend,
>(
    coach_options: Arc<CoachOptions>,
    search_options: Arc<SearchOptions>,
    mut agent: impl NeuralNet<ACTION_SIZE, State = State, BurnBackend = BurnBackend>,
    device: &Device<BurnBackend>,
    env: &impl Environment<ACTION_SIZE, State = State, BurnBackend = BurnBackend>,
)->impl NeuralNet<ACTION_SIZE, State = State, BurnBackend = BurnBackend>{
    let mut total_training_examples: VecDeque<Vec<(State, Tensor<BurnBackend, 1>, f64)>> = VecDeque::new();

    for _iterations in 0..coach_options.iterations{
        let mut training_examples: Vec<(State, Tensor<BurnBackend, 1>, f64)> = vec![];

        for i in 0..coach_options.episodes{
            let (mut v, _) = execute_episode(coach_options.clone(), search_options.clone(), &agent, &agent, device, env);
            training_examples.append(&mut v);
            print!("{i} ");
            std::io::stdout().flush().unwrap();
        }
        println!();

        total_training_examples.push_back(training_examples);

        if total_training_examples.len() > coach_options.max_iterations_for_training{
            total_training_examples.pop_front();
        }

        let training_data: Vec<_> = total_training_examples.iter().flatten().collect();

        let prospective_agent = agent.train(training_data, device);
        
        println!("training over");
        
        let (prospective_agent_wins, agent_wins, _draws) = grouped_simulations(coach_options.episodes, search_options.clone(), 0.0, &prospective_agent, &agent, device, env);

        let non_draws = agent_wins as f64 + prospective_agent_wins as f64;

        if (non_draws!=0.0) && prospective_agent_wins as f64 / non_draws >= coach_options.update_threshold{

            let (prospective_agent_wins, random_wins, _draws) = grouped_simulations_against_random(coach_options.episodes, search_options.clone(), 0.0, &prospective_agent, device, env);
            println!("{}% win against random", ((prospective_agent_wins as f64)/(prospective_agent_wins as f64 + random_wins as f64) * 100.0).round());

            agent = prospective_agent;
        }
    }
    agent
}