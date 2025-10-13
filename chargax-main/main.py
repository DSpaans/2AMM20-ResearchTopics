import jax
import jaxnasium as jym
import numpy as np
import optax
from jaxnasium.algorithms import PPO

from chargax import Chargax, get_electricity_prices  # noqa: E402

if __name__ == "__main__":
    env = Chargax(
        elec_grid_buy_price=get_electricity_prices("2023_NL"),
        elec_grid_sell_price=get_electricity_prices("2023_NL") - 0.02,
        
        minutes_per_timestep=5,
        num_discretization_levels=10,
        num_chargers=16,
        num_dc_groups=10,
        elec_customer_sell_price=0.75,
        
        capacity_exceeded_alpha = 0.0,
        charged_satisfaction_alpha = 0.0,
        time_satisfaction_alpha = 0.0,
        rejected_customers_alpha = 0.0,
        battery_degredation_alpha = 0.0,
    )
    env = jym.LogWrapper(env)
    rng = jax.random.PRNGKey(42)

    # RL Training with PPO
    total_timesteps = 1e7
    gamma = 0.99
    gae_lambda = 0.95
    max_grad_norm = 100.0
    
    clip_coef = 0.2
    clip_coef_vf = 10.0
    ent_coef = 0.01
    vf_coef = 0.25
    
    num_minibatches = 4
    num_envs = 12
    num_steps = 300

    num_epochs = 4
    num_training_iterations = (total_timesteps // num_steps // num_envs) * num_epochs
    
    optimizer = optax.adam(
        learning_rate=optax.linear_schedule(2.5e-3, 2.5e-5, num_training_iterations)
    )
    
    agent = PPO(  # Not optimized, just a simple example
        
        num_steps=num_steps,
        num_envs=num_envs,
        num_epochs=num_epochs,
        total_timesteps=total_timesteps,
        optimizer=optimizer,
        
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_coef=clip_coef,
        clip_coef_vf=clip_coef_vf,
        max_grad_norm=max_grad_norm,
        ent_coef=ent_coef,
        vf_coef=vf_coef,

        num_minibatches=num_minibatches,
        
    )

    agent: PPO = agent.train(rng, env)

    results = agent.evaluate(rng, env, num_eval_episodes=25)
    print(f"Average reward over 25 evaluation episodes: {np.mean(results)}")