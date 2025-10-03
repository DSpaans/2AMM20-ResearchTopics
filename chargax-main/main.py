import jax
import jaxnasium as jym
import numpy as np
import optax
#test
from jaxnasium.algorithms import PPO

from chargax import Chargax, get_electricity_prices  # noqa: E402

if __name__ == "__main__":
    env = Chargax(
        elec_grid_buy_price=get_electricity_prices("2023_NL"),
        elec_grid_sell_price=get_electricity_prices("2023_NL") - 0.02,
        minutes_per_timestep=5,
        num_discretization_levels=10,
        elec_customer_sell_price=0.75,
    )
    env = jym.LogWrapper(env)
    rng = jax.random.PRNGKey(42)

    # RL Training with PPO
    num_envs = 12
    num_steps = 300
    total_timesteps = int(1e7)
    num_epochs = 4
    num_minibatches = 4
    minibatch_size = 900
    batch_size = 3600
    learning_rate = 2.5e-4
    gamma = 0.99
    gae_lambda = 0.95
    max_grad_norm = 100.0
    clip_coef = 0.2
    vf_clip_coef = 10.0
    entropy_coef = 0.01
    value_coef = 0.25
    agent = PPO(  # Not optimized, just a simple example
        learning_rate=learning_rate,
        anneal_learning_rate=True,
        gamma=gamma,
        gae_lambda=gae_lambda,
        max_grad_norm=max_grad_norm,
        clip_coef=clip_coef,
        clip_coef_vf=vf_clip_coef,
        ent_coef=entropy_coef,
        vf_coef=value_coef,
        total_timesteps=total_timesteps,
        num_envs=num_envs,
        num_steps=num_steps,
        num_minibatches=num_minibatches,
        num_epochs=num_epochs,
    )

    agent: PPO = agent.train(rng, env)

    results = agent.evaluate(rng, env, num_eval_episodes=25)
    print(f"Average reward over 25 evaluation episodes: {np.mean(results)}")
