 # Import the imports from evaluate.py

import hdf5 as h5
import numpy as np
from minihack.agent.polybeast.evaluate import (
        get_action,
        load_model,
        get_env_shortcut,
)


def collect_data(env, num_episodes, max_steps, save_dir, checkpoint_path, output_path, watch):

    # Load the env
    env = gym.make(
        env,
        savedir=save_dir,
        max_episode_steps=max_steps,
        observation_keys=obs_keys.split(",")
    )

    agent_env = get_env_shortcut(env)
    # Load the model
    model, hidden = load_model(agent_env, checkpoint_path)

    obs = env.reset()
    done = False
    steps = 0
    episodes = 0
    action = None
    while episodes < num_episodes:
        if done:
            obs = env.reset()
            done = False
            steps = 0
            episodes += 1

        # Get the action
        action, hidden = get_action(model, obs, hidden, watch=watch)

        # Take the action
        obs, reward, done, info = env.step(action)
        """Save a list to in hdf5 file"""
        with h5.File(output_path, 'a') as f:
            f.create_dataset("obs", data=obs)
            f.create_dataset("action", data=action)
            f.create_dataset("reward", data=reward)
            f.create_dataset("done", data=done)
            f.create_dataset("info", data=info)







