# # WARNING, This file will not be functional until stable-baselines3 is compatible
# # with gymnasium. See https://github.com/DLR-RM/stable-baselines3/pull/780 for more information.
# import gymnasium as gym
# from stable_baselines3 import DDPG, HerReplayBuffer

# import panda_gym

# env = gym.make("PandaPush-v3")

# model = DDPG(policy="MultiInputPolicy", env=env, replay_buffer_class=HerReplayBuffer, verbose=1)

# model.learn(total_timesteps=100000)


import gymnasium as gym
import panda_gym
from stable_baselines3 import HerReplayBuffer
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
from sb3_contrib import TQC
from stable_baselines3.common.callbacks import BaseCallback
# --- IMPORTS FOR VEC ENV ---
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv

# --- 1. Create the Environments ---
N_ENVS = 4  # Number of environments to run in sequence

# Create a vectorized environment
# MUST use DummyVecEnv for HER
env = make_vec_env(
    "PandaPush-v3", 
    n_envs=N_ENVS, 
    vec_env_cls=DummyVecEnv
)

# We still need a SEPARATE, single env for rendering
eval_env = gym.make("PandaPush-v3", render_mode="human")


# --- 2. Configure HER Replay Buffer ---
goal_selection_strategy = GoalSelectionStrategy.FUTURE
n_sampled_goal = 4

# --- 3. Instantiate the TQC Model ---
model = TQC(
    policy="MultiInputPolicy",
    env=env,  # Use the vectorized env
    replay_buffer_class=HerReplayBuffer,
    replay_buffer_kwargs=dict(
        n_sampled_goal=n_sampled_goal,
        goal_selection_strategy=goal_selection_strategy,
    ),
    verbose=1,
    gamma=0.95,
    batch_size=2048,
    learning_rate=0.001,
    policy_kwargs=dict(net_arch=[512, 512, 512]),
    # We are collecting N_ENVS times as much data per step
    # so we should train more often.
    gradient_steps=N_ENVS,
    train_freq=(1, "step")
)

# --- 4. Train the Model ---
# We remove the callback for speed.
# You can add it back if you want, it will still work.
print("--- Training with 4 environments ---")
model.learn(total_timesteps=100000) # Removed callback

# Close the envs
env.close()
eval_env.close()

print("Training finished. Saving model.")
model.save("tqc_panda_push_vec")

# --- 5. Enjoy the trained agent ---
print("Loading model and testing.")
del model
model = TQC.load("tqc_panda_push_vec", env=env)

test_env = gym.make("PandaPush-v3", render_mode="human")
obs, info = test_env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = test_env.step(action)
    if terminated or truncated:
        obs, info = test_env.reset()

test_env.close()