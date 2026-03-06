import gymnasium as gym
import panda_gym
import time
from stable_baselines3 import HerReplayBuffer, SAC, DDPG
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
# --- NEW IMPORTS ---
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv

# --- 1. Configuration ---
ALGORITHM_TO_USE = "SAC"
ENV_ID = "PandaPush-v3"
MODEL_NAME = f"{ALGORITHM_TO_USE.lower()}_{ENV_ID.lower()}"
N_ENVS = 4  # --- CHANGE 1: Set number of environments ---

# --- 2. Create the Environment ---
# --- CHANGE 2: Create a vectorized env ---
# We MUST use DummyVecEnv because HerReplayBuffer requires it.
env = make_vec_env(
    ENV_ID,
    n_envs=N_ENVS,
    vec_env_cls=DummyVecEnv
)

# --- 3. Configure HER Replay Buffer ---
goal_selection_strategy = GoalSelectionStrategy.FUTURE
n_sampled_goal = 4

# --- 4. Instantiate the Model ---
if ALGORITHM_TO_USE == "SAC":
    model_class = SAC
else:
    model_class = DDPG

model = model_class(
    policy="MultiInputPolicy",
    env=env,
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
    
    # --- CHANGE 3: Tell the model to train more ---
    # We collect N_ENVS transitions per step, so we do N_ENVS gradient updates.
    # This forces the GPU to work harder.
    gradient_steps=N_ENVS,
    
    # --- CHANGE 4: Wait until buffer is filled a bit ---
    # Give it a good number of steps *per environment* before starting
    learning_starts=5000 
)

# --- 5. Train the Model ---
print(f"--- Starting training for {ALGORITHM_TO_USE} with {N_ENVS} environments ---")
start_time = time.time()
# Note: total_timesteps is now the *total* steps across all envs
model.learn(total_timesteps=100000)
end_time = time.time()

print(f"Training finished in {end_time - start_time:.2f} seconds.")
model.save(MODEL_NAME)

# --- 6. Enjoy the trained agent ---
print("Loading model and testing...")
del model
model = model_class.load(MODEL_NAME, env=env)

# We still test on a single environment
test_env = gym.make(ENV_ID, render_mode="human")
obs, info = test_env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = test_env.step(action)
    if terminated or truncated:
        obs, info = test_env.reset()

test_env.close()