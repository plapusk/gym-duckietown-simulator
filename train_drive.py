import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
import gym
import requests
import matplotlib.pyplot as plt
from gym_duckietown.envs import DuckietownEnv
from dqn import DQN
from agent import Agent
from wrapper import extract_state, PerceptionModule

training_maps = [

    "small_loop.yaml",
    "udem1.yaml",
    "straight_road.yaml",
    "loop_dyn_duckiebots.yaml",
    "loop_empty.yaml",
    "loop_obstacles.yaml",
    "loop_only_duckies.yaml",
    "regress_4way_adam.yaml",
    "regress_4way_drivable.yaml",
    "small_loop_cw.yaml",
    "small_loop_only_duckies.yaml",
    
    "4way.yaml",
    "zigzag_dists.yaml"
]


agent = Agent(input_dim=11, action_dim=4)  # vezi `extract_state`
perception_module = PerceptionModule("slow_color.pt", "fine_tune_myset.pth")

switch_every = 300  # change env every 50 episodes
warmup = 300

# Initial env setup with first map
current_map_index = 0
env = gym.make("Duckietown-udem1-v0", map_name=training_maps[current_map_index], camera_width=640, camera_height=640)
obs = env.reset()

MAX_STEPS_PER_EPISODE = 500

for episode in range(1000):  # example with 1500 epochs
#    if episode >= warmup and (episode - warmup) % switch_every == 0:
#        # Change environment map cyclically
#        current_map_index = (current_map_index + 1) % len(training_maps)
#        env.close()  # close previous env cleanly
#        env = gym.make("Duckietown-udem1-v0", map_name=training_maps[current_map_index], camera_width=640, camera_height=640)
#        print(f"Switched to new map: {training_maps[current_map_index]}")

    obs = env.reset()
    done = False
    total_reward = 0
    step = 0

    while not done and step < MAX_STEPS_PER_EPISODE:
        step += 1

        # Get perception outputs
        bboxes, seg_mask, _ = perception_module.predict(obs)

        # Get robot speed from env (linear and angular velocity)
        # Duckietown env exposes vehicle speed as a dict: env.cur_pos or env.cur_speed
        # Check your env's API, for example env.cur_pos might have 'linear' and 'angular'
        speed = None
        if hasattr(env, 'cur_pos'):
            speed = {
                'linear': env.cur_pos[2] if len(env.cur_pos) > 2 else 0.0,
                'angular': env.cur_pos[3] if len(env.cur_pos) > 3 else 0.0
            }
        elif hasattr(env, 'cur_speed'):
            speed = env.cur_speed  # or adapt accordingly

        # Extract state including speed info
        state = extract_state(bboxes, seg_mask, speed)

        # Select action
        action_id = agent.select_action(state)
        if action_id == 0:
            action = [0.0, -1.0]  # left
        elif action_id == 1:
            action = [0.44, 0.0]  # forward
        elif action_id == 2:
            action = [0.0, 1.0]   # right
        else:
            action = [0.0, 0.0]   # idle

        # Step environment
        obs, reward, done, _ = env.step(action)
        total_reward += reward

        # Next perception and state
        next_bboxes, next_seg_mask, _ = perception_module.predict(obs)

        # Get next speed
        next_speed = None
        if hasattr(env, 'cur_pos'):
            next_speed = {
                'linear': env.cur_pos[2] if len(env.cur_pos) > 2 else 0.0,
                'angular': env.cur_pos[3] if len(env.cur_pos) > 3 else 0.0
            }
        elif hasattr(env, 'cur_speed'):
            next_speed = env.cur_speed

        next_state = extract_state(next_bboxes, next_seg_mask, next_speed)

        # Store and train
        agent.store_transition(state, action_id, reward, next_state, done)
        agent.train()
        agent.decay_epsilon()

    agent.update_target()
    if episode % 50 == 0:
        print(f"Saving model at episode {episode}")
    torch.save(agent.model.state_dict(), 'agent_model_small_loop.pth')

    print(f"Ep {episode}, reward: {total_reward:.2f}")
    print(f"num steps: {step}, epsilon: {agent.epsilon:.3f}")

torch.save(agent.model.state_dict(), 'agent_model_straight_line.pth')