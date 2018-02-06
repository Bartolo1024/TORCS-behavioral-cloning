from gym_torcs import TorcsEnv
from human_agent import HumanAgent
from mlp_agent import NeuralAgent as mlpAgent
from lstm_agent import NeuralAgent as lstmAgent
import numpy as np

episode_count = 1
max_steps = 100000
reward = 0
done = False
collect_data_mode = False
step = 0

# Generate a Torcs environment
env = TorcsEnv(vision=False, throttle=True, gear_change=False, brake=True)

if collect_data_mode:
    agent = HumanAgent(max_steps)
else:
    agent = mlpAgent(max_steps)
    #agent = lstmAgent(max_steps)


print("TORCS Experiment Start.")
for i in range(episode_count):
    print("Episode : " + str(i))

    if np.mod(i, 3) == 0:
        # Sometimes you need to relaunch TORCS because of the memory leak error
        ob = env.reset(relaunch=True)
    else:
        ob = env.reset()

    total_reward = 0.

    if step == max_steps and not done and collect_data_mode:
        agent.next_dataset()

    for j in range(max_steps):
        action = agent.act(ob, reward, done, step)

        ob, reward, done, _ = env.step(action)

        total_reward += reward

        step += 1
        if done:
            break

    print("TOTAL REWARD @ " + str(i) +" -th Episode  :  " + str(total_reward))
    print("Total Step: " + str(step))
    print("")

agent.end(step == max_steps)
env.end()  # This is for shutting down TORCS
print("Finish.")

