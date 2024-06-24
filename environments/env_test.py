from custom_grid_world_env import MazeGameEnv
import time

env = MazeGameEnv()

obs, _ = env.reset()
env.render()
done = False

time.sleep(100)

while not done:
    action = env.action_space.sample()
    obs, reward, done, _, _ = env.step(action)
    env.render()
    print(reward)


