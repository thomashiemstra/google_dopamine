import os
import tensorflow as tf
from dopamine.agents.dqn import dqn_agent
from dopamine.atari import run_experiment
import matplotlib.pyplot as plt
from matplotlib import animation
tf.logging.set_verbosity(tf.logging.INFO)

slim = tf.contrib.slim

BASE_PATH = 'C:/Users/thomas/PycharmProjects/dopamine/'  # @param
GAME = 'Breakout'  # @param
# @title Create an agent based on DQN, but choosing actions randomly.

LOG_PATH = os.path.join(BASE_PATH, 'random_dqn', GAME)


class MyDQNAgent(dqn_agent.DQNAgent):

    def __init__(self, sess, num_actions):
        super(MyDQNAgent, self).__init__(sess, num_actions, tf_device='/gpu:0')

    def step(self, reward, observation):
        return super(MyDQNAgent, self).step(reward, observation)

    def _network_template(self, state):
        net = tf.cast(state, tf.float32)
        net = tf.div(net, 255.)
        net = slim.conv2d(net, 32, [8, 8], stride=4)
        net = slim.conv2d(net, 64, [4, 4], stride=2)
        net = slim.conv2d(net, 64, [3, 3], stride=1)
        net = slim.flatten(net)
        net = slim.fully_connected(net, 512, activation_fn=tf.nn.relu6)
        q_values = slim.fully_connected(net, self.num_actions, activation_fn=None)
        return self._get_network_type()(q_values)


def create_random_dqn_agent(sess, environment, summary_writer=None):
    return MyDQNAgent(sess, num_actions=environment.action_space.n)


# Create the runner class with this agent. We use very small numbers of steps
# to terminate quickly, as this is mostly meant for demonstrating how one can
# use the framework. We also explicitly terminate after 110 iterations (instead
# of the standard 200) to demonstrate the plotting of partial runs.
print("creating runner")
random_dqn_runner = run_experiment.Runner(BASE_PATH,
                                          create_random_dqn_agent,
                                          game_name=GAME
                                          )

# @title Train MyRandomDQNAgent.
print('Will train agent, please be patient, may be a while...')
random_dqn_runner.run_experiment()
print('Done training!')




def display_frames_as_gif(frames):
    """
    Displays a list of frames as a gif, with controls
    """
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi = 72)
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save('test.html', writer='html', fps=60)


def generate_results_html():
    action = random_dqn_runner._initialize_episode()
    env = random_dqn_runner._environment

    frames = []

    for _ in range(500):
      observation, reward, is_terminal = random_dqn_runner._run_one_step(action)
      action = random_dqn_runner._agent.step(reward, observation)
      frames.append(env.render(mode = 'rgb_array'))


    display_frames_as_gif(frames)