import tensorflow as tf
import gym
import numpy as np
import argparse

layers = tf.keras.layers
tf.enable_eager_execution()

class RandomAgent(tf.keras.Model):
    def __init__(self, observation_space, action_space):
        super(RandomAgent, self).__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        return
    def act(self, state):
        action = self.action_space.sample()
        return action


class REINFORCEAgent(tf.keras.Model):
    def __init__(self, observation_space, action_space):
        super(REINFORCEAgent, self).__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.hidden_layer1 = layers.Dense(10,
                                          activation='relu',
                                          kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.hidden_layer2 = layers.Dense(self.action_space.n, 
                                          activation='relu',
                                          kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.logit_layer = layers.Dense(self.action_space.n, activation='linear',
                                          kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    def act(self, state):
        out = self.call(state)
        sample_action = tf.random.multinomial(out, 1)
        return sample_action[0, 0].numpy()
    def call(self, state):
        a = self.hidden_layer1(state)
        a = self.hidden_layer2(a)
        a = self.logit_layer(a)
        return a
        
    def train_one_episode(self, trajectory):
        episode_states  = [s for _, s, _, _, _ in trajectory]
        episode_action_indexes = [a for _, _, a, _ , _ in trajectory]
        episode_actions =[]
        for a in episode_action_indexes:
            one_hot = np.zeros(self.action_space.n)
            one_hot[a] = 1
            episode_actions.append(one_hot)
        episode_rewards = [r for _, _, _, _, r in trajectory]
        discounted_rewards = discount_and_normalize_rewards(episode_rewards)
        episode_states = np.vstack(episode_states)
        episode_actions = np.vstack(episode_actions)
        
        # Compute gradients
        with tf.GradientTape() as tape:
            action_logits = self(episode_states)
            neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=action_logits,labels=episode_actions)
            loss = tf.reduce_mean(neg_log_prob * discounted_rewards)
        grads = tape.gradient(loss, self.variables)

        # Update weights
        self.optimizer.apply_gradients(zip(grads, self.variables))


# Building functions
def build_env(env_name):
    """Returns configured gym env"""
    env = gym.make(env_name)
    return env

def build_agent(observation_space, action_space, agent_type):
    """Returns initialized agent"""
    if agent_type == 'Random':
        agent = RandomAgent(observation_space, action_space)
    elif agent_type == 'REINFORCE':
        agent = REINFORCEAgent(observation_space, action_space)
    return agent


def main():
    env = build_env('CartPole-v0')
    agent = build_agent(env.observation_space, 
                        env.action_space, FLAGS.agent_type)
    total_rewards = []
    running_avg_reward = 0.0
    for i in range(FLAGS.num_episodes):
        done = False
        obs = env.reset()
        steps = 0
        total_reward = 0.0
        trajectory = []
        if FLAGS.render and i+1 == FLAGS.num_episodes:
            frames = []
        while not done:
            action = agent.act(tf.expand_dims(obs, 0))
            obs_prev = obs
            obs, reward, done, info = env.step(action)
            traj_step = (steps, obs_prev, action, obs, reward)
            trajectory.append(traj_step)
            total_reward += reward
            steps += 1
            if FLAGS.render and i+1 == FLAGS.num_episodes:
                frames.append(env.render(mode = 'rgb_array'))
        # Render to GIF
        if FLAGS.render and i+1 == FLAGS.num_episodes:
            env.render()
            display_frames_as_gif(frames)
        #train
        running_avg_reward =  (total_reward + (i * running_avg_reward))/(i+1)
        if i % 10 == 0:
            print(running_avg_reward)
        agent.train_one_episode(trajectory)

if __name__ == '__main__':
    main()
    parser = argparse.ArgumentParser()
