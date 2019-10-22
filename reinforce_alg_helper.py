import gym
import matplotlib.pyplot as plt
import numpy as np

def get_random_episode(env, gamma=0.99):
    """
        Corre un episodio muestreando el espacio de acción de forma aleatoria. Nos permite establecer un baseline
    """
    done = False
    env.reset()
    positions_of_cart = []
    velocities_of_cart = []
    angles_of_pole = []
    rotation_rates_of_pole = []
    actions = []
    rewards = []
    while not done:
        action = env.action_space.sample()
        actions.append(action)
        observation, reward, done, _ = env.step(action)
        (position_of_cart, velocity_of_cart, angle_of_pole, rotation_rate_of_pole) = observation
        positions_of_cart.append(position_of_cart)
        velocities_of_cart.append(velocity_of_cart)
        angles_of_pole.append(angle_of_pole)
        rotation_rates_of_pole.append(rotation_rate_of_pole)
        rewards.append(reward)
    states = np.array([positions_of_cart, velocities_of_cart, angles_of_pole, rotation_rates_of_pole]).T
    reward_sum = np.sum(rewards)
    return states, np.array(actions).reshape(-1,1), rewards, reward_sum, discount_rewards(rewards, gamma)
    # return positions_of_cart, velocities_of_cart, angles_of_pole, rotation_rates_of_pole, actions, rewards

def plot_episode(positions_of_cart, velocities_of_cart, angles_of_pole, rotation_rates_of_pole, actions, show_pos_thres=False):
    f, ax = plt.subplots(2, 2, figsize=(20,8))
    ax = ax.reshape(-1)
    ax[0].set_title('Positions of Cart')
    ax[0].plot(positions_of_cart, marker='x')
    if show_pos_thres:
        ax[0].hlines(2.4, xmin=0, xmax=len(positions_of_cart))
        ax[0].hlines(-2.4, xmin=0, xmax=len(positions_of_cart))
    ax[1].set_title('Velocities of Cart')
    ax[1].plot(velocities_of_cart)
    ax[1].scatter(range(len(actions)),velocities_of_cart, marker='x', c=actions.reshape(-1))
    ax[2].set_title('Angles of pole')
    ax[2].plot(np.array(angles_of_pole)*180/3.14159)
    ax[2].hlines(12, xmin=0, xmax=len(angles_of_pole))
    ax[2].hlines(-12, xmin=0, xmax=len(angles_of_pole))
    ax[3].set_title('Pole Velocity At Tip')
    ax[3].plot(rotation_rates_of_pole)
    ax[3].scatter(range(len(actions)),rotation_rates_of_pole, marker='x', c=actions.reshape(-1))
    plt.show()
    
def get_observations_stats(env, run_episode, N=5000, plot_it=True):
    states_dim = env.observation_space.shape[0]
    actions = np.empty(0).reshape(0,1)
    states = np.empty(0).reshape(0,states_dim)
    rewards = []
    for i in range(N):
        state, action, reward, reward_sum, discounted_rewards = run_episode(env)
        rewards.append(np.sum(reward))
        states = np.vstack([states, state])
        actions = np.vstack([actions, action])
    if plot_it:
        f, ax = plt.subplots(2,2, figsize=(20,10))
        ax = ax.flatten()
        titles = ['positions_of_cart', 'velocities_of_cart', 'angles_of_pole', 'rotation_rates_of_pole']
        for i in range(states_dim):
            ax[i].hist(states[:,i], 40)
            ax[i].set_title(titles[i])
        print('Media de las acciones [0,1]:', np.mean(actions))
        print('Media de rewards:', np.mean(rewards))
        
        plt.show()
    return np.mean(states[:,:4], axis=0), np.std(states[:,:4], axis=0)

import keras.layers as layers
from keras.models import Model
from keras.optimizers import Adam
import keras.backend as K
from keras.initializers import glorot_uniform
from keras.losses import categorical_crossentropy, binary_crossentropy

def get_policy_model(env, hidden_layer_neurons, lr):
    dimen = env.reset().shape
    num_actions = env.action_space.n
    inp = layers.Input(shape=dimen,name="input_x")
    adv = layers.Input(shape=[1], name="advantages")
    x = layers.Dense(hidden_layer_neurons, 
                     activation="relu", 
                     use_bias=False,
                     kernel_initializer=glorot_uniform(seed=42),
                     name="dense_1")(inp)
    out = layers.Dense(num_actions, 
                       activation="softmax", 
                       kernel_initializer=glorot_uniform(seed=42),
                       use_bias=False,
                       name="out")(x)

    def custom_loss(y_true, y_pred):
        # actual: 0 predict: 0 -> log(0 * (0 - 0) + (1 - 0) * (0 + 0)) = -inf
        # actual: 1 predict: 1 -> log(1 * (1 - 1) + (1 - 1) * (1 + 1)) = -inf
        # actual: 1 predict: 0 -> log(1 * (1 - 0) + (1 - 1) * (1 + 0)) = 0
        # actual: 0 predict: 1 -> log(0 * (0 - 1) + (1 - 0) * (0 + 1)) = 0
        log_lik = K.log(y_true * (y_true - y_pred) + (1 - y_true) * (y_true + y_pred))
        # log_lik = categorical_crossentropy(y_true, y_pred)
        # log_lik = y_true*K.log(y_pred) + (1 - y_true) * K.log((1-y_pred))
        return K.mean(log_lik * adv, keepdims=True)
        
    model_train = Model(inputs=[inp, adv], outputs=out)
    model_train.compile(loss=custom_loss, optimizer=Adam(lr))
    model_predict = Model(inputs=[inp], outputs=out)
    return model_train, model_predict

def discount_rewards(r, gamma=0.99):
    """Takes 1d float array of rewards and computes discounted reward
    e.g. f([1, 1, 1], 0.99) -> [2.9701, 1.99, 1]
    """
    prior = 0
    out = []
    for val in r:
        new_val = val + prior * gamma
        out.append(new_val)
        prior = new_val
    return np.array(out[::-1])

def score_model(model, env, num_tests, render=False):
    scores = []    
    dimen = model.input_shape[1]
    for num_test in range(num_tests):
        observation = env.reset()
        reward_sum = 0
        while True:
            if render:
                env.render()

            state = np.reshape(observation, [1, dimen])
            predict = model.predict([state])[0]
            action = np.argmax(predict)
            observation, reward, done, _ = env.step(action)
            reward_sum += reward
            if done:
                break
        scores.append(reward_sum)
    env.close()
    return np.mean(scores)

def run_episode(env, model, greedy=False, gamma=.99, get_probs=False):
    in_dimen = model.input_shape[1]
    out_dimen = model.output_shape[1]
    states = np.empty(0).reshape(0, in_dimen)
    actions = np.empty(0).reshape(0,1)
    rewards = np.empty(0).reshape(0,1)
    probs = []
    reward_sum = 0
    observation = env.reset()
    done = False
    while not done:
        # Append the observations to our batch
        state = np.reshape(observation, [1, in_dimen])

        predict = model.predict([state])[0]
        probs.append(predict)
        if greedy:
            action = np.argmax(predict)
        else:
            action = np.random.choice(range(len(predict)),p=predict)
        

        # Append the observations and outputs for learning
        states = np.vstack([states, state])
        actions = np.vstack([actions, action])

        # Determine the oucome of our action
        observation, reward, done, _ = env.step(action)
        reward_sum += reward
        rewards = np.vstack([rewards, reward])
    if get_probs:
        return states, actions, rewards, reward_sum, discount_rewards(rewards, gamma), probs
    else:
        return states, actions, rewards, reward_sum, discount_rewards(rewards, gamma)

def apply_baselines(discounted_rewards):
    discounted_rewards_unbiased = discounted_rewards - discounted_rewards.mean()
    discounted_rewards_normalized = discounted_rewards_unbiased / discounted_rewards_unbiased.std()
    return discounted_rewards_normalized.squeeze()

def actions_to_one_hot(actions, num_actions = 2):
    actions = actions.reshape(-1).astype(int)
    actions_train = np.zeros([len(actions), num_actions])
    actions_train[np.arange(len(actions)), actions] = 1
    return actions_train