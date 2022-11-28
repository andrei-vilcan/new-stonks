import os
import torch
import numpy as np
import pickle
import time

from env import TradingEnv
from agent import TradingAgent


# Fill replay memory (initialize)
def fill_memory(env, agent, memory_fill_eps):
        print(f'Filling replay memory with {memory_fill_eps} simulations.')
        percent_complete = 0

        for i in range(memory_fill_eps):
            state = env.reset()
            done = False
            a = 0
            while not done:
                action = agent.select_action(state)
                next_state, reward, done, info = env.step(action)
                agent.replay_memory.store(state, action, next_state, reward, done)
                state = next_state

            if not i % round(memory_fill_eps / 100):
                status = str(percent_complete) + "% percent complete"
                print(status, end="\r")
                time.sleep(0.001)
                percent_complete += 1

# Training
def train(env, agent, train_eps, memory_fill_eps, batchsize, results_path, update_freq, model_name, render=False):
    fill_memory(env, agent, memory_fill_eps)
    print('States in memory: ', len(agent.replay_memory))
    print('Training model')

    step_count = 0

    reward_history = []
    epsilon_history = []

    best_score = -np.inf

    for ep_count in range(train_eps):
        epsilon_history.append(agent.epsilon)

        state = env.reset()
        done = False
        ep_score = 0

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.replay_memory.store(state, action, next_state, reward, done)
            agent.learn(batchsize)

            if step_count % update_freq == 0:
                agent.update_target()

            ep_score += reward
            state = next_state
            step_count += 1

        if render:
            if ep_count % show_every == 0:
                env.render()

        agent.update_epsilon()
        reward_history.append(ep_score)

        current_avg_score = np.mean(reward_history[-100:])

        print(f'Ep: {ep_count}, Total Steps: {step_count}, Ep: Score: {round(ep_score, 3)}, '
              f'Avg score: {round(current_avg_score, 3)} Epsilon: {round(epsilon_history[-1], 3)}')

        if current_avg_score >= best_score:
            agent.save('{}/dqn_model'.format(results_path))
            best_score = current_avg_score

    with open('{}/train_reward_history.pkl'.format(results_path), 'wb') as f:
        pickle.dump(reward_history, f)

    with open('{}/train_epsilon_history.pkl'.format(results_path), 'wb') as f:
        pickle.dump(epsilon_history, f)

# Testing
def test(env, agent, num_test_eps, seed, results_path, render=False):
        step_count = 0
        reward_history = []

        for ep in range(num_test_eps):
            score = 0
            done = False
            state = env.reset()
            while not done:
                if render:
                    env.render()

                action = agent.select_action(state)
                next_state, reward, done, _ = env.step(action)

                score += reward
                state = next_state
                step_count += 1

            reward_history.append(score)
            print('Ep: {}, Score: {}'.format(ep, score))

        with open('{}/test_reward_history_{}.pkl'.format(results_path, seed), 'wb') as f:
            pickle.dump(reward_history, f)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    show_every = 20
    filename = 'poop'

    # Replay mem params
    replay_memory_size = 50_000

    # Training params
    num_episodes = 5_000
    render = True
    update_freq = 50

    # Neural net params
    num_neurons = 256
    batch_size = 32
    learning_rate = 0.001
    discount = 0.99

    # Epsilon greedy decay params
    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 1-5e-3

    # Stock env params
    period = '650d'
    min_stock_size = 1000  # minimum # candles to accept stock into env
    candle_len = '1h'
    tx_fee = 0.0005
    trading_period = 250  # Length of game in number of candles
    hold_decay = 0.0005
    ticker_set = 'large_caps2'

    # Tickers
    all_tickers = {
        'large_caps': ['pg', 'vym', 'qqq', 'cost', 'sq', 'td', 'v', 'aapl', 'ac.to', 'tsla', 'cni', 'fts.to', 'bbby',
                       'cjt.to', 'nvda', 'amd', 'twtr', 'dia', 'spy', 'ezu', 'ewj', 'googl', 'fb', 'amzn', 'msft',
                       'nok', 'phia.as', 'sie.de', 'bidu', 'baba', '0700.hk', 'jpm', 'xom', 'rdsa.as', 'ko'],
        'large_caps2': ['vym', 'cost', 'td', 'aapl', 'ac.to', 'cni', 'jpm', 'xom', 'ko',
                        'cjt.to', 'amd', 'spy', 'amzn', 'nok', 'twtr'],
        'large_caps3': ['pg', 'vym', 'cost', 'td', 'aapl', 'ac.to', 'cni', 'cjt.to', 'ko', 'amzn', 'nok'],
        'forex': ['JPYUSD=X', 'CADUSD=X', 'JPYINR=X', 'INRGBP=X', 'GBPEUR=X', 'EURUSD=X', 'JPY=X',
                  'GBPUSD=X', 'AUDUSD=X', 'NZDUSD=X', 'EURJPY=X', 'EURSEK=X', 'EURCHF=X', 'HKD=X', 'SGD=X',
                  'CNY=X', 'ZAR=X'],
        'crypto': ['btc-usd', 'eth-usd', 'hex-usd', 'link-usd', 'ada-usd', 'xlm-usd', 'eth-btc', 'ada-eth',
                   'egld-usd', 'grt-usd', 'dot-usd', 'lrc-usd', 'reef-usd', 'xrp-usd'],
        'natural_resources': ['CL=F', 'GC=F', 'SI=F', 'LBS=F'],
        'misc': ['^TNX'],
        'mid_caps': ['cspr', 'asts', 'spce'],
        'tickers': ['amd']
    }

    # Create trading agent and environment
    env = TradingEnv(candle_len, period, all_tickers[ticker_set], batch_size=batch_size, tx_fee=tx_fee,
                     min_stock_size=min_stock_size, trading_period=trading_period, hold_decay=hold_decay)
    agent = TradingAgent(env, device, num_neurons, epsilon, epsilon_min, epsilon_decay,
                         replay_memory_size, batch_size, discount=discount, lr=learning_rate)

    # Run first round of training
    train(env, agent, num_episodes, replay_memory_size, batch_size, os.getcwd(), update_freq, filename, render=render)
    agent.save(filename)

    # Retrain model with reset epsilon
    agent = TradingAgent(env, device, num_neurons, epsilon, epsilon_min, epsilon_decay,
                         replay_memory_size, batch_size, discount=discount, lr=learning_rate)
    agent.load(filename)
    train(env, agent, num_episodes + 1_500, replay_memory_size, batch_size, os.getcwd(), update_freq, filename,
          render=render)
    agent.save(filename + '2')