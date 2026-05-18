import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt
import time

# Параметры
EPISODES = 10000
ALPHA = 0.1
GAMMA = 0.99
EPSILON = 1.0
EPSILON_DECAY = 0.999
EPSILON_MIN = 0.01
WINDOW_SIZE = 100

np.random.seed(42)
random.seed(42)
env = gym.make('FrozenLake-v1', is_slippery=True)
metrics_table = {}

# Обучение агента (универсально для всех пяти алгоритмов)
def train_agent(algorithm_name):
    print(f"Обучение алгоритмом {algorithm_name}...")
    start_time = time.time()
    
    env.action_space.seed(42)
    epsilon = EPSILON
    win_rates = []
    wins_in_window = 0
    
    if algorithm_name == "Double Q-Learning":
        Q1 = np.zeros((env.observation_space.n, env.action_space.n))
        Q2 = np.zeros((env.observation_space.n, env.action_space.n))
    else:
        Q = np.zeros((env.observation_space.n, env.action_space.n))
        
    for episode in range(1, EPISODES + 1):
        state, _ = env.reset(seed=42) if episode == 1 else env.reset()
        done = False
        trajectory = [] # История шагов специально для Монте-Карло
        
        def choose_action(st):
            if random.uniform(0, 1) < epsilon:
                return env.action_space.sample()
            if algorithm_name == "Double Q-Learning":
                return np.argmax(Q1[st, :] + Q2[st, :])
            return np.argmax(Q[st, :])
            
        if algorithm_name == "SARSA":
            action = choose_action(state)
            
        while not done:
            if algorithm_name != "SARSA":
                action = choose_action(state)
                
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Если Монте-Карло, записываем шаг в историю и идем дальше
            if algorithm_name == "Monte Carlo":
                trajectory.append((state, action, reward))
            
            # Для всех методов, кроме Монте-Карло
            elif algorithm_name == "Q-Learning":
                best_next_action = np.argmax(Q[next_state, :])
                Q[state, action] += ALPHA * (reward + GAMMA * Q[next_state, best_next_action] - Q[state, action])
            elif algorithm_name == "SARSA":
                next_action = choose_action(next_state)
                Q[state, action] += ALPHA * (reward + GAMMA * Q[next_state, next_action] - Q[state, action])
                action = next_action 
            elif algorithm_name == "Expected SARSA":
                best_action = np.argmax(Q[next_state, :])
                expected_q = 0
                num_actions = env.action_space.n
                for a in range(num_actions):
                    prob = (1 - epsilon + epsilon/num_actions) if a == best_action else (epsilon/num_actions)
                    expected_q += prob * Q[next_state, a]
                Q[state, action] += ALPHA * (reward + GAMMA * expected_q - Q[state, action])
            elif algorithm_name == "Double Q-Learning":
                if random.uniform(0, 1) < 0.5:
                    best_next_action = np.argmax(Q1[next_state, :])
                    Q1[state, action] += ALPHA * (reward + GAMMA * Q2[next_state, best_next_action] - Q1[state, action])
                else:
                    best_next_action = np.argmax(Q2[next_state, :])
                    Q2[state, action] += ALPHA * (reward + GAMMA * Q1[next_state, best_next_action] - Q2[state, action])
            
            state = next_state
            if reward == 1.0:
                wins_in_window += 1
                
        # Для Монте-Карло, когда эпизод окончен
        if algorithm_name == "Monte Carlo":
            G = 0 # Суммарная награда (Return)
            # Идем по истории с конца в начало
            for s, a, r in reversed(trajectory):
                G = r + GAMMA * G
                Q[s, a] += ALPHA * (G - Q[s, a])
                
        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
        
        if episode % WINDOW_SIZE == 0:
            win_rates.append(wins_in_window / WINDOW_SIZE * 100)
            wins_in_window = 0
            
    end_time = time.time()
    
    metrics_table[algorithm_name] = {
        "Максимальный % побед": round(max(win_rates), 1),
        "Финальный % побед (в конце)": round(np.mean(win_rates[-10:]), 1),  # Среднее за последние 1000 эпизодов
        "Время обучения (сек)": round(end_time - start_time, 2)
    }

    return win_rates

# Запуск алгоритмов
q_win_rates = train_agent("Q-Learning")
sarsa_win_rates = train_agent("SARSA")
exp_sarsa_win_rates = train_agent("Expected SARSA")
double_q_win_rates = train_agent("Double Q-Learning")
mc_win_rates = train_agent("Monte Carlo")

# Таблица сравнения метрик алгоритмов
print(" Сравнительные метрики алгоритмов (FrozenLake 4x4)")
print("-"*70)
print(f"{'Алгоритм':<20} | {'Максимальный % побед':<15} | {'Финальный %':<15} | {'Время (сек)':<15}")
print("-" * 70)
for alg, data in metrics_table.items():
    print(f"{alg:<20} | {data['Максимальный % побед']:<15} | {data['Финальный % побед (в конце)']:<15} | {data['Время обучения (сек)']:<15}")
print("-"*70 + "\n")

# Построение графика
plt.figure(figsize=(12, 7))
episodes_x = np.arange(WINDOW_SIZE, EPISODES + 1, WINDOW_SIZE)

plt.plot(episodes_x, q_win_rates, label="Q-Learning", color="blue", alpha=0.7)
plt.plot(episodes_x, sarsa_win_rates, label="SARSA", color="red", alpha=0.7)
plt.plot(episodes_x, exp_sarsa_win_rates, label="Expected SARSA", color="green", alpha=0.7)
plt.plot(episodes_x, double_q_win_rates, label="Double Q-Learning", color="purple", linewidth=2)
plt.plot(episodes_x, mc_win_rates, label="Monte Carlo", color="orange", linewidth=2, linestyle='--')

plt.title("Сравнение алгоритмов обучения в стохастической среде FrozenLake 4×4)")
plt.xlabel("Количество эпизодов")
plt.ylabel("Процент побед (за последние 100 попыток)")
plt.legend()
plt.grid(True)
plt.show()