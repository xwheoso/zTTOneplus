#!/usr/bin/env python3

import warnings
import os
from threading import Thread
import time
import numpy as np
import random
import socket
import struct
import math
from collections import deque, defaultdict

# Suppress warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import tensorflow as tf
    try:
        from tensorflow.keras import backend as K
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense
        from tensorflow.keras.optimizers import Adam
    except ImportError:
        # Fallback for older TensorFlow versions
        from keras import backend as K
        from keras.models import Sequential
        from keras.layers import Dense
        from keras.optimizers import Adam
    import matplotlib.pyplot as plt

# === Configuration ===
PORT = 8702
experiment_time = 500
clock_change_time = 30
action_space = 9
target_fps = 60
target_temp = 65
beta = 2 

# TF Configuration
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.load_model = False
        self.training = 0
        self.state_size = state_size
        self.action_size = action_size
        self.actions = list(range(action_space))
        self.q_table = defaultdict(lambda: [0.0 for i in range(action_space)])
        
        # === 动作空间定义 (关键修改) ===
        self.clk_action_list = []
        
        # 定义三个档位 (低, 中, 高)
        # CPU 对应大核索引: 8 (低), 15 (中), 22 (高/满血)
        # GPU 对应索引: 2 (低), 5 (中), 8 (高/满血)
        self.cpu_tiers = [8, 15, 22] 
        self.gpu_tiers = [2, 5, 8]

        # 生成 3x3 = 9 种组合
        for i in range(3): # CPU tiers
            for j in range(3): # GPU tiers
                clk_action = (self.cpu_tiers[i], self.gpu_tiers[j])
                self.clk_action_list.append(clk_action)
        
        # 打印动作表以供检查
        print("Action Space Mapping (CPU_idx, GPU_idx):")
        for idx, act in enumerate(self.clk_action_list):
            print(f"Action {idx}: {act}")

        # Hyperparameters
        self.learning_rate = 0.05
        self.discount_factor = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.08
        self.epsilon_min = 0.0
        
        self.batch_size = 64
        self.train_start = 150
        
        self.q_max = 0
        self.avg_q_max = 0
        self.currentLoss = 0
        self.memory = deque(maxlen=500)
        
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()
        
        if self.load_model:
            self.model.load_weights("./save_model/model.h5")
            self.epsilon = 0.1

    def build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(24, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(self.action_size, activation='linear', kernel_initializer='he_uniform'))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def get_action(self, state):
        state = np.array([state])
        if np.random.rand() <= self.epsilon:
            # Exploration
            return random.randrange(self.action_size)
        else:
            # Exploitation
            q_value = self.model.predict(state)
            return np.argmax(q_value[0])

    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_model(self):
        self.training = 1
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_min

        mini_batch = random.sample(self.memory, self.batch_size)

        states = np.zeros((self.batch_size, self.state_size))
        next_states = np.zeros((self.batch_size, self.state_size))
        actions, rewards, dones = [], [], []

        for i in range(self.batch_size):
            states[i] = mini_batch[i][0]
            actions.append(mini_batch[i][1])
            rewards.append(mini_batch[i][2])
            next_states[i] = mini_batch[i][3]
            dones.append(mini_batch[i][4])

        target = self.model.predict(states)
        target_val = self.target_model.predict(next_states)

        for i in range(self.batch_size):
            # === 关键修改：Done 逻辑 ===
            if dones[i]:
                target[i][actions[i]] = rewards[i]
            else:
                # 只有未结束时才计算未来奖励
                target[i][actions[i]] = rewards[i] + self.discount_factor * (np.amax(target_val[i]))

        hist = self.model.fit(states, target, batch_size=self.batch_size, epochs=1, verbose=0)
        self.currentLoss = hist.history['loss'][0]
        self.training = 0


def get_reward(fps, power, target_fps, c_t, g_t, c_t_prev, g_t_prev, beta):
    v1 = 0
    v2 = 0
    
    # 简单的 FPS 奖励
    if fps >= target_fps:
        u = 1
    else:
        # FPS 不足时指数惩罚
        u = math.exp(0.1 * (fps - target_fps))

    # 温度惩罚 (GPU)
    if g_t > target_temp:
        v2 = 2 * (target_temp - g_t)
    
    # 温度恶化惩罚 (CPU)
    if c_t_prev < target_temp and c_t >= target_temp:
        v1 = -2

    # === 关键修改：防止除以零 ===
    safe_power = max(1, power) 
    
    return u + v1 + v2 + beta / safe_power

if __name__ == "__main__":
    
    # State Size = 7 (c_c, g_c, c_p, g_p, c_t, g_t, fps)
    agent = DQNAgent(state_size=7, action_size=9)
    
    t = 0
    ts = []
    fps_data = []
    power_data = []
    avg_q_max_data = []
    loss_data = []
    reward_tmp = []
    
    # 初始化变量
    c_c = 22 # 初始满频
    g_c = 8
    c_p, g_p = 0, 0
    c_t, g_t = 37.0, 37.0
    c_t_prev, g_t_prev = 37.0, 37.0
    fps = 60.0
    
    print(f"TCPServer Waiting on port {PORT}")
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(("", PORT))
    server_socket.listen(5)

    try:
        client_socket, address = server_socket.accept()
        print(f"Connected to client: {address}")
        
        # 绘图初始化
        plt.ion()
        fig = plt.figure(figsize=(12, 14))
        ax1 = fig.add_subplot(4, 1, 1)
        ax2 = fig.add_subplot(4, 1, 2)
        ax3 = fig.add_subplot(4, 1, 3)
        ax4 = fig.add_subplot(4, 1, 4)
        
        # 初始状态
        state = (c_c, g_c, c_p, g_p, c_t, g_t, fps)
        action = 8 # 默认最高

        while t < experiment_time:
            msg = client_socket.recv(1024).decode()
            if not msg:
                print('No received data')
                break
                
            state_tmp = msg.split(',')
            
            # 保存上一时刻温度
            c_t_prev = c_t
            g_t_prev = g_t
            
            # 解析数据
            c_c = int(state_tmp[0])
            g_c = int(state_tmp[1])
            c_p = int(state_tmp[2])
            g_p = int(state_tmp[3])
            c_t = float(state_tmp[4])
            g_t = float(state_tmp[5])
            fps = float(state_tmp[6])

            ts.append(t)
            fps_data.append(fps)
            power_data.append(c_p + g_p)

            next_state = (c_c, g_c, c_p, g_p, c_t, g_t, fps)
            
            # 记录 Q 值用于观察
            pred = agent.model.predict(np.array([next_state]))[0]
            agent.q_max += np.amax(pred)
            agent.avg_q_max = agent.q_max / (t + 1)
            avg_q_max_data.append(agent.avg_q_max)
            loss_data.append(agent.currentLoss)

            # 计算奖励
            safe_power = max(1, c_p + g_p)
            reward = get_reward(fps, safe_power, target_fps, c_t, g_t, c_t_prev, g_t_prev, beta)
            
            reward_tmp.append(reward)
            if len(reward_tmp) >= 300:
                reward_tmp.pop(0)

            # === 关键修改：Done 逻辑 ===
            done = 1 if t == experiment_time - 1 else 0
            
            # 存储经验
            agent.append_sample(state, action, reward, next_state, done)
            
            print(f'[{t}] FPS:{fps:.1f} Pwr:{safe_power} Temp:{c_t}/{g_t} R:{reward:.2f} Act:{action}')

            # 训练模型
            if len(agent.memory) >= agent.train_start:
                agent.train_model()

            # 更新状态
            state = next_state

            # === 决策逻辑 (混合策略) ===
            
            # 1. 过热保护 (强制使用 CPU 低档位)
            if c_t >= target_temp:
                print("Overheat! Cooling down...")
                # 随机选择动作 0, 1, 2 (CPU tier 0: index 8)
                action = random.randint(0, 2)
                
            # 2. 激进探索 (温度低但 FPS 不足 -> 尝试高频)
            elif target_temp - c_t >= 3 and fps < target_fps - 5:
                 if np.random.rand() <= 0.4:
                    print("Boosting performance!")
                    # 随机选择动作 6, 7, 8 (CPU tier 2: index 22)
                    action = random.randint(6, 8)
                 else:
                    action = agent.get_action(state)
            
            # 3. 正常 DQN 决策
            else:
                action = agent.get_action(state)

            # 解析动作 -> 频率索引
            c_c_next = agent.clk_action_list[action][0]
            g_c_next = agent.clk_action_list[action][1]

            # 发送回 Client
            send_msg = f"{c_c_next},{g_c_next}"
            client_socket.send(send_msg.encode())

            # 实时绘图
            if t % 5 == 0: # 减少绘图频率防止卡顿
                ax1.clear()
                ax1.plot(ts, fps_data, 'pink')
                ax1.axhline(y=target_fps, color='r')
                ax1.set_ylabel('FPS')
                ax1.set_title(f'FPS (Target {target_fps})')

                ax2.clear()
                ax2.plot(ts, power_data, 'blue')
                ax2.set_ylabel('Power (mW)')

                ax3.clear()
                ax3.plot(ts, avg_q_max_data, 'orange')
                ax3.set_ylabel('Avg Q-Value')

                ax4.clear()
                ax4.plot(ts, loss_data, 'black')
                ax4.set_ylabel('Loss')
                
                plt.pause(0.001)

            # 模型保存与参数调整
            if done:
                agent.update_target_model()
            
            if t > 0 and t % 500 == 0:
                agent.model.save_weights("./save_model/model.h5")
                print("[Model Saved]")

            t += 1

    finally:
        server_socket.close()
        plt.ioff()
        plt.show()