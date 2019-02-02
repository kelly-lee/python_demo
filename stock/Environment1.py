# -*- coding: utf-8 -*-
class Environment1:
    def __init__(self, data, history_t=10):
        self.data = data
        self.history_t = history_t
        self.reset()

    def reset(self):
        # 时间游标
        self.t = 0
        self.done = False
        # 盈利（卖出后的）
        self.profits = 0
        # 买点   价格
        self.positions = []
        # 当时持有的股票的当前损益金额
        self.position_value = 0
        # 过去90天股票价格的增加/减少值
        # 观察到的状态是一个长度为91的向量
        self.history = [0 for _ in range(self.history_t)]
        return [self.position_value] + self.history

    def step(self, act):
        #本次交易赢亏,act为持有和买入时,reward为0
        reward = 0
        # act = 0:stay,1:buy,2:sell
        if act == 1:
            # positions 记录 t时刻收盘价
            self.positions.append(self.data.iloc[self.t, :]['Close'])
        elif act == 2:  # sell
            if len(self.positions) == 0:
                reward = -1
            else:
                profits = 0
                for p in self.positions:
                    #当前价格 - 每个买点的价格
                    profits += (self.data.iloc[self.t, :]['Close'] - p)
                reward += profits

                self.profits += profits
                self.positions = []
        # set next time
        self.t += 1
        self.position_value = 0
        for p in self.positions:
            self.position_value += (self.data.iloc[self.t, :]['Close'] - p)
        self.history.pop(0)
        self.history.append(self.data.iloc[self.t, :]['Close'] - self.data.iloc[(self.t - 1), :]['Close'])
        # 一次卖的赢亏
        if reward > 0:
            reward = 1
        elif reward < 0:
            reward = -1
        print act, self.t, self.position_value, self.positions, self.profits
        return [self.position_value] + self.history, reward, self.done  # obs,reward,done


import numpy as np
import pandas_datareader.data as web

data = web.DataReader('GOOG', data_source='yahoo', start='1/1/2008', end='12/30/2018')
env = Environment1(data)
env.step(1)
env.step(1)
env.step(1)
env.step(2)
env.step(1)
env.step(2)

# for i in range(10):
#     act = np.random.randint(3)
#     obs, reward, done = env.step(act)
    # print act, obs, reward, done
