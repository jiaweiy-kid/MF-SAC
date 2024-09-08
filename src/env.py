import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import math
from utils import *

socd = SampleFromNormalDistribution(0.9, 0.1, 1, 0.85, 0.95)
sock = SampleFromNormalDistribution(9, 1, 1, 6, 12)

class Env:
    def __init__(self, data_path, d1_distribution, anxious_duration, EV_case, anx="low"):
        super(Env, self).__init__()
        self.origin_data = pd.read_excel(data_path, engine='openpyxl').to_numpy()  # 从excle文件中读取数据
        self.data = MaxMinNormalization(self.origin_data, 1)  # 进行maxmin归一化
        # data = GaussianNomalization(origin_data, 1)
        # data = DecimalNormalization(origin_data, 1)
        self.state_dim = 5
        self.start_point = 47  # 因为是使用前48个小时的电价，所以从第47个点开始
        self.t_index = random.randint(self.start_point, len(self.data) - 100)  # 随机返回一个47到数据末尾-100的值

        self.state_list = []
        self.state = np.zeros(shape=53, dtype='f8')  # 状态空间是48个电价加上6个其他值，所以是53
        self.done = False
        self.EV_case = EV_case
        self.t_a = int(self.data[self.t_index][0])  # 到达时间
        self.t = self.t_a
        self.soc = float(np.random.uniform(0, 0.95))  # 随机初始化一个soc
        place, mu, sigma = self.placeSelection(self.t_a)  # 返回位置，以及位置对应的离开概率的均值和方差
        self.place_info = [place, mu, sigma]
        self.flag = 0
        self.t_d, self.soc_d = self.depatureSim()
        self.anx = anx
        # self.k2 = SampleFromNormalDistribution(9, 1, 1, 6, 12)  # k2服从N(9，1)，并且在6-12之间
        self.k2 = sock
        self.t_x = self.anxiousTime()
        self.soc_x = self.anxiousGenerate()
        self.state = self.getState()

    def reset(self, t_tongb):
        self.start_point = 47
        self.t_index = t_tongb
        self.state_list = []
        for i in range(self.t_index - self.start_point, self.t_index + 1):
            self.state_list.append(self.data[i][1])

        self.state = np.zeros(shape=53, dtype='f8')

        self.done = False

        self.t_a = int(self.data[self.t_index][0])  # 是一个0-24之前的时间值
        self.t = self.t_a
        # self.soc = float(np.random.uniform(0, 0.95))
        self.soc = 0.5
        # if self.flag == 0:
        place, mu, sigma = self.placeSelection(self.t_a)
        self.place_info = [place, mu, sigma]
        self.t_d, self.soc_d = self.depatureSim()
        # self.t_d = 18
        self.soc_d = socd
        # self.k2 = SampleFromNormalDistribution(9, 1, 1, 6, 12)
        self.k2 = sock
        self.t_x = self.anxiousTime()
        self.soc_x = self.anxiousGenerate()
        self.state = self.getState()
        # self.flag = 1

        return self.state

    def step(self, action):
        socn, action = self.getSoc(action, mu=0.98)
        self.soc = socn
        reward_tuple = self.calculateReward(action, anx=self.anx)
        self.t_index += 1
        self.t = int(self.data[self.t_index][0])
        self.soc_x = self.anxiousGenerate()
        self.state = self.getState()
        if self.t == self.t_d:
            self.done = True
        return self.state, reward_tuple, action, self.done

    def placeSelection(self, t):  # 选择位置，家，公共场所和单位
        mu_arr = self.EV_case[0]  # 到达的概率
        sigma_arr = self.EV_case[1]
        home_prob = norm.pdf(t, mu_arr[0], sigma_arr[0])
        office_prob = norm.pdf(t, mu_arr[1], sigma_arr[1])
        public_prob = norm.pdf(t, mu_arr[2], sigma_arr[2])
        sum_prob = home_prob + office_prob + public_prob
        home_prob = home_prob / sum_prob
        office_prob = office_prob / sum_prob
        public_prob = public_prob / sum_prob   # 进行概率的归一化
        place_index = [0, 1, 2]  # 0-home, 1-office, 2-public
        place = int(np.random.choice(place_index, 1, [home_prob, office_prob, public_prob]))  # 依照概率选取一个位置
        mu_dep = self.EV_case[2]  # 离开的概率
        sigma_dep = self.EV_case[3]
        return place, mu_dep[place], sigma_dep[place]

    def depatureSim(self):
        k1 = 0.0
        t_d = 0
        mu = self.place_info[1]
        sigma = self.place_info[2]
        flag = 0
        while flag == 0:
            t_d = int(round(np.random.normal(mu, sigma)))  # 四舍五入后取整
            k1 = SampleFromNormalDistribution(0.9, 0.1, 1, 0.85, 0.95)
            if t_d > 24 or t_d < 1:
                continue
            if t_d < self.t_a and t_d + 24 - self.t_a >= 2:
                flag = 1

            elif t_d - self.t_a >= 2:
                flag = 1
        # if t_d > 24:
        #     t_d -= 24
        return t_d, k1

    def anxiousTime(self):  # 根据到达和离开的时间差来计算焦虑时间
        if self.t_d > self.t_a:
            if self.t_d - self.t_a <= 4:
                t_x = self.t_d - int(round(random.uniform(1, self.t_d - self.t_a - 1)))
            else:
                t_x = self.t_d - int(round(random.uniform(1, 4)))
        else:
            if self.t_d - self.t_a + 24 <= 4:
                t_x = self.t_d + 24 - int(round(random.uniform(1, self.t_d - self.t_a + 24 - 1)))
            else:
                t_x = self.t_d + 24 - int(round(random.uniform(1, 4)))
        if t_x > 24:
            t_x -= 24
        return t_x

    def anxiousGenerate(self):  # 公式4
        t_now = 24 if self.t % 24 == 0 else self.t % 24
        t_anx = 24 if self.t_x % 24 == 0 else self.t_x % 24
        t_dep = 24 if self.t_d % 24 == 0 else self.t_d % 24
        if t_now < self.t_a:
            t_now += 24
        if t_dep < self.t_a:
            t_dep += 24
        if t_anx < self.t_a:
            t_anx += 24

        if t_now < t_anx:
            return 0.0
        t_charge = t_dep - self.t_a
        tx_interval = t_now - self.t_a
        nominator = self.soc_d * (math.exp(-self.k2 * tx_interval / t_charge) - 1)
        denominator = math.exp(-self.k2) - 1
        soc_x = nominator / denominator
        return soc_x

    def getSoc(self, action, mu):
        action_actual = np.ndarray(shape=(1,), buffer=np.array([0.0]))
        soc = self.soc + (action * mu)
        if soc > 1:
            surplus = abs(soc - 1)
            gap = abs(1 - self.soc)
            action_actual = action * gap / (gap + surplus)
            soc = 1
        if soc < 0:
            surplus = abs(soc)
            gap = abs(self.soc)
            action_actual = action * gap / (gap + surplus)
            soc = 0
        if (soc > 0) & (soc < 1):
            action_actual = action
        return float(soc), action_actual

    def getState(self):
        self.state_list = []
        p_mean = 0.0
        for i in range(self.t_index - self.start_point, self.t_index + 1):
            # state_lst.append(0.1)
            # p_mean = p_mean + self.data[i][1]
            self.state_list.append(self.data[i][1])
        # p_mean = p_mean / 48
        # self.state_list.append(p_mean)
        info = [self.t_x / 24.0, self.t_d / 24.0, self.soc, self.soc_x, self.soc_d]
        for i in info:
            self.state_list.append(i)
        return np.array(self.state_list, dtype='f8')

    def calculateReward(self, action, kp=1.5, kx=1.7, kd=3.5, anx="low"):
        if anx == "high":
            kx = kx * 10
            kd = kd * 10
        price = self.origin_data[self.t_index][1]
        t_now = 24 if self.t % 24 == 0 else self.t % 24
        t_anx = 24 if self.t_x % 24 == 0 else self.t_x % 24
        t_dep = 24 if self.t_d % 24 == 0 else self.t_d % 24
        r = np.ndarray(shape=(1,), buffer=np.array([0.0]))
        r_anx = np.ndarray(shape=(1,), buffer=np.array([0.0]))
        r_price = np.ndarray(shape=(1,), buffer=np.array([0.0]))

        if t_now < self.t_a:
            t_now += 24
        if t_dep < self.t_a:
            t_dep += 24
        if t_anx < self.t_a:
            t_anx += 24

        if t_now < t_anx:
            r_price = -kp * action * price
            r = r_price
        elif (t_now >= t_anx) & (t_now < t_dep):
            # if t_now < t_dep:
            r_price = -kp * action * price
            r_anx = -kx * max((self.soc_x - self.soc), 0) ** 2  # price & TA
            # r_anx = 0.0  # price & TA
            r = r_price + r_anx
        # 加到倒数第二个时间上
        if t_now == t_dep - 1:
            temp_anx = -kd * max((self.soc_d - self.soc), 0) ** 2
            r_anx += temp_anx  # price & RA
            # r_anx = 0.0  # price & RA
            r += temp_anx
        return r, r_anx, r_price

    def simulation(self, agent, ev_id, t_tongb):  # , td_sim, ta_sim, tx_sim,
        self.reset(t_tongb)
        priceSim = pd.read_excel('..\\price\\testPrice.xlsx', engine='openpyxl', header=None)
        priceSim = priceSim.to_numpy()
        price = priceSim[48:215, 1]
        # priceSim = priceSim.to_numpy()
        # priceSim = GaussianNomalization(priceSim, 1)
        priceSim = MaxMinNormalization(priceSim, 1)
        # priceSim = DecimalNormalization(priceSim, 1)
        self.data = priceSim
        time = [i for i in range(1, 168)]
        td_sim = [9, 17, 32, 42, 57, 64, 79, 89, 105, 115, 131, 136, 154, 162, 167]  # depature time
        ta_sim = [1, 11, 19, 34, 43, 58, 66, 81, 91, 106, 116, 132, 137, 156, 163]  # start charging time
        tx_sim = [7, 14, 30, 38, 54, 62, 76, 88, 101, 112, 128, 133, 150, 160, 165]  # anxious time
        socd_sim = []
        k2_sim = []
        charge_interval = []

        for i in range(len(ta_sim)):
            k1 = SampleFromNormalDistribution(0.9, 0.1, 1, 0.85, 0.95)
            k2 = SampleFromNormalDistribution(9, 1, 1, 6, 12)
            socd_sim.append(k1)
            k2_sim.append(k2)
            interval = tx_sim[i] - ta_sim[i]
            t_charge = td_sim[i] - ta_sim[i]
            charge_interval.append(t_charge)

        soc_sim = [0.5]  # initial soc
        index = 47 + ta_sim[0]
        action_lst = []
        iter_times = 0
        self.soc = soc_sim[0]

        while iter_times < 15:
            self.t_index = index
            self.t_a = 24 if ta_sim[iter_times] % 24 == 0 else ta_sim[iter_times] % 24
            self.t_d = 24 if td_sim[iter_times] % 24 == 0 else td_sim[iter_times] % 24
            self.t_x = 24 if tx_sim[iter_times] % 24 == 0 else tx_sim[iter_times] % 24
            self.t = self.t_a
            self.soc_d = socd_sim[iter_times]
            self.k2 = k2_sim[iter_times]
            self.soc_x = self.anxiousGenerate()
            self.state = self.getState()
            for i in range(charge_interval[iter_times]):
                action = agent.select_action(self.state)
                next_state, _, action, _ = self.step(action, anx)
                action = action.item()
                action_lst.append(action)
                soc_sim.append(self.soc)
                self.state = next_state
                index += 1
            for k in range(len(td_sim)):
                if len(soc_sim) == td_sim[k]:
                    departFlag = 1
                    time_index = k
            if (departFlag == 1) & (time_index != 14):
                for j in range(td_sim[time_index], ta_sim[time_index + 1]):
                    action = -0.05
                    action_lst.append(action)
                    self.soc += action
                    soc_sim.append(self.soc)
                    index += 1
            if time_index == 14:
                break
            iter_times += 1

        max_value = np.max(price) + 20
        min_value = np.min(price) - 20
        price_norm = []
        for i in price:
            # price_norm.append((i - min_value) / (max_value - min_value))
            price_norm.append(i)

        fig, ax1 = plt.subplots(figsize=(10, 5))
        for i in range(len(ta_sim)):
            t_home = []
            t_office = []
            t_public = []
            t_driving = []
            rate = []
            if i < 14:
                for j in range(td_sim[i], ta_sim[i + 1] + 1):
                    t_driving.append(j)
            if i % 2 == 0:
                for j in range(ta_sim[i], td_sim[i] + 1):
                    if j != td_sim[len(td_sim) - 1]:
                        t_home.append(j)
            if (i % 2 == 1) & (i != 11) & (i != 13):
                for j in range(ta_sim[i], td_sim[i] + 1):
                    t_office.append(j)
            if (i == 11) | (i == 13):
                for j in range(ta_sim[i], td_sim[i] + 1):
                    t_public.append(j)

            # the y-axis of histogram
            if len(t_home) != 0:
                for j in t_home: rate.append(action_lst[j - 1])
                if i == 0:
                    ax1.bar(np.array(t_home), np.array(rate), color='lightskyblue', label='home')
                else:
                    ax1.bar(np.array(t_home), np.array(rate), color='lightskyblue')
            rate.clear()
            if len(t_driving) != 0:
                for j in t_driving: rate.append(action_lst[j - 1])
                if i == 0:
                    ax1.bar(np.array(t_driving), np.array(rate), color='tab:gray', label='driving')
                else:
                    ax1.bar(np.array(t_driving), np.array(rate), color='tab:gray')
            rate.clear()

            if len(t_public) != 0:
                for j in t_public: rate.append(action_lst[j - 1])
                if i == 11:
                    ax1.bar(np.array(t_public), np.array(rate), color='darksalmon', label='public')
                else:
                    ax1.bar(np.array(t_public), np.array(rate), color='darksalmon')
            rate.clear()
            if len(t_office) != 0:
                for j in t_office: rate.append(action_lst[j - 1])
                if i == 1:
                    ax1.bar(np.array(t_office), np.array(rate), color='goldenrod', label='office')
                else:
                    ax1.bar(np.array(t_office), np.array(rate), color='goldenrod')
            rate.clear()

        ax1.legend(loc=0)
        ax1.set(xlabel='Time(h)', ylabel='charging power')
        ax1.tick_params(labelsize=20)
        plt.rcParams.update({'font.size': 20})
        ax2 = ax1.twinx()
        ax2.plot(range(len(soc_sim)), np.array(price_norm), 'r', label='price')
        ax2.legend(loc='upper right')
        ax2.set(ylabel='Price')
        ax2.tick_params(labelsize=20)
        fig.savefig('../run/pic/{}_pic1.pdf'.format(ev_id))

        fig, sim = plt.subplots(figsize=(10, 5))
        sim.plot(range(len(soc_sim)), np.array(soc_sim), 'b', label='SoC')
        sim.set(ylabel='SoC')
        sim.legend(loc='upper left')
        axsim = sim.twinx()
        axsim.plot(range(len(soc_sim)), np.array(price_norm), 'r', label='price')
        axsim.legend(loc='upper right')
        axsim.set(xlabel='time', ylabel='Price')
        fig.savefig('../run/pic/pic2.pdf'.format(ev_id))
        plt.show()
