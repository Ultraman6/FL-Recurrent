from math import log2
from random import random

import numpy as np


class Client:
    def __init__(
        self, client_idx, local_training_data, local_test_data, local_sample_number, args, device, model_trainer,
    ):
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number
        self.args = args
        self.device = device
        self.model_trainer = model_trainer
        # 客户本地开销
        self.distance = np.random.rand()/2  # distance from AP in km
        # self.cos_max = cos_max
        self.tau_d = None  # 下行链路 average downlink transmitting time in sec
        self.tau_u = None  # 上行链路 average uplink transmitting time in sec
        self.tan_per_energy = 1  # 训练功率 暂时为1w
        self.com_per_energy = 1  # 通信功率 暂时为1w

        # 客户非价格质量属性
        self.tau_t = None  # 训练时延 average downlink transmitting time in sec
        self.tau = None     # 总时延 update time in sec
        self.tanTime = None  # 本次本地训练时间
        self.comTime = None  # 本次通信延迟
        self.hisRate = None  # 历史完成率
        self.coe = None # 本地训练时间与通信延迟的系数

        # 客户本地训练结果
        self.loss = None
        self.accuracy = None

        # 客户投标信息
        self.total_cost = None
    def update_local_dataset(self, client_idx, local_training_data, local_test_data, local_sample_number):
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number
        self.model_trainer.set_id(client_idx)

    def get_sample_number(self):
        return self.local_sample_number

    def train(self, w_global):
        self.model_trainer.set_model_params(w_global)
        self.model_trainer.train(self.local_training_data, self.device, self.args)
        weights = self.model_trainer.get_model_params()
        self.loss=self.model_trainer.train_loss
        self.accuracy=self.model_trainer.train_accuracy
        self.tanTime=self.model_trainer.train_time
        return weights

    def local_test(self, b_use_test_dataset):
        if b_use_test_dataset:
            test_data = self.local_test_data
        else:
            test_data = self.local_training_data
        metrics = self.model_trainer.test(test_data, self.device, self.args)
        return metrics
    def get_transmitting_times(self): # 模拟计算通信开销
        fading = 10 ** -12.81 * self.distance ** -3.76  # large scaling fading in dB
        gain = 23  # uplink and downlink transmitting power in dBm
        noise_power = -107  # std of noise in dBm
        bandwidth = 15e3  # allocated bandwidth in Hz
        message_size = 5e3  # size of transmitting model parameters (both uplink and downlink) in bits
        average_rate = log2(1 + 10 ** (gain/10) * fading / 10 ** (noise_power/10))  # average distribution rate in bits/(sec*Hz)
        self.tau_d = message_size / (bandwidth * average_rate)
        self.tau_u = self.tau_d

    # 客户投标返回
    def getBid(self, flag): # flag=1表示诚实投标，flag=0表示虚假投标，(声明成本，(本地训练时间，通信延迟，历史完成率))
        totalCost = self.com_per_energy*(self.tau_d+self.tau_u)+self.local_sample_number*self.tan_per_energy
        bid=[totalCost,(self.tanTime,self.comTime,self.hisRate)]
        if flag==1:
            return bid
        else:
            bid[0]=random.uniform(bid[0], bid[0] * 10)
            return bid