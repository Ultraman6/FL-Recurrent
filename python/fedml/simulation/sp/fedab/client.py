from math import log2
from random import random
import time
import numpy as np

from python.fedml.simulation.sp.classical_vertical_fl.party_models import sigmoid


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
        self.tau_d = None  # 下行链路 average downlink transmitting time in sec
        self.tau_u = None  # 上行链路 average uplink transmitting time in sec
        self.tan_per_energy = 1  # 训练功率 暂时为1w
        self.com_per_energy = 1  # 通信功率 暂时为1w

        # 客户非价格质量属性
        self.tau_t = None  # 训练时延 average downlink transmitting time in sec
        self.tau = None     # 总时延 update time in sec
        # self.coe = None # 本地训练时间与通信延迟的系数

        # 客户本地训练结果
        self.loss = None
        # self.accuracy = None
        self.hisRate=1 # 历史完成率
        # 客户投标信息
        self.total_cost = None
        # self.flag = False # 是否被选中
        # 初始化模拟时延
        self.get_transmitting_times()
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
        I_start = time.time()  # 记录本轮开始时间
        self.model_trainer.train(self.local_training_data, self.device, self.args)
        I_end = time.time()  # 记录本轮结束时间
        self.tau_t = I_end - I_start
        self.loss = self.local_test(b_use_test_dataset=False)["test_loss"]
        return self.model_trainer.get_model_params()

    def local_test(self, b_use_test_dataset):
        if b_use_test_dataset:
            test_data = self.local_test_data
        else:
            test_data = self.local_training_data
        metrics = self.model_trainer.test(test_data, self.device, self.args)
        return metrics

    def get_transmitting_times(self): # 模拟计算通信时延
        fading = 10 ** -12.81 * self.distance ** -3.76  # large scaling fading in dB
        gain = 23  # uplink and downlink transmitting power in dBm
        noise_power = -107  # std of noise in dBm
        bandwidth = 15e3  # allocated bandwidth in Hz
        message_size = 5e3  # size of transmitting model parameters (both uplink and downlink) in bits
        average_rate = log2(1 + 10 ** (gain/10) * fading / 10 ** (noise_power/10))  # average distribution rate in bits/(sec*Hz)
        self.tau_d = message_size / (bandwidth * average_rate)
        self.tau_u = self.tau_d

    # 客户投标价格
    def getBid(self, flag): # flag=1表示诚实投标，flag=0表示虚假投标，(声明成本，(本地训练时间，通信延迟，历史完成率))
        totalCost = self.com_per_energy*(self.tau_d+self.tau_u)+self.local_sample_number*self.tan_per_energy
        if flag==1:
            return sigmoid(totalCost)
        else:
            totalCost=random.uniform(totalCost, totalCost * 10)
            return sigmoid(totalCost)
    def getQuality(self): # 客户投标非价格质量属性 [0,1]
        return (sigmoid(self.tau_t),sigmoid(self.tau_d+self.tau_u),self.hisRate)