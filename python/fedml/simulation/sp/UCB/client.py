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

        self.distance = np.random.rand()/2  # distance from AP in km
        self.tau_max = tau_max
        self.tau_d = None  # average downlink transmitting time in sec
        self.tau_u = None  # average uplink transmitting time in sec
        self.tau = None  # update time in sec
        self.mu = None  # real reward
        self.b_t = 0  # Indicator if user was selected in time t - 1
        self.y_t = 0  # UCB estimate reward in iteration t
        self.z_t = 0  # number of times that the user was selected
        self.y_hat_t = 1  # truncated UCB estimate reward in iteration t
        self.r = None  # current user reward
        if c:
            self.c = c[user_id]  # fairness restriction
        self.virtual_queue = 0
        self.alpha = alpha  # availability probability (binomal distribution)

        self.get_transmitting_times()
        self.get_average_reward()
    def get_average_reward(self):
        avg_tau_lu = self.batch_size / ((self.user_id + 2) * 10)  # average computing time in sec
        avg_tau = min(self.tau_d + self.tau_d + avg_tau_lu, self.tau_max)
        self.mu = 1 - avg_tau / self.tau_max
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
        return weights

    def local_test(self, b_use_test_dataset):
        if b_use_test_dataset:
            test_data = self.local_test_data
        else:
            test_data = self.local_training_data
        metrics = self.model_trainer.test(test_data, self.device, self.args)
        return metrics
