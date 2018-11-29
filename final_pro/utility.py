
class Config:
    def __init__(self, num_cluster=1, num_epoch=100, debug=False, bandwidth=0.1):
        self.num_cluster = num_cluster
        self.num_epoch = num_epoch
        self.debug = debug
        self.bandwidth = bandwidth
