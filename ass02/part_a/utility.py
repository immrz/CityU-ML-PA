import numpy as np


class ParamDiffStore:
    def __init__(self, len_of_mem=10, tol=1e-5):
        self.memory = [10000] * len_of_mem
        self.counter = 0
        self.prev_param = None  # type: list[np.ndarray]
        self.tol = tol

    def is_saturate(self, new_param):
        if self.prev_param is None:
            self.prev_param = new_param
            return False
        else:
            diff = 0.
            for i in range(len(new_param)):
                a = new_param[i].flatten()
                b = self.prev_param[i].flatten()
                diff += np.linalg.norm(a - b, ord=2) / a.shape[0]

            self.memory[self.counter] = diff / len(new_param)
            self.counter = (self.counter + 1) % len(self.memory)

            self.prev_param = new_param
            return max(self.memory) < self.tol
