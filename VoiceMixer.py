import os
import random
from processing import *

class MixVoiceData():
    def __init__(self, inst_dir, sing_dir):
        self.inst_dir = inst_dir
        self.sing_dir = sing_dir
        self.rate_set = set()

        self.singers_list = [os.path.join(self.sing_dir, f) for f in os.listdir(self.sing_dir)]
        self.inst_list = [os.path.join(self.inst_dir, f) for f in os.listdir(self.inst_dir)]

        self.inst_queue = []
        self.sing_queue = []
        
        self.epochs = 500
        self.cache_size = 5

    def fill_queues(self):
        random.shuffle(self.singers_list)
        random.shuffle(self.inst_list)
        for (i, s) in enumerate(self.singers_list[:self.cache_size]):
            rate, tmp_ndarray = read(s)
            self.rate_set.add(rate)
            self.sing_queue.append(tmp_ndarray)
    
        for (i, s) in enumerate(self.inst_list[:self.cache_size]):
            print(s)
            rate, tmp_ndarray = read(s)
            self.rate_set.add(rate)
            self.inst_queue.append(tmp_ndarray)
    
    def generate_data(self):
        for i in range(self.epochs):
            print(self.rate_set)
            if (len(self.inst_queue) == 0) and (len(self.sing_queue) == 0):
                self.fill_queues()
            for _ in range(self.cache_size):
                singer = self.sing_queue.pop()
                yield concat_raw_signal(self.inst_queue.pop(), singer)
