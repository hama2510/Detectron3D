from datetime import datetime
import pickle
import os, sys

class Logger:
    
    def __init__(self):
        self.log_file = 'log_{}.pkl'.format(datetime.now().strftime("%Y%m%d_%H%M"))

    def init(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        pickle.dump([], open(os.path.join(save_dir, self.log_file), 'wb'))
        
    def log(self, item, save_dir):
        log = pickle.load(open(os.path.join(save_dir, self.log_file), 'rb'))
        log.append(item)
        pickle.dump(log, open(os.path.join(save_dir, self.log_file), 'wb'))
        