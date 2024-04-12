import numpy as np
class ReplayBuffer():
    def __init__(self, task_num, class_num, max_size=500, sample_size=200):
        self.max_size = max_size
        self.classes_per_task = class_num//task_num
        self.sample_size = sample_size
        self.class_to_task = {}
        self.task_num = task_num
        self.class_num = class_num

        # class_to_tasks : {[CLASS]: [TASK]}
        # e.g {22: 1, 23: 1}
        for c in range(class_num):
            self.class_to_task[c] = (c%task_num) + 1
        
        self.task_x = {}
        self.task_y = {}
        #task_examples : {[TASK]: [(x1,y1), (x2,y2), ..., (x_max_size, y_max_size)]}
        for task in range(1, task_num+1):
            self.task_x[task] = []
            self.task_y[task] = []
    
    def add_example(self, x, y):
        task = self.class_to_task[y]
        self.task_x[task].append(x)
        self.task_y[task].append(y)
        if len(self.task_x[task]) > self.max_size:
            self.task_x[task].pop(0)
            self.task_y[task].pop(0)
    
    def get_examples(self, task):
        sample = self.sample_size
        if len(self.task_x[task]) < sample:
            sample = len(self.task_x[task])
        indices = np.random.choice(len(self.task_x[task]), sample)
        return np.array(self.task_x[task])[indices], np.array(self.task_y[task])[indices]
    
    def get_random_sample(self):
        all_examples_x = []
        all_examples_y = []
        for task in self.task_x:
            all_examples_x += self.task_x[task]
            all_examples_y += self.task_y[task]
        sample = self.sample_size
        if len(all_examples_x) < sample:
            sample = len(all_examples_x)
        indices = np.random.choice(len(all_examples_x), sample)
        return np.array(all_examples_x)[indices], np.array(all_examples_y)[indices]
    


