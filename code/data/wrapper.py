import numpy as np


class DeterministicDataset:
    def __init__(self, dataset):
        self.dataset = dataset
        self.order = np.arange(len(self.dataset))
        self.update_order()

    def update_order(self):
        np.random.shuffle(self.order)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return self.dataset[self.order[item]]
