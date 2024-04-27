#queue
import class_queue as cq
import numpy as np
from tqdm import tqdm

queue = cq.Queue(num_sinks=6, array_utilities=[['soap', 'washing'], ['soap', 'washing'], ['towel', 'washing'], ['soap', 'washing'], ['towel', 'washing'], ['washing']])

for i in tqdm(range(int(1e6)), delay=1, miniters=10):
    queue.one_iteration(optimize=False)