from glob import glob
import matplotlib.pyplot as plt
import numpy as np

restaurant_acts = np.array([np.load(acts).squeeze() for acts in glob(f'./acts/restaurant/*mixed8')])
jeep_acts = np.array([np.load(acts).squeeze() for acts in glob(f'./acts/jeep/*mixed8')])

print('restaurant_acts[0].shape')
print(restaurant_acts[0].shape)
print('jeep_acts[0].shape')
print(jeep_acts[0].shape)
