import numpy as np


with open('record.txt', 'r') as f:
    data = f.read().split(',')[:-1]

print(len(data))
# data = data[100:]
assert len(data) == 100

acc = np.zeros(100)
for n in range(100):
    acc[n] = float(data[n])

print(acc.mean())
print(acc.std())
