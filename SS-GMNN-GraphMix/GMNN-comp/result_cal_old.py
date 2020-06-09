import numpy as np


with open('record.txt', 'r') as f:
    data = f.read().split(',')[:-1]

print(len(data))
# data = data[20:]
assert len(data) == 100

acc = np.zeros(100)
for n in range(100):
    acc[n] = float(data[n])

# acc[0] = -acc[0]
print(acc[acc<0.8])
print(acc.mean())
print(acc.std())

