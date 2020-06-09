import numpy as np


with open('record.txt', 'r') as f:
    data = f.read()[1:].split(',')[:-1]

assert len(data) == 20

acc = np.zeros(20)
for n in range(20):
    acc[n] = float(data[n])

print(acc.mean())
# print(acc.std())

with open('record_val.txt', 'a') as f:
    f.write(str(acc.mean()) + ' ')
