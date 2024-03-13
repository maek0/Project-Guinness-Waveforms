from scipy import signal
import numpy as np
import matplotlib.pyplot as plt

filepath = input("FILE")

csvFile = open(filepath)
csvArray = np.genfromtxt(csvFile, delimiter=",")
x = csvArray[2:-2,0]
y = csvArray[2:-2,1]

xN = len(x)
yN = len(y)
v = xN - yN

# find any NaN values in x and y
n = np.argwhere(np.isnan(x))
m = np.argwhere(np.isnan(y))

# if x and y are somehow different lengths, cut them to the same length
if v > 0:
    x = x[:yN]
elif v < 0:
    y = y[:xN]
    
print(y[15])

if n.size>0 and m.size>0:
    ind = max(max(n),max(m))
    ind = int(ind)
    print(ind)
    # x = x[:ind-1]
    # y = y[:ind-1]
    x = x[ind+1:]
    y = y[ind+1:]
    # if there are NaN values anywhere in x or y, cut both of them down before the earliest found NaN
elif n.size>0 and m.size==0:
    ind = max(n)
    ind = int(ind)
    print(ind)
    # x = x[:ind-1]
    # y = y[:ind-1]
    x = x[ind+1:]
    y = y[ind+1:]
    # if there are NaN values anywhere in x, cut both x and y down before the earliest found NaN in x
elif n.size==0 and m.size>0:
    ind = max(m)[0]
    ind = int(ind)
    print(ind)
    # x = x[:ind-1]
    # y = y[:ind-1]
    x = x[ind+1:]
    y = y[ind+1:]

print(y[15])
# print(y)
    
# y = signal.detrend(y, type="constant")

# y_diff2 = np.gradient(np.gradient(y))
# y_diff2 = np.abs(y_diff2)

plt.plot(x,y)
plt.show()