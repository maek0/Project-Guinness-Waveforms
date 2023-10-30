import numpy as np
import csv
import matplotlib.pyplot as plt

### Examples for Voltage Ramp Analyzing ###

headers = ['Time','Voltage']

# cycles in a 30ms pulse burst:
# 460hz = 460 cycles per second
# 30ms = 0.03s
pulse_burst_sin_cycles = int(460*0.03)
# how many datapoints to generate in the sin wave
pulse_burst_count = int(600) 

# sine wave
length_sin = np.pi * 2 * pulse_burst_sin_cycles
pulse_burst = np.sin(np.arange(0, length_sin, length_sin / pulse_burst_count))
pulse_burst = pulse_burst[:-1]
# print(pulse_burst.shape)

## 75V voltage limit ## regular ramping ##
limit = 75

# voltage ramp switch cutoff
cutoff = limit*0.66

# datapoints per second
dpps = int(pulse_burst_count/0.03)

# pulse rate = every 1/2 dpps
pr = dpps/2

# make whole file X seconds
sec = 25

# length of entire file:
file_length = dpps * sec

t = np.linspace(0,sec,file_length)
x = np.zeros(file_length)

# 3 seconds of nothing
buf = int(2)

pulse_amp = 10
ramp1 = 2.5
ramp2 = 1

i = int(buf*dpps)
while i < file_length:
    j = int(i+pulse_burst_count)

    if pulse_amp<=cutoff:
        x[i:j] = x[i:j] + pulse_amp*pulse_burst
        pulse_amp = pulse_amp + ramp1

    elif pulse_amp>cutoff and pulse_amp<=limit:
        x[i:j] = x[i:j] + pulse_amp*pulse_burst
        pulse_amp = pulse_amp + ramp2
    
    elif pulse_amp>=limit:
        x[i:j] = x[i:j] + limit*pulse_burst

    i = int(i + pr)

plt.plot(t,x)
plt.show()

with open('ramp_75V_normalrate.csv', newline='', mode='w') as f:
  writer = csv.writer(f)
  writer.writerow([headers[0], headers[1]])

  for i in range(0,np.shape(x)[0]):
    writer.writerow([t[i],x[i]])



## 120V voltage limit ## regular ramping ##
limit = 120

# voltage ramp switch cutoff
cutoff = limit*0.66

# datapoints per second
dpps = int(pulse_burst_count/0.03)

# pulse rate = every 1/2 dpps
pr = dpps/2

# make whole file X seconds
sec = 40

# length of entire file:
file_length = dpps * sec

t = np.linspace(0,sec,file_length)
x = np.zeros(file_length)

# 3 seconds of nothing
buf = int(2)

pulse_amp = 10
ramp1 = 2.5
ramp2 = 1

i = int(buf*dpps)
while i < file_length:
    j = int(i+pulse_burst_count)

    if pulse_amp<=cutoff:
        x[i:j] = x[i:j] + pulse_amp*pulse_burst
        pulse_amp = pulse_amp + ramp1

    elif pulse_amp>cutoff and pulse_amp<=limit:
        x[i:j] = x[i:j] + pulse_amp*pulse_burst
        pulse_amp = pulse_amp + ramp2
    
    elif pulse_amp>=limit:
        x[i:j] = x[i:j] + limit*pulse_burst

    i = int(i + pr)

plt.plot(t,x)
plt.show()

with open('ramp_120V_normalrate.csv', newline='', mode='w') as f:
  writer = csv.writer(f)
  writer.writerow([headers[0], headers[1]])

  for i in range(0,np.shape(x)[0]):
    writer.writerow([t[i],x[i]])



## 75V voltage limit ## ramping 2v/s first, then ramping 1V/s ##
limit = 75

# voltage ramp switch cutoff
cutoff = limit*0.66

# datapoints per second
dpps = int(pulse_burst_count/0.03)

# pulse rate = every 1/2 dpps
pr = dpps/2

# make whole file X seconds
sec = 55

# length of entire file:
file_length = dpps * sec

t = np.linspace(0,sec,file_length)
x = np.zeros(file_length)

# 3 seconds of nothing
buf = int(2)

pulse_amp = 10
ramp1 = 1
ramp2 = 0.5

i = int(buf*dpps)
while i < file_length:
    j = int(i+pulse_burst_count)

    if pulse_amp<=cutoff:
        x[i:j] = x[i:j] + pulse_amp*pulse_burst
        pulse_amp = pulse_amp + ramp1

    elif pulse_amp>cutoff and pulse_amp<=limit:
        x[i:j] = x[i:j] + pulse_amp*pulse_burst
        pulse_amp = pulse_amp + ramp2
    
    elif pulse_amp>=limit:
        x[i:j] = x[i:j] + limit*pulse_burst

    i = int(i + pr)

plt.plot(t,x)
plt.show()

with open('ramp_75V_halfrate.csv', newline='', mode='w') as f:
  writer = csv.writer(f)
  writer.writerow([headers[0], headers[1]])

  for i in range(0,np.shape(x)[0]):
    writer.writerow([t[i],x[i]])



## 120V voltage limit ## ramping 2v/s first, then ramping 1V/s ##
limit = 120

# voltage ramp switch cutoff
cutoff = limit*0.66

# datapoints per second
dpps = int(pulse_burst_count/0.03)

# pulse rate = every 1/2 dpps
pr = dpps/2

# make whole file X seconds
sec = 25

# length of entire file:
file_length = dpps * sec

t = np.linspace(0,sec,file_length)
x = np.zeros(file_length)

# 3 seconds of nothing
buf = int(2)

pulse_amp = 10
ramp1 = 5
ramp2 = 2

i = int(buf*dpps)
while i < file_length:
    j = int(i+pulse_burst_count)

    if pulse_amp<=cutoff:
        x[i:j] = x[i:j] + pulse_amp*pulse_burst
        pulse_amp = pulse_amp + ramp1

    elif pulse_amp>cutoff and pulse_amp<=limit:
        x[i:j] = x[i:j] + pulse_amp*pulse_burst
        pulse_amp = pulse_amp + ramp2
    
    elif pulse_amp>=limit:
        x[i:j] = x[i:j] + limit*pulse_burst

    i = int(i + pr)

plt.plot(t,x)
plt.show()

with open('ramp_120V_doublerate.csv', newline='', mode='w') as f:
  writer = csv.writer(f)
  writer.writerow([headers[0], headers[1]])

  for i in range(0,np.shape(x)[0]):
    writer.writerow([t[i],x[i]])