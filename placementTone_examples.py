import numpy as np
import csv
import matplotlib.pyplot as plt

### Examples for Bipolar Pulse Analyzing ###

headers = ['Time','Voltage','Audio']

limit = 5       # volts
samples = 10000 # in a second
fileTime = 3    # second(s)
freq = 2        # placement pulses per second (Hz)
delay = 0.1     # second(s)

pulseInterval = samples/freq

fileLength = int(fileTime*samples)

t = np.linspace(0,fileTime,fileLength)
x = np.zeros(fileLength)
a = np.zeros(fileLength)

placementStart = int(samples*0.5)
audioStart = int(samples*(0.5+delay))

placement_pulse = [limit/2,limit,limit,limit,limit,limit,limit/2,0,-limit/2,-limit,-limit,-limit,-limit,-limit, -limit/2]
audio_pulse = [0.5,-0.5,0.5,-0.5,0.4,-0.4,0.4,-0.4,0.4,-0.4,0.3,-0.3,0.3-0.3,0.3,-0.3,0.3,-0.3,0.2,-0.2,0.2,-0.2,0.1,-0.1,0.1,-0.1,0.1,-0.1,0.1,-0.1,0.1,-0.1,0.1,-0.1]

for i in range(4):
    x[int(placementStart+(i*pulseInterval)):int(placementStart+(i*pulseInterval)+len(placement_pulse))] = x[int(placementStart+(i*pulseInterval)):int(placementStart+(i*pulseInterval)+len(placement_pulse))] + placement_pulse
    a[int(audioStart+(i*pulseInterval)):int(audioStart+(i*pulseInterval)+len(audio_pulse))] = x[int(audioStart+(i*pulseInterval)):int(audioStart+(i*pulseInterval)+len(audio_pulse))] + audio_pulse

plt.plot(t,x)
plt.plot(t,a)
plt.show()

with open('Delayed-Placement-Tone.csv', newline='', mode='w') as f:
  writer = csv.writer(f)
  writer.writerow([headers[0], headers[1], headers[2]])

  for i in range(0,np.shape(x)[0]):
    writer.writerow([t[i],x[i],a[i]])
    
    

limit = 3       # volts
samples = 10000 # in a second
fileTime = 3    # second(s)
freq = 2        # placement pulses per second (Hz)
delay = 0.01     # second(s)

pulseInterval = samples/freq

fileLength = int(fileTime*samples)

t = np.linspace(0,fileTime,fileLength)
x = np.zeros(fileLength)
a = np.zeros(fileLength)

audioStart = int(samples*(0.5+delay))

placement_pulse = [limit/2,limit,limit,limit,limit,limit,limit/2,0,-limit/2,-limit,-limit,-limit,-limit,-limit, -limit/2]

for i in range(4):
    x[int(placementStart+(i*pulseInterval)):int(placementStart+(i*pulseInterval)+len(placement_pulse))] = x[int(placementStart+(i*pulseInterval)):int(placementStart+(i*pulseInterval)+len(placement_pulse))] + placement_pulse
    a[int(audioStart+(i*pulseInterval)):int(audioStart+(i*pulseInterval)+len(audio_pulse))] = x[int(audioStart+(i*pulseInterval)):int(audioStart+(i*pulseInterval)+len(audio_pulse))] + audio_pulse

plt.plot(t,x)
plt.plot(t,a)
plt.show()

with open('Immediate-Placement-Tone.csv', newline='', mode='w') as f:
  writer = csv.writer(f)
  writer.writerow([headers[0], headers[1], headers[2]])

  for i in range(0,np.shape(x)[0]):
    writer.writerow([t[i],x[i],a[i]])