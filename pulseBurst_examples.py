import numpy as np
import csv
import matplotlib.pyplot as plt

### Examples for Pulse Burst Analyzing ###

headers = ['Time','Voltage']

t = np.linspace(0,10,5000)
cycles_pure = 20 # how many sine cycles
resolution = 5000 # how many datapoints to generate

# pure sine wave
length_sin = np.pi * 2 * cycles_pure

sin_wave = 100*np.sin(np.arange(0, length_sin, length_sin / resolution))

with open('sin_file.csv', newline='', mode='w') as f:
  writer = csv.writer(f)
  writer.writerow([headers[0], headers[1]])

  for i in range(0,np.shape(sin_wave)[0]):
    writer.writerow([t[i],sin_wave[i]])

# sine wave with noise
cycles_noise = 500

# noise
length_noise = np.pi * 2 * cycles_noise

noise_wave = np.sin(np.arange(0, length_noise, length_noise / resolution))

sin_with_noise = sin_wave + noise_wave
plt.plot(sin_with_noise)
plt.show()

with open('sin_with_noise_file.csv', newline='', mode='w') as f:
  writer = csv.writer(f)
  writer.writerow([headers[0], headers[1]])

  for i in range(0,np.shape(sin_wave)[0]):
    writer.writerow([t[i],sin_with_noise[i]])