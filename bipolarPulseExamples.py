import numpy as np
import csv
import matplotlib.pyplot as plt

### Examples for Bipolar Pulse Analyzing ###

headers = ['Time','Voltage']

limit = 5
ten = limit*0.10
ninety = limit*0.9

fullRamp = 1    # microsecond
pulseWidth = 2  # microsecond

microsecond = 1000000   # in a second
samples = 10000000

samplesInMicrosecond = samples/microsecond

fileTime = 0.0001  # second(s)

fileLength = int(fileTime*samples)

t = np.linspace(0,fileTime,fileLength)
x = np.zeros(fileLength)

startPoint = int(fileLength*0.4)

slope = limit/(samplesInMicrosecond)

sequenceWidth = int((2*pulseWidth + 4*fullRamp)*samplesInMicrosecond)



for i in range(0,sequenceWidth):
    
    if i in range(0,int(samplesInMicrosecond*fullRamp)):
        x[startPoint+i]=x[startPoint+i-1] + slope  #positive slope
        
    if i in range(int((samplesInMicrosecond*fullRamp)),int(samplesInMicrosecond*(fullRamp+pulseWidth))):
        x[startPoint+i]=limit
    
    if i in range(int((samplesInMicrosecond*(fullRamp + pulseWidth))),int(samplesInMicrosecond*(3*fullRamp + pulseWidth))):
        x[startPoint+i]=x[startPoint+i-1] - slope   #negative slope
    
    if i in range(int((samplesInMicrosecond*(3*fullRamp + pulseWidth))),int((samplesInMicrosecond*(3*fullRamp + 2*pulseWidth)))):
        x[startPoint+i]=-limit
        
    if i in range(int((samplesInMicrosecond*(3*fullRamp + 2*pulseWidth))),sequenceWidth):
        x[startPoint+i]=x[startPoint+i-1] + slope   #positive slope

plt.plot(t,x)
plt.show()

with open('Faster-Placement-Pulse.csv', newline='', mode='w') as f:
  writer = csv.writer(f)
  writer.writerow([headers[0], headers[1]])

  for i in range(0,np.shape(x)[0]):
    writer.writerow([t[i],x[i]])


limit = 3
ten = limit*0.10
ninety = limit*0.9

fullRamp = 2    # microsecond
pulseWidth = 3  # microsecond

microsecond = 1000000   # in a second
samples = 10000000

samplesInMicrosecond = samples/microsecond

fileTime = 0.0001  # second(s)

fileLength = int(fileTime*samples)

t = np.linspace(0,fileTime,fileLength)
x = np.zeros(fileLength)

startPoint = int(fileLength*0.4)

slope = limit/(2*samplesInMicrosecond)

sequenceWidth = int((2*pulseWidth + 4*fullRamp)*samplesInMicrosecond)



for i in range(0,sequenceWidth):
    
    if i in range(0,int(samplesInMicrosecond*fullRamp)):
        x[startPoint+i]=x[startPoint+i-1] + slope  #positive slope
        
    if i in range(int((samplesInMicrosecond*fullRamp)),int(samplesInMicrosecond*(fullRamp+pulseWidth))):
        x[startPoint+i]=limit
    
    if i in range(int((samplesInMicrosecond*(fullRamp + pulseWidth))),int(samplesInMicrosecond*(3*fullRamp + pulseWidth))):
        x[startPoint+i]=x[startPoint+i-1] - slope   #negative slope
    
    if i in range(int((samplesInMicrosecond*(3*fullRamp + pulseWidth))),int((samplesInMicrosecond*(3*fullRamp + 2*pulseWidth)))):
        x[startPoint+i]=-limit
        
    if i in range(int((samplesInMicrosecond*(3*fullRamp + 2*pulseWidth))),sequenceWidth):
        x[startPoint+i]=x[startPoint+i-1] + slope   #positive slope

plt.plot(t,x)
plt.show()

with open('Slower-Placement-Pulse.csv', newline='', mode='w') as f:
  writer = csv.writer(f)
  writer.writerow([headers[0], headers[1]])

  for i in range(0,np.shape(x)[0]):
    writer.writerow([t[i],x[i]])
