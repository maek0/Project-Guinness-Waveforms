def guinnessFilter():
    import matplotlib.pyplot as plt
    from scipy import signal
    import numpy as np
    import os
    import sys
    import CheckFile

    filepath = input('Enter the name or file path of the oscilloscope .csv output: ')

    CheckFile.CheckFile(filepath)

    while True:
        voltageLimit = input('Enter the voltage limit of the Guinness generator of the captured waveform: ')
        try:
            voltageLimit = int(voltageLimit)
            if type(voltageLimit)==str:
                raise TypeError
            elif voltageLimit>150 or voltageLimit<0:
                raise ValueError
            elif voltageLimit<=150 and voltageLimit>=0:
                break
            else:
                print("Invalid input.")
                raise TypeError
        except ValueError:
            print("Not a valid input. Value must be an integer in the range from 0 to 150.")
        except TypeError:
            print("Not a valid input. Value must be an integer in the range from 0 to 150.")
    
    headers = ["Time", "Voltage"]
    csvFile = open(filepath)
    csvArray = np.genfromtxt(csvFile, delimiter=",")
    x = csvArray[2:,0]
    y = csvArray[2:,1]
    
    plt.plot(x,y)
    plt.title("Guinness Generator Output")
    plt.xlabel(headers[0])
    plt.ylabel(headers[1])
    
    xaxis = range(int(min(x)), int(max(x)), 2)
    yaxis = range(int(min(y)), int(max(y)), 5)
    plt.xticks(xaxis)
    plt.yticks(yaxis)
    
    plt.show()
    
    x_peaks, xpeak_properties = signal.find_peaks(x)

guinnessFilter()