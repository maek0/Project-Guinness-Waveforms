def guinnessFilter():
    import matplotlib.pyplot as plt
    from scipy import signal
    import numpy as np
    import os
    import sys

    filepath = input('Enter the name or file path of the oscilloscope .csv output: ')

    if os.path.exists(filepath):
        if filepath[-4:]==".csv":
            print("File found. File is a .csv file.")
        else:
            print("File found, but it is not a .csv file. Double check input file type and name.")
    else:
        print("File was not found. Double check input file path and file name.")
        sys.exit()

    while True:
        try:
            voltageLimit = input('Enter the voltage limit of the Guinness generator of the captured waveform: ')
        except type(voltageLimit)==float:
            print("Input must be an integer.")
        except type(voltageLimit)==str:
            print("Not a valid input. Value must be an integer in the range from 0 to 150.")
        except voltageLimit>150 or voltageLimit<0:
            print("Not a valid input. Value must be an integer in the range from 0 to 150.")
        else:
            break

    # if type(voltageLimit)==int:
    #     pass
    # elif type(voltageLimit)==float:
    #     print("Input must be an integer.")
    #     sys.exit()
    # elif type(voltageLimit)==str:
    #     print("Not a valid input. Value must be an integer in the range from 0 to 150.")
    #     sys.exit()
    # elif voltageLimit>150 or voltageLimit<0:
    #     print("Not a valid input. Value must be an integer in the range from 0 to 150.")
    #     sys.exit()
    # else:
    #     sys.exit()
    
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