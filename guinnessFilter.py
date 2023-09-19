def guinnessFilter():
    import matplotlib.pyplot as plt
    from scipy import signal
    import numpy as np
    import os
    import sys
    import CheckFile
    import VoltageCheck

    filepath = input('Enter the name or file path of the oscilloscope .csv output: ')

    CheckFile.CheckFile(filepath)
    VoltageCheck.VoltageCheck()
    
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
    
    y_peaks, ypeak_properties = signal.find_peaks(y, height=5,prominence=15,distance=50)   
    # the distance likely will have to change when the frequency is fixed to be 2Hz
    
    plt.scatter(x[y_peaks],y[y_peaks])
    plt.plot(x,y)
    plt.show()
    # this plot won't have to stay here, currently being used to validate settings
    
    # I have the max points... now I need to use the 66%(voltage limit) to divide the points into two sections
    # the two sections will have a linear fit ==> slope of the two sections 

guinnessFilter()