import PySimpleGUI as sg
import os
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import datetime

def VoltageCheckPlacement(voltageLimit):
    try:
        voltageLimit = float(voltageLimit)

    except ValueError:
        status = False
    
    if type(voltageLimit) == float:
        if voltageLimit > 150 or voltageLimit < 0:
            status = False

        elif voltageLimit <= 150 and voltageLimit >= 0:
            status = True
        
    elif type(voltageLimit) == int:
        
        voltageLimit = float(voltageLimit)
        
        if voltageLimit > 150 or voltageLimit < 0:
            status = False

        elif voltageLimit <= 150 and voltageLimit >= 0:
            status = True

    else:
        status = False

    return status


def guinnessPlacement(filepath, voltageLimit):
    datetime_rn = datetime.datetime.now()
    str_datetime_rn = datetime_rn.strftime("%d-%b-%Y, %X %Z")
    
    headers = ["Time", "Voltage"]
    csvFile = open(filepath)
    csvArray = np.genfromtxt(csvFile, delimiter=",")
    x = csvArray[2:-2,0]
    y = csvArray[2:-2,1]
    
    filename = os.path.basename(filepath)
    
    # find the indices of the peaks of the output energy signal (not including voltage checks)
    y_peaks_xvalues, ypeak_properties = signal.find_peaks(y, height=(voltageLimit-0.05),distance=50)

    # get the y-values of the output energy peaks
    y_peaks_yvalues = ypeak_properties["peak_heights"]
    
    first_derivative = np.gradient(y)
    second_derivative = np.gradient(first_derivative)
    
    # plot the raw signal
    plt.plot(x,y, color = 'blue')
    plt.plot(x,first_derivative, color = 'orange')
    plt.plot(x,second_derivative, color = 'red')
    
    # plotting options
    plt.title("Guinness Generator Placement Output, Voltage Limit = {}V\nInput file name: '{}'".format(voltageLimit, filename))
    plt.text(min(x)+1,max(y)-3,"ST-0001-066-101A, {}".format(str_datetime_rn),fontsize="small")
    plt.xlabel(headers[0])
    plt.ylabel(headers[1])
    
    # plotting options
    plt.xlim(min(x),max(x))
    plt.ylim(min(y)-3,max(y)+3)
    plt.legend(loc="lower left")
    
    # display the plot
    plt.show()
