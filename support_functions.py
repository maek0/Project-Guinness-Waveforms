import sys
import os
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy import signal
from scipy.fftpack import fft
from scipy.fftpack import fftfreq
import matplotlib.pyplot as plt

def CheckFile(filepath):
    if os.path.exists(filepath):
        if filepath[-1] == "/":
            filepath = filepath[:-1]
        elif filepath[-4:] == ".csv":
            print("File found. File is a .csv file.")
            return os.path.basename(filepath)
        else:
            print(
                "File found, but it is not a .csv file. Double check input file type and name."
            )
            sys.exit()
    else:
        print("File was not found. Double check input file path and file name.")
        sys.exit()


def linearRegression(x, y):
    x = np.array(x).reshape((-1, 1))

    model = LinearRegression().fit(x, y)
    r_sq = model.score(x, y)
    slope = model.coef_
    intercept = model.intercept_
    y_predict = model.predict(x)

    return r_sq, slope, intercept, y_predict


def VoltageCheck():
    while True:
        voltageLimit = input(
            "Enter the voltage limit of the Guinness generator of the captured waveform: "
        )
        try:
            voltageLimit = int(voltageLimit)
            if type(voltageLimit) == str:
                raise TypeError
            elif voltageLimit > 150 or voltageLimit < 0:
                raise ValueError
            elif voltageLimit <= 150 and voltageLimit >= 0:
                return voltageLimit
                break
            else:
                print("Invalid input.")
                raise TypeError
        except ValueError:
            print(
                "Not a valid input. Value must be an integer in the range from 0 to 150."
            )
            continue
        except TypeError:
            print(
                "Not a valid input. Value must be an integer in the range from 0 to 150."
            )
            continue

def plotting_peaks(x, y, voltageLimit, filename, str_datetime_rn, headers):
    y_peaks_xvalues, ypeak_properties = signal.find_peaks(y, height=2.5,prominence=15,distance=50)
    y_peaks_yvalues = ypeak_properties["peak_heights"]
    
    first_peakX = x[y_peaks_xvalues[0]]
    first_peakY = y_peaks_yvalues[0]
    
    cutoff = 0.66*float(voltageLimit)
    ind = np.where(y_peaks_yvalues>=cutoff)[0][0]
    
    fiveVoltRampY = y_peaks_yvalues[:ind]
    fiveVoltRampX = x[y_peaks_xvalues[:ind]]
    
    twoVoltRampY = y_peaks_yvalues[ind:]
    twoVoltRampX = x[y_peaks_xvalues[ind:]]
    
    fiveV_rsq, fiveV_slope, fiveV_intercept, fiveV_fit = linearRegression(fiveVoltRampX, fiveVoltRampY)
    twoV_rsq, twoV_slope, twoV_intercept, twoV_fit = linearRegression(twoVoltRampX, twoVoltRampY)
    
    plt.plot(x,y, color = 'blue')
    
    plt.scatter(fiveVoltRampX,fiveVoltRampY, color = 'green')
    plt.scatter(twoVoltRampX,twoVoltRampY, color = 'orange')

    plt.axhline(cutoff, label = "{:.2f}V".format(cutoff), linestyle = "--", color = "black")
    
    plt.title("Guinness Generator Output, Voltage Limit = {}V\nInput file name: '{}'".format(voltageLimit, filename))
    plt.text(min(x)+1,max(y)-3,"ST-0001-066-101A, {}".format(str_datetime_rn),fontsize="small")
    plt.xlabel(headers[0])
    plt.ylabel(headers[1])
    
    plt.plot(fiveVoltRampX, fiveV_fit, color = 'green', label = "y = {:.2f}x + {:.2f}\n$R^2$ = {:.2f}".format(fiveV_slope[0], fiveV_intercept, fiveV_rsq))
    plt.plot(twoVoltRampX, twoV_fit, color = 'orange', label = "y = {:.2f}x + {:.2f}\n$R^2$ = {:.2f}".format(twoV_slope[0], twoV_intercept, twoV_rsq))
    
    plt.plot(first_peakX,first_peakY, "x", color = "black", label = "First Peak = {:.2f}V".format(first_peakY), markersize = 8, markeredgewidth = 2)

    plt.xlim(min(x),max(x))
    plt.ylim(min(y)-3,max(y)+3)
    plt.legend(loc="lower left")
    
    plt.show()
    
def THD(x,y):
    
    xN = len(x)
    yN = len(y)
    v = np.abs(xN - yN)
    
    if v > 0:
        x = x[:len(x)-v]
        y = y[:len(y)-v]
    
    T = 1.0/xN
    yf = np.abs(fft(y))
    xf = fftfreq(xN,T)[:xN//2]
    plt.plot(xf, 2.0/xN * np.abs(yf[0:xN//2]))
    # plt.plot(xf, yf)
    
    sq_sum = 0.0
    for i in range (len(yf)):
        sq_sum += yf[i]**2
    
    sq_harmonics = sq_sum - (max(yf))**2.0
    thd = 100*sq_harmonics**0.5 / max(yf)
    plt.show()
    
    return thd

def plotTHD(thd, x, y, voltageLimit, filename, str_datetime_rn, headers):
    plt.plot(x,y, color = 'blue')
    
    plt.title("Guinness Generator Pulse Burst, THD = {}%\nVoltage Limit = {}V, Input file name: '{}'".format(thd,voltageLimit, filename))
    plt.text(min(x)+1,max(y)-3,"ST-0001-066-101A, {}".format(str_datetime_rn),fontsize="small")
    plt.xlabel(headers[0])
    plt.ylabel(headers[1])
    
    plt.xlim(min(x),max(x))
    plt.ylim(min(y)-3,max(y)+3)
    plt.legend(loc="lower left")
    
    plt.show()