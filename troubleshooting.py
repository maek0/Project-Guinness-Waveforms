import os
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy import signal
import datetime
import matplotlib.pyplot as plt

def linearRegression(x, y):
    y = signal.detrend(y, type="constant")
    
    # reshape the input x value so that the LinearRegression function can accept it
    x = np.array(x).reshape((-1, 1))

    # get the line of best fit for x and y
    model = LinearRegression().fit(x, y)

    # r^2 value = how well the model fits & predicts the values
    r_sq = model.score(x, y)

    # slope of the line of best fit
    slope = model.coef_

    # y-intercept of the line of best fit
    intercept = model.intercept_

    # predicted y values using the line of best fit
    y_predict = model.predict(x)

    return r_sq, slope, intercept, y_predict


def guinnessRampFilter():

    filepath = input("")
    voltageLimit = input("")
    
    datetime_rn = datetime.datetime.now()
    str_datetime_rn = datetime_rn.strftime("%d-%b-%Y, %X %Z")
    
    headers = ["Time", "Voltage"]
    csvFile = open(filepath)
    csvArray = np.genfromtxt(csvFile, delimiter=",")
    x = csvArray[2:,0]
    y = csvArray[2:,1]
    
    filename = os.path.basename(filepath)
    y = signal.detrend(y, type="constant")

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
    
    if n.size>0 and m.size>0:
        ind = max(max(n),max(m))
        x = x[:ind-1]
        y = y[:ind-1]
        # if there are NaN values anywhere in x or y, cut both of them down before the earliest found NaN
    elif n.size>0 and m.size==0:
        ind = max(n)
        x = x[:ind-1]
        y = y[:ind-1]
        # if there are NaN values anywhere in x, cut both x and y down before the earliest found NaN in x
    elif n.size==0 and m.size>0:
        ind = max(m)
        x = x[:ind-1]
        y = y[:ind-1]
    
    # find the indices of the peaks of the output energy signal (not including voltage checks)
    y_peaks_xvalues, ypeak_properties = signal.find_peaks(y, height=2.5,prominence=15,distance=50)

    # get the y-values of the output energy peaks
    y_peaks_yvalues = ypeak_properties["peak_heights"]
    
    # find the first peak of the output energy signal (not including voltage checks)
    first_peakX = x[y_peaks_xvalues[0]]
    first_peakY = y_peaks_yvalues[0]
    
    # find the cutoff voltage = 66% of the input voltage limit
    cutoff = 0.66*float(voltageLimit)

    # index the cutoff voltage
    ind_cutoff = np.where(y_peaks_yvalues>=cutoff)[0][0]

    # index the first point where the voltage reaches the set limit
    ind_limit = np.argmax(y_peaks_yvalues>=float(voltageLimit))
    
    # get all peaks BEFORE the indexed cutoff value - this is when the generator should be ramping at 5V/s
    fiveVoltRampY = y_peaks_yvalues[:(ind_cutoff-1)]
    fiveVoltRampX = x[y_peaks_xvalues[:(ind_cutoff-1)]]
    
    # get all peaks AFTER the indexed cutoff value and UNTIL the peaks reach the voltage limit - this is when the generator should be ramping at 2V/s
    twoVoltRampY = y_peaks_yvalues[ind_cutoff:ind_limit]
    twoVoltRampX = x[y_peaks_xvalues[ind_cutoff:ind_limit]]
    
    # find the line of best fit for the ramping section BEFORE reaching 66%(voltage limit)
    fiveV_rsq, fiveV_slope, fiveV_intercept, fiveV_fit = linearRegression(fiveVoltRampX, fiveVoltRampY)

    # find the line of best fit for the ramping section AFTER reaching 66%(voltage limit)
    twoV_rsq, twoV_slope, twoV_intercept, twoV_fit = linearRegression(twoVoltRampX, twoVoltRampY)
    
    # plot the raw signal
    plt.plot(x,y, color = 'blue')
    
    # plot the voltage peaks of the ramping signal
    plt.scatter(fiveVoltRampX,fiveVoltRampY, color = 'green')
    plt.scatter(twoVoltRampX,twoVoltRampY, color = 'orange')

    # plotting the 66%(voltage limit)
    plt.axhline(cutoff, label = "{:.2f}V".format(cutoff), linestyle = "--", color = "black")
    
    # plotting options
    plt.title("Guinness Generator Output Ramp, Voltage Limit = {}V\nInput file name: '{}'".format(voltageLimit, filename))
    plt.text(min(x)+1,max(y)-3,"ST-0001-066-101A, {}".format(str_datetime_rn),fontsize="small")
    plt.xlabel(headers[0])
    plt.ylabel(headers[1])
    
    # plot the best fit line for the ramping section BEFORE reaching 66%(voltage limit)
    plt.plot(fiveVoltRampX, fiveV_fit, color = 'green', label = "y = {:.2f}x + {:.2f}\n$R^2$ = {:.2f}".format(fiveV_slope[0], fiveV_intercept, fiveV_rsq))
    
    # plot the best fit line for the ramping section AFTER reaching 66%(voltage limit)
    plt.plot(twoVoltRampX, twoV_fit, color = 'orange', label = "y = {:.2f}x + {:.2f}\n$R^2$ = {:.2f}".format(twoV_slope[0], twoV_intercept, twoV_rsq))
    
    # mark the first peak of the output energy signal on the plot
    plt.plot(first_peakX,first_peakY, "x", color = "black", label = "First Peak = {:.2f}V".format(first_peakY), markersize = 8, markeredgewidth = 2)

    # plotting options
    plt.xlim(min(x),max(x))
    plt.ylim(min(y)-3,max(y)+3)
    plt.legend(loc="lower left")
    
    # display the plot
    plt.show()
    guinnessRampFilter()