import os
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy import signal
from scipy.fftpack import fft
from scipy.fftpack import fftfreq
import matplotlib.pyplot as plt

def CheckCSV(filepath):
    csvArray = np.genfromtxt(open(filepath), delimiter=",")
    rows = np.size(csvArray,0)
    columns = np.size(csvArray,1)

    if rows > 2 or rows < 2:
        status = False

    else:
        if columns < 500:
            status = False
            
        else:
            status = True

    return status

def CheckFile(filepath):
    if os.path.exists(filepath):

        if filepath[-1] == "/":
            filepath = filepath[:-1]

        elif filepath[-4:] == ".csv":
            status = True

        else:
            status = False

    else:
        status = False
    return status


def VoltageCheck(voltageLimit):
    try:
        voltageLimit = int(voltageLimit)

    except ValueError:
        status = False
    
    if type(voltageLimit) == float:

        voltageLimit = int(voltageLimit)

        if voltageLimit > 150 or voltageLimit < 0:
            status = False

        elif voltageLimit <= 150 and voltageLimit >= 0:
            status = True
        
    elif type(voltageLimit) == int:
        if voltageLimit > 150 or voltageLimit < 0:
            status = False
        elif voltageLimit <= 150 and voltageLimit >= 0:
            status = True

    else:
        status = False

    return status


def linearRegression(x, y):
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


def plotting_peaks(x, y, voltageLimit, filepath, str_datetime_rn, headers):
    filename = os.path.basename(filepath)
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
    plt.axhline(cutoff, label = "{:.1f}V".format(cutoff), linestyle = "--", color = "black")
    
    # plotting options
    plt.title("Guinness Generator Output, Voltage Limit = {}V\nInput file name: '{}'".format(voltageLimit, filename))
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
    
def THD(x, y, voltageLimit, filepath, str_datetime_rn, headers):
    filename = os.path.basename(filepath)

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
    
    if n!=[] and m!=[]:
        ind = max(max(n),max(m))
        x = x[:ind-1]
        y = y[:ind-1]
        # if there are NaN values anywhere in x or y, cut both of them down before the earliest found NaN
    elif n!=[] and m==[]:
        ind = max(n)
        x = x[:ind-1]
        y = y[:ind-1]
        # if there are NaN values anywhere in x, cut both x and y down before the earliest found NaN in x
    elif n==[] and m!=[]:
        ind = max(m)
        x = x[:ind-1]
        y = y[:ind-1]
        # if there are NaN values anywhere in y, cut both x and y down before the earliest found NaN in y
    
    # time step of x
    step = x[1]-x[0]

    # normalized time array
    T = 1.0/xN

    # fast fourier transform (FFT) of y
    yf = np.abs(fft(y))

    # building frequency domain array
    xf = fftfreq(xN,T)[:xN//2]

    # making the fft of y plottable by removing the mirroring and imaginary values
    yf_plottable = 2.0/xN * np.abs(yf[0:xN//2])

    # padding the fourier transform of y with its absolute min value at the beginning and end
    # this allows the find_peaks function to find potential peaks at the first and last values of the fourier transform of y
    yf_plottable = np.concatenate(([min(yf_plottable)],yf_plottable,[min(yf_plottable)]))

    # padding the frequency array; adding a step before the first value and a step after the last
    xf = np.concatenate(([min(xf)-step],xf,[max(xf)+step]))

    # plotting the fft
    plt.plot(xf, yf_plottable, label = "FFT of the Pulse Burst")
    plt.text(min(x)+1,max(y)-3,"ST-0001-066-101A, {}".format(str_datetime_rn),fontsize="small")
    plt.xlabel("Frequency (mHz)")
    plt.ylabel(headers[1])

    # finding the peaks of the fft of y; this function gives the indices of the values
    y_peaks_xvalues, ypeak_properties = signal.find_peaks(yf_plottable, height=0.10,prominence=0.2,distance=10)

    # getting the y values of the peaks (these are the amplitudes (V) of the harmonics)
    y_peaks_yvalues = ypeak_properties["peak_heights"]

    # mark the peaks on the plot
    plt.plot(y_peaks_xvalues,y_peaks_yvalues,"x", color='red', label = "Harmonic Amplitudes", markersize = 4, markeredgewidth = 1)

    # calculating the total harmonic distortion of the signal with its harmonic amplitudes
    thd = 100*(np.sum(y_peaks_yvalues)-max(y_peaks_yvalues))**0.5 / max(y_peaks_yvalues)

    # plotting options
    plt.title("Guinness Generator Pulse Burst FFT, THD = {:.3f}%\nVoltage Limit = {}V, Input file name: '{}'".format(thd,voltageLimit, filename))
    plt.xlim(min(xf),10000)
    plt.ylim(min(yf_plottable)-3,max(yf_plottable)+3)
    plt.legend(loc="upper right")

    # display the plot
    plt.show()
    
    return thd

