import PySimpleGUI as sg
import os
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy import signal
from scipy.fftpack import fft
from scipy.fftpack import fftfreq
import matplotlib.pyplot as plt
import datetime

def CheckCSV(filepath):
    csvArray = np.genfromtxt(open(filepath), delimiter=",")
    rows = np.size(csvArray,0)
    columns = np.size(csvArray,1)

    if columns != 2:
        status = False

    else:
        if rows < 500:
            status = False
            
        else:
            status = True

    return status

def CheckAudioCSV(filepath):
    csvArray = np.genfromtxt(open(filepath), delimiter=",")
    rows = np.size(csvArray,0)
    columns = np.size(csvArray,1)

    if columns != 3:
        status = False

    else:
        if rows < 500:
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
        voltageLimit = float(voltageLimit)

    except ValueError:
        status = False
    
    if type(voltageLimit) == float:

        # voltageLimit = int(voltageLimit)

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


def plotting_peaks(x, y, voltageLimit, filepath, str_datetime_rn, headers):
    filename = os.path.basename(filepath)
    y = signal.detrend(y, type="constant")
    
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
    
    
def THD(x, y, voltageLimit, filepath, str_datetime_rn, headers):
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
    plt.xlabel("Frequency Bins")
    plt.ylabel(headers[1])

    # finding the peaks of the fft of y; this function gives the indices of the values
    y_peaks_xvalues, ypeak_properties = signal.find_peaks(yf_plottable, height=0.10,prominence=0.2,distance=10)
    xlim = int(np.ceil(y_peaks_xvalues[-1] / 1000.0)) * 1000

    # getting the y values of the peaks (these are the amplitudes (V) of the harmonics)
    y_peaks_yvalues = ypeak_properties["peak_heights"]

    # mark the peaks on the plot
    plt.plot(y_peaks_xvalues,y_peaks_yvalues,"x", color='red', label = "Harmonic Amplitudes", markersize = 4, markeredgewidth = 1)

    # calculating the total harmonic distortion of the signal with its harmonic amplitudes
    thd = 100*((np.sum(y_peaks_yvalues)-max(y_peaks_yvalues))**0.5 / max(y_peaks_yvalues))

    # plotting options
    plt.title("Guinness Generator Pulse Burst FFT, THD = {:.3f}%\nVoltage Limit = {}V, Input file name: '{}'".format(thd,voltageLimit, filename))
    plt.xlim(min(xf),xlim)
    plt.ylim(min(yf_plottable)-3,max(yf_plottable)+3)
    plt.legend(loc="upper right")

    # display the plot
    plt.show()
    
    return thd


def guinnessRampFilter(filepath,voltageLimit):
    
    datetime_rn = datetime.datetime.now()
    str_datetime_rn = datetime_rn.strftime("%d-%b-%Y, %X %Z")
    
    headers = ["Time", "Voltage"]
    csvFile = open(filepath)
    csvArray = np.genfromtxt(csvFile, delimiter=",")
    x = csvArray[2:,0]
    y = csvArray[2:,1]
    
    plotting_peaks(x, y, voltageLimit, filepath, str_datetime_rn, headers)


def guinnessTHD(filepath,voltageLimit):

    datetime_rn = datetime.datetime.now()
    str_datetime_rn = datetime_rn.strftime("%d-%b-%Y, %X %Z")
    
    headers = ["Time", "Voltage"]
    csvFile = open(filepath)
    csvArray = np.genfromtxt(csvFile, delimiter=",")
    x = csvArray[2:-2,0]
    y = csvArray[2:-2,1]
    
    THD(x, y, voltageLimit, filepath, str_datetime_rn, headers)


def placementTimingLowResInput(x, y, voltageLimit, filepath, str_datetime_rn, headers):

    '''
    DO THIS FOR ALL 
    - CREATE A LINE FROM THE MIN TO THE MAX 
    - PLOT WHERE THE LINE INTERCEPTS THE PLOTTING 10% AND 90% LINES 
    - CALCULATE RISE/FALL TIME FROM THOSE POINTS
    '''

    filename = os.path.basename(filepath)
    
    y = signal.detrend(y, type="constant")
    
    # if point is so far away from the ninety point, pop and use the new last element, or just increment to -2

    y_diff2 = np.gradient(np.gradient(y))
    peak_indices, peak_info = signal.find_peaks(y_diff2,height=0.05)

    peak_heights = peak_info['peak_heights']

    highest_peak_index = peak_indices[np.argmax(peak_heights)]
    second_and_third_highest_peak_indices = [peak_indices[np.argpartition(peak_heights,-2)[-2]], peak_indices[np.argpartition(peak_heights,-3)[-3]]]

    buff = 5

    first_cutoff_index = min(second_and_third_highest_peak_indices)-buff
    second_cutoff_index = max(second_and_third_highest_peak_indices)+buff

    y_windowed = y[first_cutoff_index:second_cutoff_index]
    x_windowed = x[first_cutoff_index:second_cutoff_index]

    # plt.plot(x_windowed,y_windowed)
    # plt.show()

    five = 0.05*float(voltageLimit)
    ten = 0.1*float(voltageLimit)
    ninety = 0.9*float(voltageLimit)
    ninetyfive = 0.95*float(voltageLimit)
    half = 0.5*float(voltageLimit)
    
    positive_ten = np.where(y_windowed>=five)
    positive_ninety = np.where(y_windowed>=ninetyfive)
    negative_ten = np.where(y_windowed<=-five)
    negative_ninety = np.where(y_windowed<=-ninetyfive)
    

    ## Assigning min and max points of the pulse to variables ##
    positive_ten_rise = x_windowed[positive_ten][0]
    if x_windowed[0] < positive_ten_rise:
        positive_ten_rise = x_windowed[0]
        
    positive_ninety_rise = x_windowed[positive_ninety][0]
    positive_ninety_fall = x_windowed[positive_ninety][-1]

    negative_ten_fall = x_windowed[negative_ten][-1]
    if x_windowed[-1] > negative_ten_fall:
        negative_ten_fall = x_windowed[-1]

    negative_ninety_rise = x_windowed[negative_ninety][0]
    negative_ninety_fall = x_windowed[negative_ninety][-1]

    
    positive_rise_time = positive_ninety_rise-positive_ten_rise
    switch_time = negative_ninety_rise-positive_ninety_fall
    negative_fall_time = negative_ten_fall-negative_ninety_fall
    
    
    plt.plot(x_windowed,y_windowed,label="Placement therapy output", color = "blue")
    plt.xlabel(headers[0])
    plt.ylabel(headers[1])

    delay = 0.0001
    if negative_ten_fall != x_windowed[-1] and positive_ten_rise != x_windowed[0]:
        points = np.array([
                    [positive_ten_rise,y_windowed[positive_ten][0]],
                    [positive_ninety_rise,y_windowed[positive_ninety][0]],
                    # [positive_ten_fall,y_windowed[positive_ten][-1]],
                    [positive_ninety_fall,y_windowed[positive_ninety][-1]],
                    # [negative_ten_rise,y_windowed[negative_ten][0]],
                    [negative_ninety_rise,y_windowed[negative_ninety][0]],
                    [negative_ten_fall,y_windowed[negative_ten][-1]],
                    [negative_ninety_fall,y_windowed[negative_ninety][-1]]
                ])
    elif negative_ten_fall == x_windowed[-1] and positive_ten_rise == x_windowed[0]:
        points = np.array([
                    [positive_ten_rise,y_windowed[0]],
                    [positive_ninety_rise,y_windowed[positive_ninety][0]],
                    [positive_ninety_fall,y_windowed[positive_ninety][-1]],
                    [negative_ninety_rise,y_windowed[negative_ninety][0]],
                    [negative_ten_fall,y_windowed[-1]],
                    [negative_ninety_fall,y_windowed[negative_ninety][-1]]
                ])
    elif negative_ten_fall == x_windowed[-1] and positive_ten_rise != x_windowed[0]:
        points = np.array([
                    [positive_ten_rise,y_windowed[positive_ten][0]],
                    [positive_ninety_rise,y_windowed[positive_ninety][0]],
                    [positive_ninety_fall,y_windowed[positive_ninety][-1]],
                    [negative_ninety_rise,y_windowed[negative_ninety][0]],
                    [negative_ten_fall,y_windowed[-1]],
                    [negative_ninety_fall,y_windowed[negative_ninety][-1]]
                ])
    elif negative_ten_fall != x_windowed[-1] and positive_ten_rise == x_windowed[0]:
        points = np.array([
                    [positive_ten_rise,y_windowed[0]],
                    [positive_ninety_rise,y_windowed[positive_ninety][0]],
                    [positive_ninety_fall,y_windowed[positive_ninety][-1]],
                    [negative_ninety_rise,y_windowed[negative_ninety][0]],
                    [negative_ten_fall,y_windowed[negative_ten][-1]],
                    [negative_ninety_fall,y_windowed[negative_ninety][-1]]
                ])

    plt.scatter(points[:,0],points[:,1],marker="x",color="red")
    
    one_mark = x_windowed[-1]/second_and_third_highest_peak_indices[0]
    two_mark = x_windowed[-1]/highest_peak_index
    three_mark = x_windowed[-1]/second_and_third_highest_peak_indices[1]

    plt.axhline(ten, xmin=one_mark, xmax=two_mark, label = "10% of set voltage, {:.2f}V".format(ten), linestyle = "--", color = "magenta")
    plt.axhline(ninety, xmin=one_mark, xmax=two_mark, label = "90% of set voltage, {:.2f}V".format(ninety), linestyle = "--", color = "green")
    plt.axhline(-ten, xmin=two_mark, xmax=three_mark, label = "-{:.2f}V".format(ten), linestyle = "--", color = "magenta")
    plt.axhline(-ninety, xmin=two_mark, xmax=three_mark, label = "-{:.2f}V".format(ninety), linestyle = "--", color = "green")

    microsecond = 1000000
    
    plt.text(positive_ten_rise-delay,half,"Rise time: {:.4f} $\mu$s".format(positive_rise_time*microsecond),fontsize="small")
    # plt.text(positive_ninety_fall+delay,half,"Fall time: {:.4f} $\mu$s".format(positive_fall_time*microsecond),fontsize="small")
    plt.text(positive_ninety_fall+delay,half,"Time: {:.4f} $\mu$s".format(switch_time*microsecond),fontsize="small")
    # plt.text(negative_ten_rise-delay,-half,"Rise time: {:.4f} $\mu$s".format(negative_rise_time*microsecond),fontsize="small")
    plt.text(negative_ninety_fall+delay,-half,"Fall time: {:.4f} $\mu$s".format(negative_fall_time*microsecond),fontsize="small")
    
    plt.text(min(x_windowed)+delay/2,max(y_windowed)+0.9,"ST-0001-066-101A, {}".format(str_datetime_rn),fontsize="small")
    
    # plotting options
    plt.title("Guinness Generator Placement Bipolar Pulse\nSet Voltage = {}V, Input file name: '{}'".format(voltageLimit, filename))
    # plt.xlim(min(x_windowed),max(x_windowed))
    plt.ylim(min(y_windowed)-1,max(y_windowed)+1)
    plt.legend(loc="upper right")

    # display the plot
    plt.show()


def placementTimingNormal(x, y, voltageLimit, filepath, str_datetime_rn, headers):
    filename = os.path.basename(filepath)
    y = signal.detrend(y, type="constant")

    y_diff2 = np.gradient(np.gradient(y))
    peak_indices, peak_info = signal.find_peaks(y_diff2,height=0.05)

    peak_heights = peak_info['peak_heights']

    highest_peak_index = peak_indices[np.argmax(peak_heights)]
    second_and_third_highest_peak_indices = [peak_indices[np.argpartition(peak_heights,-2)[-2]], peak_indices[np.argpartition(peak_heights,-3)[-3]]]

    buff = 200

    first_cutoff_index = min(second_and_third_highest_peak_indices)-buff
    second_cutoff_index = max(second_and_third_highest_peak_indices)+buff

    y_windowed = y[first_cutoff_index:second_cutoff_index]
    x_windowed = x[first_cutoff_index:second_cutoff_index]

    # plt.plot(x_windowed,y_windowed)
    # plt.show()

    ten = 0.1*float(voltageLimit)
    ninety = 0.9*float(voltageLimit)
    half = 0.5*float(voltageLimit)
    
    positive_ten = np.where(y_windowed>=ten)
    positive_ninety = np.where(y_windowed>=ninety)
    negative_ten = np.where(y_windowed<=-ten)
    negative_ninety = np.where(y_windowed<=-ninety)
    
    positive_ten_rise = x_windowed[positive_ten][0]
    positive_ten_fall = x_windowed[positive_ten][-1]
    positive_ninety_rise = x_windowed[positive_ninety][0]
    positive_ninety_fall = x_windowed[positive_ninety][-1]
    negative_ten_rise = x_windowed[negative_ten][0]
    negative_ten_fall = x_windowed[negative_ten][-1]
    negative_ninety_rise = x_windowed[negative_ninety][0]
    negative_ninety_fall = x_windowed[negative_ninety][-1]
    
    positive_rise_time = positive_ninety_rise-positive_ten_rise
    positive_fall_time = positive_ten_fall-positive_ninety_fall
    
    negative_rise_time = negative_ninety_rise-negative_ten_rise
    negative_fall_time = negative_ten_fall-negative_ninety_fall
    
    plt.plot(x_windowed,y_windowed,label="Placement therapy output", color = "blue")
    plt.xlabel(headers[0])
    plt.ylabel(headers[1])

    delay = 0.0001
    points = np.array([
                [positive_ten_rise,y_windowed[positive_ten][0]],
                [positive_ninety_rise,y_windowed[positive_ninety][0]],
                [positive_ten_fall,y_windowed[positive_ten][-1]],
                [positive_ninety_fall,y_windowed[positive_ninety][-1]],
                [negative_ten_rise,y_windowed[negative_ten][0]],
                [negative_ninety_rise,y_windowed[negative_ninety][0]],
                [negative_ten_fall,y_windowed[negative_ten][-1]],
                [negative_ninety_fall,y_windowed[negative_ninety][-1]]
            ])

    plt.scatter(points[:,0],points[:,1],marker="x",color="red")
    
    # one_mark = x_windowed[-1]/second_and_third_highest_peak_indices[0]
    # two_mark = x_windowed[-1]/highest_peak_index
    # three_mark = x_windowed[-1]/second_and_third_highest_peak_indices[1]
    one_mark = 0
    two_mark = 0.6
    three_mark = 0.4
    four_mark = 1

    plt.axhline(ten, xmin=one_mark, xmax=two_mark, label = "10% of set voltage, (+/-) {:.2f}V".format(ten), linestyle = "--", color = "magenta")
    plt.axhline(ninety, xmin=one_mark, xmax=two_mark, label = "90% of set voltage, (+/-) {:.2f}V".format(ninety), linestyle = "--", color = "green")
    plt.axhline(-ten, xmin=three_mark, xmax=four_mark, linestyle = "--", color = "magenta")
    plt.axhline(-ninety, xmin=three_mark, xmax=four_mark, linestyle = "--", color = "green")
    
    microsecond = 1000000
    
    plt.text(positive_ten_rise+delay,half,"Rise time: {:.4f} $\mu$s".format(positive_rise_time*microsecond),fontsize="small")
    plt.text(positive_ninety_fall-2*delay,half,"Fall time: {:.4f} $\mu$s".format(positive_fall_time*microsecond),fontsize="small")
    plt.text(negative_ten_rise+delay,-half,"Rise time: {:.4f} $\mu$s".format(negative_rise_time*microsecond),fontsize="small")
    plt.text(negative_ninety_fall-2*delay,-half,"Fall time: {:.4f} $\mu$s".format(negative_fall_time*microsecond),fontsize="small")
    
    plt.text(min(x_windowed)+delay/2,max(y_windowed)+0.9,"ST-0001-066-101A, {}".format(str_datetime_rn),fontsize="small")
    
    # plotting options
    plt.title("Guinness Generator Placement Bipolar Pulse\nSet Voltage = {}V, Input file name: '{}'".format(voltageLimit, filename))
    # plt.xlim(min(x_windowed),max(x_windowed))
    plt.ylim(min(y_windowed)-1,max(y_windowed)+1)
    plt.legend(loc="upper right")

    # display the plot
    plt.show()


def lowresRiseFall(filepath,voltageLimit):
    datetime_rn = datetime.datetime.now()
    str_datetime_rn = datetime_rn.strftime("%d-%b-%Y, %X %Z")
    
    headers = ["Time", "Voltage"]
    csvFile = open(filepath)
    csvArray = np.genfromtxt(csvFile, delimiter=",")
    x = csvArray[2:-2,0]
    y = csvArray[2:-2,1]
    
    placementTimingLowResInput(x, y, voltageLimit, filepath, str_datetime_rn, headers)


def normalRiseFall(filepath,voltageLimit):
    datetime_rn = datetime.datetime.now()
    str_datetime_rn = datetime_rn.strftime("%d-%b-%Y, %X %Z")
    
    headers = ["Time", "Voltage"]
    csvFile = open(filepath)
    csvArray = np.genfromtxt(csvFile, delimiter=",")
    x = csvArray[2:-2,0]
    y = csvArray[2:-2,1]
    
    placementTimingNormal(x, y, voltageLimit, filepath, str_datetime_rn, headers)


def audioDelay(x, place, audio, voltageLimit, filepath, str_datetime_rn, headers):
    filename = os.path.basename(filepath)
    
    y = signal.detrend(y, type="constant")
    
    
def guinnessAudioSync(filepath,voltageLimit):
    datetime_rn = datetime.datetime.now()
    str_datetime_rn = datetime_rn.strftime("%d-%b-%Y, %X %Z")
    
    headers = ["Time", "Voltage"]
    csvFile = open(filepath)
    csvArray = np.genfromtxt(csvFile, delimiter=",")
    x = csvArray[2:-2,0]
    place = csvArray[2:-2,1]
    audio = csvArray[2:-2,2]
    
    audioDelay(x, place, audio, voltageLimit, filepath, str_datetime_rn, headers)

help_button_base64 = b'iVBORw0KGgoAAAANSUhEUgAAABgAAAAYCAYAAADgdz34AAAACXBIWXMAAAsTAAALEwEAmpwYAAABq0lEQVRIie1VPS8EURTdIJEIEvOxGyRCUPADJFtIBIVEva1CQ6NQCLHz3lMqdKJQoNNJJKKhoNDRiUjQqXwEEcy6b5LnDEE28+Zjl46b3GrmnnPvuWfupFJ/IswpVWdzL2dxuWRx2rQZbdtcrqQZjZtCNZUN3ChUjcWI25weAaj0SZ7F5Koxq5pL6xqd2UwehgMXJya7NYXsSwReL5SBorOk4F8kjFxDUE8sATTeiuj0GpKchxIxurSFqg0FTzM5ENVlRritKaGqsPCLcBI5F0qAMTeiZZDDmbzbhvduIqS6QhMVAfCOCVUNCZ5K1V6XsHA2qH3ebY8sZLSOCeYhz0Hswp3XUY3+lI0pPIK+C77GsVMwmgkQGE6hK0HhXhICk9NYgODjJJD3GwS2kENaF+Hh/k8J4KJn/8zobYrlxO7A8UYAshtKwOWaFvw9cqoSICchxcuQcPo7deD00iBUSziBv2xR6AbAQzn+19pTF7iM/SC5Sw6Os81pMhH4Z1hOoRNFO/HWladpIQdLAi8iYtSLxS3iKz6Gi+59nW3/nOPL9v90/vErG/w/3gALBuad4TTYiQAAAABJRU5ErkJggg=='

THD_eq = b'iVBORw0KGgoAAAANSUhEUgAAAWgAAABGCAYAAADhNA4nAAAAAXNSR0IArs4c6QAABAR0RVh0bXhmaWxlACUzQ214ZmlsZSUyMGhvc3QlM0QlMjJFbGVjdHJvbiUyMiUyMG1vZGlmaWVkJTNEJTIyMjAyMy0xMC0yMVQwNCUzQTQ5JTNBMzguMTYzWiUyMiUyMGFnZW50JTNEJTIyTW96aWxsYSUyRjUuMCUyMChXaW5kb3dzJTIwTlQlMjAxMC4wJTNCJTIwV2luNjQlM0IlMjB4NjQpJTIwQXBwbGVXZWJLaXQlMkY1MzcuMzYlMjAoS0hUTUwlMkMlMjBsaWtlJTIwR2Vja28pJTIwZHJhdy5pbyUyRjIyLjAuMiUyMENocm9tZSUyRjExNC4wLjU3MzUuMjg5JTIwRWxlY3Ryb24lMkYyNS44LjQlMjBTYWZhcmklMkY1MzcuMzYlMjIlMjBldGFnJTNEJTIyd0NoN3hnaFpOQm1GQXdibmxiekolMjIlMjB2ZXJzaW9uJTNEJTIyMjIuMC4yJTIyJTIwdHlwZSUzRCUyMmRldmljZSUyMiUzRSUzQ2RpYWdyYW0lMjBuYW1lJTNEJTIyUGFnZS0xJTIyJTIwaWQlM0QlMjIzMVRZSHVjRVRjVWJWOXRfRUNYWCUyMiUzRWpaTnRiNEl3RU1jJTJGRFlrdW1RRTZuYjRjNkhSWnRpVnpjUzlOcFNjMEZzcEtGZHluM3dGRlJHT3lOOXI3M1VONyUyRnpzczRzZkZYTkUwZXBNTWhPWGFyTERJMUhKZHg3RnQlMkZDdkpzU1lqMTZsQnFEZ3pRUzFZOGw4dzBPU0ZlODRnNndScUtZWG1hUmNHTWtrZzBCMUdsWko1TjJ3clJmZldsSVp3QlpZQkZkZjBtek1kMVhROHRGdSUyQkFCNUclMkJyTGhtRGJCcGtRV1VTYnpHbFV4WkdZUlgwbXA2MU5jJTJCQ0JLOFJwZDZrTFBON3luaHlsSTlIOFNpbzMlMkZlbGlNUHphZk8zaDM1MlR5RWszdjNicktnWXE5YWRnYTJWOExQTmlWYVZmOURPOTYyWSUyRlN2ZFhhdFlhek1zRnlQZnhkcmNtRiUyRmRDeEI0TkJ2NDh2d2tTbmRQU3h1QkZESHh1Rk5SVDRmaSUyRlNzVURnNERIVFN1N0FsMElxSkls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'

voltage_ramp_example = b'iVBORw0KGgoAAAANSUhEUgAAAmIAAAB6CAYAAAAcTD85AAAAAXNSR0IArs4c6QAACIB0RVh0bXhmaWxlACUzQ214ZmlsZSUyMGhvc3QlM0QlMjJFbGVjdHJvbiUyMiUyMG1vZGlmaWVkJTNEJTIyMjAyMy0xMC0yMVQxNiUzQTM4JTNBNTAuODEwWiUyMiUyMGFnZW50JTNEJTIyTW96aWxsYSUyRjUuMCUyMChXaW5kb3dzJTIwTlQlMjAxMC4wJTNCJTIwV2luNjQlM0IlMjB4NjQpJTIwQXBwbGVXZWJLaXQlMkY1MzcuMzYlMjAoS0hUTUwlMkMlMjBsaWtlJTIwR2Vja28pJTIwZHJhdy5pbyUyRjIyLjAuMiUyMENocm9tZSUyRjExNC4wLjU3MzUuMjg5JTIwRWxlY3Ryb24lMkYyNS44LjQlMjBTYWZhcmklMkY1MzcuMzYlMjIlMjBldGFnJTNEJTIyYTNIT1NLTjJVOTFxVFVYdDVHMl8lMjIlMjB2ZXJzaW9uJTNEJTIyMjIuMC4yJTIyJTIwdHlwZSUzRCUyMmRldmljZSUyMiUzRSUzQ2RpYWdyYW0lMjBuYW1lJTNEJTIyUGFnZS0xJTIyJTIwaWQlM0QlMjJaMTl2ZHpocU84NmJXSkxmcUxfciUyMiUzRTdaeGRjOXNvRklaJTJGalMlMkJUa1JENnVteWNKdTFzZDdiYjdIUm5lNmUxaUsySkxEd3lpZTMlMkIlMkJrVXh5T2JnT0VLeGNEZURMekxXRVJ6SjUzbUJBeElaQmVQNSUyQnJiT0ZyUGZhVTdLRWZMeTlTaTRIaUVVSTh6JTJGTm9iTjFvQlJzalZNNnlMZm12eWQ0YTc0U1lUUkU5YkhJaWRMcFNDanRHVEZRalZPYUZXUkNWTnNXVjNUbFZyc25wYnFWUmZabEdpR3UwbFc2dGElMkZpNXpOdHRZazlIYjJUNlNZenVTVmZVJTJCY21XZXlzREFzWjFsT1YzdW00T01vR05lVXN1MjMlMkJYcE15aVoyTWk3YmVqY3ZuRzF2ckNZVjYxTGh4N2YxUCUyRmhQZG5QNzVYUDFkVEwyNnVMcXg0WHc4cFNWaiUyQklIaTV0bEd4bUJhVTBmRjZJWXFSbFpINHA3OXE4czd1bjM1YmUlMkZscXVFMERsaDlZWVhrUklSTllSQVVDS09WN3R3UjFJTnMlMkYxUXkwdGxBdkcwZGIyTEF2OGlBbkU0S01WVGRmMEh1czElMkI4elljME0lMkY4Z1Q1OHUlMkZDRDE2TkNxdnhESXk5JTJCVk5HS0c2OW1iTTZ2Y3UzenJ6eGlWVTZhUzNqOFNBJTJGSU1SUWtWd1NwaDIwdkx1R0JzRWhiVGNxTUZVJTJCcWpBJTJCRlNsemhLeTM0N2JWVVFwVktCR0s5cEklMkYxaElnNiUyQjZLRGJ1TGpmbGhXVHduVCUyRkR4amEzJTJGekcwaGlSeEtRMUpwTlo1UjIwVVVkR3VGZTE5UVZTdWN1ckd0JTJGRlYxRzZrZnJ2WEJ5cVBmeVlNV2hPclB3cEUxZ3lXcjZRTWEwcFBWejNjRGpuNXViOW93Y0pOSEI1bktVOUxuYml3QWFCcGVCOHVuWFhGcnEzZHdPM1JGR0E2aGduJTJGWEx1dmlmcWlBeXhDN3JlYUJaWTd1Y1k4ZlphSFNVZlBwaWY4SE5tVldRT0JVWXFjQ1hzOUkzeWtEemMxNGRvQ0hHJTJGbmV0Z3hDcCUyRkpMaiUyRkRyckF2akZpVjBkdU5IZlVBZVIybzU3Y3dkJTJCTURwdmYlMkJDeUEwTWRKT3E0SHFVbm1SdEF0MEY4enJrQmN0bUNtU3FRcDA3dFlueDBLdDlWRmRCdGdPektJSFV5TUpPQm5GUUxYa25QaFRYb1IxdFNzanRJWUpjMEd1cEFObFNwZyUyRkFrZ3dSMGkzemcxbTczZ0YwT2FTZ0xyT1olMkJLWndMOXBRRmNPdEhkbVhnVWtoREdZQ2xvYlF2ZCUyQkRIaDNLeXJBT1hOQnJxQUR3UFNPSFNRazlaQUxkMkZ4YXd5eGtOVlpDcXVaN3Z3Vkc5cHd5QVg1aUMySjFSeW1BNVdYU1VSZUFqSUl1ZUMwN1FFVnh2aWtLN1F1andmcEVUd2o0JTJGOEFEaHRkY0VPdXZDOHNBZzE3c2QlMkJLN2dNVXp2VGdRZVk3dmczZnpBRUh6a0RRUGU4cXQyb1pzUUdJS1BZU3A0SXZDeDNabGc2T1lBaHVCVG1PeWRDTHowYSUyQnNOVFpmbG00SEgzakRKWGV2WEZuaVgxUnVDeDhPTThhMWZXJTJCQmRWbThJSGcyVDFiZCUyQmJZRjMlMkJ5aTRGaFdVQUdU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'

pulse_burst_example = b'iVBORw0KGgoAAAANSUhEUgAAAlAAAAChCAYAAAAIs4HQAAAAAXNSR0IArs4c6QAACAB0RVh0bXhmaWxlACUzQ214ZmlsZSUyMGhvc3QlM0QlMjJFbGVjdHJvbiUyMiUyMG1vZGlmaWVkJTNEJTIyMjAyMy0xMC0yMVQxNiUzQTM5JTNBNDUuNzg5WiUyMiUyMGFnZW50JTNEJTIyTW96aWxsYSUyRjUuMCUyMChXaW5kb3dzJTIwTlQlMjAxMC4wJTNCJTIwV2luNjQlM0IlMjB4NjQpJTIwQXBwbGVXZWJLaXQlMkY1MzcuMzYlMjAoS0hUTUwlMkMlMjBsaWtlJTIwR2Vja28pJTIwZHJhdy5pbyUyRjIyLjAuMiUyMENocm9tZSUyRjExNC4wLjU3MzUuMjg5JTIwRWxlY3Ryb24lMkYyNS44LjQlMjBTYWZhcmklMkY1MzcuMzYlMjIlMjBldGFnJTNEJTIyQlNjWWtZUzhQYnBUUnE3LVRjRlolMjIlMjB2ZXJzaW9uJTNEJTIyMjIuMC4yJTIyJTIwdHlwZSUzRCUyMmRldmljZSUyMiUzRSUzQ2RpYWdyYW0lMjBuYW1lJTNEJTIyUGFnZS0xJTIyJTIwaWQlM0QlMjJ0OWxNOG8wbURaWFRvNkR0b3lfMCUyMiUzRTdacGRjNXM0RklaJTJGalMlMkZONkJ0eG1iakpkanJkMmM1bXQ1bGU3UkJRREMwZ2l1VTQyViUyQiUyRmtpMFpLQ1RCY1p5MVd6eE1BZ2R4Qk8lMkJqYzRRa0puaVczJTJGOVdoV1h5dTR4Rk5rRWd2cCUyRmdkeE9FcGhBeCUyRmM5WUhqWVdndmpHTUslMkZTZUdPQ3RlRXElMkZWZFlJN0RXWlJxTFJhdWdrakpUYWRrMlJySW9SS1JhdHJDcTVLcGQ3RlptN1ZyTGNDNDZocXNvekxyVzZ6Uld5Y2JLS2FqdDcwVTZUMXpORU5nemVlZ0tXOE1pQ1dPNWFwand4UVRQS2luVlppJTJCJTJGbjRuTWlPZDAyVngzJTJCY2paN1kxVm9sQkRMcGglMkYlMkZ2dkRkOGhSTnYzeVIzNkJaOG5YUEpwYUwzZGh0clFQYkc5V1BUZ0Y1cFZjbHJhWXFKUzQ3OU05dkhIRlFmZSUyQjRQWnBkVE1STWhlcWV0QkZyQ1B1MjB0c0M4SE94YXJXbXdiV2xqUzFadFlZV3NienJlOWFCcjFqbGVoWDVRTmxkNWZYSUlyJTJGdWY2VFVTeSUyRmZ3bmVUekhhU1pibXN6NnA4bEQ1SHRXSzRaWlV5UE1oNUl3UXdIdyUyRllBeDJoWU93UnpqcWU0aHppQ2tOc0s0UzB3UHBDSGRyWGswZG4lMkJheXQ1Q0lJZzhFdVA2eGpuU0ljaThBclA2UiUyRjFQSkFTMVNKNWpTN0tiNU9xZWRHNVZTbmNrJTJCaGpjaSUyQnlRWHFVcGxvYyUyRmZTS1Zrcmd0azVzUjVHSDB6RElwNEpqTlo2Zk94dUEyWG1XcDRPTXZTdWJsU3lWSmIxJTJGN1BGdVVtM1FKbjBmdUpVaVl0bjVsblJaY0xGZXFyUFpXSVFyc3ZLJTJGbFZYJTJCRkZ1bXAwV1JaeiUyRlJmN0FVUlRCSUJuanZINXdHWUFINHVuTHU0R1QySndFaG9ndCUyRlhnSk1SakdQbU1Zd1lwaHdFNkZFNDg0bndKem52M0V1RUJBQWdrZkxPaFR2QWVGV3d5d3Q0RE5pY0dOa2NCMjJ6a3VHSFRFZlllc1BWNFFkT0dpTGdmUFc3YWJLUzlEMjNHRFcyMmhla2ZOMjElMkZwTDBIYlQybTE3UVI4SW5ianBzMlA3ckJ5NjZqa3plVEtoZ0RvOXR1VG5aMDRpWmdScHk3NFR6SjBRa2FNRWN6d3Y1SlJpZU83UWo3UmJCUGJIU0N4bG1tdldpZjF1Z0VqZE5NJTJCOUElMkJzZEVKSGhEYm9valB6QktvUGlwa1lYZ25LdGUxdklONmQ4MVR4QlpQVjlhbkZySkUzRm8wZmZMRmx2YXMzRGxiSlRMTiUyRjY3cHExOHFXOE1ubWVyYjJ5S2o3WVZEbjN1TnBURGdSdXZPMzBJdXEwaFlGN1g2WGElMkZCVG01VldNMkY2cmhkUTkwcXNnZm5BVkg5aTNHR0ZIZ0IwNEVJQTElMkZIS1hUejZ6dURicnRsWGl0dVFmQzJuTHR6UzU5bHBrelMlMkZCRzNUbWlxelhpaEt2bE51T1JzMjhCdG1tVSUyRm1FS2JwQ05OWDFROTJUdFA0OWhVYzc1S1VpV3V5akF5ZGE2cXNPeTBwRXFhM0wzdUxhYkI3azFyUUdKMlhKQ0glMkZjWWlheWN0czU2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'

tone_sync_example = b'iVBORw0KGgoAAAANSUhEUgAAAkQAAACYCAYAAAAFg/YnAAAAAXNSR0IArs4c6QAACcJ0RVh0bXhmaWxlACUzQ214ZmlsZSUyMGhvc3QlM0QlMjJFbGVjdHJvbiUyMiUyMG1vZGlmaWVkJTNEJTIyMjAyMy0xMS0yN1QyMSUzQTQ4JTNBMTguMzcxWiUyMiUyMGFnZW50JTNEJTIyTW96aWxsYSUyRjUuMCUyMChXaW5kb3dzJTIwTlQlMjAxMC4wJTNCJTIwV2luNjQlM0IlMjB4NjQpJTIwQXBwbGVXZWJLaXQlMkY1MzcuMzYlMjAoS0hUTUwlMkMlMjBsaWtlJTIwR2Vja28pJTIwZHJhdy5pbyUyRjIxLjYuNSUyMENocm9tZSUyRjExNC4wLjU3MzUuMjQzJTIwRWxlY3Ryb24lMkYyNS4zLjElMjBTYWZhcmklMkY1MzcuMzYlMjIlMjBldGFnJTNEJTIyWGVnYjhTMmlram9hazFUU1FxYnolMjIlMjB2ZXJzaW9uJTNEJTIyMjEuNi41JTIyJTIwdHlwZSUzRCUyMmRldmljZSUyMiUzRSUzQ2RpYWdyYW0lMjBuYW1lJTNEJTIyUGFnZS0xJTIyJTIwaWQlM0QlMjJFMm5DMDhmTGtfak5Ta2ttWXpaRCUyMiUzRTdWeGJjOW82RVA0MXpQUTh4R05KbGklMkJQdVRUdFMyYlNTU2JudkxwWUFVJTJCTlJZMElTWCUyRjlrYkVNRm5JczZrdHNpSGxnOENLdnpYN2ZybmFYaFFtNlhyeCUyQlMlMkZ6bCUyRkk0R0pKcEFNM2lkb0pzSmhKN3I4ZWRVOEpZSk1MWXl3U3dKZzB3RTlvS0g4QThSUWxOSTEyRkFWdEpDUm1uRXdxVXNuTkk0SmxNbXlmd2tvUnQ1MlRPTjVLc3UlMkZSbFJCQTlUUDFLbCUyRjRZQm0yZFNGNXQ3JTJCWGNTenViNWxZRXAzbG40JTJCV0loV00zOWdHNEtJdlIxZ3E0VFNsbjJhdkY2VGFMVWRybGRzdk51MzNsM2QyTUppZGt4SjJ4JTJCdyUyQjkzUCUyQjh1bng3JTJGJTJCN1glMkI4WGolMkZJM2lJTDJ5aDVzV1AxdUlUaTd0bGI3a0paZ2xkTDhVeWtqRHlXbVo0JTJGMmUlMkIzRlJ2RE93JTJCTHFjSm9RdkNramUlMkJKRmVFY0hhS29BaHdoSXJOM3VEWThReFhMSnNYN1owRDRRdWNaenYxZTFQd0Y4SWFmMkdaSXd4RDR1QXlwUmclMkZpbW5NaFZkenR1QVh1UUg4SlRkYUhKRDBDaVklMkZ5czRsZ1VLMnZaRXE4RkZOVnpTTnFab2xseVVrOGxuNElsJTJCMHpGVGlDdmMwNUxlelF3WlpCdlRNJTJGUU5KT0xtZVlkb1F5MHBYZEoxTWlkQlRaT09CYWd4azNVQ3Ztdm5KakRCRjlSYmRuV25xQXc1SHdEV0FLODUyTE5nYXRkYkhnNDNyZ0IzNHElMkZrV1lGQ0YlMkZJb2w5QmU1cGhGTnRvcFElMkJvbHZiOFg1OXo1akpJbTNaMElUN05ibm13czhVJTJCNmNXYkN3UiUyRjYwelI5SEV5U3dnVUJkQWxYclJ0QndGZFVkRThnWkNkUTJnWUIzZ0xMYklvTTB5dnVna0R0U3FQVTlySEtqYWNhZ2F0MTlFTWdiQ2RRMmdheURMQWkwdVl0cGxQZEJvYnhOMDFMVnBORG05dGJsSDdReEhmajliRzFjc1JDNmZmTG1BaHVPNmUwZjh1N2ltQTFZVTUwOGw2cnV1dEpHSjBLYXpBNVY5SWU5N2xhVldlMlprUVljMFo5aFNlakhzJTJGVG9hak1QR1hsWSUyQmx1ZjN5VCUyQlVtYlFjeGhGWll4NW4wc0paUndyR2slMkJLJTJGWjJTZm1qdERTcnZtRmVDYWp1R1o3dU9NSDJCWnNnMFRPbmhxTFRqbXNIdTNOWWJwckJXVDJXSVcwRk9OYTM3UTZ0UDl3Y0F5bFNSMiUyQmhuNXYlMkZRJTJCaHolMkJEMEZsb1Rua0FGQ3JwekhFQUpCelRSOEE3RDREQUVTb2lpcm5GZ0NPNkxtZVF3QkFoMVhleWZoJTJGclg3Q0lQM2ZQdGIlMkZleTBhRVhiJTJCMWtkUDJQMlBhSGVlZyUyRnRidUxMSk0yRCUyRlIlMkJySXhPVTZDR25xU056WFZ5cGM1SlhKc01qbUZ4R2lpSlVRJTJCVkU0UzVHWWNsTVRMcjlLZ1FpbmZuUXAzbGlFUWZBdUQlMkJRNDgweGpWZ1o0S244UU53dnk0MnclMkJDWUJPR1dDYkV1VDRBRm1nWU8lMkJXeEJqdXdOanBDbW0xRnIlMkJQdUlrWFpCdERYbWpFMHJHcWs4QTd0ZWkyTVQwWXZIY2paQVdFblJLRW9XMGN4dUQyRUZaYmRJJTJGaGdrTnFmbG45TTJCZ2N3RGhBWUNvU3dBMTR3bllNSzEwUUVHOEN4VndiV3hndSUyQkRnVU1YYVRkdkRxTHZRclJiWFQ3a1RtMSUyQmVUaFh4d3A1OTRYVzZhMSUyRkF3d2ElMkJJNU1BR25iNXZ1MldSM2NwbENNRHBTZGJCOThPdEUlMkJEZGl2NFNhTThmYUJ6S053VGElMkJiVTFaM2RFc1VkWjlTbzFnREI1MExiTnF6NlkydlZ1aUV3U2tKNjE1aTNXNk9mSnVZYVIyeUVlYlh1WGpDM2FuMUhmOXljaHp6TndRT0s2U3JUSEthQno1UXF0czBUdDQ2WWdnenJ3NG1pbG5WNUFYJTJGRHpmaW1zR1pBV2VCSEZHcmJ6c3lrSkh0M3lwSzJkS3I5ZmJvMXl0T3NkZ2NvVHRNMWRlT2U5Uk0xelZkd0g1JTJCcFdVZDg5JTJGbnA4VzYwYld1VTk3TnZ0enRRY1pxbzYzeXhFZW9hNWYyZzN1RlBTODQ5VzlQTlRqUkoxM1JjNlNOZlV3djRNVjhyJTJGckRZR1V6Q05wYmQyaDlYMU0lMkZYcWljbVBqNWR3N1VLN3MlMkJGZHFOOXUxcDNMOXMySHY4dlFPdUl6VEN2MU4wUDVtcSUyRjVBak14MVJORDJlalRFM0RsQjRTTmF6VzcyT2lWcHg1OW9hU3FPR3g1dGIlMkZpTEYlMkJwcVlaYnVzaFZhdFZiWDh5dkJ2dDJ4cmwlMkZXemM0NXlEM2hlYm9WNnR2QiUyRlVPJTJGekRoSE5QMTNSVHlVM3lOUjFYZWtqWWJMV0NIeE8yNHNncTZENWg0NGY3UDBmTWdOMyUyRnd5VDYlMkJqOCUzRCUzQyUyRmRpYWdyYW0lM0UlM0MlMkZteGZpbGUlM0Xw7bFuAAAerklEQVR4Xu2dfaxsV1nGn+m90bZQNEBSEhQUSkokimkKjdL+QWMo8o+GGhApmvBhIwUBg0blo6e9YjSQFAwtgmBC2oJES5QIgUarhGIoIUBNiEUtEAiRCr0IxV6w93bMu2evc9eZ7jmz9p6118fev7m5uXNm1l7vu37PnjPPXevdey3EAwIQgAAEIAAB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'

rise_fall_example = b'iVBORw0KGgoAAAANSUhEUgAAAlAAAAD2CAYAAAAZOLmfAAAAAXNSR0IArs4c6QAACfx0RVh0bXhmaWxlACUzQ214ZmlsZSUyMGhvc3QlM0QlMjJFbGVjdHJvbiUyMiUyMG1vZGlmaWVkJTNEJTIyMjAyMy0xMS0yN1QyMSUzQTQ1JTNBNTMuMDk0WiUyMiUyMGFnZW50JTNEJTIyTW96aWxsYSUyRjUuMCUyMChXaW5kb3dzJTIwTlQlMjAxMC4wJTNCJTIwV2luNjQlM0IlMjB4NjQpJTIwQXBwbGVXZWJLaXQlMkY1MzcuMzYlMjAoS0hUTUwlMkMlMjBsaWtlJTIwR2Vja28pJTIwZHJhdy5pbyUyRjIxLjYuNSUyMENocm9tZSUyRjExNC4wLjU3MzUuMjQzJTIwRWxlY3Ryb24lMkYyNS4zLjElMjBTYWZhcmklMkY1MzcuMzYlMjIlMjBldGFnJTNEJTIycGZscjBPYWZCdW11aUd3NEhRQkwlMjIlMjB2ZXJzaW9uJTNEJTIyMjEuNi41JTIyJTIwdHlwZSUzRCUyMmRldmljZSUyMiUzRSUzQ2RpYWdyYW0lMjBuYW1lJTNEJTIyUGFnZS0xJTIyJTIwaWQlM0QlMjJad2FnM3hyZzFBLU5hejdZU2F3YSUyMiUzRTdWeGJkNk0yRVA0MU9hZDlpSTR1WEIlMkJUN0tZNXA3YzAyYlBiUGlwR3RqbUx3UVY1bmZUWFZ4aUJKUVQ0RXVQRkxYcElZSUFSelBkcE5Cb0dYNUc3eGV0UEtWM09mMDBDRmwxaEdMeGVrUTlYR0NPTFdPSmZMbmtySkI0a2hXQ1dob0U4YVN0NER2OWhVZ2lsZEJVR0xOTk81RWtTOFhDcEN5ZEpITE1KMTJRMFRaTzFmdG8waWZSZWwzVEdETUh6aEVhbTlFc1k4TGw4Q2h0dTVROHNuTTNMbmhHVVJ4YTBQRmtLc2prTmtyVWlJaCUyQnZ5RjJhSkx6WVdyemVzU2czWG1tWDRycjdscVBWamFVczV2dGNNSHY2OHZUem45NDZmcGlHJTJGSyUyQm5qdyUyRlUlMkJ1T2FlSVdhYnpSYXlTZVdkOHZmU2hQTTBtUzFsS2V4bExQWEpzUFRsJTJGSjBhTjRZcWg1WDhJUWxDOGJUTjNHS1ZHVEpLeVJEVUtsaHZiVzM3VXZaWExGMWRTR1ZHTThxMVZzemlBMXBpUU9zZ25ZYmhjWEJUVTR2c1JjbnNSRGV6dmxDZFBJQmlVMWhzRGhnZVE5UTdCWFhzc0FnMnRaQUhkaVlabFB0MG1DV1VwYXlpUEx3bTk1cGs2bGtENDlKS0c2blFvVmc0S3JOYXdhcFZKZ2xxM1RDcEE2VmhUVzFOcmFBcFRTN1d5Mm42WXh4USUyQjBHMWNva3h3T05SNkQ3QXJwRjdabUFKY2NBRzlCc3ZnRVRkYUdjOFRUNVdrMEd1SkxjSlZHU2JsUVRLTnI5JTJGY2lJUG9hJTJCd0l5JTJCS2FjdDh4T3k5b2RCdmdNY3JUVjIyM2JUR0JFQWJhVlolMkJ1VXVRSllJQjl6eXI5T3REU09BOWVjbUZsQ3Z4MzYzQXY4OVZ4UHM2c2k2bWpMUndBSEtpQTE4RjI4YjBaVkJHN2lxYmV4T2JSYjBnYXJzbmRxd0E1RGF1akFYR3dXcFR1cUJMTU1EJTJGUzZpcDJ5ZXgzdDFWeVJpS3E2N0hOMmxTUDgwRGFPb0pxSlJPSXZGN2tUNEV5Ymt0M21FRm9ydzlVWWVXSVJCa0hkenU1NkhuRDB2NlNUdmN5MkNkY094VFpPWVA4dWJ3dVYlMkJFWkFqWERtemhoRHdhSGRXalRFSUlDUktxOEVGUERsb0ZmJTJGbkllQ3Izc1F5M1NIUzZWd3FPWG00NkJwUWIxWUI5cEJ4bHJqQ1BuRkZCT21qVUIlMkJHeEl6elcyWWVGVlFQMkpvaklEMkJhaTZNJTJGQkhUM1pqVzRQR0dETEYlMkZ5aUF4bHo5U0xwQ01OeElNdDF4UWc4WCUyRndwSUJPUVI0YXZQM0NFb09qaGQzaEY0N2V1bDVnVkhtcDBieW5KZzhWdmVrdmZmeUU4SzZXaGt6bm8waFJ5VVhSb2JzWklpWWM0QzI1cktPcEFpQ2hsNEhLZzJkbHk5bXpnS05FY2ZPaUFQQiUyQmhKOHlDRUhzZzJReDJWaEMlMkZBRSUyQjhEM1clMkJkJTJGalBJQVFBSGV3SDFBUzBUa2pKUEJjWk1CZGozZ3RJNyUyQmxyelFFZG5xN3N6WWptNzZuaHJNRE1QSW5oT3dCJTJGbTF1ZU00OGxqMVFFS29KYjdTenZ2dUE0MnBpJTJCWTBvMlVEVDgweTFzblFsUDBlYWh4aDVpN0duT01lSUF0ZjE3aDJHQ2JLcFZIR0pjRmhJeGszcDRZR0N2S1lKJTJCaG5jcThqV0p1Rjk1N2RNVFQwMWpJT2ZWZXA3RkhOa00zcE10JTJCY3JOTG83VGFsazYlMkJNdHd6YXhzcUdmRzhhaGNzSHVSM1JGeFk5SmxuSXd5Um5VbHFnVjdtSFgyckhLemRST3BTSVRYbVhPOUVwaVhwZFNDSWYyS3ElMkYwUEJzU1RpcUZCYmhYQVBQVkVZNzNWdzduYk13WHl0ZlBoWFNoRk41clFoZXpScWJYc25oa2lveGZKR0UyQ094TUJMaW9IeWpKJTJGeDdlMTFMMjdReVVIcnNrVG00T0hxY2tRd1dzUUNCN2VWZmwwV0dwbUp2SiUyQkl5ZE5kWTRmeTlTc29EMTlrbXFMJTJGSmcwbDclMkJicXhkM2xjYk0zayUyRjQyaWwzZHBLVmdoQWtJTUgxZFJ4a3F0NG5GZjZqMEpXWEhYcFhqd2F5SzdUNmJhWmpVTlFnU1U3ODZNcEhkdkpHdXFxaGdVeVg1ak16cVM3RjBrcXdwR04lMkZsVGszajRPeENQSEZXUjhiOHE1amUlMkJtSGxuTWY5MyUyQm1xRG1HbXh3endEYWZFTVQySHVEZUNuY05IaUZBWSUyRiUyQmttdmtiSGxnbHJSdnExbjJvamhDaXlvRjRuakJxJTJGUVhFUGZnNGRveXJTTnZPbWROOTNlb3lYNkhTcUhtbkp4cCUyQkRRUGQxMFBuS29oVU91b3hkek9KZE1vcVlzM2tpaTNrbGslMkJVaiUyRkZxeVcycm1zMmN6TSUyRkJXNHd4JTJCeUg0ZU1kRXN0bWRYcndxWDcyME5VSyUyQjh4bHpTT0RkUlNkTnN4a1QlMkZYQzBOaTV2USUyQkp4SFBmOGhBWVAlMkY1VXJGWGs3NSUyQnI2JTJCUHI0a05TQzJSQjcybWo4d2FmM1lBZyUyRnEzQm50QUszYTNQJTJGVlFMSVMyUDVoQlB2NEwlM0MlMkZkaWFncmFtJTNFJTNDJTJGbXhmaWxlJTNFgQETugAAIABJREFUeF7tnQnYNEV1tu+ORhHIHxdUFKPgbtxxwwQTxQXUuAeIBPcFFTeMJu4oRnEN7kZFxaAmqGBcARVcAFdE1LihIi6IC7gLion9X883VVg0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'

# make it pretty:
# sg.theme_previewer()
sg.theme('LightGrey1')

treatmentLayout = [
                [sg.Text()],
                [sg.Text('File:'), sg.Push(), sg.Input(key="-TREATMENT_FILE-", do_not_clear=True, size=(50,3)), sg.FileBrowse()],
                [sg.Text()],
                [sg.Text('Voltage Limit:'), sg.Push(), sg.Input(key="-TREATMENT_VOLT-", do_not_clear=True, size=(50,3))],
                [sg.Text()],
                [sg.Text('', key="-ERROR_TREATMENT-", size=(70,3))],
                [sg.Text()],
                [sg.Button('Analyze Voltage Ramp', key="-TREATMENT_RAMP-"), sg.Button('Analyze Pulse Burst', key="-TREATMENT_PULSE-"), sg.Push(), sg.Button('', image_data=help_button_base64, button_color=(sg.theme_background_color(),sg.theme_background_color()), border_width=0, key='-TREAT_INFO-')]
                ]

placementLayout = [
                [sg.Text()],
                [sg.Text('File:'), sg.Push(), sg.Input(key="-PLACEMENT_FILE-", do_not_clear=True, size=(50,3)), sg.FileBrowse()],
                [sg.Text()],
                [sg.Text('Voltage Limit:'), sg.Push(), sg.Input(key="-PLACEMENT_VOLT-", do_not_clear=True, size=(50,3))],
                [sg.Text()],
                [sg.Text('', key="-ERROR_PLACEMENT-", size=(70,3))],
                [sg.Text()],
                [sg.Button('Analyze Bipolar Pulse', key="-PLACEMENT_PULSE-"), sg.Button('Analyze Bipolar Pulse - Low Res', key="-PLACEMENT_PULSE_LOWRES-"), sg.Button('Analyze Tone Sync', key="-PLACEMENT_TONE-"), sg.Push(), sg.Button('', image_data=help_button_base64, button_color=(sg.theme_background_color(),sg.theme_background_color()), border_width=0, key='-PLACE_INFO-')]
                ]

layout_win1 = [
                [sg.TabGroup([[sg.Tab("Treatment",treatmentLayout), sg.Tab("Placement", placementLayout)]])],
                [sg.Push(), sg.Button('Exit', button_color='red')]
                ]

win1 = sg.Window(title='Guinness Waveform Analyzer (ST-0001-066-101A)', layout=layout_win1)
win2_active = False
win3_active = False

info_txt_width = 138
info_txt_size = 9

while True:
    try:
        event, value = win1.read(timeout=2000)
        
        # When 2s elapse, reset the output window to blank. This sets a "timer" on any error text that is displayed there
        if sg.TIMEOUT_EVENT:
            value["-ERROR_TREATMENT-"] = ''
            win1["-ERROR_TREATMENT-"].update(value["-ERROR_TREATMENT-"])
            value["-ERROR_PLACEMENT-"] = ''
            win1["-ERROR_PLACEMENT-"].update(value["-ERROR_PLACEMENT-"])
        
        # Close application if the window is closed or if the "Exit" button is pressed
        if event == sg.WIN_CLOSED or event == 'Exit':
            break
        
        # TREATMENT INFORMATION WINDOW
        if event == '-TREAT_INFO-' and win2_active == False:
            win2_active = True
            layout_win2 = [[sg.Text("Instructions For Use: Treatment Analysis Options", font=('None',12,'bold'))],
                           [sg.Text("Capture the treatment output from the Guinness Generator on an oscilloscope and export the data from the oscilloscope screen as a .csv file. In this application, enter the filepath of the exported .csv file and the voltage limit set during the Guinness Generator treatment output.\n\nThere are several restrictions on the input .csv file to prevent errors and inaccuracies:",size=(info_txt_width, None), font=('None',info_txt_size))],
                           [sg.Text("1.  It must have only 2 columns: the first for timestamps, the second for voltage.\n2.  It must have at least 500 rows of data.\n3.  Its headers (if applicable) must be contained to the first row.",pad=(40,0), size=(info_txt_width,None), font=('None',info_txt_size))],
                           [sg.Text("\nWith the waveform filepath and voltage limit entered, click one of the buttons to analyze the inputs. See the following sections for details surrounding the function of each button.", size=(info_txt_width,None), font=('None',info_txt_size))],
                           [sg.Text("\nAnalyze Voltage Ramp", font=('None',info_txt_size+1,'underline'))],
                           [sg.Text("This button will take the treatment waveform input and look for the lines of best fit of the voltage ramp; this is done piecewise as the Guinness Generator should ramp at a different rate before and after the output voltage reaches 66% of the set voltage limit. For this function to work properly, the input .csv file should capture the voltage ramp of the Guinness Generator from 0V to the set Voltage Limit. For this function to work as intended, the input waveform should resemble the following:", size=(info_txt_width,None), font=('None',info_txt_size))],
                           [sg.Push(), sg.Image(voltage_ramp_example), sg.Push()],
                           [sg.Text("\nAnalyze Bipolar Pulse", font=('None',info_txt_size+1,'underline'))],
                           [sg.Text("This button will take the treatment waveform input and tranform it into the frequency domain via the Fourier Transform. Frequency will be plotted against amplitude; the more the frequency is present, the higher the plotted amplitude. These frequency amplitudes are used to calculate the Total Harmonic Distortion (THD) per the following equation:", size=(info_txt_width,None), font=('None',info_txt_size))],
                           [sg.Push(), sg.Image(THD_eq), sg.Push()],
                           [sg.Text("For this function to work as intended, the input waveform should be of the pulse burst and resemble the following:", size=(info_txt_width,None), font=('None',info_txt_size))],
                           [sg.Push(), sg.Image(pulse_burst_example), sg.Push()],
                           [sg.Button("Close")]]

            win2 = sg.Window(title="ST-0001-066-101A Information", layout=layout_win2, size=(1000,810))

        # If the treatment info window is up, read events and/or values
        if win2_active == True:
            win2_events, win2_values = win2.read()
            
            # Close treatment info window if the window is closed or if the "Close" button is pressed
            if win2_events == sg.WIN_CLOSED or win2_events == 'Close':
                win2_active  = False
                win2.close()
        
        # PLACEMENT INFORMATION WINDOW
        if event == '-PLACE_INFO-' and win3_active == False:
            win3_active = True
            layout_win3 = [[sg.Text("Instructions For Use: Placement Analysis Options", font=('None',12,'bold'))],
                           [sg.Text("Capture the placement output from the Guinness Generator on an oscilloscope and export the data from the oscilloscope screen as a .csv file. In this application, enter the filepath of the exported .csv file and the voltage limit set during the Guinness Generator placement output.\n\nThere are several restrictions on the input .csv file to prevent errors and inaccuracies:",size=(info_txt_width, None), font=('None',info_txt_size))],
                           [sg.Text("1.  It must have at least 500 rows of data.\n2.  Its headers (if applicable) must be contained to the first row.",pad=(40,0), size=(info_txt_width,None), font=('None',info_txt_size))],
                           [sg.Text("\nWith the waveform filepath and voltage limit entered, click one of the buttons to analyze the inputs. See the following sections for details surrounding the function of each button.", size=(info_txt_width,None), font=('None',info_txt_size))],
                           [sg.Text("\nAnalyze Bipolar Pulse", font=('None',info_txt_size+1,'underline'))],
                           [sg.Text("This button will take the placement waveform input and calculate the rise and fall times of both the positive and negative pulses. The times are measured between the 10% and 90% voltage levels of the signal.", size=(info_txt_width,None), font=('None',info_txt_size))],
                           [sg.Push(), sg.Image(rise_fall_example), sg.Push()],
                           [sg.Text("\nAnalyze Tone Sync", font=('None',info_txt_size+1,'underline'))],
                           [sg.Text("This analysis option is unique as it requires the input .csv file to have three columns instead of two. The first two columns must still be time and generator output voltage; the third column should be the recorded voltage of the generator's speaker. The input .csv file should capture at least two placement pulses and their accompanying audio tones. This function will calculate the delay between the placement pulses and their corresponding audio tones.", size=(info_txt_width,None), font=('None',info_txt_size))],
                           [sg.Push(), sg.Image(tone_sync_example), sg.Push()],
                           [sg.Button("Close")]]

            win3 = sg.Window(title="ST-0001-066-101A Information", layout=layout_win3, size=(1000,810))

        # If the placement info window is up, read events and/or values
        if win3_active == True:
            win3_events, win3_values = win3.read()
            
            # Close placement info window if the window is closed or if the "Close" button is pressed
            if win3_events == sg.WIN_CLOSED or win3_events == 'Close':
                win3_active  = False
                win3.close()


        if event == "-TREATMENT_RAMP-":

            if value["-TREATMENT_FILE-"] != '' and value["-TREATMENT_VOLT-"] != '':
                
                fileGood = CheckFile(value["-TREATMENT_FILE-"])
                voltageGood = VoltageCheck(value["-TREATMENT_VOLT-"])

                if fileGood == True and voltageGood == True:
                    csvGood = CheckCSV(value["-TREATMENT_FILE-"])

                    if csvGood == True:
                        try:
                            guinnessRampFilter(value["-TREATMENT_FILE-"], value["-TREATMENT_VOLT-"])
                        except ValueError:
                            value["-ERROR_TREATMENT-"] = "Error:  Something went wrong. The contents of the input file are incompatible with the requested operation. Review contents of the input waveform .csv and the selected analysis option."
                            win1["-ERROR_TREATMENT-"].update(value["-ERROR_TREATMENT-"])

                        except IndexError:
                            value["-ERROR_TREATMENT-"] = "Error:  Something went wrong. The contents of the input file are incompatible with the requested operation. Review contents of the input waveform .csv and the selected analysis option."
                            win1["-ERROR_TREATMENT-"].update(value["-ERROR_TREATMENT-"])
                            
                        except TypeError:
                            value["-ERROR_TREATMENT-"] = "Error:  Something went wrong. The contents of the input file are incompatible with the requested operation. Review contents of the input waveform .csv and the selected analysis option."
                            win1["-ERROR_TREATMENT-"].update(value["-ERROR_TREATMENT-"])
                    else:
                        value["-ERROR_TREATMENT-"] = "Error:  Input file contains unexpected contents. File must contain only a column of timestamps and a column of corresponding measured voltage."
                        win1["-ERROR_TREATMENT-"].update(value["-ERROR_TREATMENT-"])

                elif fileGood == False and voltageGood == True:
                    value["-ERROR_TREATMENT-"] = "Error:  Invalid filepath or filetype. Input must be a .csv file"
                    win1["-ERROR_TREATMENT-"].update(value["-ERROR_TREATMENT-"])

                elif fileGood == True and voltageGood == False:
                    csvGood = CheckCSV(value["-TREATMENT_FILE-"])

                    if csvGood == True:
                        value["-ERROR_TREATMENT-"] = "Error:  Not a valid voltage limit input. Value must be an integer in the range from 0 to 150."
                        win1["-ERROR_TREATMENT-"].update(value["-ERROR_TREATMENT-"])
                    else:
                        value["-ERROR_TREATMENT-"] = "Error:  Not a valid voltage limit input. Value must be an integer in the range from 0 to 150.\n\nError:  Input file contains unexpected contents. File must contain only a column of timestamps and a column of corresponding measured voltage."
                        win1["-ERROR_TREATMENT-"].update(value["-ERROR_TREATMENT-"])

                elif fileGood == False and voltageGood == False:
                    value["-ERROR_TREATMENT-"] = "Error:  Invalid file and voltage limit."
                    win1["-ERROR_TREATMENT-"].update(value["-ERROR_TREATMENT-"])

            elif value["-TREATMENT_FILE-"] == '' or value["-TREATMENT_VOLT-"] == '':
                value["-ERROR_TREATMENT-"] = "Error:  Both the filepath and voltage limit must be entered."
                win1["-ERROR_TREATMENT-"].update(value["-ERROR_TREATMENT-"])


        if event == "-TREATMENT_PULSE-":
            if value["-TREATMENT_FILE-"] != '' and value["-TREATMENT_VOLT-"] != '':
                
                fileGood = CheckFile(value["-TREATMENT_FILE-"])
                voltageGood = VoltageCheck(value["-TREATMENT_VOLT-"])

                if fileGood == True and voltageGood == True:
                    csvGood = CheckCSV(value["-TREATMENT_FILE-"])

                    if csvGood == True:
                        try:
                            guinnessTHD(value["-TREATMENT_FILE-"], value["-TREATMENT_VOLT-"])
                        except ValueError:
                            value["-ERROR_TREATMENT-"] = "Error:  Something went wrong. The contents of the input file are incompatible with the requested operation. Review contents of the input waveform .csv and the selected analysis option."
                            win1["-ERROR_TREATMENT-"].update(value["-ERROR_TREATMENT-"])

                        except IndexError:
                            value["-ERROR_TREATMENT-"] = "Error:  Something went wrong. The contents of the input file are incompatible with the requested operation. Review contents of the input waveform .csv and the selected analysis option."
                            win1["-ERROR_TREATMENT-"].update(value["-ERROR_TREATMENT-"])
                            
                        except TypeError:
                            value["-ERROR_TREATMENT-"] = "Error:  Something went wrong. The contents of the input file are incompatible with the requested operation. Review contents of the input waveform .csv and the selected analysis option."
                            win1["-ERROR_TREATMENT-"].update(value["-ERROR_TREATMENT-"])
                    else:
                        value["-ERROR_TREATMENT-"] = "Error:  Input file contains unexpected contents. File must contain only a column of timestamps and a column of corresponding measured voltage."
                        win1["-ERROR_TREATMENT-"].update(value["-ERROR_TREATMENT-"])

                elif fileGood == False and voltageGood == True:
                    value["-ERROR_TREATMENT-"] = "Error:  Invalid filepath or filetype. Input must be a .csv file"
                    win1["-ERROR_TREATMENT-"].update(value["-ERROR_TREATMENT-"])

                elif fileGood == True and voltageGood == False:
                    csvGood = CheckCSV(value["-TREATMENT_FILE-"])

                    if csvGood == True:
                        value["-ERROR_TREATMENT-"] = "Error:  Not a valid voltage limit input. Value must be an integer in the range from 0 to 150."
                        win1["-ERROR_TREATMENT-"].update(value["-ERROR_TREATMENT-"])
                    else:
                        value["-ERROR_TREATMENT-"] = "Error:  Not a valid voltage limit input. Value must be an integer in the range from 0 to 150.\n\nError:  Input file contains unexpected contents. File must contain only a column of timestamps and a column of corresponding measured voltage."
                        win1["-ERROR_TREATMENT-"].update(value["-ERROR_TREATMENT-"])

                elif fileGood == False and voltageGood == False:
                    value["-ERROR_TREATMENT-"] = "Error:  Invalid file and voltage limit."
                    win1["-ERROR_TREATMENT-"].update(value["-ERROR_TREATMENT-"])
            
            elif value["-TREATMENT_FILE-"] == '' or value["-TREATMENT_VOLT-"] == '':
                value["-ERROR_TREATMENT-"] = "Error:  Both the filepath and voltage limit must be entered."
                win1["-ERROR_TREATMENT-"].update(value["-ERROR_TREATMENT-"])
        
        
        if event == "-PLACEMENT_PULSE-":
            if value["-PLACEMENT_FILE-"] != '' and value["-PLACEMENT_VOLT-"] != '':
                
                fileGood = CheckFile(value["-PLACEMENT_FILE-"])
                voltageGood = VoltageCheck(value["-PLACEMENT_VOLT-"])

                if fileGood == True and voltageGood == True:
                    csvGood = CheckCSV(value["-PLACEMENT_FILE-"])

                    if csvGood == True:
                        try:
                            normalRiseFall(value["-PLACEMENT_FILE-"], value["-PLACEMENT_VOLT-"])
                        except ValueError:
                            value["-ERROR_PLACEMENT-"] = "Error:  Something went wrong. The contents of the input file are incompatible with the requested operation. Review contents of the input waveform .csv and the selected analysis option."
                            win1["-ERROR_PLACEMENT-"].update(value["-ERROR_PLACEMENT-"])

                        except IndexError:
                            value["-ERROR_PLACEMENT-"] = "Error:  Something went wrong. The contents of the input file are incompatible with the requested operation. Review contents of the input waveform .csv and the selected analysis option."
                            win1["-ERROR_PLACEMENT-"].update(value["-ERROR_PLACEMENT-"])
                            
                        except TypeError:
                            value["-ERROR_PLACEMENT-"] = "Error:  Something went wrong. The contents of the input file are incompatible with the requested operation. Review contents of the input waveform .csv and the selected analysis option."
                            win1["-ERROR_PLACEMENT-"].update(value["-ERROR_PLACEMENT-"])
                    else:
                        value["-ERROR_PLACEMENT-"] = "Error:  Input file contains unexpected contents. File must contain only a column of timestamps and a column of corresponding measured voltage."
                        win1["-ERROR_PLACEMENT-"].update(value["-ERROR_PLACEMENT-"])

                elif fileGood == False and voltageGood == True:
                    value["-ERROR_PLACEMENT-"] = "Error:  Invalid filepath or filetype. Input must be a .csv file"
                    win1["-ERROR_PLACEMENT-"].update(value["-ERROR_PLACEMENT-"])

                elif fileGood == True and voltageGood == False:
                    csvGood = CheckCSV(value["-PLACEMENT_FILE-"])

                    if csvGood == True:
                        value["-ERROR_PLACEMENT-"] = "Error:  Not a valid voltage limit input. Value must be an integer in the range from 0 to 150."
                        win1["-ERROR_PLACEMENT-"].update(value["-ERROR_PLACEMENT-"])
                    else:
                        value["-ERROR_PLACEMENT-"] = "Error:  Not a valid voltage limit input. Value must be an integer in the range from 0 to 150.\n\nError:  Input file contains unexpected contents. File must contain only a column of timestamps and a column of corresponding measured voltage."
                        win1["-ERROR_PLACEMENT-"].update(value["-ERROR_PLACEMENT-"])

                elif fileGood == False and voltageGood == False:
                    value["-ERROR_PLACEMENT-"] = "Error:  Invalid file and voltage limit."
                    win1["-ERROR_PLACEMENT-"].update(value["-ERROR_PLACEMENT-"])
            
            elif value["-PLACEMENT_FILE-"] == '' or value["-PLACEMENT_VOLT-"] == '':
                value["-ERROR_PLACEMENT-"] = "Error:  Both the filepath and voltage limit must be entered."
                win1["-ERROR_PLACEMENT-"].update(value["-ERROR_PLACEMENT-"])

        if event == "-PLACEMENT_PULSE_LOWRES-":
            if value["-PLACEMENT_FILE-"] != '' and value["-PLACEMENT_VOLT-"] != '':
                
                fileGood = CheckFile(value["-PLACEMENT_FILE-"])
                voltageGood = VoltageCheck(value["-PLACEMENT_VOLT-"])

                if fileGood == True and voltageGood == True:
                    csvGood = CheckCSV(value["-PLACEMENT_FILE-"])

                    if csvGood == True:
                        try:
                            lowresRiseFall(value["-PLACEMENT_FILE-"], value["-PLACEMENT_VOLT-"])
                        except ValueError:
                            value["-ERROR_PLACEMENT-"] = "Error:  Something went wrong. The contents of the input file are incompatible with the requested operation. Review contents of the input waveform .csv and the selected analysis option."
                            win1["-ERROR_PLACEMENT-"].update(value["-ERROR_PLACEMENT-"])

                        except IndexError:
                            value["-ERROR_PLACEMENT-"] = "Error:  Something went wrong. The contents of the input file are incompatible with the requested operation. Review contents of the input waveform .csv and the selected analysis option."
                            win1["-ERROR_PLACEMENT-"].update(value["-ERROR_PLACEMENT-"])
                            
                        except TypeError:
                            value["-ERROR_PLACEMENT-"] = "Error:  Something went wrong. The contents of the input file are incompatible with the requested operation. Review contents of the input waveform .csv and the selected analysis option."
                            win1["-ERROR_PLACEMENT-"].update(value["-ERROR_PLACEMENT-"])
                    else:
                        value["-ERROR_PLACEMENT-"] = "Error:  Input file contains unexpected contents. File must contain only a column of timestamps and a column of corresponding measured voltage."
                        win1["-ERROR_PLACEMENT-"].update(value["-ERROR_PLACEMENT-"])

                elif fileGood == False and voltageGood == True:
                    value["-ERROR_PLACEMENT-"] = "Error:  Invalid filepath or filetype. Input must be a .csv file"
                    win1["-ERROR_PLACEMENT-"].update(value["-ERROR_PLACEMENT-"])

                elif fileGood == True and voltageGood == False:
                    csvGood = CheckCSV(value["-PLACEMENT_FILE-"])

                    if csvGood == True:
                        value["-ERROR_PLACEMENT-"] = "Error:  Not a valid voltage limit input. Value must be an integer in the range from 0 to 150."
                        win1["-ERROR_PLACEMENT-"].update(value["-ERROR_PLACEMENT-"])
                    else:
                        value["-ERROR_PLACEMENT-"] = "Error:  Not a valid voltage limit input. Value must be an integer in the range from 0 to 150.\n\nError:  Input file contains unexpected contents. File must contain only a column of timestamps and a column of corresponding measured voltage."
                        win1["-ERROR_PLACEMENT-"].update(value["-ERROR_PLACEMENT-"])

                elif fileGood == False and voltageGood == False:
                    value["-ERROR_PLACEMENT-"] = "Error:  Invalid file and voltage limit."
                    win1["-ERROR_PLACEMENT-"].update(value["-ERROR_PLACEMENT-"])
            
            elif value["-PLACEMENT_FILE-"] == '' or value["-PLACEMENT_VOLT-"] == '':
                value["-ERROR_PLACEMENT-"] = "Error:  Both the filepath and voltage limit must be entered."
                win1["-ERROR_PLACEMENT-"].update(value["-ERROR_PLACEMENT-"])

        if event == "-PLACEMENT_TONE-":
            if value["-PLACEMENT_FILE-"] != '' and value["-PLACEMENT_VOLT-"] != '':
                
                fileGood = CheckFile(value["-PLACEMENT_FILE-"])
                voltageGood = VoltageCheck(value["-PLACEMENT_VOLT-"])

                if fileGood == True and voltageGood == True:
                    csvGood = CheckAudioCSV(value["-PLACEMENT_FILE-"])

                    if csvGood == True:
                        try:
                            guinnessAudioSync(value["-PLACEMENT_FILE-"], value["-PLACEMENT_VOLT-"])
                        except ValueError:
                            value["-ERROR_PLACEMENT-"] = "Error:  Something went wrong. The contents of the input file are incompatible with the requested operation. Review contents of the input waveform .csv and the selected analysis option."
                            win1["-ERROR_PLACEMENT-"].update(value["-ERROR_PLACEMENT-"])

                        except IndexError:
                            value["-ERROR_PLACEMENT-"] = "Error:  Something went wrong. The contents of the input file are incompatible with the requested operation. Review contents of the input waveform .csv and the selected analysis option."
                            win1["-ERROR_PLACEMENT-"].update(value["-ERROR_PLACEMENT-"])
                            
                        except TypeError:
                            value["-ERROR_PLACEMENT-"] = "Error:  Something went wrong. The contents of the input file are incompatible with the requested operation. Review contents of the input waveform .csv and the selected analysis option."
                            win1["-ERROR_PLACEMENT-"].update(value["-ERROR_PLACEMENT-"])
                    else:
                        value["-ERROR_PLACEMENT-"] = "Error:  Input file contains unexpected contents. File must contain only a column of timestamps and a column of corresponding measured voltage."
                        win1["-ERROR_PLACEMENT-"].update(value["-ERROR_PLACEMENT-"])

                elif fileGood == False and voltageGood == True:
                    value["-ERROR_PLACEMENT-"] = "Error:  Invalid filepath or filetype. Input must be a .csv file"
                    win1["-ERROR_PLACEMENT-"].update(value["-ERROR_PLACEMENT-"])

                elif fileGood == True and voltageGood == False:
                    csvGood = CheckCSV(value["-PLACEMENT_FILE-"])

                    if csvGood == True:
                        value["-ERROR_PLACEMENT-"] = "Error:  Not a valid voltage limit input. Value must be an integer in the range from 0 to 150."
                        win1["-ERROR_PLACEMENT-"].update(value["-ERROR_PLACEMENT-"])
                    else:
                        value["-ERROR_PLACEMENT-"] = "Error:  Not a valid voltage limit input. Value must be an integer in the range from 0 to 150.\n\nError:  Input file contains unexpected contents. File must contain only a column of timestamps and a column of corresponding measured voltage."
                        win1["-ERROR_PLACEMENT-"].update(value["-ERROR_PLACEMENT-"])

                elif fileGood == False and voltageGood == False:
                    value["-ERROR_PLACEMENT-"] = "Error:  Invalid file and voltage limit."
                    win1["-ERROR_PLACEMENT-"].update(value["-ERROR_PLACEMENT-"])
            
            elif value["-PLACEMENT_FILE-"] == '' or value["-PLACEMENT_VOLT-"] == '':
                value["-ERROR_PLACEMENT-"] = "Error:  Both the filepath and voltage limit must be entered."
                win1["-ERROR_PLACEMENT-"].update(value["-ERROR_PLACEMENT-"])
                
                
    # Catch for value errors, application should not crash this way
    except ValueError:
        value["-ERROR_TREATMENT-"] = "Error:  Something went wrong. The contents of the input file are incompatible with the requested operation. Review contents of the input waveform .csv and the selected analysis option."
        value["-ERROR_PLACEMENT-"] = "Error:  Something went wrong. The contents of the input file are incompatible with the requested operation. Review contents of the input waveform .csv and the selected analysis option."
        win1["-ERROR_TREATMENT-"].update(value["-ERROR_PLACEMENT-"])
        win1["-ERROR_PLACEMENT-"].update(value["-ERROR_PLACEMENT-"])
        
    # Catch for index errors, application should not crash this way
    except IndexError:
        value["-ERROR_TREATMENT-"] = "Error:  Something went wrong. The contents of the input file are incompatible with the requested operation. Review contents of the input waveform .csv and the selected analysis option."
        value["-ERROR_PLACEMENT-"] = "Error:  Something went wrong. The contents of the input file are incompatible with the requested operation. Review contents of the input waveform .csv and the selected analysis option."
        win1["-ERROR_TREATMENT-"].update(value["-ERROR_PLACEMENT-"])
        win1["-ERROR_PLACEMENT-"].update(value["-ERROR_PLACEMENT-"])
        
    # Catch for type errors, application should not crash this way
    except TypeError:
        value["-ERROR_TREATMENT-"] = "Error:  Something went wrong. The contents of the input file are incompatible with the requested operation. Review contents of the input waveform .csv and the selected analysis option."
        value["-ERROR_PLACEMENT-"] = "Error:  Something went wrong. The contents of the input file are incompatible with the requested operation. Review contents of the input waveform .csv and the selected analysis option."
        win1["-ERROR_TREATMENT-"].update(value["-ERROR_PLACEMENT-"])
        win1["-ERROR_PLACEMENT-"].update(value["-ERROR_PLACEMENT-"])


'''To create your EXE file from your program that uses PySimpleGUI, my_program.py, enter this command in your Windows command prompt:

pyinstaller my_program.py

You will be left with a single file, my_program.exe, located in a folder named dist under the folder where you executed the pyinstaller command.
'''