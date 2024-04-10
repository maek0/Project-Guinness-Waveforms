import PySimpleGUI as sg
import os
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy import signal
from scipy.fftpack import fft
from scipy.fftpack import fftfreq
import matplotlib.pyplot as plt
import datetime

toolVersion = "101A"

def evenColumns(x,y):
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
        ind = int(max(max(n),max(m))[0])
        # x = x[:ind-1]
        # y = y[:ind-1]
        x = x[ind+1:]
        y = y[ind+1:]
        
    # if there are NaN values anywhere in y, cut both x and y down before the earliest found NaN in y
    # this issue typically happens in the beginning of the y array. If there are nan value at or near the end of an array, this will break the input and it will not work. It will require manual editing of the input file.
    elif n.size>0 and m.size==0:
        ind = int(max(n)[0])
        # x = x[:ind-1]
        # y = y[:ind-1]
        x = x[ind+1:]
        y = y[ind+1:]
        
    # if there are NaN values anywhere in x, cut both x and y down before the earliest found NaN in x
    elif n.size==0 and m.size>0:
        ind = int(max(m)[0])
        # x = x[:ind-1]
        # y = y[:ind-1]
        x = x[ind+1:]
        y = y[ind+1:]

    return x, y

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

def CheckPlainPlotCSV(filepath):
    csvArray = np.genfromtxt(open(filepath), delimiter=",")
    rows = np.size(csvArray,0)
    columns = np.size(csvArray,1)

    if columns < 2 or columns > 3:
        status = False

    else:
        if rows < 500:
            status = False
            
        else:
            status = True

    return status, columns

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

def treatmentVoltageCheck(voltageLimit):
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
        if voltageLimit > 150 or voltageLimit < 0:
            status = False

        elif voltageLimit <= 150 and voltageLimit >= 0:
            status = True

    else:
        status = False

    return status

def placementVoltageCheck(voltageLimit):
    try:
        voltageLimit = float(voltageLimit)

    except ValueError:
        status = False
    
    if type(voltageLimit) == float:

        if voltageLimit > 5 or voltageLimit < 0:
            status = False

        elif voltageLimit <= 5 and voltageLimit >= 0:
            status = True
        
    elif type(voltageLimit) == int:
        if voltageLimit > 5 or voltageLimit < 0:
            status = False

        elif voltageLimit <= 5 and voltageLimit >= 0:
            status = True

    else:
        status = False

    return status

def plotContents(filepath, columns):
    datetime_rn = datetime.datetime.now()
    str_datetime_rn = datetime_rn.strftime("%d-%b-%Y, %X %Z")
    
    csvFile = open(filepath)
    csvArray = np.genfromtxt(csvFile, delimiter=",")
    
    x_ = csvArray[2:-2,0]
    y_ = csvArray[2:-2,1]

    x,y = evenColumns(x_,y_)

    y = signal.detrend(y, type="constant")

    plt.plot(x,y,label="Column 2 of the input file.")

    if columns == 3:
        z_ = csvArray[2:-2,2]
        _, z = evenColumns(x,z_)
        # z = signal.detrend(x,type="constant")
        plt.plot(x,z,label="Column 3 of the input file.")
    
    plt.title("Preview plot of the input file.\n{}".format(str_datetime_rn))
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.xlim([min(x),max(x)])
    plt.ylim([min(y)-1,max(y)+1])
    plt.legend(loc="lower left")
    plt.show()

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

def guinnessRampFilter(filepath,voltageLimit):
    
    datetime_rn = datetime.datetime.now()
    str_datetime_rn = datetime_rn.strftime("%d-%b-%Y, %X %Z")
    
    headers = ["Time (s)", "Voltage (V)"]
    csvFile = open(filepath)
    csvArray = np.genfromtxt(csvFile, delimiter=",")
    x_ = csvArray[2:,0]
    y_ = csvArray[2:,1]

    x_full,y_full = evenColumns(x_,y_)
    
    filename = os.path.basename(filepath)
    y_full = signal.detrend(y_full, type="constant")
    
    # if time permits, change dist to be dependent on the time delta of the whole input file
    backup = 1000
    dist = 100
    
    # find the indices of the peaks of the output energy signal 
    y_startcutoff_xvalues = signal.find_peaks(y_full, height=10, distance=dist)
    y_endcutoff_xvalues = signal.find_peaks(y_full,height=float(voltageLimit), distance=dist)

    if len(y_endcutoff_xvalues[0])>10:
        x = x_full[(y_startcutoff_xvalues[0][0]-backup):y_endcutoff_xvalues[0][10]]
        y = y_full[(y_startcutoff_xvalues[0][0]-backup):y_endcutoff_xvalues[0][10]]
    elif len(y_endcutoff_xvalues[0])>5:
        x = x_full[(y_startcutoff_xvalues[0][0]-backup):y_endcutoff_xvalues[0][5]]
        y = y_full[(y_startcutoff_xvalues[0][0]-backup):y_endcutoff_xvalues[0][5]]
    else:
        x = x_full[(y_startcutoff_xvalues[0][0]-backup):len(x_full)]
        y = y_full[(y_startcutoff_xvalues[0][0]-backup):len(y_full)]
    
    # find the indices of the peaks of the output energy signal (not including voltage checks)
    y_peaks_xvalues, ypeak_properties = signal.find_peaks(y, height=4,prominence=10,distance=dist)

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
    if max(y_peaks_yvalues)<float(voltageLimit):
        ind_limit = y_peaks_xvalues[-1]
    else:
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
    
    # mark the first peak of the output energy signal on the plot
    plt.plot(first_peakX,first_peakY, "x", color = "red", label = "First Peak = {:.2f}V".format(first_peakY), markersize = 8, markeredgewidth = 2)

    # plotting the 66%(voltage limit)
    plt.axhline(cutoff, label = "66% Voltage Limit = {:.2f}V".format(cutoff), linestyle = "--", color = "black")
    
    # plotting options
    plt.title("Guinness Generator Output Ramp, Voltage Limit = {}V\nInput file name: '{}'".format(voltageLimit, filename))
    plt.text(min(x)+1,max(y)-3,"ST-0001-066-{}, {}".format(toolVersion,str_datetime_rn),fontsize="small")
    plt.xlabel(headers[0])
    plt.ylabel(headers[1])
    
    # plot the best fit line for the ramping section BEFORE reaching 66%(voltage limit)
    plt.plot(fiveVoltRampX, fiveV_fit, color = 'green', label = "1st Segment: y = {:.2f}x + {:.2f}\n$R^2$ = {:.2f}".format(fiveV_slope[0], fiveV_intercept, fiveV_rsq))
    
    # plot the best fit line for the ramping section AFTER reaching 66%(voltage limit)
    plt.plot(twoVoltRampX, twoV_fit, color = 'orange', label = "2nd Segment: y = {:.2f}x + {:.2f}\n$R^2$ = {:.2f}".format(twoV_slope[0], twoV_intercept, twoV_rsq))

    # plotting options
    plt.xlim(min(x),max(x))
    plt.ylim(min(y)-3,max(y)+3)
    plt.legend(loc="lower left")
    
    # display the plot
    plt.show()

def guinnessTHD(filepath,voltageLimit):

    datetime_rn = datetime.datetime.now()
    str_datetime_rn = datetime_rn.strftime("%d-%b-%Y, %X %Z")
    
    headers = ["Time (s)", "Voltage (V)"]
    csvFile = open(filepath)
    csvArray = np.genfromtxt(csvFile, delimiter=",")
    x_ = csvArray[2:-2,0]
    y_ = csvArray[2:-2,1]

    x,y = evenColumns(x_,y_)
    
    filename = os.path.basename(filepath)
    y = signal.detrend(y, type="constant")

    xN = len(x)
    
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
    plt.text(min(x)+1,max(y)-3,"ST-0001-066-{}, {}".format(toolVersion,str_datetime_rn),fontsize="small")
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
    
def averagePkAmp(filepath,voltageLimit):
    datetime_rn = datetime.datetime.now()
    str_datetime_rn = datetime_rn.strftime("%d-%b-%Y, %X %Z")
    
    headers = ["Time (s)", "Voltage (V)"]
    csvFile = open(filepath)
    csvArray = np.genfromtxt(csvFile, delimiter=",")
    x_ = csvArray[2:-2,0]
    y_ = csvArray[2:-2,1]

    x_,y = evenColumns(x_,y_)

    x = np.linspace(x_[0],x_[-1],np.size(y))

    # plt.plot(x,y)
    # plt.show()

    dist = 20
    
    filename = os.path.basename(filepath)
    
    # y = signal.detrend(y, type="constant")
    heightLim = int(voltageLimit)-5
    
    peaks_xvalues, peaks_xvalues_properties = signal.find_peaks(y, height=float(heightLim),prominence=10,distance=dist)
    # print(peaks_xvalues)
    peaks_yvalues = peaks_xvalues_properties["peak_heights"]
    # print(peaks_yvalues)

    avgAmp = np.mean(peaks_yvalues)
    # print(avgAmp)
    
    plt.plot(x,y,label="Placement therapy output", color = "blue")
    plt.scatter(x[peaks_xvalues],peaks_yvalues,marker="x",color="magenta", s=30, label="Pulse Peaks")
    plt.xlabel(headers[0])
    plt.ylabel(headers[1])

    plt.axhline(avgAmp, label = "Average Peak Amplitude {:.3f}V".format(avgAmp), linestyle = "--", color = "green")

    plt.text(min(x),max(y)+0.5,"ST-0001-066-{}, {}".format(toolVersion,str_datetime_rn),fontsize="small")
    
    # plotting options
    plt.title("Guinness Generator Average Amplitude\nSet Voltage = {}V, Input file name: '{}'\nAverage Peak Voltage = {:.3f}".format(voltageLimit, filename, avgAmp))
    
    plt.xlim(min(x),max(x))
    plt.ylim(min(y)-5,max(y)+5)
    plt.legend(loc="lower left")

    # display the plot
    plt.show()

def calcRiseFall(filepath,voltageLimit):
    datetime_rn = datetime.datetime.now()
    str_datetime_rn = datetime_rn.strftime("%d-%b-%Y, %X %Z")
    
    headers = ["Time (s)", "Voltage (V)"]
    csvFile = open(filepath)
    csvArray = np.genfromtxt(csvFile, delimiter=",")
    x_ = csvArray[2:-2,0]
    y_ = csvArray[2:-2,1]

    x,y = evenColumns(x_,y_)

    '''
    DO THIS FOR ALL 
    - CREATE A LINE FROM THE MIN TO THE MAX 
    - PLOT WHERE THE LINE INTERCEPTS THE PLOTTING 10% AND 90% LINES 
    - CALCULATE RISE/FALL TIME FROM THOSE POINTS
    '''

    filename = os.path.basename(filepath)
    
    y = signal.detrend(y, type="constant")

    y_diff2 = np.gradient(np.gradient(y))
    y_diff2 = np.abs(y_diff2)

    # plt.plot(y_diff2)

    peak_indices, peak_info = signal.find_peaks(y_diff2,height=0.06)
    # print(peak_indices)

    # peak_heights = peak_info['peak_heights']
    # highest_peak_index = peak_indices[np.argmax(peak_heights)]

    # secondThird = peak_indices[np.argmax(peak_heights)]
    # second_and_third_highest_peak_indices = [peak_indices[0], peak_indices[-1]]

    buff = 5
    delay = 0.00005

    first_cutoff_index = peak_indices[0]-buff
    second_cutoff_index = peak_indices[-1]+buff

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
    
    
    positive_rise_x = [x_windowed[positive_ten][0], x_windowed[positive_ninety][0]]
    positive_rise_y = [y_windowed[positive_ten][0], y_windowed[positive_ninety][0]]
    
    switch_x = [x_windowed[positive_ninety][-1],x_windowed[negative_ninety][0]]
    switch_y = [y_windowed[positive_ninety][-1],y_windowed[negative_ninety][0]]
    
    negative_fall_x = [x_windowed[negative_ninety][-1],x_windowed[negative_ten][-1]]
    negative_fall_y = [y_windowed[negative_ninety][-1],y_windowed[negative_ten][-1]]
    
    
    positive_rise_coefficients = np.polyfit(positive_rise_x,positive_rise_y,1)
    switch_coefficients = np.polyfit(switch_x,switch_y,1)
    negative_fall_coefficients = np.polyfit(negative_fall_x,negative_fall_y,1)
    
    positivePoly = np.poly1d(positive_rise_coefficients)
    switchPoly = np.poly1d(switch_coefficients)
    negativePoly = np.poly1d(negative_fall_coefficients)
    
    length = 500000
    positiveFitX = np.linspace(x_windowed[0]-delay, x_windowed[positive_ninety][-1], length)
    switchFitX = np.linspace(x_windowed[positive_ninety][0], x_windowed[negative_ninety][-1], length)
    negativeFitX = np.linspace(x_windowed[negative_ninety][0], x_windowed[-1]+delay,length)
    
    positiveFitY = positivePoly(positiveFitX)
    switchFitY = switchPoly(switchFitX)
    negativeFitY = negativePoly(negativeFitX)
    
    # print(negativeFitX)
    
    
    ## Assigning min and max points of the pulse to variables ##
    positive_ten_rise = positiveFitX[np.where(positiveFitY>=ten)][0]
    
    positive_ninety_rise = positiveFitX[np.where(positiveFitY>=ninety)][0]
    positive_ninety_fall = switchFitX[np.where(switchFitY<=ninety)][0]
    
    positive_ten_fall = switchFitX[np.where(switchFitY<=ten)][0]
    negative_ten_rise = switchFitX[np.where(switchFitY<=-ten)][0]

    negative_ninety_rise = switchFitX[np.where(switchFitY<=-ninety)][0]
    negative_ninety_fall = negativeFitX[np.where(negativeFitY>=-ninety)][0]

    negative_ten_fall = negativeFitX[np.where(negativeFitY>=-ten)][0]
    
    
    positive_rise_time = positive_ninety_rise-positive_ten_rise
    positive_fall_time = positive_ten_fall-positive_ninety_fall
    # switch_time = negative_ninety_rise-positive_ninety_fall
    negative_rise_time = negative_ninety_rise-negative_ten_rise
    negative_fall_time = negative_ten_fall-negative_ninety_fall
    
    points = np.array([
                    [positive_ten_rise, positiveFitY[np.where(positiveFitY>=ten)][0]],
                    [positive_ninety_rise, positiveFitY[np.where(positiveFitY>=ninety)][0]],
                    [positive_ninety_fall, switchFitY[np.where(switchFitY<=ninety)][0]],
                    [positive_ten_fall, switchFitY[np.where(switchFitY<=ten)][0]],
                    [negative_ten_rise, switchFitY[np.where(switchFitY<=-ten)][0]],
                    [negative_ninety_rise, switchFitY[np.where(switchFitY<=-ninety)][0]],
                    [negative_ninety_fall, negativeFitY[np.where(negativeFitY>=-ninety)][0]],
                    [negative_ten_fall, negativeFitY[np.where(negativeFitY>=-ten)][0]]
                ])
    
    plt.plot(x_windowed,y_windowed,label="Placement therapy output", color = "blue")
    plt.plot(x,y,color = "blue")
    plt.xlabel(headers[0])
    plt.ylabel(headers[1])

    # plt.plot(positiveFitX,positiveFitY)
    # plt.plot(switchFitX,switchFitY)
    # plt.plot(negativeFitX,negativeFitY)
    
    one_mark = 0.2
    two_mark = 0.55
    three_mark = 0.45
    four_mark = 0.8

    plt.axhline(ten, xmin=one_mark, xmax=two_mark, label = "10% of set voltage, (+/-) {:.2f}V".format(ten), linestyle = "--", color = "magenta")
    plt.axhline(ninety, xmin=one_mark, xmax=two_mark, label = "90% of set voltage, (+/-) {:.2f}V".format(ninety), linestyle = "--", color = "green")
    plt.axhline(-ten, xmin=three_mark, xmax=four_mark, linestyle = "--", color = "magenta")
    plt.axhline(-ninety, xmin=three_mark, xmax=four_mark, linestyle = "--", color = "green")
    
    plt.scatter(points[:,0],points[:,1],marker="x",color="red", s=50, label="Rise and Fall markers")
    
    plt.axhline(0, label = "Origin (0V)", color = "black")

    microsecond = 1000000
    
    plt.text(positive_ten_rise+delay,half,"Rise time: {:.4f} $\mu$s".format(positive_rise_time*microsecond),fontsize="small")
    plt.text(positive_ninety_fall+delay,half,"Fall time: {:.4f} $\mu$s".format(positive_fall_time*microsecond),fontsize="small")
    # plt.text(positive_ninety_rise+delay,ten,"Time: {:.4f} $\mu$s".format(switch_time*microsecond),fontsize="small")
    plt.text(negative_ten_rise+delay,-half,"Rise time: {:.4f} $\mu$s".format(negative_rise_time*microsecond),fontsize="small")
    plt.text(negative_ninety_fall+delay,-half,"Fall time: {:.4f} $\mu$s".format(negative_fall_time*microsecond),fontsize="small")
    
    plt.text(min(x_windowed),max(y_windowed)+0.5,"ST-0001-066-{}, {}".format(toolVersion,str_datetime_rn),fontsize="small")
    
    # plotting options
    plt.title("Guinness Generator Placement Bipolar Pulse\nSet Voltage = {}V, Input file name: '{}'".format(voltageLimit, filename))
    
    xscale = x[min([first_cutoff_index,len(x)-second_cutoff_index])]-x[0]
    plt.xlim(x[first_cutoff_index]-xscale, x[second_cutoff_index]+xscale)
    
    yscale = float(voltageLimit)/5.0
    plt.ylim(min(y_windowed)-yscale,max(y_windowed)+yscale)
    
    plt.legend(loc="upper right")

    # display the plot
    plt.show()


def guinnessAudioSync(filepath,voltageLimit):
    datetime_rn = datetime.datetime.now()
    str_datetime_rn = datetime_rn.strftime("%d-%b-%Y, %X %Z")
    
    headers = ["Time (s)", "Voltage (V)"]
    csvFile = open(filepath)
    csvArray = np.genfromtxt(csvFile, delimiter=",")
    x_ = csvArray[2:-2,0]
    place_ = csvArray[2:-2,1]
    audio_ = csvArray[2:-2,2]

    x, place1 = evenColumns(x_,place_)
    x, audio1 = evenColumns(x_,audio_)
    audio, place = evenColumns(audio1,place1)
    
    filename = os.path.basename(filepath)
    
    place = signal.detrend(place, type="constant")
    # audio = signal.detrend(audio, type="constant")    # was messing up the signal
    working_audio = audio   # modifiable copy of the placement audio to get indices of the start of the tone(s)
    
    # vertically offset the audio signal for clearer graphing/visualization
    audio = audio + 0.5
        
    cutoff = 0.5
    working_audio[(working_audio<cutoff)&(working_audio>-cutoff)] = 0
    # plt.plot(working_audio)
    # plt.show()
    
    audio_peakIndices = []
    z = 100
    
    '''
    See if the value of z can also be made dependent on the length of the input file
    '''
    
    for i in range(z,len(working_audio)-z,1):
        if sum(working_audio[i-z:i])==0 and working_audio[i+1]!=0 and working_audio[i]==0:
            audio_peakIndices.append(i)
    audio_peakIndices = np.array(audio_peakIndices)
    audio_peakHeights = audio[audio_peakIndices]
    plt.plot(working_audio)
    zeroes = np.zeros(np.shape(audio_peakIndices))
    plt.scatter(audio_peakIndices,zeroes,marker='X',c='red')
    plt.show()
    
    # plot placement and audio
    plt.plot(x, place, label = "Placement Output", color = "blue")
    plt.plot(x, audio, label = "Placement Audio", color = "orange")

    if float(voltageLimit) >= 1.0:
        placement_peakIndices, _ = signal.find_peaks(place, height=0.4, distance=500)
    else:
        placement_peakIndices, _ = signal.find_peaks(place, height=0.05, distance=500)
    
    placement_peakHeights = place[placement_peakIndices]

    # audio_peakIndices, _ = signal.find_peaks(audio_diff2, height=0.15, distance=1000)
    

    diff = []
    for i in range(0,len(audio_peakIndices),1):
        for j in range(0,len(placement_peakIndices),1):
            if np.abs(audio_peakIndices[i]-placement_peakIndices[j])<500:
                diff_temp = np.abs(x[audio_peakIndices[i]]-x[placement_peakIndices[j]])
                diff.append(diff_temp)
                plt.text(x[placement_peakIndices[j]]+0.02,-place[placement_peakIndices[j]]+float(0.25*float(voltageLimit)),"Delay = {:.4f}s".format(diff_temp))
            else:
                pass

    diff = np.array(diff)
    average_delay = np.mean(diff)

    # plot peaks of placement and audio
    plt.scatter(x[placement_peakIndices], placement_peakHeights,marker="x",color="magenta",s=50,label="Placement Pulse(s)")
    plt.scatter(x[audio_peakIndices],audio_peakHeights,marker="x",color="black", s=50,label="Audio Tone(s) Onset",zorder=5)

    # tool name
    plt.text(min(x)+0.05,max(place)+0.9,"ST-0001-066-{}, {}".format(toolVersion,str_datetime_rn),fontsize="small")

    # plotting options
    plt.title("Guinness Generator Placement Output and Audio Tones\nSet Voltage = {}V, Input File Name = '{}'\nAverage Tone Delay = {:.4f} seconds".format(voltageLimit, filename, average_delay))
    plt.xlim(min(x),max(x))
    plt.ylim(min(place)-1,max(place)+1)
    plt.legend(loc="lower right")
    plt.xlabel(headers[0])
    plt.ylabel(headers[1])

    # display the plot
    plt.show()

help_button_base64 = b'iVBORw0KGgoAAAANSUhEUgAAABgAAAAYCAYAAADgdz34AAAACXBIWXMAAAsTAAALEwEAmpwYAAABq0lEQVRIie1VPS8EURTdIJEIEvOxGyRCUPADJFtIBIVEva1CQ6NQCLHz3lMqdKJQoNNJJKKhoNDRiUjQqXwEEcy6b5LnDEE28+Zjl46b3GrmnnPvuWfupFJ/IswpVWdzL2dxuWRx2rQZbdtcrqQZjZtCNZUN3ChUjcWI25weAaj0SZ7F5Koxq5pL6xqd2UwehgMXJya7NYXsSwReL5SBorOk4F8kjFxDUE8sATTeiuj0GpKchxIxurSFqg0FTzM5ENVlRritKaGqsPCLcBI5F0qAMTeiZZDDmbzbhvduIqS6QhMVAfCOCVUNCZ5K1V6XsHA2qH3ebY8sZLSOCeYhz0Hswp3XUY3+lI0pPIK+C77GsVMwmgkQGE6hK0HhXhICk9NYgODjJJD3GwS2kENaF+Hh/k8J4KJn/8zobYrlxO7A8UYAshtKwOWaFvw9cqoSICchxcuQcPo7deD00iBUSziBv2xR6AbAQzn+19pTF7iM/SC5Sw6Os81pMhH4Z1hOoRNFO/HWladpIQdLAi8iYtSLxS3iKz6Gi+59nW3/nOPL9v90/vErG/w/3gALBuad4TTYiQAAAABJRU5ErkJggg=='
THD_eq = b'iVBORw0KGgoAAAANSUhEUgAAAWgAAABGCAYAAADhNA4nAAAAAXNSR0IArs4c6QAABAR0RVh0bXhmaWxlACUzQ214ZmlsZSUyMGhvc3QlM0QlMjJFbGVjdHJvbiUyMiUyMG1vZGlmaWVkJTNEJTIyMjAyMy0xMC0yMVQwNCUzQTQ5JTNBMzguMTYzWiUyMiUyMGFnZW50JTNEJTIyTW96aWxsYSUyRjUuMCUyMChXaW5kb3dzJTIwTlQlMjAxMC4wJTNCJTIwV2luNjQlM0IlMjB4NjQpJTIwQXBwbGVXZWJLaXQlMkY1MzcuMzYlMjAoS0hUTUwlMkMlMjBsaWtlJTIwR2Vja28pJTIwZHJhdy5pbyUyRjIyLjAuMiUyMENocm9tZSUyRjExNC4wLjU3MzUuMjg5JTIwRWxlY3Ryb24lMkYyNS44LjQlMjBTYWZhcmklMkY1MzcuMzYlMjIlMjBldGFnJTNEJTIyd0NoN3hnaFpOQm1GQXdibmxiekolMjIlMjB2ZXJzaW9uJTNEJTIyMjIuMC4yJTIyJTIwdHlwZSUzRCUyMmRldmljZSUyMiUzRSUzQ2RpYWdyYW0lMjBuYW1lJTNEJTIyUGFnZS0xJTIyJTIwaWQlM0QlMjIzMVRZSHVjRVRjVWJWOXRfRUNYWCUyMiUzRWpaTnRiNEl3RU1jJTJGRFlrdW1RRTZuYjRjNkhSWnRpVnpjUzlOcFNjMEZzcEtGZHluM3dGRlJHT3lOOXI3M1VONyUyRnpzczRzZkZYTkUwZXBNTWhPWGFyTERJMUhKZHg3RnQlMkZDdkpzU1lqMTZsQnFEZ3pRUzFZOGw4dzBPU0ZlODRnNndScUtZWG1hUmNHTWtrZzBCMUdsWko1TjJ3clJmZldsSVp3QlpZQkZkZjBtek1kMVhROHRGdSUyQkFCNUclMkJyTGhtRGJCcGtRV1VTYnpHbFV4WkdZUlgwbXA2MU5jJTJCQ0JLOFJwZDZrTFBON3luaHlsSTlIOFNpbzMlMkZlbGlNUHphZk8zaDM1MlR5RWszdjNicktnWXE5YWRnYTJWOExQTmlWYVZmOURPOTYyWSUyRlN2ZFhhdFlhek1zRnlQZnhkcmNtRiUyRmRDeEI0TkJ2NDh2d2tTbmRQU3h1QkZESHh1Rk5SVDRmaSUyRlNzVURnNERIVFN1N0FsMElxSkls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'
voltage_ramp_example = b'iVBORw0KGgoAAAANSUhEUgAAAmIAAAB6CAYAAAAcTD85AAAAAXNSR0IArs4c6QAACIB0RVh0bXhmaWxlACUzQ214ZmlsZSUyMGhvc3QlM0QlMjJFbGVjdHJvbiUyMiUyMG1vZGlmaWVkJTNEJTIyMjAyMy0xMC0yMVQxNiUzQTM4JTNBNTAuODEwWiUyMiUyMGFnZW50JTNEJTIyTW96aWxsYSUyRjUuMCUyMChXaW5kb3dzJTIwTlQlMjAxMC4wJTNCJTIwV2luNjQlM0IlMjB4NjQpJTIwQXBwbGVXZWJLaXQlMkY1MzcuMzYlMjAoS0hUTUwlMkMlMjBsaWtlJTIwR2Vja28pJTIwZHJhdy5pbyUyRjIyLjAuMiUyMENocm9tZSUyRjExNC4wLjU3MzUuMjg5JTIwRWxlY3Ryb24lMkYyNS44LjQlMjBTYWZhcmklMkY1MzcuMzYlMjIlMjBldGFnJTNEJTIyYTNIT1NLTjJVOTFxVFVYdDVHMl8lMjIlMjB2ZXJzaW9uJTNEJTIyMjIuMC4yJTIyJTIwdHlwZSUzRCUyMmRldmljZSUyMiUzRSUzQ2RpYWdyYW0lMjBuYW1lJTNEJTIyUGFnZS0xJTIyJTIwaWQlM0QlMjJaMTl2ZHpocU84NmJXSkxmcUxfciUyMiUzRTdaeGRjOXNvRklaJTJGalMlMkJUa1JENnVteWNKdTFzZDdiYjdIUm5lNmUxaUsySkxEd3lpZTMlMkIlMkJrVXh5T2JnT0VLeGNEZURMekxXRVJ6SjUzbUJBeElaQmVQNSUyQnJiT0ZyUGZhVTdLRWZMeTlTaTRIaUVVSTh6JTJGTm9iTjFvQlJzalZNNnlMZm12eWQ0YTc0U1lUUkU5YkhJaWRMcFNDanRHVEZRalZPYUZXUkNWTnNXVjNUbFZyc25wYnFWUmZabEdpR3UwbFc2dGElMkZpNXpOdHRZazlIYjJUNlNZenVTVmZVJTJCY21XZXlzREFzWjFsT1YzdW00T01vR05lVXN1MjMlMkJYcE15aVoyTWk3YmVqY3ZuRzF2ckNZVjYxTGh4N2YxUCUyRmhQZG5QNzVYUDFkVEwyNnVMcXg0WHc4cFNWaiUyQklIaTV0bEd4bUJhVTBmRjZJWXFSbFpINHA3OXE4czd1bjM1YmUlMkZscXVFMERsaDlZWVhrUklSTllSQVVDS09WN3R3UjFJTnMlMkYxUXkwdGxBdkcwZGIyTEF2OGlBbkU0S01WVGRmMEh1czElMkI4elljME0lMkY4Z1Q1OHUlMkZDRDE2TkNxdnhESXk5JTJCVk5HS0c2OW1iTTZ2Y3UzenJ6eGlWVTZhUzNqOFNBJTJGSU1SUWtWd1NwaDIwdkx1R0JzRWhiVGNxTUZVJTJCcWpBJTJCRlNsemhLeTM0N2JWVVFwVktCR0s5cEklMkYxaElnNiUyQjZLRGJ1TGpmbGhXVHduVCUyRkR4amEzJTJGekcwaGlSeEtRMUpwTlo1UjIwVVVkR3VGZTE5UVZTdWN1ckd0JTJGRlYxRzZrZnJ2WEJ5cVBmeVlNV2hPclB3cEUxZ3lXcjZRTWEwcFBWejNjRGpuNXViOW93Y0pOSEI1bktVOUxuYml3QWFCcGVCOHVuWFhGcnEzZHdPM1JGR0E2aGduJTJGWEx1dmlmcWlBeXhDN3JlYUJaWTd1Y1k4ZlphSFNVZlBwaWY4SE5tVldRT0JVWXFjQ1hzOUkzeWtEemMxNGRvQ0hHJTJGbmV0Z3hDcCUyRkpMaiUyRkRyckF2akZpVjBkdU5IZlVBZVIybzU3Y3dkJTJCTURwdmYlMkJDeUEwTWRKT3E0SHFVbm1SdEF0MEY4enJrQmN0bUNtU3FRcDA3dFlueDBLdDlWRmRCdGdPektJSFV5TUpPQm5GUUxYa25QaFRYb1IxdFNzanRJWUpjMEd1cEFObFNwZyUyRkFrZ3dSMGkzemcxbTczZ0YwT2FTZ0xyT1olMkJLWndMOXBRRmNPdEhkbVhnVWtoREdZQ2xvYlF2ZCUyQkRIaDNLeXJBT1hOQnJxQUR3UFNPSFNRazlaQUxkMkZ4YXd5eGtOVlpDcXVaN3Z3Vkc5cHd5QVg1aUMySjFSeW1BNVdYU1VSZUFqSUl1ZUMwN1FFVnh2aWtLN1F1andmcEVUd2o0JTJGOEFEaHRkY0VPdXZDOHNBZzE3c2QlMkJLN2dNVXp2VGdRZVk3dmczZnpBRUh6a0RRUGU4cXQyb1pzUUdJS1BZU3A0SXZDeDNabGc2T1lBaHVCVG1PeWRDTHowYSUyQnNOVFpmbG00SEgzakRKWGV2WEZuaVgxUnVDeDhPTThhMWZXJTJCQmRWbThJSGcyVDFiZCUyQmJZRjMlMkJ5aTRGaFdVQUdU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'
pulse_burst_example = b'iVBORw0KGgoAAAANSUhEUgAAAlAAAAChCAYAAAAIs4HQAAAAAXNSR0IArs4c6QAACAB0RVh0bXhmaWxlACUzQ214ZmlsZSUyMGhvc3QlM0QlMjJFbGVjdHJvbiUyMiUyMG1vZGlmaWVkJTNEJTIyMjAyMy0xMC0yMVQxNiUzQTM5JTNBNDUuNzg5WiUyMiUyMGFnZW50JTNEJTIyTW96aWxsYSUyRjUuMCUyMChXaW5kb3dzJTIwTlQlMjAxMC4wJTNCJTIwV2luNjQlM0IlMjB4NjQpJTIwQXBwbGVXZWJLaXQlMkY1MzcuMzYlMjAoS0hUTUwlMkMlMjBsaWtlJTIwR2Vja28pJTIwZHJhdy5pbyUyRjIyLjAuMiUyMENocm9tZSUyRjExNC4wLjU3MzUuMjg5JTIwRWxlY3Ryb24lMkYyNS44LjQlMjBTYWZhcmklMkY1MzcuMzYlMjIlMjBldGFnJTNEJTIyQlNjWWtZUzhQYnBUUnE3LVRjRlolMjIlMjB2ZXJzaW9uJTNEJTIyMjIuMC4yJTIyJTIwdHlwZSUzRCUyMmRldmljZSUyMiUzRSUzQ2RpYWdyYW0lMjBuYW1lJTNEJTIyUGFnZS0xJTIyJTIwaWQlM0QlMjJ0OWxNOG8wbURaWFRvNkR0b3lfMCUyMiUzRTdacGRjNXM0RklaJTJGalMlMkZONkJ0eG1iakpkanJkMmM1bXQ1bGU3UkJRREMwZ2l1VTQyViUyQiUyRmtpMFpLQ1RCY1p5MVd6eE1BZ2R4Qk8lMkJqYzRRa0puaVczJTJGOVdoV1h5dTR4Rk5rRWd2cCUyRmdkeE9FcGhBeCUyRmM5WUhqWVdndmpHTUslMkZTZUdPQ3RlRXElMkZWZFlJN0RXWlJxTFJhdWdrakpUYWRrMlJySW9SS1JhdHJDcTVLcGQ3RlptN1ZyTGNDNDZocXNvekxyVzZ6Uld5Y2JLS2FqdDcwVTZUMXpORU5nemVlZ0tXOE1pQ1dPNWFwand4UVRQS2luVlppJTJCJTJGbjRuTWlPZDAyVngzJTJCY2paN1kxVm9sQkRMcGglMkYlMkZ2dkRkOGhSTnYzeVIzNkJaOG5YUEpwYUwzZGh0clFQYkc5V1BUZ0Y1cFZjbHJhWXFKUzQ3OU05dkhIRlFmZSUyQjRQWnBkVE1STWhlcWV0QkZyQ1B1MjB0c0M4SE94YXJXbXdiV2xqUzFadFlZV3NienJlOWFCcjFqbGVoWDVRTmxkNWZYSUlyJTJGdWY2VFVTeSUyRmZ3bmVUekhhU1pibXN6NnA4bEQ1SHRXSzRaWlV5UE1oNUl3UXdIdyUyRllBeDJoWU93UnpqcWU0aHppQ2tOc0s0UzB3UHBDSGRyWGswZG4lMkJheXQ1Q0lJZzhFdVA2eGpuU0ljaThBclA2UiUyRjFQSkFTMVNKNWpTN0tiNU9xZWRHNVZTbmNrJTJCaGpjaSUyQnlRWHFVcGxvYyUyRmZTS1Zrcmd0azVzUjVHSDB6RElwNEpqTlo2Zk94dUEyWG1XcDRPTXZTdWJsU3lWSmIxJTJGN1BGdVVtM1FKbjBmdUpVaVl0bjVsblJaY0xGZXFyUFpXSVFyc3ZLJTJGbFZYJTJCRkZ1bXAwV1JaeiUyRlJmN0FVUlRCSUJuanZINXdHWUFINHVuTHU0R1QySndFaG9ndCUyRlhnSk1SakdQbU1Zd1lwaHdFNkZFNDg0bndKem52M0V1RUJBQWdrZkxPaFR2QWVGV3d5d3Q0RE5pY0dOa2NCMjJ6a3VHSFRFZlllc1BWNFFkT0dpTGdmUFc3YWJLUzlEMjNHRFcyMmhla2ZOMjElMkZwTDBIYlQybTE3UVI4SW5ianBzMlA3ckJ5NjZqa3plVEtoZ0RvOXR1VG5aMDRpWmdScHk3NFR6SjBRa2FNRWN6d3Y1SlJpZU83UWo3UmJCUGJIU0N4bG1tdldpZjF1Z0VqZE5NJTJCOUElMkJzZEVKSGhEYm9valB6QktvUGlwa1lYZ25LdGUxdklONmQ4MVR4QlpQVjlhbkZySkUzRm8wZmZMRmx2YXMzRGxiSlRMTiUyRjY3cHExOHFXOE1ubWVyYjJ5S2o3WVZEbjN1TnBURGdSdXZPMzBJdXEwaFlGN1g2WGElMkZCVG01VldNMkY2cmhkUTkwcXNnZm5BVkg5aTNHR0ZIZ0IwNEVJQTElMkZIS1hUejZ6dURicnRsWGl0dVFmQzJuTHR6UzU5bHBrelMlMkZCRzNUbWlxelhpaEt2bE51T1JzMjhCdG1tVSUyRm1FS2JwQ05OWDFROTJUdFA0OWhVYzc1S1VpV3V5akF5ZGE2cXNPeTBwRXFhM0wzdUxhYkI3azFyUUdKMlhKQ0glMkZjWWlheWN0czU2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'
tone_sync_example = b'iVBORw0KGgoAAAANSUhEUgAAAkQAAACYCAYAAAAFg/YnAAAAAXNSR0IArs4c6QAACcJ0RVh0bXhmaWxlACUzQ214ZmlsZSUyMGhvc3QlM0QlMjJFbGVjdHJvbiUyMiUyMG1vZGlmaWVkJTNEJTIyMjAyMy0xMS0yN1QyMSUzQTQ4JTNBMTguMzcxWiUyMiUyMGFnZW50JTNEJTIyTW96aWxsYSUyRjUuMCUyMChXaW5kb3dzJTIwTlQlMjAxMC4wJTNCJTIwV2luNjQlM0IlMjB4NjQpJTIwQXBwbGVXZWJLaXQlMkY1MzcuMzYlMjAoS0hUTUwlMkMlMjBsaWtlJTIwR2Vja28pJTIwZHJhdy5pbyUyRjIxLjYuNSUyMENocm9tZSUyRjExNC4wLjU3MzUuMjQzJTIwRWxlY3Ryb24lMkYyNS4zLjElMjBTYWZhcmklMkY1MzcuMzYlMjIlMjBldGFnJTNEJTIyWGVnYjhTMmlram9hazFUU1FxYnolMjIlMjB2ZXJzaW9uJTNEJTIyMjEuNi41JTIyJTIwdHlwZSUzRCUyMmRldmljZSUyMiUzRSUzQ2RpYWdyYW0lMjBuYW1lJTNEJTIyUGFnZS0xJTIyJTIwaWQlM0QlMjJFMm5DMDhmTGtfak5Ta2ttWXpaRCUyMiUzRTdWeGJjOW82RVA0MXpQUTh4R05KbGklMkJQdVRUdFMyYlNTU2JudkxwWUFVJTJCTlJZMElTWCUyRjlrYkVNRm5JczZrdHNpSGxnOENLdnpYN2ZybmFYaFFtNlhyeCUyQlMlMkZ6bCUyRkk0R0pKcEFNM2lkb0pzSmhKN3I4ZWRVOEpZSk1MWXl3U3dKZzB3RTlvS0g4QThSUWxOSTEyRkFWdEpDUm1uRXdxVXNuTkk0SmxNbXlmd2tvUnQ1MlRPTjVLc3UlMkZSbFJCQTlUUDFLbCUyRjRZQm0yZFNGNXQ3JTJCWGNTenViNWxZRXAzbG40JTJCV0loV00zOWdHNEtJdlIxZ3E0VFNsbjJhdkY2VGFMVWRybGRzdk51MzNsM2QyTUppZGt4SjJ4JTJCdyUyQjkzUCUyQjh1bng3JTJGJTJCN1glMkI4WGolMkZJM2lJTDJ5aDVzV1AxdUlUaTd0bGI3a0paZ2xkTDhVeWtqRHlXbVo0JTJGMmUlMkIzRlJ2RE93JTJCTHFjSm9RdkNramUlMkJKRmVFY0hhS29BaHdoSXJOM3VEWThReFhMSnNYN1owRDRRdWNaenYxZTFQd0Y4SWFmMkdaSXd4RDR1QXlwUmclMkZpbW5NaFZkenR1QVh1UUg4SlRkYUhKRDBDaVklMkZ5czRsZ1VLMnZaRXE4RkZOVnpTTnFab2xseVVrOGxuNElsJTJCMHpGVGlDdmMwNUxlelF3WlpCdlRNJTJGUU5KT0xtZVlkb1F5MHBYZEoxTWlkQlRaT09CYWd4azNVQ3Ztdm5KakRCRjlSYmRuV25xQXc1SHdEV0FLODUyTE5nYXRkYkhnNDNyZ0IzNHElMkZrV1lGQ0YlMkZJb2w5QmU1cGhGTnRvcFElMkJvbHZiOFg1OXo1akpJbTNaMElUN05ibm13czhVJTJCNmNXYkN3UiUyRjYwelI5SEV5U3dnVUJkQWxYclJ0QndGZFVkRThnWkNkUTJnWUIzZ0xMYklvTTB5dnVna0R0U3FQVTlySEtqYWNhZ2F0MTlFTWdiQ2RRMmdheURMQWkwdVl0cGxQZEJvYnhOMDFMVnBORG05dGJsSDdReEhmajliRzFjc1JDNmZmTG1BaHVPNmUwZjh1N2ltQTFZVTUwOGw2cnV1dEpHSjBLYXpBNVY5SWU5N2xhVldlMlprUVljMFo5aFNlakhzJTJGVG9hak1QR1hsWSUyQmx1ZjN5VCUyQlVtYlFjeGhGWll4NW4wc0paUndyR2slMkJLJTJGWjJTZm1qdERTcnZtRmVDYWp1R1o3dU9NSDJCWnNnMFRPbmhxTFRqbXNIdTNOWWJwckJXVDJXSVcwRk9OYTM3UTZ0UDl3Y0F5bFNSMiUyQmhuNXYlMkZRJTJCaHolMkJEMEZsb1Rua0FGQ3JwekhFQUpCelRSOEE3RDREQUVTb2lpcm5GZ0NPNkxtZVF3QkFoMVhleWZoJTJGclg3Q0lQM2ZQdGIlMkZleTBhRVhiJTJCMWtkUDJQMlBhSGVlZyUyRnRidUxMSk0yRCUyRlIlMkJySXhPVTZDR25xU056WFZ5cGM1SlhKc01qbUZ4R2lpSlVRJTJCVkU0UzVHWWNsTVRMcjlLZ1FpbmZuUXAzbGlFUWZBdUQlMkJRNDgweGpWZ1o0S244UU53dnk0MnclMkJDWUJPR1dDYkV1VDRBRm1nWU8lMkJXeEJqdXdOanBDbW0xRnIlMkJQdUlrWFpCdERYbWpFMHJHcWs4QTd0ZWkyTVQwWXZIY2paQVdFblJLRW9XMGN4dUQyRUZaYmRJJTJGaGdrTnFmbG45TTJCZ2N3RGhBWUNvU3dBMTR3bllNSzEwUUVHOEN4VndiV3hndSUyQkRnVU1YYVRkdkRxTHZRclJiWFQ3a1RtMSUyQmVUaFh4d3A1OTRYVzZhMSUyRkF3d2ElMkJJNU1BR25iNXZ1MldSM2NwbENNRHBTZGJCOThPdEUlMkJEZGl2NFNhTThmYUJ6S053VGElMkJiVTFaM2RFc1VkWjlTbzFnREI1MExiTnF6NlkydlZ1aUV3U2tKNjE1aTNXNk9mSnVZYVIyeUVlYlh1WGpDM2FuMUhmOXljaHp6TndRT0s2U3JUSEthQno1UXF0czBUdDQ2WWdnenJ3NG1pbG5WNUFYJTJGRHpmaW1zR1pBV2VCSEZHcmJ6c3lrSkh0M3lwSzJkS3I5ZmJvMXl0T3NkZ2NvVHRNMWRlT2U5Uk0xelZkd0g1JTJCcFdVZDg5JTJGbnA4VzYwYld1VTk3TnZ0enRRY1pxbzYzeXhFZW9hNWYyZzN1RlBTODQ5VzlQTlRqUkoxM1JjNlNOZlV3djRNVjhyJTJGckRZR1V6Q05wYmQyaDlYMU0lMkZYcWljbVBqNWR3N1VLN3MlMkJGZHFOOXUxcDNMOXMySHY4dlFPdUl6VEN2MU4wUDVtcSUyRjVBak14MVJORDJlalRFM0RsQjRTTmF6VzcyT2lWcHg1OW9hU3FPR3g1dGIlMkZpTEYlMkJwcVlaYnVzaFZhdFZiWDh5dkJ2dDJ4cmwlMkZXemM0NXlEM2hlYm9WNnR2QiUyRlVPJTJGekRoSE5QMTNSVHlVM3lOUjFYZWtqWWJMV0NIeE8yNHNncTZENWg0NGY3UDBmTWdOMyUyRnd5VDYlMkJqOCUzRCUzQyUyRmRpYWdyYW0lM0UlM0MlMkZteGZpbGUlM0Xw7bFuAAAerklEQVR4Xu2dfaxsV1nGn+m90bZQNEBSEhQUSkokimkKjdL+QWMo8o+GGhApmvBhIwUBg0blo6e9YjSQFAwtgmBC2oJES5QIgUarhGIoIUBNiEUtEAiRCr0IxV6w93bMu2evc9eZ7jmz9p6118fev7m5uXNm1l7vu37PnjPPXevdey3EAwIQgAAEIAAB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'
rise_fall_example = b'iVBORw0KGgoAAAANSUhEUgAAAlAAAAD2CAYAAAAZOLmfAAAAAXNSR0IArs4c6QAACfx0RVh0bXhmaWxlACUzQ214ZmlsZSUyMGhvc3QlM0QlMjJFbGVjdHJvbiUyMiUyMG1vZGlmaWVkJTNEJTIyMjAyMy0xMS0yN1QyMSUzQTQ1JTNBNTMuMDk0WiUyMiUyMGFnZW50JTNEJTIyTW96aWxsYSUyRjUuMCUyMChXaW5kb3dzJTIwTlQlMjAxMC4wJTNCJTIwV2luNjQlM0IlMjB4NjQpJTIwQXBwbGVXZWJLaXQlMkY1MzcuMzYlMjAoS0hUTUwlMkMlMjBsaWtlJTIwR2Vja28pJTIwZHJhdy5pbyUyRjIxLjYuNSUyMENocm9tZSUyRjExNC4wLjU3MzUuMjQzJTIwRWxlY3Ryb24lMkYyNS4zLjElMjBTYWZhcmklMkY1MzcuMzYlMjIlMjBldGFnJTNEJTIycGZscjBPYWZCdW11aUd3NEhRQkwlMjIlMjB2ZXJzaW9uJTNEJTIyMjEuNi41JTIyJTIwdHlwZSUzRCUyMmRldmljZSUyMiUzRSUzQ2RpYWdyYW0lMjBuYW1lJTNEJTIyUGFnZS0xJTIyJTIwaWQlM0QlMjJad2FnM3hyZzFBLU5hejdZU2F3YSUyMiUzRTdWeGJkNk0yRVA0MU9hZDlpSTR1WEIlMkJUN0tZNXA3YzAyYlBiUGlwR3RqbUx3UVY1bmZUWFZ4aUJKUVQ0RXVQRkxYcElZSUFSelBkcE5Cb0dYNUc3eGV0UEtWM09mMDBDRmwxaEdMeGVrUTlYR0NPTFdPSmZMbmtySkI0a2hXQ1dob0U4YVN0NER2OWhVZ2lsZEJVR0xOTk81RWtTOFhDcEN5ZEpITE1KMTJRMFRaTzFmdG8waWZSZWwzVEdETUh6aEVhbTlFc1k4TGw4Q2h0dTVROHNuTTNMbmhHVVJ4YTBQRmtLc2prTmtyVWlJaCUyQnZ5RjJhSkx6WVdyemVzU2czWG1tWDRycjdscVBWamFVczV2dGNNSHY2OHZUem45NDZmcGlHJTJGSyUyQm5qdyUyRlUlMkJ1T2FlSVdhYnpSYXlTZVdkOHZmU2hQTTBtUzFsS2V4bExQWEpzUFRsJTJGSjBhTjRZcWg1WDhJUWxDOGJUTjNHS1ZHVEpLeVJEVUtsaHZiVzM3VXZaWExGMWRTR1ZHTThxMVZzemlBMXBpUU9zZ25ZYmhjWEJUVTR2c1JjbnNSRGV6dmxDZFBJQmlVMWhzRGhnZVE5UTdCWFhzc0FnMnRaQUhkaVlabFB0MG1DV1VwYXlpUEx3bTk1cGs2bGtENDlKS0c2blFvVmc0S3JOYXdhcFZKZ2xxM1RDcEE2VmhUVzFOcmFBcFRTN1d5Mm42WXh4USUyQjBHMWNva3h3T05SNkQ3QXJwRjdabUFKY2NBRzlCc3ZnRVRkYUdjOFRUNVdrMEd1SkxjSlZHU2JsUVRLTnI5JTJGY2lJUG9hJTJCd0l5JTJCS2FjdDh4T3k5b2RCdmdNY3JUVjIyM2JUR0JFQWJhVlolMkJ1VXVRSllJQjl6eXI5T3REU09BOWVjbUZsQ3Z4MzYzQXY4OVZ4UHM2c2k2bWpMUndBSEtpQTE4RjI4YjBaVkJHN2lxYmV4T2JSYjBnYXJzbmRxd0E1RGF1akFYR3dXcFR1cUJMTU1EJTJGUzZpcDJ5ZXgzdDFWeVJpS3E2N0hOMmxTUDgwRGFPb0pxSlJPSXZGN2tUNEV5Ymt0M21FRm9ydzlVWWVXSVJCa0hkenU1NkhuRDB2NlNUdmN5MkNkY094VFpPWVA4dWJ3dVYlMkJFWkFqWERtemhoRHdhSGRXalRFSUlDUktxOEVGUERsb0ZmJTJGbkllQ3Izc1F5M1NIUzZWd3FPWG00NkJwUWIxWUI5cEJ4bHJqQ1BuRkZCT21qVUIlMkJHeEl6elcyWWVGVlFQMkpvaklEMkJhaTZNJTJGQkhUM1pqVzRQR0dETEYlMkZ5aUF4bHo5U0xwQ01OeElNdDF4UWc4WCUyRndwSUJPUVI0YXZQM0NFb09qaGQzaEY0N2V1bDVnVkhtcDBieW5KZzhWdmVrdmZmeUU4SzZXaGt6bm8waFJ5VVhSb2JzWklpWWM0QzI1cktPcEFpQ2hsNEhLZzJkbHk5bXpnS05FY2ZPaUFQQiUyQmhKOHlDRUhzZzJReDJWaEMlMkZBRSUyQjhEM1clMkJkJTJGalBJQVFBSGV3SDFBUzBUa2pKUEJjWk1CZGozZ3RJNyUyQmxyelFFZG5xN3N6WWptNzZuaHJNRE1QSW5oT3dCJTJGbTF1ZU00OGxqMVFFS29KYjdTenZ2dUE0MnBpJTJCWTBvMlVEVDgweTFzblFsUDBlYWh4aDVpN0duT01lSUF0ZjE3aDJHQ2JLcFZIR0pjRmhJeGszcDRZR0N2S1lKJTJCaG5jcThqV0p1Rjk1N2RNVFQwMWpJT2ZWZXA3RkhOa00zcE10JTJCY3JOTG83VGFsazYlMkJNdHd6YXhzcUdmRzhhaGNzSHVSM1JGeFk5SmxuSXd5Um5VbHFnVjdtSFgyckhLemRST3BTSVRYbVhPOUVwaVhwZFNDSWYyS3ElMkYwUEJzU1RpcUZCYmhYQVBQVkVZNzNWdzduYk13WHl0ZlBoWFNoRk41clFoZXpScWJYc25oa2lveGZKR0UyQ094TUJMaW9IeWpKJTJGeDdlMTFMMjdReVVIcnNrVG00T0hxY2tRd1dzUUNCN2VWZmwwV0dwbUp2SiUyQkl5ZE5kWTRmeTlTc29EMTlrbXFMJTJGSmcwbDclMkJicXhkM2xjYk0zayUyRjQyaWwzZHBLVmdoQWtJTUgxZFJ4a3F0NG5GZjZqMEpXWEhYcFhqd2F5SzdUNmJhWmpVTlFnU1U3ODZNcEhkdkpHdXFxaGdVeVg1ak16cVM3RjBrcXdwR04lMkZsVGszajRPeENQSEZXUjhiOHE1amUlMkJtSGxuTWY5MyUyQm1xRG1HbXh3endEYWZFTVQySHVEZUNuY05IaUZBWSUyRiUyQmttdmtiSGxnbHJSdnExbjJvamhDaXlvRjRuakJxJTJGUVhFUGZnNGRveXJTTnZPbWROOTNlb3lYNkhTcUhtbkp4cCUyQkRRUGQxMFBuS29oVU91b3hkek9KZE1vcVlzM2tpaTNrbGslMkJVaiUyRkZxeVcycm1zMmN6TSUyRkJXNHd4JTJCeUg0ZU1kRXN0bWRYcndxWDcyME5VSyUyQjh4bHpTT0RkUlNkTnN4a1QlMkZYQzBOaTV2USUyQkp4SFBmOGhBWVAlMkY1VXJGWGs3NSUyQnI2JTJCUHI0a05TQzJSQjcybWo4d2FmM1lBZyUyRnEzQm50QUszYTNQJTJGVlFMSVMyUDVoQlB2NEwlM0MlMkZkaWFncmFtJTNFJTNDJTJGbXhmaWxlJTNFgQETugAAIABJREFUeF7tnQnYNEV1tu+ORhHIHxdUFKPgbtxxwwQTxQXUuAeIBPcFFTeMJu4oRnEN7kZFxaAmqGBcARVcAFdE1LihIi6IC7gLion9X883VVg0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'

toolIcon = b'iVBORw0KGgoAAAANSUhEUgAAAu4AAALuCAYAAADxHZPKAAAAAXNSR0IArs4c6QAATFp0RVh0bXhmaWxlACUzQ214ZmlsZSUyMGhvc3QlM0QlMjJFbGVjdHJvbiUyMiUyMG1vZGlmaWVkJTNEJTIyMjAyNC0wMS0wMlQyMSUzQTIyJTNBMzguODY3WiUyMiUyMGFnZW50JTNEJTIyTW96aWxsYSUyRjUuMCUyMChXaW5kb3dzJTIwTlQlMjAxMC4wJTNCJTIwV2luNjQlM0IlMjB4NjQpJTIwQXBwbGVXZWJLaXQlMkY1MzcuMzYlMjAoS0hUTUwlMkMlMjBsaWtlJTIwR2Vja28pJTIwZHJhdy5pbyUyRjIyLjAuMyUyMENocm9tZSUyRjExNC4wLjU3MzUuMjg5JTIwRWxlY3Ryb24lMkYyNS44LjQlMjBTYWZhcmklMkY1MzcuMzYlMjIlMjBldGFnJTNEJTIyYmpTZFNpTl9lV1VjVU1TVU80T3YlMjIlMjB2ZXJzaW9uJTNEJTIyMjIuMC4zJTIyJTIwdHlwZSUzRCUyMmRldmljZSUyMiUzRSUzQ2RpYWdyYW0lMjBuYW1lJTNEJTIyUGFnZS0xJTIyJTIwaWQlM0QlMjJzeGRCQW45MFZDaWN5UDdYeWJYUyUyMiUzRXBielpzcXU4MGkzNE5QdXlJZ0JqbWt2UnQ2YXhhY3hOQmFadlRHJTJCdyUyRmZSSDZiVyUyQnZmOWRwMDVFUmRTTVdHdmFUQkJTS25Qa3lGUkslMkZ6cUp6N2U2cEZOdGozblIlMkY0c2k4dmUlMkZUdEslMkZLSXFrVHpUJTJCQlZjJTJCZjY1dzlOOEwxZExrZjIlMkY2ejRWcjh5MyUyQlhpVCUyQlhuMDFlYkglMkIxNDNiT1BaYk0lMkYzM3hXd2NoaUxiJTJGdXRhdWl6ajhkJTJCM2xXUCUyRjMyJTJCZDBxcjQzeTVjczdUJTJGMzY5R1RiN1ZmMGR4SnY1elhTdWFxdjduelNUeDl5JTJGUDlKJTJCYiUyRjE1WTZ6UWZqJTJGOXg2U1QlMkY2eVF1NDdqOSUyQmZSOGkwVVB3dnRITG4lMkJlVSUyRjRQZiUyRjEzeDVaaTJQNiUyRlBQRFFtNnRiVjgyJTJCSEhxdkMlMkYlMkIzSTR6cSUyRjNYNjA4cWU5cSUyQiUyRkElMkY0WHhmUzRQV0hDSHlyNDhNJTJCRjVsbmhPJTJCdiUyRkRKYVpYOUIzNFV4UyUyRiUyRm1DYnpuJTJCTGFmJTJGMHgzcmt2M1gzJTJGTjBTJTJGOTFRczBUSm9OU3BxSENGeCUyRnBXakJZV2NRbUZCeiUyRklFeTFHaEglMkJ1VnlEV2c0cSUyRkFuJTJCSVIySjZJNSUyRlMwUnh1VEZ3QmNXWHEwJTJGb2FGbnBqUEhnRHNId1pTVW9GSFk3RmNwSHV6NXZTNlN4JTJCRTNOdnlpQk1iNyUyQnE0eE9WdGNtbnFIZVJKdmxua2Z0ZCUyRnR6QzljaG5iN25aJTJCbmltdzhYVmZnSElWRlUzaHJ0NUtRV3JYZms0UiUyQkVCRUdqZFNSJTJGb0VmNFIwSzE1OGtWMmhUeXE0djRCNjdqJTJGN2clMkZ6NHZDb1h1eVFNdVd2OTBQOUd0WFIlMkJoQU5uNFdSbFpuVnhtaGx5SVExdFg1MHlZOCUyRlh2MjExWXMxd2hadFo5JTJGanI5OXd1SndmOCUyRmlIMXRVQkhSRkloSEhkcVglMkZiVk5IMGtmODA1YUdhdVJwbGJlcFpQdTN6UXBKdUFQJTJGSHAlMkZ1SVpFT05HJTJGVDZUOXQ0cWVQUDJQRDMlMkYlMkZwbm5LeSUyRnZibnolMkJEJTJCdG5Yb2dTem95UEx6N1BPUHpIN2QlMkI5dSUyRlRKUUZWQWdpOFlqc1ElMkI5a0xOeDlHSVlUajZjRFByJTJCbmFTTEdaVHFMb2xZbjNpS2xDcExsc0w3UTRpSHN5VEVLTTlleHkyY1VVQ3JPMld0U2tRNk4lMkZFYWhLMGc3WkdIVWNXTXg2VlZWWjBqbzE3QVE3eUloaUZseURYcXNTcDF5VCUyQnVyaWp4eGhYYkNYZ3Z3TTRxY0klMkJRZ1ZSRFFlTVBQU0NmOGJmRzA3bng1aUhxRnhFTzAlMkZGYVc4Unl0a3ZZaE9kUTR5b3F0VWJHeERLUWhUY2RXTU9rR2ExMW1QUEtuOFJQbFE5eUVRQlROWThENnVpYTRYVmEyRHl4VXRSeG50TXBMRXglMkJSSngwSkxTeHNLcjJsU2hjOWxLM1h0NEowb1JDV3p4bSUyRkFWM0YlMkJEWXZ4alZBNkdNJTJCcmJuR25kUlZiQlA0dTYlMkZMZTRNRllDRVZWVTFnWEJIcWpkQ2h2UTdteHZBVm1JT1dBRG0xTDI3QXY4cHZ5ZWVWeHJ5R3BkRXFKQnd0ZjJmOEFvMm9zbEFRRzNOUzJRS0trSlRMV1lNJTJGZnVGNlJpZlV5M2docElrWVBKUmwwa1k4S1ZoJTJCYTQ0bTElMkJzUTdxUXAyOUF6ZzBTZm45VDM0ektlcWFFZndDYWtqU1MlMkZnVGpoZmlFejdvUFRjVVdWZ2VSR0RTb3NLUmJ1MFpJb2JLeER3VFpTU1pubm1TaERubG9OcW1mZU9xejRjalUyayUyQjU1anJDZ2xvcDMzQkhPcVF6Ync2OHpab1JzbEdNYjh4bzh5NEU4dnQ3WCUyQjFPSENlJTJGeVNSU2w0NEwxb3hKWDd5bGp3Um10UkZ4cFhiZXczdG90Umt5RlRDUDhQM1JkVU9pMmJUUWRhN1B3NlpydVhzV3IwV0s4MFF1aCUyQk9JYjJxekVzQUIyazJFTVFBWHUyaVB5NmhIdFNIMTJBYXBFMFRxd0xvaWMlMkZycVZjbFRCJTJGQk1ycTE4eGJsRHlVTU5ieXZWMnlJOWFNNmdJNjVjWDFyT2kzeVhFSHRkQ3FGQmRHZHh2WEE2TXk0ZHhWYjl4NlUlMkZmdiUyQk8yOGNmdU5ZbVh0M20zSHoxNzdHZ3BXSjhuTTJ4a1VvRkZLdkFJeTlUMXJ1WUQ2d2VXNDZqV3B2OUNGRnBzc1RJNXRIMTdiZWM1WDBZU2pRNVpKbXJiUXhmRG1tejhOb3dqMWF4UDZsMTlpUGFIdzVMUkNPRmxkJTJCZGlJQlVQQ2JTZllSaFZkRnBDSWRiYVEzbHdVb3Y3ZXdlTldxMDZoYzRqR2I5TGduY1piQzFLWDdBVjlEVXZvNFM4Mnc5OTdsRzlsVklIdVBCQlZGVlcyU0hyRFdoOWZHQmxaeGlxQjgwd0ltSzRZJTJGWHZDVUNZVnl6Y3NVcUgyR0piTWJuSllHR2RMQVNvUk9GOVRzM3ZXbVBycyUyRlI1JTJGR0NzYlRLdktEc0xHY1hqQ2pDRCUyRjkwSWhHZlRrMlc1TnZCWWRQUCUyQmFxUkdoNjUzVTMlMkYzSFAwSm1uNTlEdGpxWkxDNnhueGkzYWJsbkVDdU1DQXhaUWZzZmpKcmxYZ2R5N25TRjJ2Z2NFUGFjZTBuRTdkNWI3YXR4bHJuZ1cya3B0SkpudVhDaUxIJTJCS1BWWnJRQkxBUDBWNzNFZjhJdXhhT1MxcW1RMG8zbDlnVU9ubE9KWjR2OVpMdldZWnpZS1RWZ3IyQnNyU3RRTmR4dHI5azhhejV0d3J6bWpRY2hGOTlFalBFOUZUJTJCUXBsZEdSckY0aUVzODljdDRDUnBRQVQ2c1lhdnA0QnNQSUxKR09PcWRTUnkwRUElMkJCWURmN1BQU2thWmkxVG9RMmhDMW53YXhpUHNMMWY3VXVuVmcwZU9hSGZETDN5dkVoVzhCQnFnZkk3UUZHQVNnMFFwJTJCbXdycVBKd0ZmOHpBT3ZJU3RQUzFTOGFuWDJUMG1WN1g0NEl5M281TGZ4S21SJTJGdXZUS3lkZyUyQm9XUDJDTGhhb2M2RzhkVVA0U3dpWFV0QUo2bmd0dFdjZEJLUWg2UU9pN0FnNUx6JTJGU1V0NEhCSW5SaUlyJTJCZGNEbWxtRmhxYnh1R21WRXpHRUlOa1drJTJCMHRlMGc5MzJLTDB3M1JBOE1MJTJGR0RRNURkQzdaVGZxNlB0RE96T3o1ZGNyNHNNYk1NaFNGOFREYSUyRnBuUGpEZndvVUxJblBZdiUyQkJWZ3hmVm9MZG9QSkdVeWRXQWpxVzJSNFllSWVEbmRsVGw5cyUyRld0VlAxTjJVZmV3UXgyY2ZJdCUyRlJ6OUxHN1psQ1MxZU5LVElXJTJGSFNKJTJGZlFoM2FGZGlrZG45WEM1dHk2Y29YMWVrRERnSzV6QTc0UHdpVllEak1zOCUyQnFXOHliSlVYVUI3cExWYXdmdGhnaUM2ZTJMN0wyTkM1THFYQ2kwdWo3Z1B1Y05DWVd3c3lpaHpJRXRYbWNlelRPUDNpZm5ydGFZZ1N4c2pXbnlvbUZYYWdnaWVDcTJWYyUyQmlDNE5KWXBNSWhlQVlMWWdQQzFySGFYVjdFNlBwSFdRRWZMbWFFVVg1RHlqRzg3bmVma25uc3BacTk4a2FzQXdMbUZMTGMxODZJUDNOQ1dMJTJGRUkzdUI0bHJvOFhJUTdqQXA5Vnlkb2NkUXpvQ1FoSVdjZkZsa1c4Ymt6ME9LWll4Slk4SDdGZnExNXg1JTJCQVI1dTNnc25vekxBaWVOWmZHRTlGNHo3ZFJZODdQOFZvYSUyRm1XNUJ0MCUyQjgxOTVkamV5JTJCalIxdTJsY2hxazZVb1QxWWxaZ3NyTyUyRmdsa2k2TzhKVDdQWE5VMloyZUhUS3JxOUFBQnVuZDhRQU0ycmhiWTMxJTJCNmw0OHpVTWZnSlBkZWolMkI4SFpoMENBTHBQWGlnczZERWhmdFJyc1pWQTJlQWVWeFQxdGl6ajBhTW1tcGxtUEZ6Y3I4cXdUJTJCN0hQV0JibXRCT09WWHc5TmswZnl3Z3hKbFFVOW9aOUg5dU1EVEpBd1lxMzVEemJxZDMzNXYzYkVCVk42UGN5bWRRQW1IUThoand6NHZEQU51JTJGbnd1ODR6ZGl0ZUN2MGslMkYlMkIlMkIwWXBqSUZFdmxSdkw1OXpFZVVZVWVLOGJocHNnUWFnYUtxOER4UmNBSFA4QVJFSWR3VCUyQnFzd0N6RjRFJTJCWE56Q2dWOU9PNTdvWWZEMFBVUmNiTVlNJTJCSE12MFVxRjFZU2JRNml0QnllJTJGcWpGJTJCV0wzUGhUemN5M0xxNGM0TDlTTHprJTJGWGMlMkJndngxUWN2cUpjVm00MkFZMkk0eU53bWhNRGRORmxhciUyRk1XJTJGVnNlbmRHUHRETGpHdnZBR2pmTXJ3ZExOODRYa2ZrQ3M1WE42cUpIR2h0RW9DOUVJblg5TUhaRlc1TFI3bVhtWDMxd3ZjOTBjQ282eEhoc0ZXM0ZkcGhRalhzRXFuVE13czlERFdsWURVbGk3JTJGM3BMM3JPaHduSU9vcVJYRU96aUU1N0Y0SWVaVm5QJTJCUlp2UmtlVW5neVhMZFklMkJGN3hrUktCMmFFM2Nja2o5aVZudno4SU4zbmh0SnJsY2lnMkMyZ3J6RlVGSmFYY3NmeVl0dEJPVFdxMmolMkZlZVVXTE8lMkJpYmNtN0l4c0dPZVJVRFA5OVptTlI5bGliR1cyc3J3bzUyQlMlMkJGJTJGZHF1MnI2RGRacTBnd1VaMzFWZHFBMnNhaEdCY1ZMYXNGYzh2WSUyQnJJNWQ1N1lSSUd2bXdEN1FIaVhGVmNkcVAlMkZmTlJvcTB3WCUyQll1SGhLWjhpY3Z6VXVORWtieDlHemRrRmhkTDZxdURWZ2VjZmZjRSUyQmElMkZqdkt0cWpKdHJPM3J4enpwOCUyRmVQM0hscXBtR2xoUEhKJTJCZ1BDVXAwQTdldnZFaGQlMkZMOUYzdGw4RHdtN2x3WkNONU4zQU42cDVmJTJGWXlMSERCS3VWN1pWYWlMYXlmc2VqdWtwaG5WYWRIbUswSlNuTTNTTXdlRkQlMkZjdkd5N0tzNTdSS2QlMkJjYUtMTFpyJTJCUnpnc1lENUlWYSUyRldRUVVDYjglMkJpMXV5Q2h3bWlEOTdoZEl2NXpjOUxvSXpyUE9zTzhHZmh3MXQlMkZJNGlUSXRWQWMzaTI5WFMwZjZ5JTJGZlBHbGl0cEJCaTdScUt0eCUyRnNxWHhPTG44VFZoamlJOGI3VmE2RlJnaU1DNiUyRmUzQmE2VVUyc0xkNUNnSEJ1NldhNiUyQlNXZmwxeU5QQWloblk2c0NMWno2c1JEcDBxJTJGZ1I1UGxxWFhCYlNWQ0d4aUhRR2pOaURSTm5jeW5lajBxcEFHZSUyQnNrOEI0dDR0TiUyQjdKT3AxOEVuTU1SSWVYWnR6NldaMFZYZzVLJTJGOHZ5ZTlNbXNwUHMzVFF2NXlUJTJCMUNxMkJiVzYlMkZ2dk4zUGFNOUclMkI5M0VFbSUyQmthZFRNc2kwc2NUUmpOZzdSckl0alUlMkZzYTN6QXVaUTJtekVEd1o2ZDBkcHZSZkpaQWV4VHZyUGd1bk40ZXFzcm41QkZzUmRSbmwlMkZGOXVqJTJGaEhiTVNiZVhHdHUyMmx1TjBaTDRubTZKaDhPUE9NZGVFR3JpQ3h2UEF2NjFyTVpkTVRjNWR2RGZiRGpJMnFvOXJPRTFMcWJSZVV6b0hGNzVyM2lqZnU2YXVheW1OWlBXSzhFJTJGdjB1JTJCRzl3bzAwc3FFYVoyQ0o4dWg3aXdQSTBjZzJYZ2k2ZTk3aHhvM2ptaVdCOGZqbzNaSjZuUHZHSmY5dFh6N0M3bGt6JTJCZ1dXN2dSc3pod2U3aDJyNnZmamFQRVdQejdvemIybUVxSW9YVERiWFRqdWhIJTJCbHBLcHNidjFITGgyZlVjZmdFZHBRV2JKMDBCRnFBYWZmNVRXR0l1NVpzSHRySnhzeE55aTFtcCUyQiUyQkNGckRxRXVSUmJGczJ0WHFNS09NSSUyQmVhOWh0dkdXSWlRYk83ME5hdFRQS2tHMzdIaHElMkJmZldDc0ZEczhIcGp6YU5NYVBpY3FWV3BXQUNGcm5vQTM1R2QlMkZmQmo2YlVIUCUyQmdQekdlZVA4JTJGN1FzYlpnb0ptN203YjE0Q1Q4RU5NZWxzNTg2ZkxEc1JPajhvVSUyRk9QeEV4YWNiTFVMRW1JOFBsJTJCWHBNMU1ET2JZcSUyRnBKRVRKY3dlRGJZdU4xZUlGJTJCVEZVaUFNNW9pUGVrNm5WdXRPa3VnZmhydGpMZVlFVHl3T0hQRnAlMkIzbjZubmxzc2pIMk41YXJmRE5KNVhSa3U3UjdiJTJGJTJGTkxoT0slMkZ0dzVIRzVhWCUyRjlLaVdGQ1BlamlWT2hNRkM2dVolMkY3MThOQlh4Y0tSaFhvMlolMkJHMlRxU2dKbWVhd255R0pUJTJGWEd1TFRYaXV2MTElMkZrNEFiRHBiVyUyQnRrb1hjJTJCMiUyQlZYR3hqdGJtN2F6Y2lmcFA1cUUlMkZtSTVncDRxVWZWVlVEMTROSG1hT3ZnVyUyQlQ5TVAzRFA2TWhKTXpuaGtpVXdqV0J0T1hHQlVKcGZFajc0NVQzaFVOdk1oNVBmT3VsJTJCUXhpbWhRcEhmd1daZFBuclElMkJQZUt2U2d0JTJCeUZ3N0R0UmZLcTBScHh5TFIlMkYwUTRNWjJkalNNcGlTN0xqajllbDZHJTJCSVN5MlplUktPcjlNelclMkI3alhjNUdyaFd3ajV4SnMlMkJKbDFjakFMMXBkTlZQRzN4bFIzSkI1ZTFpRDk0M0RZZW1WUkRTMUJHQ0hxd3NUOW5NcHRYQWwzOUkyMWElMkJNeUJzTnRuREE0MnJsalYzejNPY29aOFRTck12Ym1INzByUlQlMkJBc1BpZDlweGhwZGZPQndZZyUyQjAwaVZuMWFTOHdsOFlWVGxKR0Q4N25NRjFvJTJCVVVYQm5id1ZidXNCVUtaVDZxaTB3JTJGeDg4JTJGMnF4JTJCcTdLOWw3R1NvcSUyRjBqWmhmT3BvQjlGeUFPZ1d2MzBBaUloOFdBaG9FN3p4elcxZCUyRmY2cEcxUnVqMXpibnYzRFZrVjFUcjMyOGhuRlphZ0dhMmMlMkJud0hKWXQ5dFZTb0xjTkRmRGM2cEsyc2tMZDRtSWJDMDJ0YTF2dzdFUVlLTTU4MEhOTE9aUTFhbnhmTmNzbk5waUhoUTBEYkY4TmMyZlNZSFhsQTVqZTBpNWt3TmI1NzFlJTJCZGtUUVQ1aTdkSUE4elk2VGd6Ym10Snc3WkNqVmIlMkZnWnpVRGc0MUpMY21GeFhob2VNREI0WmM3djFoSGZod1l6cHhGJTJGb0NhYU9tVmxlTDZuYjF5T3ZPMVhPMmRrN3NhU0l3JTJCZkxMWG9RQ3hrRUZ4Y1pWUVB4WW5MUmQ4cG5JOU5tMnZBeDhOdGx1aU1JNXBBaVdhdDF4UlJNTzRWOXBOTGhINDB1Q01ibHVseldMaEJKcDh6aEx0OFBqcWlNejVDR3VSWjQ3djc2NkI5cHAxJTJCcGtXM3RidWglMkJ6Y3Q4T1QyZjl5OGZQNU9lamVrS3NsbVBjZENySXJNOGpmejY1ZGd5ZEhBbll6Y3FYeTgyUkVVTHZRaGlVM09GakdTMVEzQnN1NEw1S0lXaG9GdzdlV3loOG8lMkJIOElwWEslMkZTRU5hNUg3TWs3dnpFZmclMkJ0cXJDc0s1RjVXM0huJTJGamolMkJBeHVEUjQlMkZsNzN4bGRIYnp0S2wxS0pKN0lFNHVlUDdOTmVJeDBBc20lMkJWMjRKTXFGQ0pTUkxmSFJtbExQd0hRNWhOeGUlMkJIJTJGQkZsNWR6WUNMcGhmbnk3TW5LMmVqODRjUzlFakxzYWtlVUZ5YnhFVmJqRWU1eDRtczQwTWMzWjMlMkZreUdwTzJaR0FJNWRZN3M1SGZDa0VocG8zVEVOdlJNWXZxYW56QWVHZW5xY08zUFFhZzY4RTdoVTdzek1qTGE1WVYlMkZSejZmWWwlMkJnVThPZWw3WlFrOWgyazFQcWZSUjFxTDR6ZlRQeGhDeSUyRlhJR3MyaTdQaEVMZWFyeWZheVBlJTJGQWJic0pIdWJhVG9nUDlNcmxSM0IzTmlYMUpVcnVYeWQzSGR2JTJCZHRQdXhxa2NyS3NHJTJCVW9ZeFI0RmpLMUdGSCUyRnJyNHgwMnBoNUFTajZYTVg0WXFmdDY3NkQySFhjMzEwMENNcmlOZWtSJTJCamNqZU5VWHVaaSUyRlJNd2tpbG04THQ3M01jenNOY0FleGNQZEFvUEo1dVl5bWoyaDM3OUVDQXc5QVZtUXdtdzExVG5LUG45OWlhZ3RZJTJGVklUWU14TVl5TUN4a1ZrRGZDJTJCcFpTNzlreDB2eE9RMnlvemtLSUtjRjhkelBGNTBLREprQXJxT1JqODBHdkxhWkFoMUYwT3kzNzh1TUVNRVZmMDYxbDhtdlFvdTE5U25uNm5rb1R6OTlydHQlMkZmOTA1ZHV4S1FhUGdrdFJJRXp2ckpsWkE5N3NOJTJCQmZubjNkaVN0MGcxWEx2dDQ0ZDhnQWU2TUhYQ1hOU24lMkZwVVZUQWNFVFgwUUElMkZWMFRRT1ByRzV1TjNURyUyRnZQQUViTmdQOGZoMU5SVFk1UHpxNVFHbXdHSGJHa1RsbHY2Zml0R2FkNHNndzdyMDclMkJ6UGsxWTNLYTVzUW54THJTTzNnSkUzczZZYUR6WHg3c0lMM014cmUlMkZ0OVB4NEI3QTEyNnh1c1N3TktKbkNsMnpUamZNNHU3M01PNFNRUHJLUVA5c05oUEhpZnRuc3VYbFBCWEVzeEpXcW5wOVo1ZldXYUpCMmJCJTJCdlglMkJ2cU80eSUyQkIyMEc5dXBGOG0zSUV2S2ZOaE4za25PSGtGTHplJTJGVHJ1JTJGNE84MyUyRmVIOGxTUkNkVHZ1SjduOVNTbDclMkYzM3lMemU5N002U3RoMlBxR3FnZEEwUWVpZWRUWklGMnNiYTBHbVNMS0J6Y081MDluQm8lMkJ0WE15UEFIR2RuamdIR2FwU2VxcXI1WHptc0pUMDBYQkpnJTJGRzF0amVJVCUyQjBZWDlYazVwcSUyRlElMkY1QlVqTHNoWVFqdlR6RTV0bmElMkJzaUIlMkJlJTJGQUFic2J3VjR2JTJCbks3M3IlMkJVYzJtbThRQmVwM3lFaExIVXZ2WVglMkZ5VUhXR3NFWGJyeCUyQnBzYmlITzdaYVZ4ZXhFOWJwcVFteGZoRGtibFIlMkZJaG1xNWVMdVFPSXlRRkRCN2xtYmpYQjBqWVlYTll4SHVFRVhuWnF5JTJCSlF6amxiZTl0U3FxRnB0Qzl6dTd1T2dCa0VQVVdQTDUwMlhtWEx4WW5CUDFHMGdMalNtdkI4RkNueEo5akVadHkwT2Fhdm9YakJUSTJQY1FQazNIcmkxRHhLMkpFaFhtJTJCVERYVmxLS09lSFBqekFVaUJIazYzMzNzZXlJNVpEQno5ajBnR3Fjdk95N2Z2dEZaRzFNMHFYb1diJTJCJTJCRUlzR1c0SzBYNEV2djdkbVhOMlAyWEtKQUZ4YklNMU4lMkZKVnNueVhHNXNzekZKWWZybDUyODhURjhJRHZhMmJCT2NJWE0xWGVzelFyM1hNeVM1STVuOHcxaUx4U0tITFlqJTJCMmJpb1dxemYxREQ5OFNHZmtKaTdpTlUwZlVkVzRSSzJrMno3Vk5Nc0dxcXJYTTMlMkZySSUyRnA0T25sNXAwZzhnejBLNnB6V1poa1k5NzglMkY0dEFaMkdVaXhoRm0wM3c3cDNEd2RhZURuQmxZcWV2QWtwbml3YnRYcmpIZjQ3blJVUGlYZiUyRlpnbHVyMGpNWTdXY1lyR2pEM09OSDJsTmZkM1BieUduaFVpdHk4QlBhZEprejloYm1xaERvRlclMkJzWEtkTXpEJTJCVlBaNnJjcTBtQ2F2RGU1Y3luYlFsbk5yeXBnVWVUdUtWU0F4ZSUyRjhLa3FYa21Tc0clMkZqOGVRaXR5JTJGTDlUQW50QTU3ZCUyRjhZNmlqWVA4VExLTiUyQm41UVkzdGxyZFFTdVd6amh4akRQY1laJTJCWHAzMk1KVXUzUkhxJTJGVFlpU0l2ODVleXJzQjdlR09TMURVN2F3NzFhV21FSUtlbjhLVGhZSzRoWTY0UlBkbGszM2g3b001TkZsZE8ySTFXOG9CSU90TFBFTUNPS2l1UUdnenZRYVBCS1lBUyUyQktmdGxHaXcwcVlZSjBoQTJKQmwlMkZKYk5zOEROaXRialR3Z0FyJTJCSmtRZ09HSXRhcXFRaGEzNGU3cEdKWnQwQyUyRjdSdndYcFRkZFdyVSUyQmZDTGdsdWNGWXNpVTRuaFpOelFPRWt4bkhMSWtVZ1I5NHVSSXh3RE1jdkE2U3duaWtqNldvYXIlMkZKcGlxSVc3VGZ0S0JyWk1RN1ptVUNqVFlHMTdOQ3RHMElYelpYWjkxMGpCRGRQJTJCSWlrRiUyQjRhSXc5R1QlMkJreDdsJTJGak5JYThKSDluJTJCREVMbTdLYTMlMkJyWEdqalNSQ1BRQ2FWcXM5S29FanJQOFFOeDEzQTY1MmlDMWZ0TkxsWDR1MU8xNVVZeUhjdGtmRlIlMkZwT0lpYzlBJTJCTzlIQUhONHpVblA0dVhyMUdxdnJOZmZoRiUyRjAzcmVCaUtJRG55bGFPYXVtdHBFU3NTQm8zcnNoNVVNV3VqWERrRk5YTGt0OTRncEt4VWJvMXhpQW1KaWZ2U3FPeDdKRE9ENWNieE5WODlnWDQwWllxajJoRmhCN2E1R3dPTXAzZ2x0OXQ0cEl6UVNZZmhwTk9LWG52WHE4dmJoemloTDglMkZCJTJGYU9jZUNmcjVyV0V0VUZ0UFlVJTJCb3JMcG5HT3VSYWozaDAzR010dk9lbHEycnNWRElpSiUyRnYwVFpqVHNGSWJFYUFhb1M3RHVqQjVLRVYyak1ObEN2N1N5JTJGc2lMYWtpY0VESWtpMWl1ZGp0N05iVWtpQ1I3WVhpY0NDRlhzTSUyRjM3JTJCT3l0RmdVODA0M2s4aW1CQWJoejczVkk4Q05iYmIxdlBTeFBxcHBncWg5MVY3UnZsYkxlZ3FLdXdxaDBvdmhkTWVpSiUyQmFIZDklMkZZOWZ1d1BaUEh5YiUyQkpmJTJGQmtXS3d1bTlXRWg5a3RjaXlheWVlNUM1Q3pIZjM2ckFQZGxpbFRKd01iMnk5WkdhZiUyRm5YZ09HQ2lzazkydVJrTmtOODRwSVB2RjB2RjNScyUyQndXblpsVWxFcXVxOGpNMSUyRlhObUNqUElaRzlZYjNweEZGOXNXSjRIU2NGTTBxajR3blRuUyUyRm94bjZGdVQ4b1FFWmVpOExpRkFRWno3cEx2SWRNZFVYZU9pbFNVaWFJSmMzaEhuWGFaeVN2VlJFVGVmN3hKUlVXY082MG8yYVQ5SVoxQVl6ejBmTkolMkI1ZDQ1cDdkeVVLbkZJZFFXVk51OW5QSE5BbEwzciUyQlklMkJVc3JnMEFmbU1SWHd1V1F2R1FlQVJ4NnNOSzBlTTF2JTJGSjFmM2lYNEFONE1zUno4JTJGR3FxemVTakx6WEhuWE44SHhLOW5WUXowa05ld016SHI4RXN6RkY0SGNrbHZNN0NYUnlMeXU0b2R1cDdiUFhycGZqNGl3YkluNWZyVWtBU1R5SzRmUDY4WGhOcTB1VmhsRmRlU2gzNlJVdHBERWd4U2pGd3VkaXpsJTJGaU5MVk9odjlSQ04lMkJsVXkwQ0dydU9yZ29CT1lpRzE4bXl3NzNHWFNRSnlQJTJGekJtNDB5ZFh1NExjREhIb3FzTjl6TnREbVo3TCUyQlNFNUdRcWpWQjVmZDhtTTVVNVlXZUM4NHJPTyUyQjBGYW5YRHFNYiUyRnE1WTMyMjlaTjA0cnBXQUhReWIybHd5WUklMkZTRGslMkYyMWN2cHoyN0lBb0xzWTkzUFZOT25qdnlzJTJCZkNJVUpEcG1udko1NjRwTnIlMkJpcnNvRFlsNG0zdU9WVkxtUUpMZ1hXeXVEOE9yZjNyTDdmRkFjJTJCYk1vMzVlUng2RmFaaWxzMUowJTJCZ3FJRnV0SmpvaTZjb1V2NUYzUHBEJTJGc3B6JTJGd2p1cjlHYW5mRzdIVHZsJTJGTGlJNmw5Q1B6Nngzc1E0RHl1endQOHBucjYlMkJCWDUydDFYSGlwUGE3NHNyQmxQRFpZbHlPY3pvdmFDZU5yb28lMkJyNzNpN1I3QzBtWlBiVGtsc0dtbSUyRllYN1hLVTU5RCUyRnQ2czVmM3puRTZqdnRsbkRUSmlUS2NzMFhINmJFNnNldzl6cnBuTURsdks1YSUyQllBUlE1RnBJRlFKazJLY1gwa0RaT2ZEUiUyRk9ia3lpMDc4c21VNm5wTUZoNlQ5SmVTJTJGanU1dm9IZ081bFlhejFEak5ieGQ2TGZIVThHcnltOTNFMWxrSjVIdDM2ZnUlMkZ6JTJCZldnTXVvREVGeSUyRnh6dkpqNnE3dEZMb1JiZmlmUldwU1RkOFNZQ25OY3B1ak92ZGRMYkZhc1V3JTJGR3RDVUZKWE5RN0hQanl0UWttaEtxUVVMejlKM3o1ZTBjNW5hQ2Fwb2RJODVYZWQ3cExyV3NuWGxtcDRHSldkcHRnanQlMkZPdnBIYW1CaWRWbk9IQWZTVkZBQjVRMEs2NlgzYnhyTnR2cVpBVDZTMG1CZndFejVXelZvYnNqcnozbTcweThNQ1VocnFpYVRuTSUyRlhtRmZudTg3ZXdGUmQ2RDh1dkFtc3B6aFhSa3cwNlpkMGdvN2RCR01zSTU1SFVZWHhTR1RDJTJGaUp2d25pT3RBZG1QJTJCcWNjOXk0TGZtWHplTWFmRmtZWHZSNXpqNkpWYkRFazNDWGwxaUplJTJCUXozM0VMZ3NBak5ZeE9lUm5FNzNRallwSk1JTERoazVBUW0lMkZrOW5BTDJTNTN6SzVhJTJCa2RQVEpmQzZXOU1JM1VjOU5lYVpMZXVlJTJCZWxWTEFkTElVRTY4TDEwSFpzbjRxZ0ZUN0ZDRzFiS08wSE4lMkJrZEoyMyUyQmVFejJSWld5YXZYQUxhOU9YTk5GUDMlMkJ0bG51cEFXbTZDaDZKblZhem1JNVFORmF6dUxsbWtrbTJyajlUelU1M0RQdDk1amZSTGhXRFBwVXpWZklrdFZTUyUyRlFUb21vMkxJa3RHOVhNS0RyQzBMWWE3Q0syblJsVk5QTnZPOFlWUG02enZtMEtMZ21VbXR0ZUZvNyUyRkZkdDUlMkJCRWNwcmNqcEh3MmJoUUdTVG1Wc01ZeUpWVXJzYSUyQmV1V3BHNGhXNTdVQXdFNnR0Y2xnTEYlMkZkOUxGJTJGalc5U1NjcWVQY3NXdGYzWTNrVnpXWXMlMkZsTzBGcFdJMmZQdVBoSkslMkIzcGFsVXcxTXhCWmxXNGlvUTduQW1qcmRqb0djSVhDUW9USkpWaFRrdUpMODdUJTJGNFJzcU1OVkdGV1paJTJGdEprNlIxOEdya2oyd2JiOXlhTiUyQnhXRERQcjJTM3h4RWNROTB4VnFRdTZmbzd0ZmZqa1QlMkY2dHlWbER4SVdQczlhZk5CJTJCeVhXaWttZzBmSVFCQml5d3RtWnpZVzNMdk9YWjZWYlhlUXRIS1Z6ZmpSd1pKeEZPYk9MYkFNNmZuZVhRVmhUcUNST0lCNU9Oa2hINlY5UHNpRFRUWjdpNzZmYyUyRjVvM3pFVW9mRVp3V1B1andrTjBNVkw0MDVPRGVCdTlkOUp3ZE1CdWFmWmNzaVZaenlQZllsSW9GWGhiYlA1azlFJTJGaTh6RnI3NUM1RzFnblk3VW1OYTdvbG1aNlBSJTJCanNjaDg1R3BHNUMyNFc4SXM3aUU4dlJ3c3ZVWE5ZTVl3Mzd2TmJydHFYTmczUXF1WnNNY25BVm1VQVdTRlclMkJuN3E1b04lMkZRamQ2eDF2VkVQMzViWFBJcXY1OFdqQlZwNktEJTJGOFdEajlNT1FzM002WHNvMVFVUlNON1UyUnU4UGFKN2Ztc2FSb1VOc2lzQyUyRnl6S1dYdUhaMnJtbGVibGhvbHd5eUNLS0tMR2w2em9WZDMwYk9oWlZnWVZYT3JOUUpCMU9VZXZUZ3Z4SkVDbklhdHJkRGRPWk92WCUyQmRCanZLOVlxY2FjekRUeSUyRjBZamtwS2xjMHpHVWZaMkNPNHhuMUJyUEI4cW4wNHRpJTJCZFhGSWZyYzRGZHR2aGR6UTBMalFRazU3dSUyQmVQMlpRSXBWSTZjNEFldVVnVDhhJTJGM3MyTmJZVVlUZVI2RHh6dTE2QSUyRmhOMU5LcVRTZm4lMkZreXROTDJnNjJrbmN4djNEcHFZRzNQeUlxJTJCODRJJTJCVjlDcENGbmZMNHVXOUtTWUxXN1RJcm9CdHpqSjRDYW91UzJzSU91N2JzdG9Qd25pWDBvVHZOR0xvM2x2dVVwUExBbXBOciUyRllGYWxCc3MyT0ZiN25KWmw3aG1HJTJGTSUyRldFS3FxQnhFZzFtT2NQdHpPMWtGJTJGR25jTFIlMkYzS2lKMlBKd2RPWEJzVThsdVhrbXFLV3AxQ1JjeCUyQkYlMkJHSGZ3VnpDRDJWJTJCcWE1dVNxQlloaXZrTDMzJTJGOUxLUWZPZ0FBeERGdCUyRk0zc2dVZHRhMVAzVWtIdWdkeERyUExkN3QlMkZFcEMwV0N1SDFWMXNSMWR0RlN1Rlg1eWNsY255WEpZaDdYem1oWWFWekVNb0g2SmY2MmdWS0ptaTE3c0Y3Q2VFNE9Za3AxUWJ2ekd6VXVyT29kV3hpbSUyRmlpVzNlVk1TNkhES2k5UDBtaDRZMzZUQk5TaHdCMndJMVNaaDBxUEpjQnZUZVpKMU9xN3VTa00lMkJpeWQwemJRWkZmaEI4TXBRaEgyUEdmY3NwWiUyRjVPczYzODNubTNyQ3lFUVd5bUVqb3p4NXdQTm1TR2JFS1hXOHpwbFNPWktSa2Y0aXd3ZG50eiUyQllmTTdCeXNyTThaVUdQdnZQbWNRQ3QlMkJEN3d0Q0M5Sk5aZm5SWjZNeHRET1YlMkZiNmVvM1pjVFBCVHhwN2llTnJDclVsRVhOWDIlMkZnZUwlMkZlWlJ3N2t3SldvekJaVkpVOVZoTkp4ekdPd3lkckFnZUVENkUlMkI4QkRIcm5wYmYwZ3FaZGpvREs3aE9ONnlXUFFQZ0VsJTJGRzlBdGdXazBDbXBCMFBxalZmajY3QXp3RmZRM082NDM3clpoQVglMkIlMkZCSlVWelBFeDNSWXJmRld1MkYwZ1VwNyUyQlFGc0p3dXZreEtpRUVPN0ZSdTVmQVlyNjJxMTZTMDE0JTJGJTJGVmRYdnNvMFQlMkZuSEdtSExXd3JGMCUyQnJOMjUxWXBqJTJGYjQlMkZJTmJ0T3AlMkY2SkROa3FFVXV5U2V4bkpWVk5MUmxZNjJEaXRRNWFjUFh1UWM4ZlhhczZSdGt6R0lTdFcwaEpKcjZrS3BBQ1pCekZMd21qYmMzMWxUS0xsakpCNVVPYTlGODd4b1Zick1HcGtUWiUyQlJ1ZHJ6cDFMREVrZzljVWx5Qzl2UWlReGxtaktMdExxR3VkR2hUcm5lbjl6TFZFY1UxbjBZc0JSVXlzNGJjT3Z2NDF4eTE2S2k0dmM1YlIlMkYlMkJWcTRHVXJlencxQ0NkJTJGSDZTRlh1ZmhtOGtPUmgySGhlcnhBbjJRJTJGckJ3N2Y2JTJGZjBkazl5SlRpaEZkSTNVcFpXMjBLdlpUalJTNXFIQXhNSXpCM3lCZzZmUGZTbFFRaGQ1YUNYdmRCWGxad1pxQnRWYmhPSE1tTnVnTVo2TjhjVHZ4TG1kS2ZGUFpWM1ladll1M2UlMkJPNGxmV3NITm1GZ2VtdzglMkZDQ2VtV3NYQXN3JTJCT0YwMW1HeUFPT00xN3hIaHhzbW0lMkZDakwyd3V3YVVXSnZRTHlERndYRkh3SENEMnltd1NpVk5PJTJGZTFZJTJCbldvR2JIZlppNHBzaExlUWolMkJSeSUyRlZLQURmOXRuVm96bmZIWHNEamRhNjdHQUt4ZWdkVldGMWxYU2UxR3EyJTJCRzJtUjIzJTJGZDJtM1F2WGwzVWJUT25ycExUWkpZcGI3cm9uclNTQmxQZ3Q2bmdta3ByTWd6ODFKJTJCU0o0dlgyOU83bFJRWDNVMCUyQjNMMFRXMWlGbHN6YjdBeWFIZCUyRnRSNTZHeiUyQjhYbnRHZkhlT1BlbyUyQlhZdU44T3BEQ1YlMkI2OGFiVmZFdiUyRjJHOUx6SXpIa3VLdHM3VFNkdkdObzA1RXJ5VEczTzlqMVd0SDJtY2dRcmc3NXNsUnlUVEFtZUZpV3dqbXRLMmxWbExzUzZnRlRHRTBtemElMkJhaU83S0kydFExallGbnRqSzlrOUhMRXZXRmVGTkQlMkJzeXBnaEIlMkJ4USUyQndGcERKeG94NldiSXZ0dmRpTkkwVUV5dFdQc0hsekJDRE0lMkJSRmJtckpFU0N0cTUwJTJGSXBrYTZNSDFmbEtUNlVQdm4lMkZma3FEWkxzRGdvVjVaamdScHpyak44ZEVLV2JLOXolMkI5ZG9DZjdGNnV3SHklMkZBYVpPbCUyQjQ4Q2tBYzJuJTJCWVRXamNQUmoxRW5Xbm9EdmZjUmNBdkp4czh2bkwyWDJxNGh3OWZxNTlUT3ZEbzdJZTlYa2o3VEo4Q2lKc0JHZ0w2UzVkd0t5cTBjNzlxNGtuR043OGtwbWYwWFZuOVlPYTFhMTVQSDZsR3hUemtpazl6ZGltNDdoV0UxeWZpRmpMakxRNldjTXliUHNLJTJGdCUyQnVDSkxSeXoxOEpjelNxdFQ2TUpmVDBWdkhWalR0RUdGV3FYa1k0ZWtlODV0NFA2RHNHNEx5bVZYOVRuJTJCODNuaGg4WHMlMkZRT29UQjJVNSUyQnJVZ2tqRTRkOFR5SXBjM0xmWVoyZUdZJTJCNzBMaUdlV003RmhQVndwbkpEZW9STjh2NVpJMkRQYnBLcDB5dDhLWDlKRkttdUd3TFZOMmFOaVM0Y3Qxc0dUNjk1bUZpSEZKQUZNdDh4VHBJbSUyRmlCTGNMTVFIcWxiMnQ2JTJCc244NG5GZ3NjMnJlclVycEwlMkZmSE84M3FOZWtCNTk3R0VQS2Q1elMzRktOWCUyRmFHVG01VWJCZiUyRiUyQmIxVm1DbUsyWEROWWl5MnJHeGZha1QlMkJRcnJUbEtTNnM3VDA4cm1oTGRpZmN4M0NldWdGS1lLN1V1dlMxak9hNmZ3Umdwb0JaRURNeEtmZGpUblgwUm1DYzBXZlR3djg4UTNyS3locmh5VHhSRHhna3k0ZWMwbXNnZkR4VDBjbE8xRWVNa0s4ZjV3ZEl3VTJLdXpJTlp1bW5vRlFwVzI5bldaNnlDME5yVjNlWWFLOGFHQlRqY1U0c0NyV0xqajR4SmJ3YXV1NGJBRjl4M3VoMmNQQ29mc1BVazd2c0slMkJrdzdPa3JqMUc5UEhDNjJuaWZsb3lOM0ljd3BTYnN1MUF2dW1vNVY5cUdWRm11U3lUalZuWXFSdHVtd3hXRVZZY2dscXZzWlJYVWdxT1h0NDlXenlVYTRMaiUyRlNjeXJ0NDEwWGJCTk5VVE0xcWJ0dkNEMVpTZWNkVUVYZUtwbVhsQkZmRXRmNSUyRmI0MDhPUUJsbFJUenpYZEVVY1RWWFFWV1cxd095JTJGQk4xRjlma2VZSjFTdlhjYVZ0OWdSNzRGZjRiT3clMkZ4VG03N3VYeWJPcmNwdzJDNnBaZ3ZMNmhwWVoxNDU1SDhRJTJCUmlxOE1sM3BBJTJCY3JKTXQ1JTJCdTRkQ1I2VDFGMHVncXBuclcwN2NHOWdjSkhYN04lMkZMcWZva2toZCUyRkloTlRVaDYlMkZ1SiUyQkpYR2Mxdmc1bm1VbWNqWjdnN1ZYV0ZPbE8lMkJ3anM4UkIzRnBNbThWMWNWbnhuOXJaN2UlMkJTV1gwT2FOVmZCTVQlMkI0aXczVFR2aVFsdU9QNklYNiUyQkNZMGZMbmpBS1V5VU40QkJaT0xJSmV5dWFyanEyY0xHMU5uJTJGbTAlMkZzMSUyQnFxbFBMYWZYdVFRJTJGNjRkSEo5b0RHTXpMUkFMUVlZZTl5bk1IU3hkZm5JdCUyQjNZUHRsa2klMkJueHQ2UXJQUm5ETEZmOGdPRVNieEN5WVZvRkNQSDl0V3hVSklSNmFvJTJCWmJqWFROb3I4NGdnZ2VzYk1NanRUcVo4NFRKVE04c2J6N0Zudkt0RVZQMG9UZVhjc3F4ZTdWM1pvbDFJY0hIN1NIc3BNWGV3dFdUJTJGU3VIa1htZ2pyYTdxZEpkMjlac25WJTJCZnhpSHcyNWMxajRyc29NSHJFaUpweHBrNmtkcVRBY3A4TDdFNDNQMlBMT3pDUG1zTlVidTR3TTVwRkg2V1M5a0IzZUdxU0JwSGhqczNmaEkybmc2RklyTEV3UEZpMWNoSHNmVGlZbjk1UWx6ZkwyWkgySEMlMkIlMkIlMkJsT2VZRUZpU0pBNzNzWE1ZV3lFZmdUQk9tJTJCTmI4Y1o3ZXRXalBIWUdjaTRWWnp1bkNiNTNOSHR5SjV0UWVjckxUbzhvdjA0NTFTb2wxRDkwSjY1S0hHbVFvcUlKREpSOVRIcmdrRW5YSXRhTDlKaWNWdUFIOFpDN29DNE1pQVVyY0JHRDU3aG43NTRJRm9GdzNOemtnWVhHNnlweDZ6MU9tYmElMkZDTkxBMnQ2MjM3OW1wdFYlMkIxNlBscDdtZDB2enpnNXFPbFBqZkpqZ1MzSGVTdDZxRldwaWpjYyUyRlo2NVpSUmlSbllKVUVPR1pHV2xZVXV5ZjI4YkVyQzBsQ1ZlWkQ3azlTeCUyRmxqc3gyRUhUS3NUekhKbmJTbVkxcWxhOW5iUVRQSXM3dUNOdEc3ejA4ckc3JTJGZXpacWZzdzhTeTRvZENQJTJCeDM5Q1d6N0NpRWlhVWZYRG5taVEyMFBhZ1FIZkt2eFJuV1g2ak1pSmViSHRnOERnTzk3Z1V6bkU0c2tYSmxPM3k0RjlRJTJGQnoxcG8wZVZsdEx0a0tvUkVFVWVGbHBDblVBQ2tONWl6RGNJYVB3aWU1WGgxdUFzVnlRSVljMyUyQmNiUEcxOCUyRlh5U3lLanNPVyUyRkJpclFaY3JGSG1hcjdpME00REVHRlljVnZVJTJCM2M4MDNiSDYxUkVoYVlHQmIxaVUwNlVNYmpLU21HZmNOZVpkSG80UHp5Skp6THJ6JTJCSXJDWjhzZTBBJTJGRHYzeUNXd2NicGJ4MjdBWmJXcmk0anlQN2ZYeEpWNXdoYm9DdVdlT3ZXQmVpR1FCbTVjM3dwJTJCQ0pMaDlZV25qSWVYdUFZRUpDU0d2S2hWVXNLRkVndCUyRnh4ekRVWWNObW1RZ0JPNTBQeXp6JTJCYzN4eFpWQWpPQ1JCaUxZWWJ2ZDh0MzZuZnA3OXBab2c1ZnUlMkJHY0VKakUzZ0JZa1QlMkI1T0c0NmRQeXBacTMxUyUyQm1iS29OZlMxWTN6WFNMN3o0ZGJEREllN1lIUU1BMnp6dDdXajNwNUtNaDUxVFh3cEhUYm8lMkZZY2NGcTJxTHQ2VTh1SVNXcEp0dDEwJTJGckMxRGt1OFhVWnowek5ja3RrNUpMOW12S1VDcUVlckhLU0dEJTJGcVM3OHVtMGt6UHBHaDRkQTU5RjV0djNycVEwa3QlMkJDRWxxSnVhYmtndENxRUtHUkhSMGd1WGF3d3JFY3ZxRU8wMGp3eiUyRlpJYUwwRURMdkNoTEtxUVBlbEZsdXo5Tzd6VUdPWlpYbzZrSjdYdWp3JTJCRzNPTjRoekIyRjVTUlowcWdUaFNzNmw1T2pNYjMlMkZJOXhMWSUyRjRUTUo4RiUyQiUyRm9zJTJGOXEycUFFVDN5JTJGdmtTTDVRZjVpck9XaTRlZ2ZlWjV1bFg2eHRJeUl0QkFyazRnSHFvcXBuTzVzeEF2ODd0RU8xJTJGalhsT25IZmpCYVpHSzVKcWRTWkN1YzZEaXhhOExXcVVmbllaNHZjeHZpb1hOTWowUHJUcnNwTFMwWEh0MnVOY0hXNzN3U29UUDZCVlN6UG5EZHNmaHFuMG1WY1c5bE16cDBBN1Y1SFFrdlE4RHdaN0xPc0Y0Wk5EV2ZzTE1iNnkxb2NvcXJmSVVsY1FzdjBRbGlPZGlYc1QzQkhCUnQzYjglMkZTRFltVklWRGwxZ0QzV21lamFuN290VlpFRVMwaTh3eFF6cWFpVk4lMkZHMEdnSXJjYUdQWUolMkYxaXhndUEyeENVZThubXpHdmlCVnA4UVU1ajFZbUt3dDdDZzRoTXFKZVp4azBUZms3d0xpUDdFVktyak9mTVNySjFCUVFyNGFoeEglMkI4TUFPUDQyeGNGTnNBUG1BYkFteE1mREQlMkI1SFlqMmlWZGVpc2dyZ2w0N3lvZGw2Y3U2bDZ1SjNCejBtWm0lMkJDUHMwdmVTTFp5ZVNNbFFzVzhuajg0S3U3NVhOOWIzNXRJTEFRRmpublpNWHdvcXNFZzFESXY0dzliUldTUFdqOUVsdWFCWFNVM2QlMkJZRkJBRWZwbTJiWDdGVEFrZVYlMkYwbUdQRm5xNzQyQ2dWSkhIMnVCSEhSckpZTzA2Z1RYeklIenFLMzI5M2xFZGRScUlEMGphaGFnQTlpdnFtZlE0MEdZbVczZjdaS3phM3pxaWVlRUxTRjVOdFR1Nm9WRmJIc1VucWVYcUE3aElhTHZhNU5yd08xVWZIakQ0aDNKRWhZOXNWdkNlZSUyRm83c2NsSVV4YVAlMkJyYVVYYlBEJTJCN1dLTkR0aFFWaldCJTJGdHN4cVpoandPa0o0c0F1cXR2aEt5ZFh4MDVWeExwejE2OXB3bzFRWTQ1aWwlMkIxWjA2OEVIYXJqVUthbjlqUCUyQjdVMmhmZTdGd1I3YkIlMkJ5eHZhekJ6b1laZWlrJTJCJTJGYmc2NkRCJTJCJTJCJTJGS2VMJTJGd09zcGJ3RzZRNFVzUldORDNZSjloMW5mM3VQTzhqVkdKY01yZUc3JTJCNm9LampZMXNObEhnRjdETlA1RnBrSTZSZnd0a0xYUkFQc3ZaSnF5U2VKcXJKUkJxdkhtbjZtekVDdmdxT0ZXUU5Db0tmSjFUaE1HWFl6U25VZVIlMkJmRGs3R2xJS2pIS0tIdlVsdWJ2ejE4NFNkOTN3WEJvbkcwTGg1dkV2YVdpTXY1Y2w5JTJGZTlXa0R6cEVUb3c4RTBHVkpmODFrSzZ6NHFNU09lRWxKTEFzTXNpaUNycGxHV1ZlZVZYRndUNVZHMTFKV1BzMG1WdHMyN3J1SWdQMnBGa3AzYTdTQ25iZ3d2cnhVQVJFS3dzbVZFMWUlMkJ5NVFnaDRQdCUyQmxTQ2pOU3N4WTlKSTdqNURIVDI3cXJuc0docEpxbkRQYTdRVVVZN3BwOWJqZllYMnJGWlBBbFlNdm5BQlU0Q2RWb1ZlVmpoUkQlMkZnRjcxU3YyczB6VjVBdmtGa1VyeXNDT1FGckJBckNhSHZTR3Z3MzZ4QTQlMkI1Q0o1WjFaSlMlMkZIZ0M1V1RxVlRUVkxBZnIwS0oxME84Q2NtSFhyblMlMkJoTlpWUDV3UjBSZXdVZjBwNTlXQnhCdXdjakd4bFltRUVoRTZDQU1lbmpKJTJGTlpGUmtHZXd5MHgwT3RLWGZ2c0ZBNFFZWGVpWnVKUERNVUIlMkZmSmppY2pUallDazRJSVZsbGhnTkpDcXdOZmRGc0RrVU5yd0Z2N1QyOUlVYURSc1BHek1qVERLeHRnc3pNeiUyRm1YejIwM0ZVUDJEdjRDa0Mzc09LN3h3Tlp2cDVKMEdrcVc1SVdTOXVvQmFpaDhISzUlMkJQZU8zZThCZTd6QmR0UWpIbFlOS2xta1BtVXFUeEF1STBZQWFWdk1nJTJCdGxBVE04Q2FDS2xHNHBlZ25vdDQlMkJzMEdjOUhnVFRFQ3NrSGJHd3NycU9MUUxibDY2blMzeGZRUW9tRGhBeUFmdHB3SlpDJTJGM3BDQTdzSEJTZFFaUGNuSFR4UGpLbjA3RU9XOWNyODFkVUJFYU82c0JXRnR3QVM5cnJrNG9vQnZLWm50QTUwUndXcFRWUE5IRWRyU3hXNEhxVDdSaExaTU5ZJTJGOVNJWE9Vb2drMnV6WHVCbk9yWXolMkZKUXcwdFF0RkR6JTJGaVMzeno3c2E3bWlGdTIlMkZLWjZpbGVJMnVOY0RPdnZPaUV2S2g2MmN4cTVCemk2OTZZSW5aWWFJSGpTemhkZ213TmRod285Q0s1MW01NGI5aEl3T0tGbUJLYmgwM0IzSXZneGtOajVvR0tSZG85JTJCaiUyRlFxanJDcGk0bU5JcUlZd1ZVRlZpejAxVjk1NDhqSGZzd2d6ZFNQTUdkZ1NLc0NOUVhiSDlhYWpEdkpGN1pvTWdpQ1BzOUpsTk0zQ21uOVpGWTN6M1lFODdBUGhEdiUyQjJNQWJWJTJGWXZhYXNYQzkyeE5XTk02N1NWQ0hMUHVWODVNM3B6VW1wd2ppMzkzdVZoSlJ4OU9HcFklMkZSb1BZZWolMkZQUCUyRmtRODdCSzlnNTl6TUFSZnVZSUVqQWpiSEFvbmpuUmdYSU5wZGk3c3JCVlgyQUcwQXpaVU9ZMW9GeHliJTJCbEdSSndWUVpVTyUyQjh5dUwwY0VWY2NQOU91QjNlOTRUNUdsVU10RSUyQmZuUHczUzZCZjJEVUVUZ0IzZEZEZGh2UU51eFVxeXFUZE5obDNPVmxURFd5c1B6VjBiZnlsRDRZbjhYZnZzc25wek9ibHlJYzBWVG84dkFEdzlOJTJGcHdQMGNEcUElMkZnOHlDbmllaHolMkZvRVVZeGVkRkxOQ08zSmQwaiUyQnQ1OGVQVSUyRk8lMkJqTENFWkpYaFJ6d3FNbmZxY010RzFTaHIlMkJZNlhOZzlsN3BNZlRkcW8zSFRObnlYcTAlMkZIJTJCUWQ3ODIlMkZnZWNhd0hNcG5JUUZPSEJLZ0RLM0d4VGlmQWFBczJDdnZDeDdCZmdDRmZrQ0ZtVUhPNU5OY3k0bTJJa3MzcHZ0OHNUdGVEVnNocGM5UFNtOEp5ZkpRRlRkRHc2JTJGZ1c3RHp2TVVhbURWVFEwYjZmQU9vVUtWNXprNkJSRlpPdTZqQVNPcFZaRTBzVXdlQUxucVI0YWx2NHFCc1RqOVBNRjVEUlI0Q2hVRGlXeFY1WWxUT1JmaTB3VHJYZVglMkZmUG93bTQxbWVMZGl4MUpFOG5ZNEIwWVFYUWNFRVNvVnRrOUlHTGxvNGRqeU8lMkI0eVpBc0ZFV3RhSmFNV2R0ZklpZCUyRiUyRjEzNzRxJTJGbm5mQXRKak1FUDlFeVp5OWlBNEl3T1UlMkY1VlBOWGRSJTJGZCUyRjUzWEFCajZNSVU5N3Z6MyUyQk9kZkRnQUpXTWVqa3AlMkYlMkZQdVJaZndBJTJGNzI2ZTNUUDdURmpaNHJIblh0ZWxnRSUyRiUyQnZMZWRYNSUyRmJrRnUlMkZ4ejNrWGY1b0t1NkR4JTJGcDdaWVh4QlZObHBTQ1hZY0M2REhPQmtBWFMxJTJGellGZSUyRlZmZ00zQWElMkI3M1AyZDV5TmhIWWo4b0p0RFUzJTJGTTFsRiUyRmxjWVk1bVFTJTJCVzVaaHl6SFUxUHE0S2U4M1h1eFdYJTJCQ2JMS0Q0ckg3JTJGYyUyRkpHb0dNSllZczFjR04lMkZUdXZBVXNKYVpSdkZjTUw2JTJCbXROQk14SG81N1lqUno4a2Q1VDN1Q2tCSXVnNkwlMkJuaVRUZ05qQWFDVWJuJTJGWldlRFhXdWtuMk91aE1XNzYlMkJ0SzBDcXZPZ1Q5amQlMkZwUGNmSkZHUHY2ZUolMkZHMHFVZlR1SCUyQmxsdjcxQ2x6TlZFY0t2TFYyJTJCZ203SjdkJTJCbVFIb1BZTDdpMlhVbzljJTJGSkpBSVNQVGczNCUyRmxyNnM5SktNS1BEY1hHbWZJTzlHdHJSQjVVcE1zMzNOUWY2UW1WaGZYc2pwdDZZWno1dGFXSUdDOEV6MnR3VSUyQk1mMldFWllUc1g0NGtuJTJGc3c2YmdwOHRDbXFONzFXJTJGc2pPcUMyd3RBOSUyRmVmMmpRVDJVZmYlMkJ5ZlBKZjJiVTJSSHBmSjV3dXgxOE5rbiUyRjdvczJQNmV2JTJGeU80TVp4YzRqODhtclAlMkJjJTJCdEtMN2xIS24lMkIyRWJFbnhzdDlwTiUyQmdhaEk1dm5zVTdCcWolMkZlYXJPJTJGJTJGTUFIMkFFJTJGN24yTzVKbyUyQiUyRnh6emxIUjk4MjA0aSUyRkNVVGRiY1ozU0RQNXlMT21FcjlYYnM4ZmZTUHd4WGFjJTJGNXl5Vnpidkk4WVUlMkJmUlM5a0daZHRZeXZJUmZIZmx4JTJCYlo0SWd1WHpFNzZsYlByJTJCJTJGJTJCMzZ1aTFqViUyRnp6bDJFY2luOTNiUyUyQldyWGolMkZINDg2SXY5OWdOTHpyUmJqczlnV2pGN0Uzd2Y0djBjdSUyRlQxejZwJTJGRG12NXpNQkhGJTJGblB0ZjV4bjlKJTJCTDZkOURvNnAlMkZ0JTJGeWZjNVh3aDc5SEslMkYzejlUOUhPUDMlMkI5ajhPd2pySiUyRndzJTNEJTNDJTJGZGlhZ3JhbSUzRSUzQyUyRm14ZmlsZSUzRfM6rC8AACAASURBVHhe7N0JmGVVfe/93zpFd2MArxETwOEVZRK7TjV5GURxQgEDgUSM4ISoETDqC+YqQ53TKIXadYpBo/A6gYlRcQIj3kAgAooTioBv6DpVIFPExAES9eYKKN1NnfU+p5uWpunus6f/3mvt/eV5eO7z3N57rf//8/9X5dfHotuJfxBAAAEEwhA44aZFeuLibTXmttUabauW/wM5/wfyrcfJ+8dpzC3RwmBrubHF0sJitVqLJG2lgd9Kzo1JriUNWvLOqSWngbyc91JrIPmBvF9Qyz0k6SENBmuksdXyC6s11npQC36VnPud3OB38u63GrjfapHu14K/X79efb8u2GdNGEhUgQACCDRXwDW3dTpHAAEEjASm5rfVgwt/LDf2R9Lgj+RaT5IfPElOT5TXEyX/RMk9QdIT5PUEOT1eWvvv44wqKuLY30n6jbx+I6f/lob/+v+W3K/l9Gt5/Vqu9Uv5wS+l1n/JL/yXth77T00tvb+IyzkDAQQQQEAiuLMFCCCAQFKBqWu31oN//FS11jxNGnuKBv4pcu7Jkn+y5HaStKOkHSRtm/TIBjw3DO73SrpH8r+Q3M/l/c/Vcj+TFn6mwaL/0Nb/+VNNHfhgAyxoEQEEEMglQHDPxcfLCCBQK4Hhj6psv3gXubFnaOCfoZbbWd7vLPmnS/q/Hg7mtWo5oGbukfTvkvuJnLtbA3+3Wu7H8gs/1q9W38WP6gQ0KUpBAIHKBAjuldFzMQIIVCZw+i27aWFhD8nvLu92k9Oukt9VcjtXVhMXjxDwd0vuTnndKefvkNztGhu7Te9/9h3QIYAAAk0RILg3ZdL0iUDTBLx3muyPy7lnr/1X2lMa7Cm5PSQN/6NO/qmHwBrJ3ya1bpV0q7y/Ze2/M+05Oefr0SJdIIAAAusECO5sAgIIxC+w/OanyC+a0GBhQs61JQ3/HZfUir85OsgoMJA0J6kv7/tqjc3KrZnVir1+lvE8XkMAAQQqFyC4Vz4CCkAAgVQCk7PPlBv7v6XBn8i7P5Hze0ka/oeh/INAEoFfyLub5fy/Sq1/lV/4/zQz8W9JXuQZBBBAoGoBgnvVE+B+BBDYvMAp8ztqsd9XXvtqoH3k/N6S/hgyBAoW+E9590O1dJOcbtRqd6POWTr8j2X5BwEEEAhKgOAe1DgoBoGGC0zO7auW319ez5HW/rtrw0VovzqBOyX9QE4/0MBdr5nxG6srhZsRQACBdQIEdzYBAQSqETh55TbaqnWAnDtA8s9b96/7g2qK4VYERgn430rue2v/9f46PTS4Tucue2DUW/w6AgggUKQAwb1ITc5CAIHNC0xd/3it2uaFknuB5F8g6blwIRC5wPcl9x3Jf0dLHvi2pvb/TeT9UD4CCAQuQHAPfECUh0C0AsO/zOiJSw5USy+W14sJ6tFOksKTC3xfTt/UQN/Ur1ddy18alRyOJxFAIJkAwT2ZE08hgEASgeHPqLvBSyX3Enn/Ejk3luQ1nkGgdgLeL8i5b0j+G/Ktr/Mz8rWbMA0hUIkAwb0Sdi5FoCYCwz/1pfXQIXKtg+V0kKQda9IZbSBQtMA98rpGfnC1BltdxZ9aUzQv5yHQDAGCezPmTJcIFCewfOX+8q0/1cC/TM7tX9zBnIRAgwS8v14t9zW5wb9oxbLrG9Q5rSKAQA4BgnsOPF5FoBECJ16xRNs87TBJh8npUElPaUTfNIlAeQI/k9eVkq7QA/9xhc4/bFV5V3MTAgjEJEBwj2la1IpAWQLd/g7yOlxa+++fSVpU1tXcg0DDBdZI+mdJl8vpck237224B+0jgMAGAgR31gEBBNYJTN66s1oP/YW8//O1/3Ep/yCAQAAC/hty7p802Op/aWbPuwMoiBIQQKBCAYJ7hfhcjUDlAqf3d9GCjpT0ckkHVF4PBSCAwJYErpP0VY3pUr2/fRdUCCDQPAGCe/NmTsdNF1h+y9M1eOgvJfcKwnrTl4H+Ixa4TvJfUWurf9SKZ/8k4j4oHQEEUggQ3FNg8SgC0QpM3fYkrVp9lORfyY/BRDtFCkdgMwL+G5L7spYsvkRTe/wSJgQQqK8Awb2+s6WzpgscdfGYdnn20XL+6Id/FKbpIvSPQBMEvirvLtZdt1ysS45eaELD9IhAkwQI7k2aNr02Q6C78qUatF4tp1dJ2q4ZTdMlAghsJHCfvL6k1uCLml72dXQQQKAeAgT3esyRLpou0Ll5d7mx18q710h+96Zz0D8CCGwo4G6X81+QX/i8envdjg0CCMQrQHCPd3ZU3nSBqamWfnfkMWq1XifpkKZz0D8CCCQSuEqDwef0uEsv0tTUINEbPIQAAsEIENyDGQWFIJBQYHJuXzn/ejm9Xl5PSPgWjyGAAAKPCDj9t7w+K+8+q5nxG6FBAIE4BAjuccyJKpsu8ImbFunuxW+U3Bv4Ixybvgz0j0DhAsM/WvLT2nn1P+gt+wz/5lb+QQCBQAUI7oEOhrIQWCvQnd9LfvAmScN/+Q9NWQsEELAUuE/Sp+Ran9L00pstL+JsBBDIJkBwz+bGWwjYCnTnXq3B4M1y7iDbizgdAQQQ2KTA1XLu7zU9/kV8EEAgHAGCezizoJKmC0zN76gHF46Xa71Z8k9vOgf9I4BACALuJ/KDv9PWYxdqauk9IVREDQg0WYDg3uTp03sYAstX7q+BO0Fywx+H4R8EEEAgUAH/KbX8BVqx7PpAC6QsBGovQHCv/YhpMFiBzvwrpcFbJPHjMMEOicIQQGATAtdIrU+ot/TL6CCAQLkCBPdyvbmt6QJT126l1U96q7z7a8k/u+kc9I8AAjELuFvk/Me1+Jcf09SBD8XcCbUjEIsAwT2WSVFn3AKnzO+oRf5t8v5tkraPuxmqRwABBB4l8Cs591GtcR/VOfwcPLuBgKUAwd1Sl7MROLW/p8bc2yX/djAQQACB+gu4j2jBf0Rnt2+tf690iED5AgT38s25sQkCy+f308LgRDkd04R26REBBBB4lIDXRRprna8VS29ABgEEihMguBdnyUkISJP9F6ulk+R1JBwIIIBA4wWcLtVA52mm/c3GWwCAQAECBPcCEDkCAU32D5Hc38j5Q9FAAAEEENhIwLsrJf8hzbSvwgYBBLILENyz2/EmAlJ3/lD5hXdK/A2nrAMCCCAwWsBfIzf2QU0vvXL0szyBAAIbCxDc2QkEsgh0Zw+V17sk99Isr/MOAggg0GwB/3U5fUDTEwT4Zi8C3acUILinBOPxhgt0+wfL62RJhzRcgvYRQACBIgSuktO5mm5fXcRhnIFA3QUI7nWfMP0VI3Da7AvVcqdIOryYAzkFAQQQQGADgcs18OforIlvo4IAApsXILizHQhsSeDU/j4ac6dK/iigEEAAAQSsBdwlWvBn6+z2TdY3cT4CMQoQ3GOcGjXbC3Ru3l1qTUruTfaXcQMCCCCAwKMF/KekwYx6e92ODAIIPCJAcGcbENhQYOq2J+nBVV059z+BQQABBBCoWMD7v9XWS6Y1tccvK66E6xEIQoDgHsQYKCIIgU6/K2n47zZB1EMRCCCAAAJDgQckTavXnoYDgaYLENybvgH0L02uPE5ubLnkd4YDAQQQQCBUAXe3/MIKzSz7ZKgVUhcC1gIEd2thzg9XYO1fnjR4t6TnhlsklSGAAAIIbCTwfbnW+/hLnNiLJgoQ3Js49ab33F05Lj/2Hv6kmKYvAv0jgEDcAu4SuYX3anrZXNx9UD0CyQUI7smteDJ2gZNXbqPFrTPkNfzz2PkHAQQQQKAOAk7naPXgTJ27bPiz8PyDQK0FCO61Hi/N/V6gM/sWyZ0paQdUEEAAAQRqJ3Cv5M9Qb+ITteuMhhDYQIDgzjrUW6A7d6C8zpT8C+rdKN0hgAACCEjuO3I6Q9Pj16KBQB0FCO51nCo9Sd3+DvL+fZI7Hg4EEEAAgaYJ+Avl3Ls13b63aZ3Tb70FCO71nm8zu+v0T5L0fknbNROArhFAAAEEJN0n6XT12uehgUBdBAjudZkkfUid2RdJboWkA+BAAAEEEEDgYYHrJL9cvYlvIYJA7AIE99gnSP3S1Py2Wj2YlteJcCCAAAIIILBJAafztbjV1dTS+xFCIFYBgnusk6PudQLduWPlBz3JPRkSBBBAAAEEtizgfy7X6mh6/DNIIRCjAME9xqlRs9S5eXe5sRl5HQkHAggggAACqQScLpVfmFRvr9tTvcfDCFQsQHCveABcn0Gg0z9Z0lmSWhne5hUEEEAAAQSGAgNJp6nXPhcOBGIRILjHMinqlJbP76eFhXPk3AvhQAABBBBAoBAB77+tsbFTtGLpDYWcxyEIGAoQ3A1xObpAgc7c8C9Rek+BJ3IUAggggAACGwi496o3fgYkCIQsQHAPeTrUJi2ffaEG7gOS9oEDAQQQQAABY4Gb1PLv0oqJbxvfw/EIZBIguGdi46VSBCb703LqlHIXlyCAAAIIILBewKunmXYXEARCEyC4hzYR6pE68y+QBn8raW84EEAAAQQQqEjgh1Lrf6q39DsV3c+1CDxGgODOUoQlwM+yhzUPqkEAAQQaL8DPvjd+BQICILgHNIxGlzI5u7ec+7CkAxrtQPMIIIAAAiEKXCfv36GZiR+GWBw1NUeA4N6cWYfb6bo/l/2ccAukMgQQQAABBNYKnMKf+84mVClAcK9Sv+l3n7zyGVrUOk/S4U2noH8EEEAAgWgELteawUk6d9mPo6mYQmsjQHCvzSgja6Q7d6y8P1/S4yOrnHIRQAABBBD4jZw7UdPjn4ECgTIFCO5lanOXdMJNi7T94o9I7ng4EEAAAQQQiFvAX6hfrX67LthnTdx9UH0sAgT3WCZVhzo7sy+SWh+R/NI6tEMPCCCAAAIISG5eGrxdvYlvoYGAtQDB3VqY89cJdOdOk/czcCCAAAIIIFBLAecmNT1+Vi17o6lgBAjuwYyipoV0+ztI+pi8jqxph7SFAAIIIIDAOgGnSyW9VdPteyFBwEKA4G6hypnrBDr94Z8W83FJT4EEAQQQQACBhgj8TNJfq9e+vCH90maJAgT3ErEbdVW3PyWvMxrVM80igAACCCCwXsDpTE23pwBBoEgBgnuRmpwlnTK/oxYNLpDXEXAggAACCCDQaAGny7SmdYLOWXpPox1ovjABgnthlBykyfk/lRtcKOmpaCCAAAIIIIDAWoGfyreO18zSf8EDgbwCBPe8gry/ToA/NYZNQAABBBBAYPMC/KkzbEcBAgT3AhAbfcTUj7fWqvv+TnKvbbQDzSOAAAIIIDBSwH9eS7Z7s6ae8eDIR3kAgU0IENxZi+wCy/v7aKC/kzSR/RDeRAABBBBAoFECs2rpzVrRvqlRXdNsIQIE90IYG3hId+5Yef/3ksYa2D0tI4AAAgggkEdgQc79labHP5PnEN5tngDBvXkzz99xZ7Ynucn8B3ECAggggAACTRbwM+pNdJosQO/pBAju6bya/fTUHY/X6gf/gb8FtdlrQPcIIIAAAgUKDP+21cVbv1FTu/2mwFM5qqYCBPeaDrbwtiZn95ZrfVrySws/mwMRQAABBBBotICblx+8QTMTP2w0A82PFCC4jyTiAU32XyWn4c/hLUYDAQQQQAABBEwEVsvrWM20v2RyOofWQoDgXosxGjbR6XclrTC8gaMRQAABBBBA4BGB5eq1pwFBYFMCBHf2YvMC3bkL5f1xECGAAAIIIIBAiQLOfVLT48eXeCNXRSJAcI9kUKWW2e3vIK+LJB1U6r1chgACCCCAAALrBa6R0zGabt8LCQLrBQju7MKjBSbn9pX85+S0GzQIIIAAAgggUKGA1x2Se51mxm+ssAquDkiA4B7QMCovpdN/heQ+J/mtK6+FAhBAAAEEEEBAkntQ8q9Tr/0VOBAguLMD6wS6/RPldR4cCCCAAAIIIBCggNNJmm6fH2BllFSiAMG9ROxgr+JvQg12NBSGAAIIIIDAIwL8TatN3waCe9M3oDv3KXn/xqYz0D8CCCCAAAJRCDj3D5oef1MUtVJk4QIE98JJIznw1B9tp9ZDX5Lzh0ZSMWUigAACCCCAwFDAuys12OpVOvtZ9wHSLAGCe7Pmva7bydlnyrmLJe3dxPbpGQEEEEAAgRoI/FDeH62ZiX+rQS+0kFCA4J4QqjaPLZ/fT95fLO+fXpueaAQBBBBAAIEmCjj3Ezl3tFYsvaGJ7TexZ4J7k6Y+Of+ncgtfltw2TWqbXhFAAAEEEKivgH9AfuyVmln6L/Xtkc7WCxDcm7ILk3OvkfOfb0q79IkAAggggECjBLx7rWbGv9ConhvYLMG9CUPv9N8q6aNNaJUeEUAAAQQQaLDA29Rrf6zB/de+dYJ73UfcnTtN3s/UvU36QwABBBBAAIHhX7TqJjU9fhYW9RQguNdzruu66vbfL6/ldW6R3hBAAAEEEEBgIwGnFZpun45L/QQI7vWb6bqOJvsfktM76toefSGAAAIIIIDAFgS8PqyZ9t9gVC8Bgnu95rmum87sBZI7vo6t0RMCCCCAAAIIJBXwF6o3cULSp3kufAGCe/gzSlfhZP+zcjom3Us8jQACCCCAAAK1FPC6SDPt19eytwY2RXCv09A7cxdL/qg6tUQvCCCAAAIIIJBXwF2i3vjReU/h/eoFCO7Vz6CYCrpzX5X3f1HMYZyCAAIIIIAAArUScO5/aXr85bXqqYHNENxjH/pRF49p1z3/SdJhsbdC/QgggAACCCBgKnCF7rz1z3XJ0Qumt3C4mQDB3Yy2hIOn5hdr1eAySYeUcBtXIIAAAggggED8AldpSesITS1dHX8rzeuA4B7rzE+8Y4m2/d3lkjso1haoGwEEEEAAAQSqEPDX6P7HHa7zd1tVxe3cmV2A4J7drro314b2By+XRGivbgrcjAACCCCAQMwC1+j+rQnvkU2Q4B7ZwLT2x2MW/plP2mMbHPUigAACCCAQmoC/RkvG/owfmwltLpuvh+Aez6ykdf8h6hX8THtMQ6NWBBBAAAEEgha4Snfeehj/wWrQM/p9cQT3OOa0rspO/5/502NiGhi1IoAAAgggEIXAFeq1/yyKShteJME9lgXgz2mPZVLUiQACCCCAQHwC/DnvUcyM4B7DmPgbUWOYEjUigAACCCAQuQB/w2roAyS4hz6hyf5n5XRM6GVSHwIIIIAAAgjUQMDrIs20X1+DTmrZAsE95LF2Zi+Q3PEhl0htCCCAAAIIIFA3AX+hehMn1K2rOvRDcA91ipP9D8npHaGWR10IIIAAAgggUGMBrw9rpv03Ne4wytYI7iGOrdt/v7yWh1gaNSGAAAIIIIBAQwScVmi6fXpDuo2iTYJ7aGPqzp0m72dCK4t6EEAAAQQQQKCBAs5Nanr8rAZ2HmTLBPeQxtLpv1XSR0MqiVoQQAABBBBAoPECb1Ov/bHGKwQAQHAPYAhrS5ice42c/3wo5VAHAggggAACCCDwewHvXquZ8S8gUq0Awb1a/3W3T87/qdzgyhBKoQYEEEAAAQQQQGCTAr51qGaW/gs61QkQ3KuzX3fz8vn9NFj4huS2qboU7kcAAQQQQAABBDYv4B9Qa+wlWrH0BpSqESC4V+P+8Cfts89Uq/UNef/0KsvgbgQQQAABBBBAIJGAcz/RYPASzUz8W6LneahQAYJ7oZwpDjv1R9tpbM21kvZO8RaPIoAAAggggAACVQv8UAuLDtTZz7qv6kKadj/BvaqJT85dIecPrep67kUAAQQQQAABBDILeHelZsYPy/w+L2YSILhnYsv5UnfuU/L+jTlP4XUEEEAAAQQQQKA6Aef+QdPjb6qugObdTHAve+ad2Z7kJsu+lvsQQAABBBBAAIHiBfyMehOd4s/lxE0JENzL3Itu/0R5nVfmldyFAAIIIIAAAgiYCjidpOn2+aZ3cPhaAYJ7WYvQ6b9C0j+WdR33IIAAAggggAACJQr8pXrtr5R4XyOvIriXMfbJuX3l9G3Jb13GddyBAAIIIIAAAgiUK+AelNcLNTN+Y7n3Nus2grv1vLv9HTTQd+S0m/VVnI8AAggggAACCFQm4HWHWnqBptv3VlZDzS8muFsPuNO/WtJB1tdwPgIIIIAAAgggEIDANeq1Dw6gjlqWQHC3HGt37kJ5f5zlFZyNAAIIIIAAAggEJeDcJzU9fnxQNdWkGIK71SA7/a6kFVbHcy4CCCCAAAIIIBCwwHL12tMB1xdlaQR3i7FN9l8lpy9aHM2ZCCCAAAIIIIBAFAJer9ZM+0tR1BpJkQT3ogc1Obu3nPuepMVFH815CCCAAAIIIIBARAKr5f3zNDPxw4hqDrpUgnuR45m64/Fatep7kl9a5LGchQACCDRZwE+Py3XnmkxA7whELODmtWTJ8zS1228ibiKY0gnuRY6i2/+KvI4s8kjOQgABBJouMAzuw38I703fBPqPVsDpUk23h38RJf/kFCC45wT8/eud2Z7kJos6jnMQQAABBNYJENzZBATqIOBn1Jvo1KGTKnsguBeh3507Vt5/uoijOAMBBBBA4BGB9aF9+P/DJ+5sBgKRCzj3Bk2PfybyLiotn+Cel395fx8NdL2ksbxH8T4CCCCAwKMFCO5sBAK1ElhQS/trRfumWnVVYjME9zzYUz/eWqvu/4GkiTzH8C4CCCCAwKYFNgzufOrOliBQC4FZLdn2OZp6xoO16KbkJgjuecA7s5+T3GvzHMG7CCCAAALJQjvBnU1BoC4C/vPqTbyuLt2U2QfBPat2d+40eT+T9XXeQwABBBDYssDGn7YT3NkYBGok4NykpsfPqlFHpbRCcM/CPDn/p3KDK7O8yjsIIIAAAskECO7JnHgKgWgFfOtQzSz9l2jrr6Bwgnta9FPmd9RWgxslPTXtqzyPAAIIIJBcYFPBnU/dk/vxJAIRCPxUD7X21TlL74mg1iBKJLinHUO3/0/yOiLtazyPAAIIIJBcYHOhneCe3JAnEYhCwOkyTbf/PIpaAyiS4J5mCN3+lLzOSPMKzyKAAAIIpBcguKc34w0EohVwOlPT7alo6y+xcIJ7UuxO/3BJlyV9nOcQQAABBLILENyz2/EmApEKHKFe+/JIay+tbIJ7Eupufwd5/VDSU5I8zjMIIIAAAtkFthTa15/K36Ka3Zc3EQhU4Gdy2lvT7XsDrS+IsgjuScbQ7X9FXkcmeZRnEEAAAQTyCRDc8/nxNgLRCjhdqun2K6Ktv4TCCe6jkPnz2kcJ8esIIIBAoQIE90I5OQyBuAT48923OC+C+5Z4OrMvktw349p4qkUAAQTiFiC4xz0/qkcgv4B/sXoT38p/Tv1OILhvbqYn3LRI22/9r5JfWr+x0xECCCAQpkCS0L6+cn7OPcwZUhUC+QXcvH714J/ogn3W5D+rXicQ3Dc3z87sBZI7vl7jphsEEEAgbAGCe9jzoToEyhPwF6o3cUJ598VxE8F9U3Pqzh0r7z8dxwipEgEEEKiPAMG9PrOkEwRyCzj3Bk2Pfyb3OTU6gOC+8TBPXvkMLWrdLOnxNZozrSCAAAJRCBDcoxgTRSJQlsBvtGawl85d9uOyLgz9HoL7xhPq9Id/ydLwL1viHwQQQACBEgXShPb1ZfFz7iUOiKsQqEbgcvXaR1RzdXi3Etw3nEmnf7Kkc8IbExUhgAAC9RcguNd/xnSIQEaBU9Rrn5vx3Vq9RnBfP87J2b3l3E21mi7NIIAAAhEJENwjGhalIlC2gPf7aGZi+LfYN/ofgvv68Xf635V0QKO3geYRQACBigSyhPZhqfyoTEUD41oEyhe4Tr3288u/NqwbCe7DeXTmzpT8e8IaDdUggAACzRHIGtwJ783ZETpFQHLvVW/8jCZLENw78y+QBt9u8hLQOwIIIFC1AMG96glwPwKxCLReqN7S78RSbdF1Etw7/eHPte9dNCznIYAAAggkFyC4J7fiSQQaLvBD9dr7NNWg2cF9sj8tp05Th0/fCCCAQAgCeUL7sH5+zj2EKVIDAiUKePU00+6WeGMwVzU3uC+ffaEG7lvBTIJCEEAAgYYK5A3uhPeGLg5tN1ug5V+kFRON+1Hn5gb3Tv9GSY39n1qa/dVO9wggEJIAwT2kaVALAtEI3KRee99oqi2o0GYGd/4UmYLWh2MQQACBfAJFhHY+cc83A95GIF6B5v0pM80L7svn99Ng8IN4l5TKEUAAgfoIENzrM0s6QaASgVbrOVqx9IZK7q7g0uYF98nZb8m5F1ZgzZUIIIAAAhsJFBXc+dSd1UKgoQLef1szEy9qSvfNCu6d/smSzmnKcOkTAQQQCF2A4B76hKgPgSgETlGvfW4UleYssjnBvXPz7tLYrZJaOc14HQEEEECgAIEiQzufuBcwEI5AIF6BgbSwp3p73R5vC8kqb05w7/a/Iq8jk7HwFAIIIICAtQDB3VqY8xFokIDTpZpuv6LuHTcjuHfnjpX3n677MOkPAQQQiEmg6ODOp+4xTZ9aETAQGLg36KzxzxicHMyR9Q/uU/PbatXCbZJ7cjDqFIIAAgggIII7S4AAAsUK+J9rydgemlp6f7HnhnNa/YN7t3+evE4Mh5xKEEAAAQQsQjufuLNXCCAgp/M13T6prhL1Du6d2RdJ7pt1HR59IYAAArEKENxjnRx1IxCDgH+xehPfiqHStDXWPLj3vyvpgLQoPI8AAgggYCtgFdz51N12bpyOQCQC16nXfn4ktaYqs77BvdMf/s8kH06lwcMIIIAAAuYClqGd4G4+Pi5AIBaBd6jXPi+WYpPWWc/g3u3vIK87JG2XFILnEEAAAQTKESC4l+PMLQg0XOA+Oe2m6fa9dXKoZ3DvzF4guePrNCh6QQABBOoiQHCvyyTpA4HQBfyF6k2cEHqVaeqrX3Dvzh0o77+RBoFnEUAAAQTKE7AO7sNOXHeuvIa4CQEEwhVw7iWaHr823ALTVVa/4N6Z+7bkX5COgacRQAABBMoQKCO0E9zLmCR3IBCLgPuOeuMvjKXaUXXWK7h3Zt8iuY+PHOK93wAAIABJREFUappfRwABBBCoRoDgXo07tyLQbAH/1+pNfKIOBvUJ7iev3EaLWndJ2qEOg6EHBBBAoI4CBPc6TpWeEAhe4F6tGeyic5c9EHylIwqsT3Dv9s+W1ymxD4T6EUAAgboKlBXa1/vxc+513ST6QiCDgNM5mm6fmuHNoF6pR3DvrhyXb/WDkqUYBBBAAIFHCRDcWQgEEKhUwA3aml4W9X+5Xo/g3pm7WPJHVboMXI4AAgggsEUBgjsLggAC1Qq4S9QbP7raGvLdHn9w784fKj+4Ih8DbyOAAAIIWAsQ3K2FOR8BBEYKuNZhml565cjnAn0g/uDe6X9P0nMD9aUsBBBAAAFJZYf29ej8nDvrhwACGwl8X73282JViTu4T648Tq51Yaz41I0AAgg0RYDg3pRJ0ycCEQj4wfGaWfbJCCp9TIlxB/fO3I8lv3OM8NSMAAIINEmA4N6kadMrAqELuLvVG39G6FVuqr54g3un35W0IkZ0akYAAQSaJkBwb9rE6ReB4AWWq9eeDr7KjQqMM7hP3fYkrVp9t6RtYgOnXgQQQKBpAlWF9vXO/Jx70zaOfhFIJPCAlizeWVN7/DLR04E8FGdwn5z9oJz7n4EYUgYCCCCAwBYECO6sBwIIBCng/d9qZuKdQda2maLiC+6dm3eXxm6LCZlaEUAAgSYLENybPH16RyB0gYU91Nvr9tCr/P3/ghhLob+vszP795J7U3R1UzACCCDQQIGqQ/uQnB+VaeDi0TICiQX8p9Sb+KvEj1f8YFyfuJ/a30djurFiM65HAAEEEEgoEEJwJ7wnHBaPIdBUgQXtq7PbN8XQflzBvTN3seSPigGWGhFAAAEEqvuLlza251N3thEBBDYv4C5Rb/zoGITiCe6nzb5QLfetGFCpEQEEEEBgnQCfuLMJCCAQhcDAv0hnTXw79FrjCe6d/mWSDg8dlPoQQAABBMIK7cNq+MSdrUQAgRECl6vXPiJ0pTiCe7d/sLyuCh2T+hBAAAEEHhEI5dP29RUR3tlOBBDYooDTIZpuXx2yUhzBvdP/mqRDQoakNgQQQACBRwsQ3NkIBBCITOAq9dovC7nm8IN7d/ZQeXdFyIjUhgACCCAQdmgfVscn7mwpAgiMFHD+ME1PXDnyuYoeCD+4d2avkdxLK/LhWgQQQACBDAKhfdpOcM8wRF5BoJEC/uvqTRwUauthB/fu/KHyAz5tD3V7qAsBBBDYjECIwZ3wzroigEAiAdc6TNNLg/zUPezg3pm9WnLB/q4n0fB5CAEEEGigAMG9gUOnZQRqI+CvUW/i4BDbCTe4T/YPkdPwP0rlHwQQQACBiARCDe184h7RElEqAlULeL1MM+3g/kTDgIP73BVy/tCq58b9CCCAAALpBAju6bx4GgEEAhTw7krNjB8WWmVhBvfJ/ovldG1oWNSDAAIIIDBaIOTgzqfuo+fHEwgg8LCA14GaaX8zJI8wg3u3/xV5HRkSFLUggAACCCQTILgnc+IpBBAIXMDpUk23XxFSleEF9+Xz+2kw+EFISNSCAAIIIJBMIPTQzifuyebIUwgg8LBAq/UcrVh6Qyge4QX3yf5n5XRMKEDUgQACCCCQXIDgntyKJxFAIAIBr4s00359KJWGFdxP7e+pMd0SCg51IIAAAgikE4ghuPOpe7qZ8jQCjRdY0LN1dvvWEBzCCu6duf9X8m8PAYYaEEAAAQTSCcQS2gnu6ebK0wgg4D6i3vj/E4JDOMH9lPkdtdXgFyGgUAMCCCCAQHoBgnt6M95AAIFIBB5q7aRzlt5TdbXhBPfu3Hvl/burBuF+BBBAAIFsAgT3bG68hQACEQg49z5Nj7+n6krDCO5T126lVU8a/i5m+6pBuB8BBBBAIJtATMF92KHrzmVrlLcQQKCJAr/Skl/uqKkDH6qy+TCCe7d/orzOqxKCuxFAAAEEsgvEFtoJ7tlnzZsINFbA6SRNt8+vsv8wgntnbl7yz64SgrsRQAABBLILWAX34afilmdn75g3EUCgeQLuFvXGl1bZd/XBvTP/SmlwSZUI3I0AAgggkE/AMlxbnp2va95GAIHmCbSOUm/pl6vqO4Dg3r9a0kFVAXAvAggggEA+AatgPazK8hP39efn6563EUCgYQLXqNc+uKqeqw3uy1fur0Hr+1U1z70IIIAAAvkFrIP7sEKrO/gPVPPPnxMQaJxAa/BcrVh2fRV9VxvcO7N/L7k3VdE4dyKAAAIIFCNQRqgu445iNDgFAQTqL+A/pd7EX1XRZ3XBfWp+R63iL1yqYujciQACCBQpUEaoLuOOIk04CwEEai6wpLWTpsr/C5mqC+6Ts++Wc++t+VhpDwEEEKi1gFWgHqJt+GMsZd1T62HRHAIIFCfg/Xs0M/G+4g5MdlJ1wb0zd7fkn56sTJ5CAAEEEAhRoMxAbXUXP+ce4mZREwKhC7ifqDe+c9lVVhPcu3OvlvdfKLtZ7kMAAQQQKFagzDBd5l3FKnEaAgjUUsC512h6/Itl9lZNcO/0r5JU2R+lUyYwdyGAAAJ1FigzTJd5V51nRm8IIFCYwNXqtQ8p7LQEB5Uf3Lvze8kP/jVBbTyCAAIIIBCwgFWQHra8qR9fKfu+gOkpDQEEQhFwrT/R9NKbyyqn/ODe6X9Y0kllNcg9CCCAAAI2AlZBeks/c17FnTZ6nIoAAjUROE+99jvK6qXc4P6Jmxbp7iW/krRdWQ1yDwIIIICAjUAVIbqKO230OBUBBGoicJ92XrW93rLPmjL6KTe4d2aPl9wFZTTGHQgggAACdgJWAXpYMZ+4282NkxFAwELAn6DexIUWJ298ZsnBvf9dSQeU0Rh3IIAAAgjYCdQtuI/6DYOdJCcjgEANBK5Tr/38MvooL7hPzu0r528ooynuQAABBBCwFbAK7kn+TPUq77ZV5XQEEIhWwLv9NDN+o3X95QX3Tv88SSdaN8T5CCCAAAL2AlWG5yrvtpflBgQQiFTgfPXa5n/4SjnBfWqqpdV/+St5PSHSYVA2AggggMDDAlbBeXg8n7izZgggEKWA039r8T9ur6mpgWX95QT301Yeq1br05aNcDYCCCCAQDkCVsE9SWgfdmh1f9LfOJSjzC0IIBCdwGDwBp217DOWdZcT3Dv9r0kq9W+WskTjbAQQQKDJAlbBOWlwtwzvaWpo8g7QOwIIbFLgKvXaL7O0sQ/unZt3l8Zus2yCsxFAAAEEyhGwCu1pP+22qoPgXs4ecQsC9RVY2EO9vW636s8+uHf7U/I6w6oBzkUAAQQQKE/AKjAT3MubITchgIChgNOZmm5PWd1gH9w7c7dJfnerBjgXAQRsBNYHND6BtPGN9VSr4J52z6zqSPsbiFjnSN0IIGAl4G5Xb3wPs9OtDl57bnflS+Vb15jeweEIIGAiQHA3YY3+UKvAnDa4DyFDqiX6wdIAAggUJ+AGB2l62deLO/CRk2w/cZ/sXyin4ywK50wEELAVILjb+sZ4ulVQzvopt1U9WX4TEeM8qRkBBIwEvD6pmfbxFqfbBfejLh7Trnv+b0nbWRTOmQggYCewcSAiyNhZx3SyVVAmuMe0BdSKAAIJBO7Tnbf+oS45eiHBs6kesQvuk3OvkfOfT1UNDyOAQBACBPcgxhBcEVbBPetvDK3qyfobieAGRkEIIFCdgHev1cz4F4ouwC64d/qXSnp50QVzHgII2AsQ3O2NY7zBKihnDe5DwxBrinG21IwAAoULfFW99pFFn2oT3Kdue5JWrf6voovlPAQQsBfYVBDKE6zsK+aGMgSsAnLeT7et6mLny9gq7kCg5gJLFv+Rpvb4ZZFd2gT3Tv+tkj5aZKGchQAC5QgQ3Mtxju2WUANyqHXFNl/qRQABE4G3qdf+WJEnGwX32a9L7iVFFspZCCBQjsDmghCfQJbjH+otoQZkq7ry/i8Boc6RuhBAoEwB/w31Jl5a5I3FB/fltzxdg4W7iyySsxBAoByBLYUggns5MwjxltDDsVV97HyI20hNCEQm0BrbWSue/ZOiqi4+uHdm3ym5DxRVIOcggEB5AgT38qxjuskqGBf1qbZVfQT3mLaUWhEIVcC/S72JDxZVnUFw739X0gFFFcg5CCBQngDBvTzrmG4KPRiHXl9Ms6ZWBBAoXOA69drPL+rUYoP76f1dtKA7iyqOcxBAoDyBJOGHTyDLm0dINyXZjSz1FrVPVvUV9b8IZLHhHQQQqJHAmHbV+9t3FdFRscG90z9Z0jlFFMYZCCBQrkCS8FNU0Cq3M27LI5BkL7KeX+Q+WdVZZI1ZnXgPAQSiFzhFvfa5RXRRdHDnx2SKmApnIFCBQJLgQ4ipYDAVX5lkL7KUWPQuxVJnFiveQQCB6AUK+3GZ4oL75K07yz304+hpaQCBBgokDT1Fh60GUkfXctLdSNtY0bsUS51pnXgeAQRqIuC3eoZm9sz9py4WF9y7/XfI60M14aUNBBolkCb0FB24GgUdWbNp9iJta0XvUUy1prXieQQQqIGA099ouv3hvJ0UF9w7/KVLeYfB+whUJZAm9BQduKrqmXtHC6TZi9GnPfoJiz2yqtei1rRePI8AArELFPOXMRUT3Lv9HeR1T+yk1I9AEwXShh1CTHO2JO1uJJWx2qHY6k3qxXMIIFATAacdNd2+N083xQT3Tv/Nkj6ZpxDeRQCBagTShh2r0FVN99y6JYG0u5FU02qHYqs3qRfPIYBAbQSOU6/9d3m6KSq4Xyrp5XkK4V0EEKhGIG3YsQpd1XTPrZsTSLsXaSStdijGmtO48SwCCEQv8FX12kfm6SJ/cD/xiiXa9mn3SVqUpxDeRQCB8gWyBh2r4FW+ADfWKbgPe8m606M2gZ0fJcSvI4BAAoE1uv8/ttP5h61K8OwmH8kf3Cf7R8rpK1kL4D0EEKhOIGvIIcRUN7Oybs66G6Pqs96dWOse5cavI4BATQS8XqGZ9vAnVTL9U0Rwv1BOx2W6nZcQQKBSgawhxzp8VYrC5WsFsu7GKD7r3Ym17lFu/DoCCNREwOuTmmkfn7Wb/MG90/+ppKdkLYD3EECgGoE8Acc6fFUjwq3rBfLsxihF692JufZRdvw6AgjUQuBn6rWfmrWTfMF9+cr9NWh9P+vlvIcAAtUJ5A041gGsOhluzrsbmxMsa2dir58NRACBmgu0Bs/VimXXZ+kyX3Dv9qfkdUaWi3kHAQSqFcgbbsoKYdUqNfP2vLtBcG/m3tA1AggkFHA6U9PtqYRPP+qxfMF9cvb7cm7/LBfzDgIIVCdQRDAjuFc3P8ubi9gNgrvlhDgbAQSiF/D+es1MPDdLH9mD+ynzO2qrwS+yXMo7CCBQrUAR4YzgXu0MrW4vYjfqGtyHfbH3VpvHuQg0TOCh1k46Z+k9abvOHtxPW3msWq1Pp72Q5xFAoHqBosIZIab6WRZdQVG7sXFdZe9KXfooer6chwACgQgMBm/QWcs+k7aa7MF9sv9ZOR2T9kKeRwCBagWKDDRlh7Fq5Zpxe5H7saFY2btSlz6asXV0iUADBbwu0kz79Wk7zx7cO/3hj8nsmPZCnkcAgWoFigw0ZYexauXqf3uRu8En7vXfFzpEAIFcAveo194p7QnZgvvk3L5y/oa0l/E8AghUL1BkOCO4Vz/PIisocjeq/LR9eLdVL8Oz2fsit46zEGiwgHf7aWb8xjQC2YJ7Z3ZScr00F/EsAghUL1B0mCHAVD/TIisoej/W11bVntStnyJnzVkIIBCCgO+oNzGTppKMwb1/laSD01zEswggUL2ARZCpKpRVr1mvCix2g+Berx2hGwQQKFzgavXah6Q5NX1wP+GmRXri4t/JubE0F/EsAghUL2ARzgju1c+1iAosdoPgXsRkOAMBBGor4P2Cfr36cbpgnzVJe0wf3Cf7h8jpa0kv4DkEEAhDwCqYEdzDmG/eKuq4H1Y9Da3Z+7wbx/sIILBWwOtlmmkPf5Il0T/pg3u3Py2vTqLTeQgBBIIRsAoxBJhgRpyrkLruR137yjVsXkYAgXAEnHqabneTFpQ+uHf635OU6a9pTVoUzyGAQPECVgGGTx+Ln1XZJ9Z5N6x64zesZW8p9yFQW4Hvq9d+XtLu0gX3qesfr1Xb/J+kh/McAgiEIWAVXtZ3R4gJY85Zq7Dcj6p3w6q3qvvKOmveQwCBAAWWPPA/NLX/b5JUli64d/qHS7osycE8gwAC4QhYhReCezgzzlOJ1X6EEG6teuN/acqzcbyLAAIbCRyhXvvyJCopg/vcWZI/NcnBPIMAAuEIWIYXAkw4c85aidV+hBDchyZ17y/r3HkPAQRCEXBnqzd+WpJqUgZ3fr49CSrPIBCSgFVo2bjHUEJaSPYx1GK5H6HshFWPofQXw55RIwIIbFEg8c+5Jw/uJ6/cRota9wOPAAJxCViFFoJ7XHuwuWqt9iOkUNuEHuuxjXSBQIMF1gy21bnLHhglkDy48+e3j7Lk1xEIUsAqtBDcgxx36qKs9qMJwX2IHVKfqYfPCwggEI5Awj/PPXlw78ydKfn3hNMhlSCAwCgBq1C2qXsJMKOmEd6vW+5HaPtg1WtofYa3ZVSEAALJBNx71Rs/Y9SzKYJ7/2pJB406kF9HAIFwBKzCCsE9nBnnqcRyP0ILtFa9htZnnn3gXQQQqFTgGvXaB4+qIEVwn31Acn8w6kB+HQEEwhGwCiub65AQE87sk1RitR8h7kGTek0ye55BAIHQBPxv1ZvYZlRVyYL75Ny+cv6GUYfx6wggEI6AVVDZUochBrZwJhJeJVY7EuIeWPU6nGqI/Ya3bVSEAAIjBbzbTzPjN27x/86OPGT4QLd/orzOS/QsDyGAQBAClkGFT9yDGHGuIiz3I9Qga9VzqP3mWhBeRgCB8gWcTtJ0+/z8wb3Tv0jS68rvgBsRQCCrgFVI4RP3rBMJ6z2r/Qg5xDax57C2jmoQQGCEwOfUax9TRHC/Q9KucCOAQBwCVgElSfchB7ck9TflGasdCXn+Tey5KftMnwjUROBO9dq75Qvup8zvqK0Gv6gJCG0g0AgBq4CSBC/k4Jak/iY8Y7kfIc+/qX03YafpEYHaCDzU2knnLL1nc/2M/o9Tl88doYH/p9qA0AgCDRCwDCij+EIObqNqb8qvW+5H6PO36j30vpuy2/SJQPQCLffnWjF+Wfbg3p17r7x/d/QQNIBAQwSsgklSPgJMUqnqnrPakRhm3+Teq9s4bkYAgcQCzr1P0+Ob/QtPR3/iPjl3hZw/NPGFPIgAApUKWAWTNE3FEODS9FO3Z612JIa5N7n3uu0x/SBQSwHvrtTM+GHZP3Hv9O+V9Me1xKEpBGooYBVM0lDFEODS9FOnZy33I4a5N73/Ou0yvSBQU4H/VK+9Q7bgPjn7TDl3V01haAuB2glYhpI0WDEEuDT91OlZyx2JZe5WBrH0X6d9phcEaing/S6amfi3TfW25R+V6cy/UhpcUksUmkKghgJWgSQtFQEmrVh5z1vtSEwzx6C8feMmBBDIItA6Sr2lX84Q3Psrhn9vapYreQcBBMoXsAokaTuJKcSl7S325612JKaZYxD7FlM/ArUXmFavvTx9cOc/TK39ZtBgfQSswkhWoZiCXNYeY3vPckdimjcOsW0u9SLQMIEt/AeqI35Upv9zSTs1jIt2EYhSwDKMZAGJKchl6S/Gd6x2JMZZYxHjBlMzAo0R+IV67Sen+8R9+c1P0WDsp40holEEIhewCiJZWWIMc1l7jeU9qx2JcdZYxLK11IlAQwVaC0/Vir1+tnH3m//EvTt/qPzgioZy0TYCUQlYhZA8CDGGuTz9hv6u5Y7EOGsrjxgtQt9d6kOgkQKudZiml16ZPLhPzp4m52YaiUXTCEQmYBVC8jIQYvIKFve+5Y7EOGc8itstTkIAAQMB7yc1M3FW8uDe6V8k6XUGpXAkAggULGAZQvKUGmOgy9NvyO9a7UjMM8Yk5I2lNgQaL/A59drHpAnuKyVNNJ4NAAQCF7AKH0W0HXOoK6L/kM6w2pOYZ4xJSBtKLQggsJHArHrtZcmCu/dO3bmHJLVgRACBsAWswkcRXccc6oroP5QzLHck5hlbucRsEsrOUgcCCGig6fGt5Jzf0GLT/3HqabNttdwsaAggEL6AVfgoqnNCTFGS2c+x2pHYZ2vlMpxU7DbZt403EUCgMIGBn9BZE/3RwX2y/yo5fbGwizkIAQRMBCyDR1EFE2CKksx+jtWe1GG22GTfK95EAAFjAa9Xa6b9pdHBvTN3puTfY1wOxyOAQE4Bq9CRs6xHvV6HcFekR9lnWe5IHWZr5VMHm7J3lfsQQGBjAfde9cbPSBLcL5b8UQAigEDYAlaho8iuCTBFaqY/y3JH6jBbK5862KTfNt5AAIFiBdwl6o0fnSC4z/YlN17s5ZyGAAJFClgFjiJrHJ5FgClaNN15VntSl7la+bD76faUpxFAYFMCfk69iXaC4N5fLWkRiAggEK6AZeAouuu6hLyiXco4z2pP6jRTjMrYRO5AAIEMAmvUay/ecnA//ZbdtLBwe4bDeQUBBEoUsAobFi3UKeRZ+FidabkjdZqplVOdjKx2lHMRQGCEwNjY7nr/s+9Y/9Rj/zjITv9wSZcBiQAC4QpYBQ2rjgkwVrJbPtdyT+o0UyunOhlVs8HcigACko5Qr335FoL77Dsl9wGoEEAgXAGroGHVMQHGSraa4F63eVp+PdXNqppN5lYEmizg36XexAc3H9wn+x+T0183mYjeEQhdwDJoWPVOgLGS3fy5VntSx1liVf5+ciMCCCQQ8Pq4Ztpv3cIn7v2rJR2U4CgeQQCBCgSsAsawlWEgszq/jmGvgvEnvtJqjuv3JHEhkTxo5cXeR7IAlIlAuALXqNc+eAvBffbHkts53PqpDIFmC1gFDIJ7vfbKak/qGkTxqtf+0w0C9RHwd6s38YxNB/cTblqk7ZcM/yhI/kEAgUAFrAIGwT3QgWcsy2pPCO7pB1JXs/QSvIEAApkEfrVqsS7YZ83a/zv9qANO/9EeWljzo0yH8hICCJgLWIWx9aF9+P+WcYc5VMMvYIbZFsDKjeCebR68hQACDwuMLXqW3v+s2x4b3Lvzh8oPrgAKAQTCFLAKFhsGd8vwToApZ6/K2pNyuinvFis39r68GXITArUUcK3DNL30yscG98m5t8n5j9SyaZpCoAYCVsGC4F6D5digBas9qXsAxa1eXwd0g0BtBLx7u2bGP7qpT9zPlh+cUptGaQSBGglYhYqNQzufuMe/NFa7QnDPvht1t8suw5sIIDBSwLXO0fTSUx8b3DtzF0v+qJEH8AACCJQuYBXGCO6lj9L0wjL3xLSRig638iO4VzRQrkWgFgLuEvXGj95EcO//QNJ+teiRJhComYBVoCgzuG/qrpqNqfJ2rPakKcETv8pXmAIQQOCxAjeo137OpoL7LyTtiBgCCIQlYBUmthSkre5sSgCsaoOYWz55/PL58TYCCJgI3KNee6dHB/epa7fWqif9zuQ6DkUAgVwCVmGC4J5rLMG9XMWeBIeQsyAMcwLyOgII2Ags+eXjNHXgg4/8Oe6T87vKDe6wuY1TEUAgj0AVYcLqTj5xz7MJW37XamZb+g2eXTfVnWzlyO5XN1NuRiB6Ad/aTTNL73wkuHdvPlB+7BvRN0YDCNRMwCpEjApjVd1bs/GV2o7VzJoWOHEsdW25DAEEkgi4hZdoeq9rNwju/WPk9dkk7/IMAgiUJ2AVIkYF9+GvW93dtCBY1rYwr2KkcSzGkVMQQKBAAafXa7p90QY/KjN7mpybKfAKjkIAgQIErEIEwb2A4QR0RJV7EhBDIaVgWQgjhyCAQJEC3k9qZuKsR4J7p/9hSScVeQdnIYBAPoGqA4TV/Xzinm8vNvW21ayS/Aav+G6qP9HKk92vfrZUgECkAuep137HBsF99hLJvTLSZigbgVoKWIWHpGGs6vtrOVSjpqxm1dSgiafRonIsAghkFPBfVm/iqA0/cf+upAMynsZrCCBgIGAVHpIG9+FzVjU0NRAarMnaI5lTsbJ4FuvJaQggkFvgOvXaz98wuN8paZfcx3IAAggUImAVHNKEdgJhIaM0PySUXTFvtMQLMC0Rm6sQQCCJwF3qtXfdMLjfJ2nbJG/yDAII2AuEEhys6uAT9+J2iBkVZ7nhSbjauHIqAghkErhfvfZ264L71Py2WjUYBnf+QQCBQASsQsOwvTSh2aqONDUEMpJgy2BGNqPB1caVUxFAIKPAktbDwX1y9ply7q6Mx/AaAggULGAVGNKG9uHzIdVSMHMtjmM+dmO0suU3rXYz42QEai3g/S7rPnHvzD9HGlxf62ZpDoGIBKwCQ5bgbhneCTD5lzK0XcnfUTgnYBvOLKgEAQSGAq39Hw7u/cMlXQYKAgiEIRBaYLCqh+Cef9+YTX7DLZ2Ar60vpyOAQCqBI9YF9+78G+UHn0r1Kg8jgICJgFVQyPppO5+4m4y5sEOt9oXfVK0bEb6FrSoHIYBAXgHXetP6T9xPlnRO3vN4HwEE8gtYBYUQg3uemvJLx39CiLsSv+qjO7Ay5jdGddsU+kGgFIFTHv7EvT8tr04pV3IJAghsUcAqKOQNyVZ1EWCyf0Ewk+x2Sd+0Ms779Zi0fp5DAIEaCTj11n/i/nFJb6lRa7SCQJQCIYcEq9oI7tlXlZlkt0vzJs5ptHgWAQQMBT7xcHCfvVhyRxlexNEIIJBAwCogFPHpnlVtBPcEi7GJR6zmUcSuZOso3LesrNn9cGdOZQiEKeAvWf+J+1WSDg6zSKpCoDkCVgGhiDAWcm3N2ZBHOmUe5U3dyprgXt4MuQmBmghcvT643yBp35o0RRsIRClgFQ6KCO3rQa1qJMCkX1lmkd4s6xtW1kV+bWbtjfcQQCAqgRvXBffJ/u1y2i2q0ikWgZoJxBAOrGokuKdfZmaR3izPG3jn0eNdBBAoRMDrjvWfuN8jaYdCDuUQBBDIJGAVDIr8VM8/ku4KAAAgAElEQVSqRoJ7upWxmkORu5Kuo/CftjJn98OfPRUiEJDAveuD+28lPS6gwigFgUYJWIWCooOYVZ2El3TrbjWHovclXVdhP21lzu6HPXeqQyAwgd85nXDTIm2/ZHVghVEOAo0SsAoFRQexWOqs+/JYzYEQufnNsTIv+mu07rtPfwg0XcBpcvYP5dyvmw5B/whUKRBTKLCqldCYfAOZQXKrIp/EvUhNzkIAgSwCTsv7T9NA/57lZd5BAIH8AlZhwOqTPKt6Ce7JdsnK32pfknUVx1NW9ux+HPOnSgRCEHA6tb+nxnRLCMVQAwJNFLAKA1ZBzKpewkuy7cc/mZPFU9hbqHImAgikERj+qMzecu6mNC/xLAIIFCdgFQZiC+5W9RY3qTBOstoXfuM0er5W9uz+aHueQACBdQJOnZUvkFrfBgQBBMoXiDUIWNVNeNzyDlq5ExyTf+1bzYDdTz4DnkSgyQJOk/1D5PS1JiPQOwJVCViFAOsgZlU34YXgXtXXYtJ72f2kUjyHAAIWAk6nzf6FWu6rFodzZhwCVv+HKI7u61ulZQi22hnLmuswadyrn2IdZmDVQ/XTqX8FfI+s/4xHdejU7R8try+NepBfr68A38TrN1vrb+6WO2Nde8zTtnLHPPlWWM3A+n8lW9+hZf3JFXkyqwBfq1nl6vOe02krj1Wr9en6tEQnaQX4Rp5WLPzny/jmbrU3ZdQe/gQfW6GVd1mBMUbzzdVsNYsydt+q9jrNN+ReytiRkPuntuF/nDo5d5ycvxCM5grwjbx+sy/jm7vV3pRRe4wTxzucqcU8C6vaw5lOvSvh+2O955ukO6fJlW+Ta30kycM8U08BvpHXa65lfWO32puy6o9t6niHM7GYZ2FVezjTqXclfH+s93yTdDf8Gfd3yOtDSR7mmXoK8I28XnMt6xu71d6UVX9MU7ey5sdksm1BzPOwrD2bJm+lEeD7Yxqtej47DO7vkte59WyPrpII8I08iVI8z5T1jd1yb8rqIZapYh3epKxmYrn7VjWHN536VmS5H/VVq1dnw7859TQ5N1OvtugmjQDfzNNohf1s2d/UrXan7D7CnqqEc3gTinEmVjWHN536VsT3xvrONmlnTp1+V9KKpC/wXP0E+GZen5mW/U3danfK7iP0DcA5vAnFOBOrmsObTn0r4ntjfWebtDOnztzpkn9f0hd4rn4CfDOvz0zL/qZutTtl9xHyBlgZD3vGOfvkY5yLZc3ZJXkzjQBfs2m06vmsU2f2PZI7s57t0VUSAb6ZJ1EK/5kqvqFb7k4V/YQ4ZYxDnMq6mqxmY7H7VrWGO516VmaxG/WUqm9XTpP9M+Q0Vd8W6WyUAN/QRwnF8etVfUO32p+q+glt2viGNpFH6olpNla1hjudelbG98V6zjVNV8PgPiWnM9K8xLP1EuAbej3mWdU3dKv9qaqf0LYB39AmQnAPdyL1r4zvi/Wf8agO+cR9lFADft0qGDSALpgWq/xmbrU/VfYUymCtbIf94Zt/yjHNx7LW/JKckFSAr9ukUvV9jp9xr+9sE3fGN/TEVME+WOU3c8v9qbKvEIZtZdt01yJnG8uMrOos0pKzRgvwtTvaqO5P8KfK1H3CCfrjG3oCpMAfqfqbudUOVd1X1WPHteoJjL4/hhlZ1ThahyeKFmj698SiPWM8jz/HPcapFVwz39QLBi35uBC+kVvtUAi9lTzO319nZTq8oMmuRc/Tak5FzsiqxqItOW+0QJF7Mfo2nghRgL85NcSplFwT39RLBi/4uhC+kVvtUAi9FTyuxMdZmRLcE48g0YMxzMmyxkRIPFSYQJO/JxaGGPlBTt3+u+R1buR9UH4OAb6p58AL4NUQvpFb7VAIvVU1Ykyrkk9/b+izsqovvRRv5BVo8vfEvHZ1eX8Y3N8hrw/VpSH6SC/AN/X0ZqG8Eco3ccsdCqXHsmduZdpUT8v5hTwrq9osPTl78wJ8/bIdTpMr3ybX+ggUzRXgG3u8sw/pm7jVHoXUY1mbYmU5rL+JntZzs5pXEbOyqs3alPM3LVDETmAbt4DT5Nxxcv7CuNug+jwCfGPPo1ftuyF9E7fao5B6LGvaWJYlXcw9VvMq4jdalrUVo8cpaQSa+P0wjU8TnnU6beWxarU+3YRm6REBBOwErAJCE/8PFZZ2e2p1cqgzC7UuqzlwLgJ1Fxj+jPvR8vpS3RulPwQQsBWwCghFfOpo23mxp+NYrGdZp1nNLe9vXEOtq6y5cA8CdRNwOm32L9RyX61bY/SDAALlCxAS8ptbGTbtN0D5J5HuBKu55QnuIdaUTpWnEUBgYwGnyf4hcvoaNAgggEBeAYJCXkEJw/yGVZxgNbc8v+GyqinPbyaqmA13IlAnAafOyhdIrW/XqSl6QQCBagQICvndMcxvWNUJoc0utHqqmgv3IlAngeHfnLq3nLupTk3RCwIIVCNgFRTyfOpYjUS2W/HL5hbKW1bzy/oJd2j1hDIn6kAgZgGnU/t7aky3xNwEtSOAQDgChIXss7Cya8pvfLLLF/Om1fyyBHerWtilYnaFUxDIKuC0vP80DfTvWQ/gPQQQQGBDAavAkCW8xDYZ7GKb2KPrtZpflrBsVUsTvg7j3kKqr7vA8Edl/lDO/brujdIfAgiUI0BgyO6MXXa7UN4MZYah1BHKXKgDgboIOJ1w0yJtv2R1XRqiDwQQqFaAwJDN38oty6e12TrgraGA1RzTftIdSh1sBQIIFCvg1h7X6f9W0uOKPZrTEECgiQJWgaHuAdTKLW3ga+LOFtlzKHMMpY4ibTkLAQT0u/XB/R5JOwCCAAIIFCFAaEiviFl6sxDfsJpjmt+4WtXAbwJD3DhqapjAveuC+2T/djnt1rDmaRcBBIwECA7pYK280oS9dBXz9JYErOaZNDhXfT/bgQACRgJed6z/xP0GSfsaXcOxCCDQMAGCQ7qBW3kR3NPNoainreZJcC9qQpyDQLQCN64P7ldJOjjaNigcAQSCErAKLnUNolZeSYNeUMtTg2KqnmfV99dghLSAQKgCVz8c3GcvltxRoVZJXQggEJ8A4SH5zLBKbhXDk1bzTPIb1yrvjmE21IhA3AL+kvWfuH9c0lvibobqEUAgJAGrAFG3T5GtnJKEvJD2pW61WM111P5XdW/d5kc/CAQq8Il1wb3bn5ZXJ9AiKQsBBCIUIEAkGxpOyZxie6qquVZ1b2zzoV4EohRw6q3/xP1kSedE2QRFI4BAkAJWAaJunyRbOY36ZDbIpalRUVXNtap7azQ6WkEgZIFTHv7Eff6N8oNPhVwptSGAQHwChIgtz8zKp26/uYlv8+3+BtVRs7XaKX4jGOMWUnPtBFzrTes/cT9c0mW1a5CGEECgUgFCBMG90gWs+PKy97/s+yrm5XoEmihwxMPBff450uD6JgrQMwII2AkQJKoJ7nw6arfTaU4ue//Lvi+NBc8igEARAq39H/6bU2efKefuKuJIzkAAAQTWCxAkCO5N/mooe//Lvq/Js6V3BCoR8H6XdcF9an5brRrcV0kRXIoAArUVsAoSQ7DYP1XGprZr//vGyp6x1X2xf63Vf9PosDECS1rbrQvuw386/WFw37YxzdMoAgiUIkCY2DSzlUsdflNTymKWdInVnDcO01b3sE8lLQrXIDBa4H712o8K7ndK2mX0ezyBAAIIJBewChSxfwqIS/IdivnJsuZc1j0xz4LaEYhc4C712rtu+In7dyUdEHlTlI8AAoEJECjK/cQ99t/QBLa+ucspa//Luic3CAcggEBWgevUaz9/g+A+e4nkXpn1NN5DAAEENiVgFSiGd8UaUjFpztdKWbO2uifWr7HmbBidNkfAf1m9iaM2/MT9w5JOag4AnSKAQFkChIpHS+NR1uaFcU8Z8y7jjjA0qQKBxgqcp177HY8E98nZ0+TcTGM5aBwBBMwECBUEd7PliuBg6/23Pj8CYkpEoP4C3k9qZuKsR4J7t3+MvD5b/87pEAEEyhYgWDwibmUxvIEfayh7s5PdZzXz9fO2Pj9ZlzyFAAKmAk6v13T7og2C+80Hyo99w/RSDkcAgUYKWAWLGMMqFs37ErCeudX5/EawebtKxwELuIWXaHqvazf4UZn5XeUGdwRcMqUhgEDEAoSLdcPDIeIlzlG65dwtz87RMq8igECRAr61m2aW3vlIcJ+6dmutetLviryDsxBAAIH1AoQLgnuTvxqs9t/SlE/cLXU5G4GUAkt++ThNHfjgI8F9+H6n/wtJO6Y8iscRQACBkQJWwSWmcGFlMMSPyWHkstTwAcvZW3CxTxaqnIlAZoF71GvvtPZ7/aOO6PR/IGm/zMfyIgIIILAZAavgElPAwKC5Xx5Ws7cSjenrysqAcxEISOAG9drP2URwn7tY8kcFVCilIIBATQQsg0ssIcPKIJb+a7LKmduwmn/mgrbwIjtlocqZCGQVcJeoN370Y4N7d/5s+cEpWY/lPQQQQGBLAlbBJYaQYdX72m/k3TkWLwIByx0oun12qmhRzkMgh4BrnaPppac+NrhPzr1Nzn8kx9G8igACCGxWwCq4xBAyrHonuMfzBWe5A0UqxPD1VGS/nIVA8ALevV0z4x/d1Cfuh8oPrgi+AQpEAIEoBayCSwxBo8m9R7msBkVb7UDRpcbw9VR0z5yHQNACrnWYppde+djgfvqP9tDCmh8FXTzFIYBAtAKWwSX0sGHVe+h9R7usRoVb7UGR5bJTRWpyFgIFCIwtepbe/6zbHhvcT7hpkbZfsrqAKzgCAQQQ2KSAVXAJOWxY9bz2mzg/3x7VV5rlLhQFwU4VJck5CBQk8KtVi3XBPmseG9yH/z+d2R9LbueCruIYBBBA4FECVsEl5LBh1TPBPb4vLstdKEoj5K+lonrkHATiEfB3qzfxjPX1PvrPcV8b3PtXSzoonoaoFAEEYhKwCi4hh40m9hzTTpZZq9UuFNVDyF9HRfXIOQhEJnCNeu2DNx/cJ/sfk9NfR9YU5SKAQCQClsEl1NBh1XOo/UayipWVabUPRTTEThWhyBkIFCjg9XHNtN+6hU/cZ98puQ8UeCVHIYAAAo8SsAouIYYOq16HoCH2y6qPFrDcidG3b/kJdiqvIO8jULSAf5d6Ex/cQnDvHy7psqKv5TwEEEBgvYBVcAkxdDSpVzY8mYDVTiS7neBehBNnIFCiwBHqtS/ffHA//ZbdtLBwe4kFcRUCCDRMwCq4ENwbtkiRtmu1/3k5Qvz6ydsT7yMQvcDY2O56/7Pv2HxwH/5Kpz/8IyEXRd8sDSCAQJACVsEltOBh1edwqKH1GuSiBVyU5W5kbZudyirHewiYCaxRr714w9Mf+6fKrA3us33JjZuVwcEIINBoAcvQElL4aEqfjV7mjM1b7kbGkvjNYFY43kPATMDPqTfRThDc5y6W/FFmdXAwAgg0XsAquDQhuIfUY+MXOSOA1f5nLGfta+xVHj3eRcBCwF2i3vjRSYL7mZJ/j0UJnIkAAggMBayCS0jhowk9ss3ZBKx2I1s1694K6WsnTx+8i0B9BNx71Rs/Y3Rwn+y/Sk5frE/jdIIAAqEJWAWXUMKHVX8ErNA2OXs9ljuStqpQvm7S1s3zCNRawOvVmml/aXRwP222rZabrTUGzSGAQKUClqElhBBi1V8IvVW6ODW63GpHshCxV1nUeAcBY4GBn9BZE/3Rwd17p+7cQ5JaxiVxPAIINFjAKriEEELq3FuDV7bQ1q12JEuRIXzNZKmbdxCoscBA0+NbyTk/OrgPn+j0V0qaqDEIrSGAQMUCVsGl6hBi1ddwXFX3VvHK1Op6yz1JC8VepRXjeQTMBWbVay/b+JZN/3GQ64L7RZJeZ14WFyCAQGMFrIJL1SHEqi+Ce/2+VCx3JalW1V8vSevkOQQaJvA59drHJA/uk7OnybmZhiHRLgIIlChgGVqqDCNWfVXZU4lr0airrHYlDSJ7lUaLZxEoScD7Sc1MnJU8uHfnD5UfXFFSeVyDAAINFbAKLlWGkTr21ND1NG/balfSFF7l10qaOnkWgUYJuNZhml56ZfLgvvzmp2gw9tNGIdEsAgiULmAVXKoKI1b9DAdTVU+lL0WDLrTcl6SM7FVSKZ5DoESB1sJTtWKvnyUP7sMnO/2fS9qpxDK5CgEEGiZgFVyqCiNW/RDc6/uFYbkzSdSq+lpJUhvPINBQgV+o137ypnrf/H+cOnx6cu4KOX9oQ9FoGwEEShCwCi1VhZG69VPCCjT+CqudSQJb1ddJktp4BoHGCnh3pWbGD0sf3Dv9FZK6jYWjcQQQMBewDC1VhBKrfqroxXz4XLBWwGpnkvCyV0mUeAaB0gWm1WsvzxDc518pDS4pvVwuRACBRglYBZeyQ4lVH8NlKLuXRi1gxc1a7s2o1tirUUL8OgJVCLSOUm/pl9MH98nZZ8q5u6oomTsRQKA5AlbBpexQUpc+mrN54XRqtTujOiz7a2RUPfw6AggM/2c4v4tmJv4tfXAfvtHp3yvpj4FEAAEErASsQkvZoaQufVjNmXM3L2C1O1syL/vrg/kjgEAigf9Ur73D5p7c8n+cOnyL/0A1kTIPIYBAdgHL0FJWOKlDD9knyJt5BSz3Z7MBoDuXt2zeRwCBogW28B+mDq8aHdy7c++V9+8uui7OQwABBDYUsAouBHf2LAYBq/3nE/cYpk+NCGwg4Nz7ND3+nuyfuC+fO0ID/0+gIoAAApYCVsEl9uBeVv2Ws+XsZAJWXwN84p7Mn6cQCEKg5f5cK8Yvyx7cT5nfUVsNfhFEMxSBAAK1FbAKLWUF39jrr+1iRdSY1Q4R3CNaAkpF4KHWTjpn6T3Zg/vwzU7/Dkm7ookAAghYCViGFuvwHnPtVvPk3PQClnu0cTXWXxPpu+cNBBCQdKd67d22JDH6Z9zXBfeLJL0OUgQQQMBSwCq4WIeUWOu2nCVnpxew2qNNVWL9NZG+e95AAAFJn1OvfUz+4N7tnyiv8yBFAAEELAWsgot1SIm1bstZcnY2Aatd4hP3bPPgLQRKFXA6SdPt8/MH98m5feX8DaUWz2UIINA4AavQYhncrWoeDt+y7sYtVyQNW+7ThgTsViQLQZnNEvBuP82M35g/uA9P6Mw+ILk/aJYg3SKAQJkCVqHFMqRY1UxwL3PzwrnLcp/Wd2n59RCOJJUgEJuA/616E9uMqjrZz7ivDe79qyUdNOpAfh0BBBDIKmAZWqzCilXNVvVmnQ3vlSNgtU982l7O/LgFgRwC16jXPnjU+ymC+9yZkt/sHwg/6iJ+HQEEEEgiYBVcrIJwbPUmmQHPVCtgtVN84l7tXLkdgS0LuPeqN37GKKXkwX2yf4icvjbqQH4dAQQQyCNgFVosgrtVrUM/i3rzzIV3yxOw3Ct2q7w5chMCqQS8XqaZ9lWj3kke3E9euY0Wte4fdSC/jgACCOQRsAotFkHYqlbCVZ4Niv9dy71it+LfDzqoqcCawbY6d9kDo7pLHtyHJ3X635P03FGH8usIIIBAVgHL0FJ0eLeqteg6s86C96oRsNorQns18+RWBBIIfF+99vMSPKeUwX3uLMmfmuRgnkEAAQSyClgFl6IDcSx1Zp0D71UnwG5VZ8/NCJQv4M5Wb/y0JPemDO79wyVdluRgnkEAAQSyCsQQWqxq5FPRrFtTr/es9qvo37zWS51uEKhM4Aj12pcnuT1dcJ+6/vFatc3/SXIwzyCAAAJZBWIILTHUmNWf96oXYL+qnwEVIFCawJIH/oem9v9NkvvSBffhifycexJXnkEAgRwCVqGlyE+zrWrkE9Eci1OjVy32i92q0YLQSp0EEv98+9r/G5a6825/Wl6d1O/xAgIIIJBCwCK4FBXcrWorqr4UzDwasEDRe0ZwD3jYlNZcAaeeptvdpADpgzt/nntSW55DAIEcAkWHlvWlFBFerGojuOdYmBq+WvSeFbH7NWSmJQSqFUj457f//v+Gpa72hJsW6YmLfyfnxlK/ywsIIIBAQoGiQ0sMwZ1glXA5GvJY0V8D7FdDFoc24xHwfkG/Xv04XbDPmqRFp//EfXhypz/8m50OTnoJzyGAAAJpBYoOLQT3tBPg+aoFiv4aILhXPVHuR+AxAler1z4kjUvG4D47Kblemot4FgEEEEgjUHRo2fDuPAEm1LrS2PJsPAJF7VuenY9Hi0oRiE3Ad9SbmElTdbbgPjm3r5y/Ic1FPIsAAgikFSgqtGx8b54QE2JNaV15Ph6BovYtz87Ho0WlCEQm4N1+mhm/MU3V2YL78IZO/xeSdkxzGc8igAACaQSKCi0E9zTqPBuSQFFfAwT3kKZKLQisFbhHvfZOaS2yB/fJ/mfldEzaC3keAQQQSCpQVGgpKrhb1TOsj2CVdCua9VxRO8d+NWtv6DYCAa+LNNN+fdpKswf301Yeq1br02kv5HkEEEAgqUBRoWVT92UJMqHVk9SR5+IWyLt3WXY9bjGqRyACgcHgDTpr2WfSVpo9uJ8yv6O2Ggx/XIZ/EEAAATOBvKFlc4VlCTMh1WIGzsHBCeTduyy7HhwCBSFQN4GHWjvpnKX3pG0re3Af3jQ5+305t3/aS3keAQQQSCqQN7QQ3JNK81yoAnm/BgjuoU6Wuhor4P31mpl4bpb+8wX3bn9KXmdkuZh3EEAAgSQCeUNLUcHdqo5hfQSrJJvQ3Gfy7h771dzdofNABZzO1HR7Kkt1+YL78pX7a9D6fpaLeQcBBBBIIpA3tGzpjjSBJpQ6kpjxTP0E8uxfmj2vnxwdIRCgQGvwXK1Ydn2WyvIF9+GNnf5PJT0ly+W8gwACCCQRyBNaQg/uhKokG8AzWb8G2C92B4HgBH6mXvupWavKH9wn+xfK6bisBfAeAgggMEoga2gZdW6aUBNCDaP64dfrK5B1/9LseH316AyBgAS8PqmZ9vFZKyoiuB8pp69kLYD3EEAAgVECWUPLqHOThhqr+4f1Ja1hVC/8er0Fsu4g+1XvvaC7CAW8XqGZ9qVZK88f3E+8Yom2fdp9khZlLYL3EEAAgS0JZA0to1SThpqq7x/VB7/eDIEse5h0x5shSJcIVC6wRvf/x3Y6/7BVWSvJH9yHN3f6w985vDxrEbyHAAIIVBHck37inSUwJZkooSqJEs+sF0i7h+wXu4NAcAJfVa99ZJ6qigrub5b0yTyF8C4CCCBQRXgfFW7ShqU0Uxx1d5qzeLb+Aml3kf2q/07QYXQCx6nX/rs8VRcT3Lv9HeSV+m9/ylM47yKAQLME0oaWpDqjwo3VvUk/7U/aB8/VXyDtLo7a7fqL0SECgQk47ajp9r15qiomuA8r6Mx+XXIvyVMM7yKAAAKbE0gbWpJKjgo3Vd2btH6ea5ZAmn0ctdvNkqNbBKoW8N9Qb+KleasoLrh3+++Q14fyFsT7CCCAwKYE0gSWtIJbCjhW9xKq0k6J54cCafaRHWNnEAhIwOlvNN3+cN6Kigvuk7fuLPfQj/MWxPsIIIBAKJ+6pwlJaadGqEorxvNpgjv7xb4gEJiA3+oZmtnz7rxVFRfch5V0+t+VdEDeongfAQQQKPNT982FHKvgTqhiv7MKJN1JdiyrMO8hYCJwnXrt5xdxctHB/WRJ5xRRGGcggAACGwskDS1p5QjuacV4vkqBJF8HBPcqJ8TdCDxG4BT12ucW4VJscD+9v4sWdGcRhXEGAgggUFZwH96zcdBJEo6yTohQlVWO94YCSXaTHWNXEAhIYEy76v3tu4qoqNjgPqyIH5cpYi6cgQACmxFIElqy4BHcs6jxThUCo74GCO1VTIU7EdisQGE/JrP2Q6bCoTuz75TcBwo/lwMRQACBhJ82ZoEqK7gTqrJMh3c2FCC4sw8IxCTg36XexAeLqrj44L78lqdrsJD7v5otqkHOQQCBegmMCi1ZuyW4Z5XjvSoEtvR1wG8Oq5gIdyKwGYHW2M5a8eyfFOVTfHAfVsZfxlTUfDgHAQQ2EigjuFvdMWyFUMVKFyFAcC9CkTMQsBYo5i9d2rBKo+Def6ukj1pzcD4CCDRPoIxQXcYdzZscHRcpQHAvUpOzEDATeJt67Y8VebpNcJ+67Ulatfq/iiyUsxBAAIH1AlbBev2n4dbnM0kE8gpsbkf5X3TyyvI+AgUKLFn8R5ra45cFnmjwH6eur67Tv1TSy4sslrMQQACBoYB1sLY+nykiUITApvaU4F6ELGcgUIjAV9VrH1nISRscYvOJ+/CCybnXyPnPF10w5yGAAAKWwdrq7OHUCFXsbpECBPciNTkLgYIFvHutZsa/UPCphp+4H3XxmHbd839L2q7oojkPAQSaLWAZrq1kCe1Wss09l+De3NnTefAC9+nOW/9Qlxy9UHSldp+4r/3UvX+hnI4rumjOQwABBGIL7wR3drZogY2/BtixooU5D4GMAl6f1Ez7+Ixvb/E12+DeXflS+dY1FoVzJgIINFuA4N7s+dP9OoENvw4I7mwFAoEIuMFBml72dYtqbIP7sOLO3G2S392ieM5EAIHmChDcmzt7On9EgODONiAQmoC7Xb3xPayqsg/u3f6UvM6waoBzEUCgmQIxBXc+CW3mjpbRNcG9DGXuQCCFgNOZmm5PpXgj1aP2wb1z8+7S2G2pquJhBBBAIIFALOGd4J5gmDySSYDgnomNlxAwFFjYQ729bre6wD64Dyvv9L8m6RCrJjgXAQSaKUBwb+bc6frRAsOvA35zyFYgEITAVeq1X2ZZSTnB/bS5Y9Xyn7ZshLMRQKB5AjEEdwJV8/ay7I4J7mWLcx8CmxEYuDforPHPWPqUE9ynplpa/Ze/ktcTLJvhbAQQaJYAwb1Z86ZbBBBAIFgBp//W4n/cXlNTA8saywnuww46/fMknWjZDGcjgECzBAjuzZo33SKAAAIBC5yvXvsk6/rKC+6Tc/vK+RusG+J8BBBolkDo4Z0flWnWPtItAgg0VMC7/TQzfqN19+UF93Wfun9X0gHWTXE+Agg0RyDk4BzSlD0AACAASURBVE5ob84e0ikCCDRa4Dr12s///9u78yDJyjLf47/3FL14WcJQxm7FuSwKDFRmdXtpEGXYZNFuwRkYwQ1R1AaXC+gMDZlZKglSmQnNjArjAq0iixvtgKNt40DLOihLc6Urs2AAZbkjCg4aBsu1F+q8N7JYxm56yco6y/u+51sR9Vef8z7P83kOEb9IsjKzEMg4uI8ulMzFWQxGDQQQKIYAwb0Ye2ZKBBBAwF0Be6KaQ0uy6C/b4H7Ryml6eMbvJW2bxXDUQACB8AUI7uHvmAkRQAABhwWe0k5rXqmT5q3Losdsg3t3omr7i5JSf/N+FnjUQAABNwRcDe+8VcaN54MuEEAAgRQFLlCzfGqK5693dPbBvTY2Vzb+RVYDUgcBBMIXcDG4E9rDf+6YEAEEEJCJ3qDG4N1ZSWQf3LuTVUavkzGHZjUkdRBAIGwBgnvY+2U6BBBAwFGB69QsH55lb/kE91rn3bL2O1kOSi0EEAhXgOAe7m6ZDAEEEHBWwJj3qFH6bpb95RPcuxNWOw9Ldscsh6UWAgiEK+BSeOdtMuE+Z0yGAAIIPCdgHlGztFPWGvkF98roZ2TM2VkPTD0EEAhTgOAe5l6ZCgEEEHBSwNrPqjX0uax7yy+418dma03826wHph4CCIQpQHAPc69MhQACCDgpMCN6teqDj2XdW37BvTtpdfQbkjkh66GphwAC4QkQ3MPbKRMhgAACbgrYS9Qc+lAeveUb3IdX7as4+nkeg1MTAQTCEnAluPP+9rCeK6ZBAAEEXiIQxW/SyJzb8pDJN7hPvOrevk4SHw2Zx/apiUBgAi6Ed4J7YA8V4yCAAALrC6xQs3xYXigOBPexd0rx0rwAqIsAAuEIENzD2SWTIIAAAm4KRMeoOfj9vHrLP7hPvOreGZPsnnkhUBcBBMIQyDu482p7GM8RUyCAAAIbFzD3qFkazFPHjeBea58sqwvyhKA2Agj4L0Bw93+HTIAAAgg4K2B0ihrlC/Psz43gXr9hK63ZvvuROq/ME4PaCCDgv0Ce4Z1X3P1/fpgAAQQQ2ITA7zXjidmqH/xsnkJuBPeuQK1ztqz9TJ4Y1EYAAf8FCO7+75AJEEAAAecEjPmcGqXP5t2XO8F90dhsbcUXMuX9QFAfAd8F8gruvNru+5ND/wgggMBmBJ6NXq3F2X/h0oYduRPcu51VO/8s2U/w4CCAAAL9ChDc+5XjPgQQQACBjQuYL6lZ+t8u6LgV3E9v76EB3eMCDD0ggIC/AnmEd15x9/d5oXMEEEBgswLj2lPnle91Qcmt4N4VqbQvl9FxLuDQAwII+ClAcPdzb3SNAAIIOCdgdYVa5fe70pd7wX14bB/F8e2uANEHAgj4J5B1cOfVdv+eETpGAAEEehKIojdqZPCOnq7N4CL3gnt36Fr7KlkdlcH8lEAAgQAFCO4BLpWREEAAgawFjK5Wo3x01mU3V8/N4F5pHySjG1yCohcEEPBHgODuz67oFAEEEHBWwOpgtco3utSfm8G9K1TpLJex813CohcEEPBHIKvwzttk/Hkm6BQBBBDoWcCaa9QqLej5+owudDi4tw+X0b9l5EAZBBAITIDgHthCGQcBBBDIUsDqrWqVr82yZC+13A3u3e6ro9dJ5tBeBuEaBBBA4M8FCO48DwgggAAC/QnYFWoOHdbfvene5XZwr43Nl42Xp0vA6QggEKIAwT3ErTITAgggkIGAiRaoMXhNBpUmXcLt4P7cq+4rJHPIpCfjBgQQKLxA2uGd97cX/hEDAAEEghOwP1VzyNl3e7gf3Guj82UNr7oH9x8GAyGQvgDBPX1jKiCAAAJBCRi7QI0hJ19t7zq7H9wnXnVvd/9I9fCgHgyGQQCB1AUI7qkTUwABBBAISeBaNctvdXkgP4J7rX2YrJz7y16XF0tvCCAgpRnceZsMTxgCCCAQmIDR4WqUr3N5Kj+C+3Ovuv9I0hEuY9IbAgi4J5BWeCe4u7drOkIAAQSmILBMzfKRU7g/k1v9Ce5njB6gyNyUiQpFEEAgGAGCezCrZBAEEEAgPYHYHqhzh25Or0AyJ/sT3Cdede9cKdljkhmdUxBAoAgCBPcibJkZEUAAgakImKVqlo6dyglZ3etXcD+9PU8DujMrHOoggID/AmkEd94m4/9zwQQIIIDAiwLj2lvnlVf6IOJXcJ941X30G5I5wQdcekQAgfwFCO7574AOEEAAAXcF7CVqDn3I3f7W78zD4H73btLAfb4A0ycCCOQvkHR45xX3/HdKBwgggEAyAuO7qzn3/mTOSv8U/4J716Qy+k8y5lPp81ABAQRCECC4h7BFZkAAAQQSFrD282oN/X3Cp6Z6nJ/BvX7f9lqz9mFJW6eqw+EIIBCEQJLBnVfbg3gkGAIBBBB4RjOm76T67k/4ROFncO8KV9s1SSM+YdMrAgjkI0Bwz8edqggggIDDAsNqlhsO97fR1vwN7hPhvfOQZHfyDZ1+EUAge4GkwjuvuGe/OyoigAACyQqYh9Us7Zzsmdmc5ndwr6z6iEy0JBsqqiCAgM8CSQR3QrvPTwC9I4AAAs8L2HihWnO+5qOH38F94lX39s8kvclHfHpGAIHsBAju2VlTCQEEEHBY4Odqlt/scH+bbc3/4F4bmy8bL/d1AfSNAALZCBDcs3GmCgIIIOC0gIkWqDF4jdM9bqY5/4P7xKvunSsle4yvS6BvBBDIRmCq4Z23ymSzJ6oggAAC6QiYpWqWjk3n7GxODSO411aVZKN2NmRUQQABXwWmEtwJ7b5unb4RQACB5wVMXFZjTsdnjzCCe3cDtfZ5slrk8zLoHQEE0hUguKfry+kIIICAswJGi9Uon+5sfz02Fk5wP23V1poW/UrSrB5n5zIEECiYAMG9YAtnXAQQQOA5gce1Ln6dzp/zjO8g4QT37iaqoydJ5qu+L4X+EUAgHYF+gztvk0lnH5yKAAIIZCNgP6rm0EXZ1Eq3SljBfSK8d26W7P7psnE6Agj4KtBPeCe4+7pt+kYAAQTMLWqWDgjFIbzgXuscLGuvD2VBzIEAAskKENyT9eQ0BBBAwGkBY96iRukGp3ucRHPhBfeJV91HL5bMwkk4cCkCCBREgOBekEUzJgIIICC7RM2hE0OCCDO419qzZPWApG1DWhazIIDA1AUmG9x5m8zUzTkBAQQQyEHgKRntqkb58Rxqp1YyzOA+8ap7+xRJX0xNjoMRQMBbgcmEd4K7t2umcQQQKLbAqWqWLwiNINzg/lx4/3dJ+4W2NOZBAIGpCRDcp+bH3QgggIDjAreqWf5rx3vsq73Ag/vogZK5sS8ZbkIAgWAFCO7BrpbBEEAAAUn2IDWHbgqRIuzg3t1YrX2BrE4OcXnMhAAC/Qn0Gtx5m0x/vtyFAAII5CZgdKEa5e7bpYP8CT+418e20Zrx+yTzmiA3yFAIINCXQC/hneDeFy03IYAAAjkJ2N9oxsDuqg8+nVMDqZcNP7hPvOreOV7WXpq6JgUQQMAbAYK7N6uiUQQQQKA3AWM+oEbpst4u9vOqYgT3ifDevkpWR/m5JrpGAIGkBbYU3Hm1PWlxzkMAAQRSFDC6Wo3y0SlWcOLo4gT36t27SQP3SoqckKcJBBDIVYDgnis/xRFAAIEkBWJpfA81596f5KEunlWc4N7Vr7ZPk7TYxUXQEwIIZCtAcM/Wm2oIIIBAigKL1Cyfn+L5zhxdrODeZa+M3iRjDnBmAzSCAAK5CWwuvPNWmdzWQmEEEECgdwFrb1Zr6MDeb/D7yuIF9+GxfRTHt/u9NrpHAIEkBDYV3AntSehyBgIIIJCBQBS9USODd2RQyYkSxQvuXfZq5yzJftaJDdAEAgjkJkBwz42ewggggEACAuZsNUtnJnCQN0cUM7hPhPf2nZLmebMpGkUAgcQFCO6Jk3IgAgggkJXASjXLe2dVzJU6xQ3uw6MHKDZBfh2uKw8XfSDgg8CG4Z23yfiwNXpEAIHCC0T2QI0M3Vw0h+IG9+6mK+2GjKpFWzrzIoDAfwsQ3HkaEEAAAc8ErJpqlWuedZ1Iu8UO7l3CanulpL0S0eQQBBDwToDg7t3KaBgBBIotcJea5cK+1ZngXh3bX4oL979aiv3fPNMjwCvuPAMIIICAnwLRAWoO3uJn71PvmuA+8ao7nzIz9UeJExDwV+CFV915f7u/O6RzBBAogkDxPkVmw60S3F8Qqbb/XdJ+RXjsmREBBNYXILjzRCCAAALOC9yqZvmvne8y5QYJ7i8AV0b3kjHd97vzgwACBRMguBds4YyLAAL+CVg7T62hu/xrPNmOCe5/7lltnyZpcbLEnIYAAq4LENxd3xD9IYBAwQUWqVk+v+AGE+MT3Dd8CqrtH0k6gocDAQQQQAABBBBAIHeBZWqWj8y9C0caILhvuIjTVu2sadHdkrZzZEe0gQACCCCAAAIIFFHgSa2L5+r8OQ8VcfiNzUxw35hKrXO8rL2UhwQBBBBAAAEEEEAgJwFjPqBG6bKcqjtZluC+qbVURy+WzEInt0ZTCCCAAAIIIIBA0AJ2iZpDJwY9Yh/DEdw3hXbiyml65cxfSHawD1duQQABBBBAAAEEEOhLwIzp96vfoIvnrevr9oBvIrhvbrnV0QMlc2PA+2c0BBBAAAEEEEDAMQF7kJpDNznWlBPtENy3tIZa5wxZ29rSZfw7AggggAACCCCAwBQFjKmoUTp3iqcEezvBvZfV1tpXyeqoXi7lGgQQQAABBBBAAIE+BIyuVqN8dB93FuYWgnsvq661Z8mq+21dO/RyOdcggAACCCCAAAIITErgURntpUb58UndVbCLCe69Lrza7n4pU/fLmfhBAAEEEEAAAQQQSFbgSDXLy5I9MrzTCO6T2WmtXZfVmZO5hWsRQAABBBBAAAEENiNgdJYa5TpGWxYguG/ZaP0rau0fyoqv3p2sG9cjgAACCCCAAAIbChj9SI3yO4DpTYDg3pvTf1+1aGy2torvlPTayd7K9QgggAACCCCAAAIvCvxaz0Z7a/HgY5j0JkBw781p/asqY2+Tia/p51buQQABBBBAAAEEEJBko/lqDf4Ei94FCO69W61/JZ/v3q8c9yGAAAIIIIBA0QX4vPa+ngCCe19sz99UHf2WZN47lSO4FwEEEEAAAQQQKJaA/baaQ+8r1szJTEtwn4pj/aGZWvP07ZKGpnIM9yKAAAIIIIAAAgURGNWMbd6o+s6rCzJvomMS3KfKOdyep1i3SRqY6lHcjwACCCCAAAIIBCwwrkj7aqS8MuAZUx2N4J4Eb61zvKy9NImjOAMBBBBAAAEEEAhSwJgPqFG6LMjZMhqK4J4UdHW0KZlKUsdxDgIIIIAAAgggEI6Abak5VA1nnnwmIbgn6V5rXyWro5I8krMQQAABBBBAAAGvBYyuVqN8tNczONI8wT3JRdQf2E5r1vxMsoNJHstZCCCAAAIIIICAnwJmTDNmvFn1XZ/0s3+3uia4J72PyuheMuZnkqYnfTTnIYAAAggggAACHgmslbVvVmvoLo96drpVgnsa66m03yWj76ZxNGcigAACCCCAAAJeCFi9W63y97zo1ZMmCe5pLararkkaSet4zkUAAQQQQAABBBwWGFaz3HC4Py9bI7inubZaZ4ms/UiaJTgbAQQQQAABBBBwSsCYr6lRWuhUT4E0Q3BPe5HV9nWSDk27DOcjgAACCCCAAAIOCKxQs3yYA30E2QLBPe211tqzFOsWGe2adinORwABBBBAAAEEchOwekCR9lej/HhuPQRemOCexYIrnb1ldLNkZ2ZRjhoIIIAAAggggEC2Ama1rA5Qq3RntnWLVY3gntW+q+3uFw/8S1blqIMAAggggAACCGQo8Hdqlq/KsF4hSxHcs1x7rX2yrC7IsiS1EEAAAQQQQACBVAWMTlGjfGGqNTh8QoDgnvWDUB1tSqaSdVnqIYAAAggggAACyQvYlppD1eTP5cSNCRDc83guap1LZO0H8yhNTQQQQAABBBBAIBEBY76pRumERM7ikJ4ECO49MaVwUaWzXMbOT+FkjkQAAQQQQAABBNIVsOYatUoL0i3C6RsKENzzeiZO/49tNbDuBkl75dUCdRFAAAEEEEAAgT4E7tL4tIN13l891ce93DIFAYL7FPCmfGtldBdF0fWydscpn8UBCCCAAAIIIIBA2gLGPKI4fotaQw+mXYrzXypAcM/7qRge20c2/qmstsm7FeojgAACCCCAAAKbFDB6WiY6RCODd6CUjwDBPR/39atWxt4mE1/jQiv0gAACCCCAAAIIbFTARvPVGvwJOvkJENzzs98gvHfeI2O/7Uo79IEAAggggAACCLwoYM171Sp9B5F8BQju+fqvX73a/pikL7vUEr0ggAACCCCAQOEFPq5m+SuFV3AAgODuwBLWa6HWOUPWtlxri34QQAABBBBAoIACxlTUKJ1bwMmdHJng7uJaau1zZDXsYmv0hAACCCCAAAIFETAaUaP86YJM68WYBHdX11Rpf0FGp7raHn0hgAACCCCAQMACVl9Uq/zJgCf0cjSCu8trq45eLJmFLrdIbwgggAACCCAQmoBdoubQiaFNFcI8BHfXt1hpXy6j41xvk/4QQAABBBBAIAABqyvUKr8/gEmCHIHg7sNaq50rJXuMD63SIwIIIIAAAgj4KmCWqlk61tfui9A3wd2XLdc6P5C1f+NLu/SJAAIIIIAAAh4JGPOvapT+1qOOC9kqwd2ntVfbP5a0wKeW6RUBBBBAAAEEnBdYrmb57c53SYMiuPv0EBxz5YBev8dySYf71Da9IoAAAggggICzAtfql/cu0NJjx53tkMZeFCC4+/Yw1Mema834jyVzqG+t0y8CCCCAAAIIuCRgV2jGwNtVH1zrUlf0smkBgruPT8fJD8zQNquXSSK8+7g/ekYAAQQQQCB/gRV6euYRunDXNfm3Qge9ChDce5Vy7bqJ8P6nZbzy7tpi6AcBBBBAAAHXBewKPf0yQrvra9pIfwR3D5f2YssTb5uJf8R73n1eIr0jgAACCCCQqcC1mhEdydtjMjVPrBjBPTHKnA567g9Wf8inzeTkT1kEEEAAAQT8EViuX977Dv4Q1Z+Fbdgpwd3f3a3fOZ/zHsommQMBBBBAAIHkBfic9uRNcziR4J4Demol+YbV1Gg5GAEEEEAAAX8F+EZUf3e3fucE91A2+cIclfblMjoutLGYBwEEEEAAAQT6ELC6Qq3y+/u4k1scFCC4O7iUKbdUHb1YMgunfA4HIIAAAggggIDHAnaJmkMnejwArW8gQHAP9ZGotL8go1NDHY+5EEAAAQQQQGAzAlZfVKv8SYzCEiC4h7XP9aeptc+R1XDIIzIbAggggAACCLzkZdkRNcqfxiU8AYJ7eDvdILx3zpC1rdDHZD4EEEAAAQQQkGRMRY3SuViEKUBwD3Ov609VbX9M0peLMCozIoAAAgggUGCBj6tZ/kqB5w9+dIJ78Ct+fsBK5z0y9ttFGZc5EUAAAQQQKJSANe9Vq/SdQs1cwGEJ7kVaemXsbYripbLapkhjMysCCCCAAALBChg9rTg6Rq3BnwQ7I4O9KEBwL9rDMDy2j2J7pWR3LNrozIsAAggggEBYAuYRReZYjQzeEdZcTLMpAYJ7EZ+NyuguMuZKSXsVcXxmRgABBBBAIACBu2TtsWoNPRjALIzQowDBvUeo4C47/T+2VfTs92Ts/OBmYyAEEEAAAQRCFrDmGsVbvUvn/dVTIY/JbC8VILgX/amodS6RtR8sOgPzI4AAAggg4IWAMd9Uo3SCF73SZOICBPfEST08sDralEzFw85pGQEEEEAAgQIJ2JaaQ9UCDcyoGwgQ3HkknhOotU+W1QVwIIAAAggggICDAkanqFG+0MHOaClDAYJ7htjOl6q2j5bMtyQ70/leaRABBBBAAIFCCJjVkn2fmuWrCjEuQ25WgODOA7K+QKWzt2S/JaNdoUEAAQQQQACBHAWsHpDM+9Qq3ZljF5R2SIDg7tAynGml1p4lqyskHepMTzSCAAIIIIBAsQRWyOg4NcqPF2tspt2cAMGd52PTArXOEln7EYgQQAABBBBAIEMBY76mRmlhhhUp5YkAwd2TReXWZrVdkzSSW30KI4AAAgggUCyBYTXLjWKNzLS9ChDce5Uq8nWV9rtkdJmk6UVmYHYEEEAAAQRSFFgrq+PVKn8vxRoc7bkAwd3zBWbWfmV0L5noUskOZlaTQggggAACCBRCwIzJxh9Qa+iuQozLkH0LENz7pivgjfUHttPa1d+U1VEFnJ6REUAAAQQQSF7A6GpNn/lB1Xd9MvnDOTE0AYJ7aBvNYh6+aTULZWoggAACCAQvwDehBr/ihAckuCcMWpjjap3jZe03JA0UZmYGRQABBBBAIBmBcRnzITVK3b8f4weBngUI7j1TceFLBIbb8xTr65KG0EEAAQQQQACBngRGFenDGimv7OlqLkLgzwQI7jwOUxOoPzRTa576umTeO7WDuBsBBBBAAIHQBey3NWPbD6u+8+rQJ2W+dAQI7um4Fu/UWucMWdsq3uBMjAACCCCAQA8CxlTUKJ3bw5VcgsAmBQjuPBzJCVTG3iYTL5H02uQO5SQEEEAAAQS8Fvi1bLRQrcGfeD0FzTshQHB3Yg0BNbFobLamxRfL6siApmIUBBBAAAEEJi9g9COti07U4sHHJn8zdyDwUgGCO09FOgK1dl1WZ6ZzOKcigAACCCDguIDRWWqU6453SXueCRDcPVuYV+1W20dI+qqkHbzqm2YRQAABBBDoX+BRSR9Vs7ys/yO4E4GNCxDceTLSFai1Z0n6Ct+2mi4zpyOAAAIIOCDQ/RZU6WNqlB93oBtaCFCA4B7gUp0ciU+dcXItNIUAAgggkJAAnxqTECTHbE6A4M7zkZ1AdfRAKfqSZAezK0olBBBAAAEE0hQwY1L8CTWHbkqzCmcj0BUguPMcZCtw4sppeuX0L0lmYbaFqYYAAggggEDSAnaJfr/2E7p43rqkT+Y8BDYmQHDnuchHoNY5XtZeKGm7fBqgKgIIIIAAAn0LPCljTlajdFnfJ3AjAn0IENz7QOOWhAROW7WzpkUXSOp++gw/CCCAAAII+CCwTOviU3T+nId8aJYewxIguIe1Tz+nqbZPk7TYz+bpGgEEEECgQAKL1CyfX6B5GdUxAYK7YwspbDuV0b1kzBcl7VdYAwZHAAEEEHBV4FZZe6paQ3e52iB9FUOA4F6MPfszZbVzlmQ/60/DdIoAAgggELaAOVvNEt8EHvaSvZmO4O7NqgrUaHVsfyn+vKS9CjQ1oyKAAAIIuCVwlxR9Ss3BW9xqi26KLEBwL/L2XZ+90m7IqOp6m/SHAAIIIBCYgFVTrXItsKkYJwABgnsASwx6hOHRAxSbf5Q0L+g5GQ4BBBBAwAWBlYrsP2hk6GYXmqEHBDYUILjzTPghwHvf/dgTXSKAAALeCvBedm9XV6DGCe4FWrb3ow6P7aPx8cUy5gDvZ2EABBBAAAE3BKy9WQMDizQyeIcbDdEFApsWILjzdPgn8Nznvp8rKfKveTpGAAEEEHBEIJZ0Bp/L7sg2aKMnAYJ7T0xc5JxA9e7dZAZasjrKud5oCAEEEEDAbQGjq2XHK2rOvd/tRukOgfUFCO48EX4L1DrHy8ZNybzG70HoHgEEEEAgfQH7G5moqkbpsvRrUQGB5AUI7smbcmLWAvWxbbQ2bsjq5KxLUw8BBBBAwBMBows1PaqpPvi0Jx3TJgIvESC481CEI1AdPVAyI5L2C2coJkEAAQQQmKLArZIdVnPopimew+0I5C5AcM99BTSQuEC1fYqkcyRtm/jZHIgAAggg4IvAU5I+rWb5Al8apk8EtiRAcN+SEP/up0CtPUvWfk4yC/0cgK4RQAABBPoXsEtkzGfUKD/e/xnciYB7AgR393ZCR0kK1DoHy+osye6f5LGchQACCCDgooC5RUZnqlG6wcXu6AmBqQoQ3KcqyP1+CFRHT5LMWZJm+dEwXSKAAAIITELgccmeqebQRZO4h0sR8E6A4O7dymi4b4HTVm2t6dGZslrU9xnciAACCCDgloDRYq2Nz9L5c55xqzG6QSB5AYJ78qac6LpAbVVJduCzkj3G9VbpDwEEEEBgUwJmqcz42WrM6WCEQFEECO5F2TRzvlSgNjZfNv6MpDfBgwACCCDgjcDPZaLPqTF4jTcd0ygCCQkQ3BOC5BiPBSqrPiIzMCzZnTyegtYRQACBwAXMw7LjI2rN+VrggzIeApsUILjzcCDwgkC1XZPU/d0aFAQQQAABZwS6711vqFluONMRjSCQkwDBPSd4yjoqUL9ve61eU5Mxn3K0Q9pCAAEEiiNg7ec1c0ZD9d2fKM7QTIrApgUI7jwdCGxMoHr3blJUkcwJACGAAAIIZC1gL5Hilppz78+6MvUQcFmA4O7ydugtf4HT2/M0YE7nE2jyXwUdIIBAEQTMUo3b83ReeWURpmVGBCYrQHCfrBjXF1PgjNEDFJnu578fUUwApkYAAQRSFVim2C7WuUM3p1qFwxHwXIDg7vkCaT9jgVr7MFmdJunwjCtTDgEEEAhR4FoZna9G+boQh2MmBJIWILgnLcp5xRCojc6X1T9I5pBiDMyUCCCAQJIC9qcy+kc1hvgs9iRZOSt4AYJ78CtmwFQFJr7EafzvJXNoqnU4HAEEEAhCwK6QGfgnvjwpiGUyRA4CBPcc0CkZoEClfbhkPilj5wc4HSMhgAACUxOw5hrJfkGt8rVTO4i7ESi2AMG92Ptn+qQFKu2DFOkUWR2V9NGchwACCHgnYHS1Yl2gVvlG73qnYQQcFCC4O7gUWgpAYHhsH43HJ8vouACmYQQEEEBgcgJWV2ggulAjg3dM7kauRgCBzQkQ3Hk+EEhT4PT2Hhown5DsJ9Isw9kIIICAGwLmSxq3X9J55Xvd6IcuEAhLgOAe1j6ZxlWBRWOzNc1+XNZ+XNIrXW2Thxgy/AAAB1pJREFUvhBAAIE+BH4vY76sdebLWjz4WB/3cwsCCPQoQHDvEYrLEEhEoH7DVlq7/cdkzUclu2ciZ3IIAgggkIuAuUfGflXTn/iK6gc/m0sLFEWgYAIE94ItnHEdEqiOvVOKT5LER0k6tBZaQQCBLQqskKKL1Bz8/hav5AIEEEhUgOCeKCeHIdCHwPCqfRWbEyVzQh93cwsCCCCQkYC9RJG9WCNzbsuoIGUQQGADAYI7jwQCrgjUx2Zr9fhCmejDkt3RlbboAwEEiixgHpGNv66ZA0tU5/3rRX4SmN0NAYK7G3ugCwTWF6h13q04/rAM38jKo4EAAjkIWLtCUfR1NUrfzaE6JRFAYBMCBHceDQRcFqiNzZWNu2+h6f5u63Kr9IYAAt4LPCXpEpnoEjUG7/Z+GgZAIEABgnuAS2WkAAUuWjlND0//oGQ+IGm/ACdkJAQQyE/gVsleqp3WflMnzVuXXxtURgCBLQkQ3LckxL8j4JpApbO3jH2/jN4vq5e71h79IICABwJGf5TV5bLmcrVKd3rQMS0igIAkgjuPAQK+CtTrkf501HGKovdJOtzXMegbAQQyFbhWcfwtvezqK1Svx5lWphgCCExZgOA+ZUIOQMABgerdu8kMvFfWvEeyuznQES0ggIAzAuZ+Gfsd2fFvqzn3fmfaohEEEJi0AMF90mTcgIDjArVVhyiO3i2jd/EHrY7vivYQSE/gKVl9T1H8XTXm/DS9MpyMAAJZChDcs9SmFgJZChxz5YBet+exMvZYSX+bZWlqIYBAbgI/kDVX6lf3XKmlx47n1gWFEUAgFQGCeyqsHIqAYwL1+7bXmrXHSPadknmLY93RDgIITEnAXi+Z72vG9KWq7/7ElI7iZgQQcFqA4O70emgOgRQEhu/ZUfGzfyeZo/loyRR8ORKBbAS6H+F4laKt/kUjez6STUmqIIBA3gIE97w3QH0E8hT4dPt1GtdRz7+Vhs+Hz3MX1EZgywK3SvqBBnS1zin/asuXcwUCCIQmQHAPbaPMg0C/ApV7d1L07N/I2nfwdpp+EbkPgaQF7PUy5oeKt/pXtfZ4OOnTOQ8BBPwSILj7tS+6RSAbgVp7lqyOkCZ+3y5pWjaFqYJA4QW631z6Y0nLZLRMjfLjhRcBAAEEXhQguPMwIIDA5gVOXj5DW//lAkkLZDRf0g6QIYBAogKPyuoaScv1zH8u14UL1iR6OochgEAwAgT3YFbJIAhkJDC8al/Z6G2K7VtlzL4ZVaUMAmEJWHubIvNvMvFPNDLntrCGYxoEEEhLgOCeliznIlAEgUVjsxU9e7hMdJiMDpU0uwhjMyMCfQg8JqsVsvF1ire6VosHH+vjDG5BAIGCCxDcC/4AMD4CiQpUOnvLxIdM/HGrtW+RMQOJns9hCPgiYO24jLlestfLRj9Vq3SnL63TJwIIuCtAcHd3N3SGgN8CJ66cplfMOFiRDpLVQZLe5PdAdI/AFgV+LqMbFetG/WHNDbp4XvcPTflBAAEEEhMguCdGyUEIILBZgfpt22nN1gdIZn/J7k+Q53kJQODnkrlFsrdoxjM3q77vkwHMxAgIIOCwAMHd4eXQGgJBC5y2amttFe0nY/aT7Juf+zX/I+iZGc5jAfv/JPOziV9rb9Wz8a06f84zHg9E6wgg4KEAwd3DpdEyAsEKdN8jH9l9ZfVGaeL39cHOymCuC/xS0u0yul2xuY33qLu+LvpDoBgCBPdi7JkpEfBToPupNdPt3rLaW7Hmydi9JL3Kz2Ho2mGB38mauxRppYzu1FpzJ5/64vC2aA2BAgsQ3Au8fEZHwEuByuguMgP/S4rfIGveIGPnSnq1l7PQdB4Cv5U1d8vYX0jRL2TH/49aQw/m0Qg1EUAAgckKENwnK8b1CCDgnsDw3TvIThtSPD4kY8qSur8lSZF7zdJRRgKxpI6ktqxtKxoYlVk3qpG5j2ZUnzIIIIBA4gIE98RJORABBJwQsNao0i7JmD0nfqU9pHgPyewuaZoTPdJEEgLrJHufFN0r6V5Ze8/Eb6vckTE2iQKcgQACCLgiQHB3ZRP0gQAC2Ql8+p5dNT6+u2R3kzW7ynT/CNa+XjI7ZdcElSYnYB+WzC9l9UsZ+4Bk7tfAwH06Z88HJncOVyOAAAL+ChDc/d0dnSOAQNIC3S+NetU2uyge30Wx3VmR2UnW7iTZHSX9T0mzky7JeS8KPCbp/0rmERnzsGL7sCLzkKKBB/W7px/ky4x4UhBAAAGJ4M5TgAACCPQqUL9hpla/6rWK1v2lNLCDYruDjHmNZF8jme4fyHaD/SxJ2/R6ZAGue1rS45Iek+xvJfMbWfsbReZRafxRxdP+UzN/92vVD15dAAtGRAABBKYkQHCfEh83I4AAAhsRqI9to9Xjr5IZ+Asp/guZaHvZeHsZvUJWr5DsKyTzckkvl9XLZbSdNPH7Moc9/yTpSVk9KaM/St1f+0fJ/EFGf5DVH2SiJ2TjJ6Tov2TH/0szB36n+mA3uPODAAIIIJCAwP8H39XnliQxzSAAAAAASUVORK5CYII='

# make it pretty:
# sg.theme_previewer()
sg.theme('LightGrey1')

treatmentLayout = [
                [sg.Text()],
                [sg.Text('File:'), sg.Push(), sg.Input(key="-TREATMENT_FILE-", do_not_clear=True, size=(50,3)), sg.FileBrowse()],
                [sg.Text()],
                [sg.Text('Voltage Limit:'), sg.Push(), sg.Input(key="-TREATMENT_VOLT-", do_not_clear=True, size=(50,3))],
                [sg.Text()],
                [sg.Button('Plot Input File',key='-T_PLOT-')],
                [sg.Text()],
                [sg.Text('', key="-ERROR_TREATMENT-", size=(70,2))],
                [sg.Text()],
                [sg.Button('Analyze Voltage Ramp', key="-TREATMENT_RAMP-"), sg.Button('Calculate THD', key="-TREATMENT_THD-"), sg.Button('Average Peaks', key="-TREATMENT_Pk_AVG-"), sg.Push(), sg.Button('', image_data=help_button_base64, button_color=(sg.theme_background_color(),sg.theme_background_color()), border_width=0, key='-TREAT_INFO-')]
                ]

placementLayout = [
                [sg.Text()],
                [sg.Text('File:'), sg.Push(), sg.Input(key="-PLACEMENT_FILE-", do_not_clear=True, size=(50,3)), sg.FileBrowse()],
                [sg.Text()],
                [sg.Text('Voltage Limit:'), sg.Push(), sg.Input(key="-PLACEMENT_VOLT-", do_not_clear=True, size=(50,3))],
                [sg.Text()],
                [sg.Button('Plot Input File',key='-P_PLOT-')],
                [sg.Text()],
                [sg.Text('', key="-ERROR_PLACEMENT-", size=(70,2))],
                [sg.Text()],
                [sg.Button('Analyze Bipolar Pulse', key="-PLACEMENT_PULSE-"), sg.Button('Analyze Tone Sync', key="-PLACEMENT_TONE-"), sg.Push(), sg.Button('', image_data=help_button_base64, button_color=(sg.theme_background_color(),sg.theme_background_color()), border_width=0, key='-PLACE_INFO-')]
                ]

layout_win1 = [
                [sg.TabGroup([[sg.Tab("Treatment",treatmentLayout), sg.Tab("Placement", placementLayout)]])],
                [sg.Push(), sg.Button('Exit', button_color='red')]
                ]

win1 = sg.Window(title='Guinness Waveform Analyzer (ST-0001-066-101A)', icon=toolIcon, layout=layout_win1)
win2_active = False
win3_active = False

info_txt_width = 138
info_txt_size = 9


# ERROR MESSAGES
incompatible_contents = "Error:  Something went wrong. The contents of the input file are incompatible with the requested operation. Review contents of the input waveform .csv and the selected analysis option."
unexpected_contents = "Error:  Input file contains unexpected contents. File must contain only a column of timestamps and a column of corresponding measured voltage."
invalid_path_or_type = "Error:  Invalid filepath or filetype. Input must be a .csv file."
invalid_treatment_V = "Error:  Not a valid voltage limit input. Value must be an integer in the range from 0 to 150."
invalid_treatment_V_and_File = "Error:  Not a valid voltage limit input. Value must be an integer in the range from 0 to 150.\n\nError:  Input file contains unexpected contents. File must contain only a column of timestamps and a column of corresponding measured voltage."
invalid_placement_V = "Error:  Not a valid voltage limit input. Value must be an integer in the range from 0 to 5."
invalid_placement_V_and_File = "Error:  Not a valid voltage limit input. Value must be an integer in the range from 0 to 5.\n\nError:  Input file contains unexpected contents."
bothInvalid = "Error:  Invalid file and voltage limit."
bothEmpty = "Error:  Both the filepath and voltage limit must be entered."
needFileToPlot = "Error:  A filepath must be entered to plot."


while True:
    try:
        event, value = win1.read(timeout=2000)
        
        # When 2s elapse, reset the output window to blank. This sets a "timer" on any error text that is displayed there
        if sg.TIMEOUT_EVENT:
            win1["-ERROR_TREATMENT-"].update("")
            win1["-ERROR_PLACEMENT-"].update("")
        
        # Close application if the window is closed or if the "Exit" button is pressed
        if event == sg.WIN_CLOSED or event == 'Exit':
            break

        if event == '-T_PLOT-':
            if value["-TREATMENT_FILE-"] != '':
                
                fileGood = CheckFile(value["-TREATMENT_FILE-"])

                if fileGood == True:
                    csvStatus, columns = CheckPlainPlotCSV(value["-TREATMENT_FILE-"])

                    if csvStatus == True:
                        try:
                            plotContents(value["-TREATMENT_FILE-"], columns)
                        except ValueError:
                            value["-ERROR_TREATMENT-"] = incompatible_contents
                            win1["-ERROR_TREATMENT-"].update(value["-ERROR_TREATMENT-"])

                        except IndexError:
                            value["-ERROR_TREATMENT-"] = incompatible_contents
                            win1["-ERROR_TREATMENT-"].update(value["-ERROR_TREATMENT-"])
                            
                        except TypeError:
                            value["-ERROR_TREATMENT-"] = incompatible_contents
                            win1["-ERROR_TREATMENT-"].update(value["-ERROR_TREATMENT-"])
                    else:
                        value["-ERROR_TREATMENT-"] = unexpected_contents
                        win1["-ERROR_TREATMENT-"].update(value["-ERROR_TREATMENT-"])

                elif fileGood == False:
                    value["-ERROR_TREATMENT-"] = bothInvalid
                    win1["-ERROR_TREATMENT-"].update(value["-ERROR_TREATMENT-"])

            elif value["-TREATMENT_FILE-"] == '':
                value["-ERROR_TREATMENT-"] = needFileToPlot
                win1["-ERROR_TREATMENT-"].update(value["-ERROR_TREATMENT-"])

        if event == '-P_PLOT-':
            if value["-PLACEMENT_FILE-"] != '':
                
                fileGood = CheckFile(value["-PLACEMENT_FILE-"])

                if fileGood == True:
                    csvStatus, columns = CheckPlainPlotCSV(value["-PLACEMENT_FILE-"])
                    if csvStatus == True:
                        try:
                            plotContents(value["-PLACEMENT_FILE-"], columns)
                        except ValueError:
                            value["-ERROR_PLACEMENT-"] = incompatible_contents
                            win1["-ERROR_PLACEMENT-"].update(value["-ERROR_PLACEMENT-"])

                        except IndexError:
                            value["-ERROR_PLACEMENT-"] = incompatible_contents
                            win1["-ERROR_PLACEMENT-"].update(value["-ERROR_PLACEMENT-"])
                            
                        except TypeError:
                            value["-ERROR_PLACEMENT-"] = incompatible_contents
                            win1["-ERROR_PLACEMENT-"].update(value["-ERROR_PLACEMENT-"])
                    else:
                        value["-ERROR_PLACEMENT-"] = unexpected_contents
                        win1["-ERROR_PLACEMENT-"].update(value["-ERROR_PLACEMENT-"])

                elif fileGood == False:
                    value["-ERROR_PLACEMENT-"] = bothInvalid
                    win1["-ERROR_PLACEMENT-"].update(value["-ERROR_PLACEMENT-"])

            elif value["-PLACEMENT_FILE-"] == '':
                value["-ERROR_PLACEMENT-"] = needFileToPlot
                win1["-ERROR_PLACEMENT-"].update(value["-ERROR_PLACEMENT-"])
        
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
                           [sg.Text("\nCalculate THD", font=('None',info_txt_size+1,'underline'))],
                           [sg.Text("This button will take the treatment waveform input and tranform it into the frequency domain via the Fourier Transform. Frequency will be plotted against amplitude; the more that a frequency is present, the higher its plotted amplitude. These frequency amplitudes are used to calculate the Total Harmonic Distortion (THD) per the following equation:", size=(info_txt_width,None), font=('None',info_txt_size))],
                           [sg.Push(), sg.Image(THD_eq), sg.Push()],
                           [sg.Text("\nAverage Peaks", font=('None',info_txt_size+1,'underline'))],
                           [sg.Text("This button will take the treament waveform input and calculate the average of the pulse amplitudes.")],
                           [sg.Text("\nFor the 'Calculate THD' or 'Average Peaks' functions to work as intended, the input waveform should be of the pulse burst and resemble the following:", size=(info_txt_width,None), font=('None',info_txt_size))],
                           [sg.Push(), sg.Image(pulse_burst_example), sg.Push()],
                           [sg.Button("Close")]]

            win2 = sg.Window(title="ST-0001-066-101A Information", layout=layout_win2, size=(1000,890))

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
                voltageGood = treatmentVoltageCheck(value["-TREATMENT_VOLT-"])

                if fileGood == True and voltageGood == True:
                    csvGood = CheckCSV(value["-TREATMENT_FILE-"])

                    if csvGood == True:
                        try:
                            guinnessRampFilter(value["-TREATMENT_FILE-"], value["-TREATMENT_VOLT-"])
                        except ValueError:
                            value["-ERROR_TREATMENT-"] = incompatible_contents
                            win1["-ERROR_TREATMENT-"].update(value["-ERROR_TREATMENT-"])

                        except IndexError:
                            value["-ERROR_TREATMENT-"] = incompatible_contents
                            win1["-ERROR_TREATMENT-"].update(value["-ERROR_TREATMENT-"])
                            
                        except TypeError:
                            value["-ERROR_TREATMENT-"] = incompatible_contents
                            win1["-ERROR_TREATMENT-"].update(value["-ERROR_TREATMENT-"])
                    else:
                        value["-ERROR_TREATMENT-"] = unexpected_contents
                        win1["-ERROR_TREATMENT-"].update(value["-ERROR_TREATMENT-"])

                elif fileGood == False and voltageGood == True:
                    value["-ERROR_TREATMENT-"] = invalid_path_or_type
                    win1["-ERROR_TREATMENT-"].update(value["-ERROR_TREATMENT-"])

                elif fileGood == True and voltageGood == False:
                    csvGood = CheckCSV(value["-TREATMENT_FILE-"])

                    if csvGood == True:
                        value["-ERROR_TREATMENT-"] = invalid_treatment_V
                        win1["-ERROR_TREATMENT-"].update(value["-ERROR_TREATMENT-"])
                    else:
                        value["-ERROR_TREATMENT-"] = invalid_treatment_V_and_File
                        win1["-ERROR_TREATMENT-"].update(value["-ERROR_TREATMENT-"])

                elif fileGood == False and voltageGood == False:
                    value["-ERROR_TREATMENT-"] = bothInvalid
                    win1["-ERROR_TREATMENT-"].update(value["-ERROR_TREATMENT-"])

            elif value["-TREATMENT_FILE-"] == '' or value["-TREATMENT_VOLT-"] == '':
                value["-ERROR_TREATMENT-"] = bothEmpty
                win1["-ERROR_TREATMENT-"].update(value["-ERROR_TREATMENT-"])


        if event == "-TREATMENT_THD-":
            if value["-TREATMENT_FILE-"] != '' and value["-TREATMENT_VOLT-"] != '':
                
                fileGood = CheckFile(value["-TREATMENT_FILE-"])
                voltageGood = treatmentVoltageCheck(value["-TREATMENT_VOLT-"])

                if fileGood == True and voltageGood == True:
                    csvGood = CheckCSV(value["-TREATMENT_FILE-"])

                    if csvGood == True:
                        try:
                            guinnessTHD(value["-TREATMENT_FILE-"], value["-TREATMENT_VOLT-"])
                        except ValueError:
                            value["-ERROR_TREATMENT-"] = incompatible_contents
                            win1["-ERROR_TREATMENT-"].update(value["-ERROR_TREATMENT-"])

                        except IndexError:
                            value["-ERROR_TREATMENT-"] = incompatible_contents
                            win1["-ERROR_TREATMENT-"].update(value["-ERROR_TREATMENT-"])
                            
                        except TypeError:
                            value["-ERROR_TREATMENT-"] = incompatible_contents
                            win1["-ERROR_TREATMENT-"].update(value["-ERROR_TREATMENT-"])
                    else:
                        value["-ERROR_TREATMENT-"] = unexpected_contents
                        win1["-ERROR_TREATMENT-"].update(value["-ERROR_TREATMENT-"])

                elif fileGood == False and voltageGood == True:
                    value["-ERROR_TREATMENT-"] = invalid_path_or_type
                    win1["-ERROR_TREATMENT-"].update(value["-ERROR_TREATMENT-"])

                elif fileGood == True and voltageGood == False:
                    csvGood = CheckCSV(value["-TREATMENT_FILE-"])

                    if csvGood == True:
                        value["-ERROR_TREATMENT-"] = invalid_treatment_V
                        win1["-ERROR_TREATMENT-"].update(value["-ERROR_TREATMENT-"])
                    else:
                        value["-ERROR_TREATMENT-"] = invalid_treatment_V_and_File
                        win1["-ERROR_TREATMENT-"].update(value["-ERROR_TREATMENT-"])

                elif fileGood == False and voltageGood == False:
                    value["-ERROR_TREATMENT-"] = bothInvalid
                    win1["-ERROR_TREATMENT-"].update(value["-ERROR_TREATMENT-"])
            
            elif value["-TREATMENT_FILE-"] == '' or value["-TREATMENT_VOLT-"] == '':
                value["-ERROR_TREATMENT-"] = bothEmpty
                win1["-ERROR_TREATMENT-"].update(value["-ERROR_TREATMENT-"])
        
        if event == "-TREATMENT_Pk_AVG-":
            if value["-TREATMENT_FILE-"] != '' and value["-TREATMENT_VOLT-"] != '':
                
                fileGood = CheckFile(value["-TREATMENT_FILE-"])
                voltageGood = treatmentVoltageCheck(value["-TREATMENT_VOLT-"])

                if fileGood == True and voltageGood == True:
                    csvGood = CheckCSV(value["-TREATMENT_FILE-"])

                    if csvGood == True:
                        try:
                            averagePkAmp(value["-TREATMENT_FILE-"], value["-TREATMENT_VOLT-"])
                        except ValueError:
                            value["-ERROR_TREATMENT-"] = incompatible_contents
                            win1["-ERROR_TREATMENT-"].update(value["-ERROR_TREATMENT-"])

                        except IndexError:
                            value["-ERROR_TREATMENT-"] = incompatible_contents
                            win1["-ERROR_TREATMENT-"].update(value["-ERROR_TREATMENT-"])
                            
                        except TypeError:
                            value["-ERROR_TREATMENT-"] = incompatible_contents
                            win1["-ERROR_TREATMENT-"].update(value["-ERROR_TREATMENT-"])
                    else:
                        value["-ERROR_TREATMENT-"] = unexpected_contents
                        win1["-ERROR_TREATMENT-"].update(value["-ERROR_TREATMENT-"])

                elif fileGood == False and voltageGood == True:
                    value["-ERROR_TREATMENT-"] = invalid_path_or_type
                    win1["-ERROR_TREATMENT-"].update(value["-ERROR_TREATMENT-"])

                elif fileGood == True and voltageGood == False:
                    csvGood = CheckCSV(value["-TREATMENT_FILE-"])

                    if csvGood == True:
                        value["-ERROR_TREATMENT-"] = invalid_treatment_V
                        win1["-ERROR_TREATMENT-"].update(value["-ERROR_TREATMENT-"])
                    else:
                        value["-ERROR_TREATMENT-"] = invalid_treatment_V_and_File
                        win1["-ERROR_TREATMENT-"].update(value["-ERROR_TREATMENT-"])

                elif fileGood == False and voltageGood == False:
                    value["-ERROR_TREATMENT-"] = bothInvalid
                    win1["-ERROR_TREATMENT-"].update(value["-ERROR_TREATMENT-"])
            
            elif value["-TREATMENT_FILE-"] == '' or value["-TREATMENT_VOLT-"] == '':
                value["-ERROR_TREATMENT-"] = bothEmpty
                win1["-ERROR_TREATMENT-"].update(value["-ERROR_TREATMENT-"])
                
                
        if event == "-PLACEMENT_PULSE-":
            if value["-PLACEMENT_FILE-"] != '' and value["-PLACEMENT_VOLT-"] != '':
                
                fileGood = CheckFile(value["-PLACEMENT_FILE-"])
                voltageGood = placementVoltageCheck(value["-PLACEMENT_VOLT-"])

                if fileGood == True and voltageGood == True:
                    csvGood = CheckCSV(value["-PLACEMENT_FILE-"])

                    if csvGood == True:
                        try:
                            calcRiseFall(value["-PLACEMENT_FILE-"], value["-PLACEMENT_VOLT-"])
                        except ValueError:
                            value["-ERROR_PLACEMENT-"] = incompatible_contents
                            win1["-ERROR_PLACEMENT-"].update(value["-ERROR_PLACEMENT-"])

                        except IndexError:
                            value["-ERROR_PLACEMENT-"] = incompatible_contents
                            win1["-ERROR_PLACEMENT-"].update(value["-ERROR_PLACEMENT-"])
                            
                        except TypeError:
                            value["-ERROR_PLACEMENT-"] = incompatible_contents
                            win1["-ERROR_PLACEMENT-"].update(value["-ERROR_PLACEMENT-"])
                    else:
                        value["-ERROR_PLACEMENT-"] = unexpected_contents
                        win1["-ERROR_PLACEMENT-"].update(value["-ERROR_PLACEMENT-"])

                elif fileGood == False and voltageGood == True:
                    value["-ERROR_PLACEMENT-"] = invalid_path_or_type
                    win1["-ERROR_PLACEMENT-"].update(value["-ERROR_PLACEMENT-"])

                elif fileGood == True and voltageGood == False:
                    csvGood = CheckCSV(value["-PLACEMENT_FILE-"])

                    if csvGood == True:
                        value["-ERROR_PLACEMENT-"] = invalid_placement_V
                        win1["-ERROR_PLACEMENT-"].update(value["-ERROR_PLACEMENT-"])
                    else:
                        value["-ERROR_PLACEMENT-"] = invalid_placement_V_and_File
                        win1["-ERROR_PLACEMENT-"].update(value["-ERROR_PLACEMENT-"])

                elif fileGood == False and voltageGood == False:
                    value["-ERROR_PLACEMENT-"] = bothInvalid
                    win1["-ERROR_PLACEMENT-"].update(value["-ERROR_PLACEMENT-"])
            
            elif value["-PLACEMENT_FILE-"] == '' or value["-PLACEMENT_VOLT-"] == '':
                value["-ERROR_PLACEMENT-"] = bothEmpty
                win1["-ERROR_PLACEMENT-"].update(value["-ERROR_PLACEMENT-"])

        if event == "-PLACEMENT_TONE-":
            if value["-PLACEMENT_FILE-"] != '' and value["-PLACEMENT_VOLT-"] != '':
                
                fileGood = CheckFile(value["-PLACEMENT_FILE-"])
                voltageGood = placementVoltageCheck(value["-PLACEMENT_VOLT-"])

                if fileGood == True and voltageGood == True:
                    csvGood = CheckAudioCSV(value["-PLACEMENT_FILE-"])

                    if csvGood == True:
                        try:
                            guinnessAudioSync(value["-PLACEMENT_FILE-"], value["-PLACEMENT_VOLT-"])
                        except ValueError:
                            value["-ERROR_PLACEMENT-"] = incompatible_contents
                            win1["-ERROR_PLACEMENT-"].update(value["-ERROR_PLACEMENT-"])

                        except IndexError:
                            value["-ERROR_PLACEMENT-"] = incompatible_contents
                            win1["-ERROR_PLACEMENT-"].update(value["-ERROR_PLACEMENT-"])
                            
                        except TypeError:
                            value["-ERROR_PLACEMENT-"] = incompatible_contents
                            win1["-ERROR_PLACEMENT-"].update(value["-ERROR_PLACEMENT-"])
                    else:
                        value["-ERROR_PLACEMENT-"] = unexpected_contents
                        win1["-ERROR_PLACEMENT-"].update(value["-ERROR_PLACEMENT-"])

                elif fileGood == False and voltageGood == True:
                    value["-ERROR_PLACEMENT-"] = invalid_path_or_type
                    win1["-ERROR_PLACEMENT-"].update(value["-ERROR_PLACEMENT-"])

                elif fileGood == True and voltageGood == False:
                    csvGood = CheckCSV(value["-PLACEMENT_FILE-"])

                    if csvGood == True:
                        value["-ERROR_PLACEMENT-"] = invalid_placement_V
                        win1["-ERROR_PLACEMENT-"].update(value["-ERROR_PLACEMENT-"])
                    else:
                        value["-ERROR_PLACEMENT-"] = invalid_placement_V_and_File
                        win1["-ERROR_PLACEMENT-"].update(value["-ERROR_PLACEMENT-"])

                elif fileGood == False and voltageGood == False:
                    value["-ERROR_PLACEMENT-"] = bothInvalid
                    win1["-ERROR_PLACEMENT-"].update(value["-ERROR_PLACEMENT-"])
            
            elif value["-PLACEMENT_FILE-"] == '' or value["-PLACEMENT_VOLT-"] == '':
                value["-ERROR_PLACEMENT-"] = bothEmpty
                win1["-ERROR_PLACEMENT-"].update(value["-ERROR_PLACEMENT-"])
                
                
    # Catch for value errors, application should not crash this way
    except ValueError:
        value["-ERROR_TREATMENT-"] = incompatible_contents
        value["-ERROR_PLACEMENT-"] = incompatible_contents
        win1["-ERROR_TREATMENT-"].update(value["-ERROR_PLACEMENT-"])
        win1["-ERROR_PLACEMENT-"].update(value["-ERROR_PLACEMENT-"])
        
    # Catch for index errors, application should not crash this way
    except IndexError:
        value["-ERROR_TREATMENT-"] = incompatible_contents
        value["-ERROR_PLACEMENT-"] = incompatible_contents
        win1["-ERROR_TREATMENT-"].update(value["-ERROR_PLACEMENT-"])
        win1["-ERROR_PLACEMENT-"].update(value["-ERROR_PLACEMENT-"])
        
    # Catch for type errors, application should not crash this way
    except TypeError:
        value["-ERROR_TREATMENT-"] = incompatible_contents
        value["-ERROR_PLACEMENT-"] = incompatible_contents
        win1["-ERROR_TREATMENT-"].update(value["-ERROR_PLACEMENT-"])
        win1["-ERROR_PLACEMENT-"].update(value["-ERROR_PLACEMENT-"])


'''To create your EXE file from your program that uses PySimpleGUI, my_program.py, enter this command in your Windows command prompt:

pyinstaller my_program.py

You will be left with a single file, my_program.exe, located in a folder named dist under the folder where you executed the pyinstaller command.
'''

'''
OR:
pip install pysimplegui-exemaker

then:
python -m pysimplegui-exemaker.pysimplegui-exemaker
'''