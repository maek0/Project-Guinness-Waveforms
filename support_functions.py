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
    place_diff2 = np.abs(np.gradient(np.gradient(place)))
    plt.plot(x,place_diff2)
    audio_diff2 = np.abs(np.gradient(np.gradient(audio)))
    plt.plot(x,audio_diff2)
    plt.show()
    # audio = signal.detrend(audio, type="constant")
    
    # vertically offset the audio signal for clearer graphing/visualization
    audio = audio + 0.5

    # plot placement and audio
    plt.plot(x, place, label = "Placement Output", color = "blue")
    plt.plot(x, audio, label = "Placement Audio", color = "orange")

    placement_peakIndices, _ = signal.find_peaks(place, height=0.4, distance=500)
    placement_peakHeights = place[placement_peakIndices]
    # print(placement_peakIndices)
    # print(placement_peakHeights)

    audio_peakIndices, _ = signal.find_peaks(audio_diff2, height=0.15, distance=2500)
    audio_peakHeights = audio[audio_peakIndices]
    # print(audio_peakIndices)
    # print(audio_peakHeights)

    diff = []
    print(len(audio_peakIndices))
    print(audio_peakIndices)
    print(len(placement_peakIndices))
    print(placement_peakIndices)
    
    for i in range(0,len(audio_peakIndices),1):
        for j in range(0,len(placement_peakIndices),1):
            if np.abs(audio_peakIndices[i]-placement_peakIndices[j])<1500:
                diff_temp = np.abs(x[audio_peakIndices[i]]-x[placement_peakIndices[j]])
                diff.append(diff_temp)
                # print(diff_temp,x[placement_peakIndices[i]],place[placement_peakIndices[i]])
                plt.text(x[placement_peakIndices[j]]+0.02,-place[placement_peakIndices[j]]+float(0.25*float(voltageLimit)),"Delay = {:.4f}s".format(diff_temp))
            else:
                pass
            print("placement index",j)
        # print(diff)
        print("audio index",i)

    print(diff)
    diff = np.array(diff)
    average_delay = np.mean(diff)

    # plot peaks of placement and audio
    plt.scatter(x[placement_peakIndices], placement_peakHeights,marker="x",color="black",s=50,label="Placement Pulse(s)")
    plt.scatter(x[audio_peakIndices],audio_peakHeights,marker="x",color="red", s=50,label="Audio Tone(s)")

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