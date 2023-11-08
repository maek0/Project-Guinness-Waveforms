import PySimpleGUI as sg
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

    if columns > 2 or columns < 2:
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
    plt.axhline(cutoff, label = "{:.2f}V".format(cutoff), linestyle = "--", color = "black")
    
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
    thd = 100*(np.sum(y_peaks_yvalues)-max(y_peaks_yvalues))**0.5 / max(y_peaks_yvalues)

    # plotting options
    plt.title("Guinness Generator Pulse Burst FFT, THD = {:.3f}%\nVoltage Limit = {}V, Input file name: '{}'".format(thd,voltageLimit, filename))
    plt.xlim(min(xf),xlim)
    plt.ylim(min(yf_plottable)-3,max(yf_plottable)+3)
    plt.legend(loc="upper right")

    # display the plot
    plt.show()
    
    return thd


def guinnessRampFilter(filepath,voltageLimit):
    import numpy as np
    from support_functions import plotting_peaks
    import datetime
    
    # need matplotlib, scipy, numpy, scikit (pip install numpy scikit-learn statsmodels), pysimplegui

    datetime_rn = datetime.datetime.now()
    str_datetime_rn = datetime_rn.strftime("%d-%b-%Y, %X %Z")
    
    headers = ["Time", "Voltage"]
    csvFile = open(filepath)
    csvArray = np.genfromtxt(csvFile, delimiter=",")
    x = csvArray[2:,0]
    y = csvArray[2:,1]
    
    plotting_peaks(x, y, voltageLimit, filepath, str_datetime_rn, headers)


def guinnessTHD(filepath,voltageLimit):
    import numpy as np
    import datetime
    
    # need matplotlib, scipy, numpy, scikit (pip install numpy scikit-learn statsmodels)

    datetime_rn = datetime.datetime.now()
    str_datetime_rn = datetime_rn.strftime("%d-%b-%Y, %X %Z")
    
    headers = ["Time", "Voltage"]
    csvFile = open(filepath)
    csvArray = np.genfromtxt(csvFile, delimiter=",")
    x = csvArray[2:-2,0]
    y = csvArray[2:-2,1]
    
    THD(x, y, voltageLimit, filepath, str_datetime_rn, headers)

help_button_base64 = b'iVBORw0KGgoAAAANSUhEUgAAABgAAAAYCAYAAADgdz34AAAACXBIWXMAAAsTAAALEwEAmpwYAAABq0lEQVRIie1VPS8EURTdIJEIEvOxGyRCUPADJFtIBIVEva1CQ6NQCLHz3lMqdKJQoNNJJKKhoNDRiUjQqXwEEcy6b5LnDEE28+Zjl46b3GrmnnPvuWfupFJ/IswpVWdzL2dxuWRx2rQZbdtcrqQZjZtCNZUN3ChUjcWI25weAaj0SZ7F5Koxq5pL6xqd2UwehgMXJya7NYXsSwReL5SBorOk4F8kjFxDUE8sATTeiuj0GpKchxIxurSFqg0FTzM5ENVlRritKaGqsPCLcBI5F0qAMTeiZZDDmbzbhvduIqS6QhMVAfCOCVUNCZ5K1V6XsHA2qH3ebY8sZLSOCeYhz0Hswp3XUY3+lI0pPIK+C77GsVMwmgkQGE6hK0HhXhICk9NYgODjJJD3GwS2kENaF+Hh/k8J4KJn/8zobYrlxO7A8UYAshtKwOWaFvw9cqoSICchxcuQcPo7deD00iBUSziBv2xR6AbAQzn+19pTF7iM/SC5Sw6Os81pMhH4Z1hOoRNFO/HWladpIQdLAi8iYtSLxS3iKz6Gi+59nW3/nOPL9v90/vErG/w/3gALBuad4TTYiQAAAABJRU5ErkJggg=='

THD_eq = b'iVBORw0KGgoAAAANSUhEUgAAAWgAAABGCAYAAADhNA4nAAAAAXNSR0IArs4c6QAABAR0RVh0bXhmaWxlACUzQ214ZmlsZSUyMGhvc3QlM0QlMjJFbGVjdHJvbiUyMiUyMG1vZGlmaWVkJTNEJTIyMjAyMy0xMC0yMVQwNCUzQTQ5JTNBMzguMTYzWiUyMiUyMGFnZW50JTNEJTIyTW96aWxsYSUyRjUuMCUyMChXaW5kb3dzJTIwTlQlMjAxMC4wJTNCJTIwV2luNjQlM0IlMjB4NjQpJTIwQXBwbGVXZWJLaXQlMkY1MzcuMzYlMjAoS0hUTUwlMkMlMjBsaWtlJTIwR2Vja28pJTIwZHJhdy5pbyUyRjIyLjAuMiUyMENocm9tZSUyRjExNC4wLjU3MzUuMjg5JTIwRWxlY3Ryb24lMkYyNS44LjQlMjBTYWZhcmklMkY1MzcuMzYlMjIlMjBldGFnJTNEJTIyd0NoN3hnaFpOQm1GQXdibmxiekolMjIlMjB2ZXJzaW9uJTNEJTIyMjIuMC4yJTIyJTIwdHlwZSUzRCUyMmRldmljZSUyMiUzRSUzQ2RpYWdyYW0lMjBuYW1lJTNEJTIyUGFnZS0xJTIyJTIwaWQlM0QlMjIzMVRZSHVjRVRjVWJWOXRfRUNYWCUyMiUzRWpaTnRiNEl3RU1jJTJGRFlrdW1RRTZuYjRjNkhSWnRpVnpjUzlOcFNjMEZzcEtGZHluM3dGRlJHT3lOOXI3M1VONyUyRnpzczRzZkZYTkUwZXBNTWhPWGFyTERJMUhKZHg3RnQlMkZDdkpzU1lqMTZsQnFEZ3pRUzFZOGw4dzBPU0ZlODRnNndScUtZWG1hUmNHTWtrZzBCMUdsWko1TjJ3clJmZldsSVp3QlpZQkZkZjBtek1kMVhROHRGdSUyQkFCNUclMkJyTGhtRGJCcGtRV1VTYnpHbFV4WkdZUlgwbXA2MU5jJTJCQ0JLOFJwZDZrTFBON3luaHlsSTlIOFNpbzMlMkZlbGlNUHphZk8zaDM1MlR5RWszdjNicktnWXE5YWRnYTJWOExQTmlWYVZmOURPOTYyWSUyRlN2ZFhhdFlhek1zRnlQZnhkcmNtRiUyRmRDeEI0TkJ2NDh2d2tTbmRQU3h1QkZESHh1Rk5SVDRmaSUyRlNzVURnNERIVFN1N0FsMElxSkls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'

voltage_ramp_example = b'iVBORw0KGgoAAAANSUhEUgAAAmIAAAB6CAYAAAAcTD85AAAAAXNSR0IArs4c6QAACIB0RVh0bXhmaWxlACUzQ214ZmlsZSUyMGhvc3QlM0QlMjJFbGVjdHJvbiUyMiUyMG1vZGlmaWVkJTNEJTIyMjAyMy0xMC0yMVQxNiUzQTM4JTNBNTAuODEwWiUyMiUyMGFnZW50JTNEJTIyTW96aWxsYSUyRjUuMCUyMChXaW5kb3dzJTIwTlQlMjAxMC4wJTNCJTIwV2luNjQlM0IlMjB4NjQpJTIwQXBwbGVXZWJLaXQlMkY1MzcuMzYlMjAoS0hUTUwlMkMlMjBsaWtlJTIwR2Vja28pJTIwZHJhdy5pbyUyRjIyLjAuMiUyMENocm9tZSUyRjExNC4wLjU3MzUuMjg5JTIwRWxlY3Ryb24lMkYyNS44LjQlMjBTYWZhcmklMkY1MzcuMzYlMjIlMjBldGFnJTNEJTIyYTNIT1NLTjJVOTFxVFVYdDVHMl8lMjIlMjB2ZXJzaW9uJTNEJTIyMjIuMC4yJTIyJTIwdHlwZSUzRCUyMmRldmljZSUyMiUzRSUzQ2RpYWdyYW0lMjBuYW1lJTNEJTIyUGFnZS0xJTIyJTIwaWQlM0QlMjJaMTl2ZHpocU84NmJXSkxmcUxfciUyMiUzRTdaeGRjOXNvRklaJTJGalMlMkJUa1JENnVteWNKdTFzZDdiYjdIUm5lNmUxaUsySkxEd3lpZTMlMkIlMkJrVXh5T2JnT0VLeGNEZURMekxXRVJ6SjUzbUJBeElaQmVQNSUyQnJiT0ZyUGZhVTdLRWZMeTlTaTRIaUVVSTh6JTJGTm9iTjFvQlJzalZNNnlMZm12eWQ0YTc0U1lUUkU5YkhJaWRMcFNDanRHVEZRalZPYUZXUkNWTnNXVjNUbFZyc25wYnFWUmZabEdpR3UwbFc2dGElMkZpNXpOdHRZazlIYjJUNlNZenVTVmZVJTJCY21XZXlzREFzWjFsT1YzdW00T01vR05lVXN1MjMlMkJYcE15aVoyTWk3YmVqY3ZuRzF2ckNZVjYxTGh4N2YxUCUyRmhQZG5QNzVYUDFkVEwyNnVMcXg0WHc4cFNWaiUyQklIaTV0bEd4bUJhVTBmRjZJWXFSbFpINHA3OXE4czd1bjM1YmUlMkZscXVFMERsaDlZWVhrUklSTllSQVVDS09WN3R3UjFJTnMlMkYxUXkwdGxBdkcwZGIyTEF2OGlBbkU0S01WVGRmMEh1czElMkI4elljME0lMkY4Z1Q1OHUlMkZDRDE2TkNxdnhESXk5JTJCVk5HS0c2OW1iTTZ2Y3UzenJ6eGlWVTZhUzNqOFNBJTJGSU1SUWtWd1NwaDIwdkx1R0JzRWhiVGNxTUZVJTJCcWpBJTJCRlNsemhLeTM0N2JWVVFwVktCR0s5cEklMkYxaElnNiUyQjZLRGJ1TGpmbGhXVHduVCUyRkR4amEzJTJGekcwaGlSeEtRMUpwTlo1UjIwVVVkR3VGZTE5UVZTdWN1ckd0JTJGRlYxRzZrZnJ2WEJ5cVBmeVlNV2hPclB3cEUxZ3lXcjZRTWEwcFBWejNjRGpuNXViOW93Y0pOSEI1bktVOUxuYml3QWFCcGVCOHVuWFhGcnEzZHdPM1JGR0E2aGduJTJGWEx1dmlmcWlBeXhDN3JlYUJaWTd1Y1k4ZlphSFNVZlBwaWY4SE5tVldRT0JVWXFjQ1hzOUkzeWtEemMxNGRvQ0hHJTJGbmV0Z3hDcCUyRkpMaiUyRkRyckF2akZpVjBkdU5IZlVBZVIybzU3Y3dkJTJCTURwdmYlMkJDeUEwTWRKT3E0SHFVbm1SdEF0MEY4enJrQmN0bUNtU3FRcDA3dFlueDBLdDlWRmRCdGdPektJSFV5TUpPQm5GUUxYa25QaFRYb1IxdFNzanRJWUpjMEd1cEFObFNwZyUyRkFrZ3dSMGkzemcxbTczZ0YwT2FTZ0xyT1olMkJLWndMOXBRRmNPdEhkbVhnVWtoREdZQ2xvYlF2ZCUyQkRIaDNLeXJBT1hOQnJxQUR3UFNPSFNRazlaQUxkMkZ4YXd5eGtOVlpDcXVaN3Z3Vkc5cHd5QVg1aUMySjFSeW1BNVdYU1VSZUFqSUl1ZUMwN1FFVnh2aWtLN1F1andmcEVUd2o0JTJGOEFEaHRkY0VPdXZDOHNBZzE3c2QlMkJLN2dNVXp2VGdRZVk3dmczZnpBRUh6a0RRUGU4cXQyb1pzUUdJS1BZU3A0SXZDeDNabGc2T1lBaHVCVG1PeWRDTHowYSUyQnNOVFpmbG00SEgzakRKWGV2WEZuaVgxUnVDeDhPTThhMWZXJTJCQmRWbThJSGcyVDFiZCUyQmJZRjMlMkJ5aTRGaFdVQUdU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'

pulse_burst_example = b'iVBORw0KGgoAAAANSUhEUgAAAlAAAAChCAYAAAAIs4HQAAAAAXNSR0IArs4c6QAACAB0RVh0bXhmaWxlACUzQ214ZmlsZSUyMGhvc3QlM0QlMjJFbGVjdHJvbiUyMiUyMG1vZGlmaWVkJTNEJTIyMjAyMy0xMC0yMVQxNiUzQTM5JTNBNDUuNzg5WiUyMiUyMGFnZW50JTNEJTIyTW96aWxsYSUyRjUuMCUyMChXaW5kb3dzJTIwTlQlMjAxMC4wJTNCJTIwV2luNjQlM0IlMjB4NjQpJTIwQXBwbGVXZWJLaXQlMkY1MzcuMzYlMjAoS0hUTUwlMkMlMjBsaWtlJTIwR2Vja28pJTIwZHJhdy5pbyUyRjIyLjAuMiUyMENocm9tZSUyRjExNC4wLjU3MzUuMjg5JTIwRWxlY3Ryb24lMkYyNS44LjQlMjBTYWZhcmklMkY1MzcuMzYlMjIlMjBldGFnJTNEJTIyQlNjWWtZUzhQYnBUUnE3LVRjRlolMjIlMjB2ZXJzaW9uJTNEJTIyMjIuMC4yJTIyJTIwdHlwZSUzRCUyMmRldmljZSUyMiUzRSUzQ2RpYWdyYW0lMjBuYW1lJTNEJTIyUGFnZS0xJTIyJTIwaWQlM0QlMjJ0OWxNOG8wbURaWFRvNkR0b3lfMCUyMiUzRTdacGRjNXM0RklaJTJGalMlMkZONkJ0eG1iakpkanJkMmM1bXQ1bGU3UkJRREMwZ2l1VTQyViUyQiUyRmtpMFpLQ1RCY1p5MVd6eE1BZ2R4Qk8lMkJqYzRRa0puaVczJTJGOVdoV1h5dTR4Rk5rRWd2cCUyRmdkeE9FcGhBeCUyRmM5WUhqWVdndmpHTUslMkZTZUdPQ3RlRXElMkZWZFlJN0RXWlJxTFJhdWdrakpUYWRrMlJySW9SS1JhdHJDcTVLcGQ3RlptN1ZyTGNDNDZocXNvekxyVzZ6Uld5Y2JLS2FqdDcwVTZUMXpORU5nemVlZ0tXOE1pQ1dPNWFwand4UVRQS2luVlppJTJCJTJGbjRuTWlPZDAyVngzJTJCY2paN1kxVm9sQkRMcGglMkYlMkZ2dkRkOGhSTnYzeVIzNkJaOG5YUEpwYUwzZGh0clFQYkc5V1BUZ0Y1cFZjbHJhWXFKUzQ3OU05dkhIRlFmZSUyQjRQWnBkVE1STWhlcWV0QkZyQ1B1MjB0c0M4SE94YXJXbXdiV2xqUzFadFlZV3NienJlOWFCcjFqbGVoWDVRTmxkNWZYSUlyJTJGdWY2VFVTeSUyRmZ3bmVUekhhU1pibXN6NnA4bEQ1SHRXSzRaWlV5UE1oNUl3UXdIdyUyRllBeDJoWU93UnpqcWU0aHppQ2tOc0s0UzB3UHBDSGRyWGswZG4lMkJheXQ1Q0lJZzhFdVA2eGpuU0ljaThBclA2UiUyRjFQSkFTMVNKNWpTN0tiNU9xZWRHNVZTbmNrJTJCaGpjaSUyQnlRWHFVcGxvYyUyRmZTS1Zrcmd0azVzUjVHSDB6RElwNEpqTlo2Zk94dUEyWG1XcDRPTXZTdWJsU3lWSmIxJTJGN1BGdVVtM1FKbjBmdUpVaVl0bjVsblJaY0xGZXFyUFpXSVFyc3ZLJTJGbFZYJTJCRkZ1bXAwV1JaeiUyRlJmN0FVUlRCSUJuanZINXdHWUFINHVuTHU0R1QySndFaG9ndCUyRlhnSk1SakdQbU1Zd1lwaHdFNkZFNDg0bndKem52M0V1RUJBQWdrZkxPaFR2QWVGV3d5d3Q0RE5pY0dOa2NCMjJ6a3VHSFRFZlllc1BWNFFkT0dpTGdmUFc3YWJLUzlEMjNHRFcyMmhla2ZOMjElMkZwTDBIYlQybTE3UVI4SW5ianBzMlA3ckJ5NjZqa3plVEtoZ0RvOXR1VG5aMDRpWmdScHk3NFR6SjBRa2FNRWN6d3Y1SlJpZU83UWo3UmJCUGJIU0N4bG1tdldpZjF1Z0VqZE5NJTJCOUElMkJzZEVKSGhEYm9valB6QktvUGlwa1lYZ25LdGUxdklONmQ4MVR4QlpQVjlhbkZySkUzRm8wZmZMRmx2YXMzRGxiSlRMTiUyRjY3cHExOHFXOE1ubWVyYjJ5S2o3WVZEbjN1TnBURGdSdXZPMzBJdXEwaFlGN1g2WGElMkZCVG01VldNMkY2cmhkUTkwcXNnZm5BVkg5aTNHR0ZIZ0IwNEVJQTElMkZIS1hUejZ6dURicnRsWGl0dVFmQzJuTHR6UzU5bHBrelMlMkZCRzNUbWlxelhpaEt2bE51T1JzMjhCdG1tVSUyRm1FS2JwQ05OWDFROTJUdFA0OWhVYzc1S1VpV3V5akF5ZGE2cXNPeTBwRXFhM0wzdUxhYkI3azFyUUdKMlhKQ0glMkZjWWlheWN0czU2V2gzeXZiMlg4OWROd2Q1TG9yelElMkZWbWF2anNqMW5aeDd0QjJqR0huTmRYR24welBZb0g3ajhnTlFSJTJGZUJzSkclMkJyS3JsQWR1WUF4JTJGVFBGVkh5ckVSZSUyQmJ3VmhiSzFUWkJHS3glMkYxbTYlMkZxT0lIcEElMkI3MyUyQnVRdnBERTNxRyUyQmppQURGdDElMkI5azRTdDNzelJMekdPd3NMQ0glMkJWVHZJNXR3ZnVKVWwzdlcwNlJ1MWVVZXNTdHU0eFNmY2pxTGNPNUFFcmJMOVlJSnM1SjBveDh4R0dESE1DWHllUW4zUDc0a0RXaCUyRlUzclp2aTlaZkIlMkJPSSUyRiUzQyUyRmRpYWdyYW0lM0UlM0MlMkZteGZpbGUlM0XAzemoAAAgAElEQVR4Xu3dCdi2fzkn8NNStoiQZSKSiKjETGoyKUvWZItoskWibFmiVJYMRcb8ZSmGhCHapVQmWpRmVNOi0jIjWUaUKCXDHJ+e39V7Pfdz7ffzPs/9f//f33E8x/sez3Pd133e3/s8f9f5O5fv+XaVFQSCQBAIAkEgCASBILAKgbdbdXUuDgJBIAgEgSAQBIJAEKg4UFGCIBAEgkAQCAJBIAisRCAO1ErAcnkQCAJBIAgEgSAQBE7bgXqXqvrqqvrlqnptg/fLq+pZVfUnPbjfq6puVVUPq6prVdVL81UEgSAQBIJAEAgCQeDygsC+DtSXVtXje86Sz/25VfWXVfWHVcVRukNV/VRVXan9/F37/edX1W9V1RdX1QOr6l+q6t2r6i1V9Q8NwKu0f/3tTVX19u2aN1TVP11eQI6cQSAIBIEgEASCwKWFwFYH6h2bI8M5emxVvbqq/rFB86FVdcuq+rmquklVvVNVvbn9/2VV9RFV9YtV9dlV9QdVJULFgXKvZ1fV9avqCVX1blV1g6r626r69Kq6b1Xdtl3z8VX1oKr6q97Xcb2qesGl9fXk0wSBIBAEgkAQCAKHiMBWB+r9q+pLquoTq+qPWxTqme0DvkNV3bGqHlFVX1RVv9miUg+vqr9uKb7nVdUNq8rvRKI4Q9eoqmtX1adW1eOqikPk7yJWd62qF7XfiWxdtzltj+6B+q9VKYo/RCWLTEEgCASBIBAELjUEtjpQHQ4cJU7O3+wAc9PmDPn1Q5pDJVL1F1X1DVX1tKr6hJ4DJeL0BVX10Kr66JbG86/XqKW6W0sJikr9XlVdtar+vjlk3VvHgbrUtDOfJwgEgSAQBILAgSKwrwM19rHUPt2vqh5QVS9saTupOk7PG1vxuFqpX6uq+1TVE6vqxlX13Kq6WVU9qaqk++7Urv+gqrp7VX1Fizx9WFVdtpPCiwN1oEoWsYJAEAgCQSAIXGoIXCwH6jRwEsV6ZVUpGL99VT24FZKP3TsO1GmgnnsEgSAQBIJAEAgCswgcsgN1taq6TVWhRnhUVb1q5tPEgZr9unNBEAgCQSAIBIEgcBoIHLIDtfbzxYFai1iuDwJBIAgEgSAQBDYhEAdqE2x5URAIAkEgCASBIHBFRuBiOVDvXFWfV1WP7NUtoSd4eVW9YgTwfdnJE4G6ImtyPnsQCAJBIAgEgTNE4GI5UOqWFH7/Uo9gk1P1z1V15cbX5F9dedjHFYr7m2ves8dOvoZtPA7UGSpO3ioIBIEgEASCwBUZgbN0oG5dVS+uKv/idnrXqrpOoyy4UVX9TJuP17GT37+qXrPiy4kDtQKsXBoEgkAQCAJBIAhsR+A8HKhbtK46kSds45jKEXI+papuvsNOvuaTxYFag1auDQJBIAgEgSAQBDYjcB4O1Cc1J+l9quojmzMVB2rzV5gXBoEgEASCQBAIAmeNwMV0oO7R6p/e0ob8GkAshTfnQHXs5A+rqmesACQRqBVg5dIgEASCQBAIAkFgOwIXy4HaLtH2V8aB2o5dXhkEgkAQCAJBIAisQCAO1AqwcmkQCAJBIAgEgSAQBCAQByp6EASCQBAIAkEgCASBlQjEgVoJWC4PAkEgCASBIBAEgsB5OlD3rqqrn+JX8PWnHFF7h6pS+P4vjeRTjdW+y/3c9/+1n9O4Z+QMnkhoT0OXop9HOGp8OQ082aYf9zqt76iz98h5et8RvQ+eVyw8932Wv/X15+lAcXhOcz3wFD8Pg7pWVX1UVf1NVf3PqnrjnsJeqaquW1UfWlV/UVUvbAzs+9w2cgbP6OfRJIN91ttX1YdU1fWr6vVV9ayq+od9btgOXx9eVX4QArP3v9vznpEzeB6yfvIn2NEN2pSRZ56SHV272dHfNjt63Z52dGovP08H6tQ+RLvRaXbhGSnzyW2e38uq6r+uZEUf+myY1z+3qv5DVT2nzQn8v3uCEDmDZ/Szal87EtW5WVV9WVW9uqocxva9J9v8tPbzJ4377lV72nvkPMI0eB6mfnLw2dGXV9WfNzv6qz11nh19SlV9epul+/Cq+tM973lqL48DNQyleXw20//YPN4faka7D/Dv2xjXP6eqjKuxSRuuvE+qIHJWBc+q6Od+dmST/tKqunNV/e+q+t7GWbePbV6tPUi+qO0h7P0FrSRg6z4SOauC5+Hq5ztV1ZdU1Tc2J+eeVfXHez7j2NHt2nxc92JHz9/Tjrba34nXxYE6CaVTnvTdV1fVu1WVgcYcnke0uqWt4N+knUYxsKuBemRVCXH+48YbRs6j02jwjH7uY0f2wA+sqm9og83VPP6PqvqVPe39hlX1mVX1Ae0+v1NVv7dHSiNyBk/jzw5VPz3G2BHnqZPzj6rqoXvakXQgO3Jvz80ntrFv+6bYNz52j78sDtRJGNUq/dt2Gn18VV2zqnjW39eKS7cCL5pldI3w88dV1bOr6tFVtTWfGzmDZ/RzfzuyB35MVd2rzeW8RlX5uVs7PG2198+rqhu3KLM6qFe0NN7W1GDkDJ7mxh6yfn50e07+RpPzg6vqW/e0I2Uvgg+yNWqh/k9Vuf9WO9pqz4OviwN1EhZh8s9uIUOpEc6UjfCb9ygCvXJVfVtVuffjWm3Va6vql1pB+ZYvNXIGz+hn1b52pG5DJJN9f0dzpqSJvrKO7r1lOdyIYGsYeWxV/fuq8juncY7UlhU5g+ch6ydf4lPbc+7bq+p6LZ13hz3tiB06gAg23LQ9Qz03OVTnvuJAnfwK3qOqvq634VGEW1XVz7Yahi1f2vu3+qe/rKonN4fMqfcXquolW25YVZGzKngebVTRz+12pJP1m6rqvarqp1v63qZ/WWv22GKe0soKaa3fag+Tj28RKGmNLbVVkTN4HrJ+cvC/pareu9Up6cb7qmZHdH7Lci925N7sSFf8J7TyF2n2LXa0RY7R18SBOgmNze/728nxKW1DVWD6+1UlZbJloS9QXKc9+qntnneqqp9rFAlbFCFyBs/oZ9W+diQy9DPN3tn3+1UVipWu7nGLvX9YHRU7v7Kqfruq/k17mKiDetLGjT9yBs9D1k81uQ9qjg6d1zSlHood6ZzbstQisyNpO5kbdiQiJQjBlrY8N7fIEQdqIWocyg+qqvu1mogXV5Xo0e1bGPLBC++ze5nWzlu0TRqnlM4C6Zdfa44Zkr01K3IGTzU70c/97IjNKXhlh9IjuuREdu/YClZ/fI1R9q69UYsKPr0VvF61qu7eq3tE1Ll2Rc6q4Hm4+ommh6MkfadL7iotk6Po/QFrlb1dr1b4M5oT9rtVxY6+s6qe25q6ttjRRlGGX5YI1HFchAqFCb+nhSOliFAFKAjlWP3ARq/3CxvBmM47vD26++5RVX/YPOm1RICR84iwLXhGP/exI9avKFcEitOEu+ZdGl+bVAGnygNg7cJZowvPqfl/tXtKE7o/nUWGuHZFzuB5yPqpS05G5Wsal5oaXQXg6oc1ZGyxIzVVDiMiWs9rdnSXqsItxY72JaZda4Mnro8DdRwSdQZdBx5FeFPrwLtlHfENaXXeogjd63QP/HUb76CoXCfBozYU2UXOo+8heB51iEY/t9kR61dD9p+qyiGHvXe2xeHBP6N1eu3yOg+UX20PE+kN+4n91sbvYLZ2Rc7gecj6KfBw/6r6gkbNw47U/XGebrvRjpS9CFywoz9rz011Ve7Njkz0ONcVB+o4/LrlnB79yN9aNj9etBOq363ln/BlC2s6PWhD7V4vlytd8OsbFCFyBs/o55F97mNHXs/5dI+u6NueiG6EU+V3f79yh+6cJSdwTSLdKVkth4eBjX9LJ17kDJ5U8VD1U4mKQ4LDQyen7rkfbb/bYkecJanAn+/ZEQcNtZDAw7l34sWBOr47yuNiIFegrUbJki4TjudA/WBVrR3HIF3HC1evYvN8c7svcjAeupZMxaZrVuQMntHPI4vZx47sf1/R+GWk7buNH/2AdP13N2byNbYpBWivQMDLgRI1sG7e2rC1Y6sRWbMiZ/A8ZP2ky+zoI1qtX2dHHB37FDta+4xjR6hAZBqMUusIp41CQwvymJYeX2NHp35tHKjjkCrUxAdjnAPHplsIwr62qhSRr938FIxL16l38qV3KUC5Xd198sYo6tesyBk8o59HFrOPHdn/jG2RUlMH1S3dPor0/U7Tx5ql0NVeYcgzB6qz944XBy8U5vQ1K3IGz0PWT7rMXtQmoVroFhZ+He1+h3ZgzeqaOURwRaA6O/IsltpTF/WMNTe8GNfGgTqOqoJxjOMcHZTx3dKWLJzY0civ+S6wsdpQ0SBoveyWU67UHn4pBXJrWjIjZ/CMfh5Z0j52ZP+zubN3jk23tGArINf5Y6Neszw0TB3Qev3fei+UvpN21YqOfmKNvUfO4HnI+knNPcfYkZ9uyeTomvvvraFijR113e+Gexur1C3NFOzI/sc+19jRmvdfdG0cqOMwIe76L63+QfdMt2x+Uns66BQur1kIM2/Tvuyn9V7ovX6kKZ7o1BpFiJxHxhM8jxQq+nm0ga+1I/ufWor77ESakGoi01Wk+otrjL2qrtPsHSUC8r9uuaeOQTxQT1jZjBI5jygmgueRNh2afpKJg3fvnUiTgz6eNpEpabg1S/2U56bsTN8pc0/pTIcQh5stTV1r5Ji8Ng7UcXicPH+5nSD7nTJXb4RevqyfWom+fC0eKJtmn5EVMZ7QJPIxjsAaRYicwTP6eWSI+9iR+kbktrduTR6daUuRS6/brB1y1iwpxU9p3DWizt3STKJLyUBhdVBruvsi5xEhY/A80qZD008yebbpVBcx6pYCcHbkwK8pY81Sd4zGwKGIs9S3ox9u6Ts1xWvsaM37L7o2DtQFmGAhPKhuAe9Tv2vARmqT9XeF5GvWZzVuKR56v9bJ+8lrIx8TTVF0umRFziOuruB5QVuin+vtCHqaMdRRKEztc8r4vXmYmjyk8tYs99K1K9LUr59it/dtha82/q4odsm9I2fwPGT91HHK0REo6MupEJwdsQd1wGuWexkijHW8Xz/FjjR4vKg9O9fY0Zr3X3RtHKgLMDnlCRtK4fGku245V+ikw4jakYItArddpOBNXcQjdjp6YO906wQsFLmUTDNyBs/o5/GDzxY7cgeRXDVJhpR23XJ+j1tL19wXt06gNfbuwKRgXGpQ521/6UaSFrQXvG7FTSNn8Dxk/VTrpCbpE3fsCN2OA4XyF116a5b5njdodrTbZKWu6jWNFmiNHa15/0XXxoE67kBdv50StUb3a5KkCYTlke1prVyzFJDbkB82QKCnJgJBmBPpUkXgQEXO4Bn9vGCFW+zIqxWgG+Py73bsHZcTJvLvatHoNfbO6TLDS+Hrn+68UPGrlIOos7qQpStyBs++vR+afqIrwHHIZnblNI5Fh55I1JrlWSugoaRm147u3G7EjraQ0q6RY/LaOFAX4FGjIGzI4ZG37S8Ky5PWXaO4tB+dmvsyvrV55dhUXzuwofqVTRxD+ZIVOY+iBcHzgrZEP4+wWGNHrpeik5JHnNtfDikf21qwEWD2T/9zNuqkrYMIPcmuTd+hzcH0sNl9KEzdN3IGz0PWT3V/apzULO3K2THof/5KOzJ/VnOMGuFdO/I3US8OlG7Xc1txoC5AL9zIS8b4a/TKriII84s+GfGwdAYPfHUMGNny0Kp64859KYKRDzh9MJUvWZEzeEY/j1vKFjtyB2l5bONSDP3Fbq/bHgqcnt2Dz5idep2uI1xQlw1MLdBVhGyQA/UnS4y9XRM5g+ch66d0GzvZDTywB12pP9bsbI0dIaPlJCmp2WUxV6NsWgAH6qUr7OjUL40DdQFSaTYbsXDkPQeQFuYXOlRUujT83hWO2iy1Q+92DKhlMXsPQedSTzpyHj18gudxJY1+rrMj6IkoOz07FO2ua7cxFF+/4nDD3r+lqhxydNz9885NlQHoykWFoi1/6YqcwfPQ9VP6zqDf3SX9zAmS2VkaJGBH7qWLT33jrh0JcsgWcaD6dENL7enUrosDdQFKHQNdbtX8nt0llyt9d7/GB7XkS3BPTMc6BuRyd6kKpAVNrH7girk+kTN4Rj+PW98WO3IHzg7GYzxQu+tDWj0kJuWlkwIcbmz8Dkr/ecDeOblOz+oh+5Qmc3tJ5Ayeh66fOoHVOu0uRNKcIATVnoNLFjvyLFaaIHq1+9wUdGBHHKi1DOdL3n/xNXGgLkClVVjRqDEu+Jl2F0JMwxIfsmK8A6XSMfDcNjR4lywT1wWn7McHOnbGvsTIGTyjn8etY4sduYN2aNFkJ+Tdpf5CJBoBIA6iJYszJmJlYLhD0a69G0OhDIADtfSekTN4Xh70U5nKTwwYibFIDij2rKWjV/BcdbXGUuG7doTGxmQQ3axPX2KYF+uaOFAXkBUuVFCK6I5nu7vkXE1t1/aMmn7JUkwqPYAjw5e9u4yI4bXz0JeG9CNn8Ix+HrekLXbkDkhxkdiKDu8utYmGgGvPXjrORc2G2g1Fr9Lyu0t5gPFNHCj7zNIVOYPnIeunwwLnSJ3v7vIMFETAi9Znkp/SfXbEQVJr3J8B2L1GVItteqYufRYvtbVV18WBugAXr5enr6AbedeQs6MWQciwTy0/Bbh2ZidOyjWkPJRLqF8Hw3MWfnORM3hGP48byxY7cgez6tj7kG2+X0vHOdj0Z9pNmSmiXXuEDruhhwk+J1EvNVAeKEvHN0XO4Lmrd4emn/S9P6+vk5fOCyK8sHVOL3nMsSPNHWqmZHx2l3tKCaL/MV92qR0tee9V18SBugCXzhnh+h+qqmcPoOj0qFtHiq8/3HAKcC2ct22nTRvm7uIM2cBFvjhmSxQhch6d3oPncW2Kfq6zI+jZfM3vGkotdNGkv2nzKpdsrHhr8EC9ZGRmJkJeaQ4RxMcttPfIGTwvD/opTTeUTjPGRQG5DjyDu5csDRzoQ17eSl92X8OOlL0YicRpW/LcXPK+q6+JA3UBMvVKToaK14ZaI4X0sYpjDB8KKw6BrzMBNYKH/VMHLlAkRwmMeGAkSxQhcgbP6OdxY9piR+7goKT9eqi41cBWXbkKWjWOLFnqJI18UvM4dBonJ34oJ2d2v3T+ZeQMnrv6d2j6if9MlGl3eV6xI81PS+dKqhVUJK7Dbijbw44MD3cIkcZbakdLbHjVNXGgLsB1tRbK/4KRdstuoLC5P0NdUEPAf1JVfXKrmzKyZXch7EP+xzMXVdlt1xy6Z+Q8qkMLnse1I/q5zo7sfbrrkP+9asDQFISLJqm30Em7ZCG8xNmkQHwoQsref7LVbXCilsy/jJzB89D108giFB1DcsqysCN0BiYGLFmoRdiR2mFR4qHnpnIb9YuiuWuIrZe8/+Jr4kBdgErIXm0JfonXDyDIceFcmWsn/7pkYTjWcuk0OlTjBH98Rr/eilWXKELkDJ7Rz+PWt8WOODPGKCHMHCLG1ayBYkQUGY3AkoXjyYFJYavNfXeRU1u2IcM2/l1i3aH3iJzB8/KgnzrjhsaRSbexI3Py7rrEiNoQYVxPggq/P2JHosIiVEhpl86RXfj2yy+LA3UBK4WoNj4jHN4yAKETqbCiuqalE9pdLxzptDkU3vQ2ok9Oq8KRSzbUyBk8o58nDXStHSG7NNiXPQ3hKeWAYdlwYPQlS5ZN38QCXXtDdZTuIV3/srbxL5loEDmD56Hrp3l0itqH5JSxEUjwLNTFvmQ5hMjesCNRqKGlGUM9so7WJXa05H1XX3PWDpRN6QGN40EdERB4kgo5FV7us9QPbf08XqdjTg2UMPwuYzi58C9hDvfF7o7SGJNbzZTQpfuOjW6QDlQzwcnapawfOsFGzuAZ/TxpcWvsyKtFlJ1gFd8P4clxuUXb9DWCLFmGkItYqcsYY0iWxjBJ3savQH1uRc7geej6+fyW6h6TE9GtQnKF4UuWg8uNmx09b+QF390iXrI37Olc1laHY4uwnfP06pay0t6PA8mQQWkzYfJ/3HLj9pp9HChhcpEiTp0T5FAxtw3VaVQ4cqknrbCOZ64NeWxUi+6FDpOhEGgfksgZPKOfw5vEGjtyB0SZv9tmdQ3Zu6HdsBZtZvdLlll312/t2mMHQsPF1Tra+JdMko+cR4PDg+dxDTwk/ZS50YE6ZkcY+JHScoyWLI0YJn/43tVXDa1vbu/HjkSSz2WdpQOlnVG7vuG66ng6B0pkp/v9khPZGFD7OlBOjsLrwvBDS+U/BUBlgKNiSeW/waKK6BD1jc0BskGrfcKjMff5OVCRM3hGP08isMaOvFrNBjqSG4zYe2dr9iYpiKHT9e5LRarUVP1CSy8M3dqeoL5K88hQ0e3uayJn8BzSo0PRT/ouQODgMCYnZ0imSaH5EjsSqdLRir1cmm5oiWih9GFHeNfOZZ2lA+UD8kJRu6tXuF3jRMHYiz/CzKl91j4OFOfo5o04T652TBF07Gh75vwtiZbxkt1bofhYmFE60IaKgwod/tSKnMEz+jlsIWvsyB2cig38Ff0eWvZGGz+SWyfiJfWJ2rVx2Njfxk7F9g97oNP1KxdseJEzeB66fkqfa6AYk1NdseYJJTBL7EiQwuQPDPxjgQe2JjrLeXvFAju6KJectQPlQwiL9ztURHOGRims/cD7OFDCocL0PF/yjCmC4jYFpbii5tJt7oHCXqsyT3qs0E06UKs0Xou5UGTkDJ7Rz2H7XGNH7uAw5IAzlZ6z8XOy1DL+7YINyd5gUzddYOx6e0wXVRqri+y/VeQMnmOqdwj6KaokLa3+b2zRdwSy6AyW2BFbNp5JvfRYVuYLW9kNB2rf+ukFpj3uFGx+8YG9cB8H6krNeVLsZojh2FJALnT4bW0I6RQEnFMFo1rOHzTheVMqw1Dxw2irnlqRM3hGP4ctZI0duYPaSym3qQJxG7/UA5tXpzhn79Jzuvo4XWMNIYh1Fcg6NOKhmluRM3hOOSbnrZ/q/r60OUdjcn5EYw53wFhiR+zNQeSHJ+zIwQc1gkjuWIf7nG3t/fezjECpgbJpqCcYWqrtbWZbvcl9HSgF38KGnKOxdZMWgZJunAu/y1ErbJWWU5w+RppnQ1V3hZp+rNC8k4cDFTmDZ/TzJAJr7MirpdKk76YoCuwHaqC+a6KLtpOEvX9j6+6T9nvTyJfE1u2BRjjpXppbkfMojRo8T2rKIein2Y8CD8paxpa0NkdPRmZoikL/dexIhsdBRJf+2PcuG6QmWR3jWKfenG3t/fezdKAIK7epDb9f79T9TkeMvObWbrx9HCgddr40LcNTrMNIMSmKEP3Q+If+FyLdRgEUuKmJGCs6pwhOzxRsLpcbOYNn9HN421tjR+7A2bGxS+ONLZ1FKFbYJqqRqaU+0b00jdjfxoplHcKk8ThQf7RgB4+cwfOQ9VPt4XXawOAxOT3zHURElJbY0V2qSsCF7Y3ZkegTO+JAmSN7LussHah+F14/r9n9XpGZXKouvblutCGw9nGgzLu6W2sv9iWPLUWlX9UKvrEJTy0EYhjLDURU3zQ2506RqLRMR7A3dc/IGTyjn8MWssaO3OHuVYUleWq8hI3ffmR+3dBA174kDjccKP96WIzZu3S9Q5gp82Mkgf37Rs7gOeWYnLd+iiohmSbH2MK1xhnqGsamnnHshwPlICKDM2ZHumfZkazWM8/Fe9qDeHKLvB0PlPx/l6qTG9WGCACU7EalnEcEimPiZK+I+7KJD6dVUwqNrEOjGvovpVQ2ZzwWisjHlnZNmyRlmUtfRs7gGf0ctqQ1duQOnBx1StJDY+tDGg8UYswnzmx6nDHRIhv+1NBUaRd7HKoDM/PmVuQMnoesn6KtOut+aEKR1TMhvmRHQ7Pt+i9lR7JBylUEFcYW38GBhQM19yyes7HNfz/LCFQn5G4XnvZHToYHwz6M5PtEoESLKICahClnB9mmmgSb6dyGiuvq26vqBS1cP/Yl6Taw4aJ4mEsLRs7gGf0ctqQ1duQO0vC64KYcUhs/Z8dMrkfN7LKmzn99q9nQPTS2nMYRCKMteeqCnTtyHnVjBc+TCByCfvpulJ4Y7ju20HbI8LAjEzemFjuSkZG604wx9b17ZuJPfMoCO7ool5yHA3VRPkg7+W39PKJjirjVYYmIjS1er6K5Z1XVo2c+yAe24Ylyvlotx5brFJmLQnG2plbkPMqhB89hLYl+LrMj6EknOLmKBE1t/KJKxrLo9plaV28F6ehKdNSOLZMJ1FRxoDA4z63IGTwPWT91mEtv0+ex9QHtWShAoWZpar1vS82JaqE+mLIj5QwcKLNkz2VtdTjORdiZN90nAsUxUaeEFv4xE++j6BQPj/bjKUfLLczA046pwE3Kb2y9V7uXaNVcN0HkDJ7Rz2FLWmNH7sAJZ5dm0o0tnUCoCTCGq4OaWk7Z0vu6bj1Uxhb2ZASBHjhzUezIGTwPXT85RKJKnp1jy+FCWg5NjwPB1BJQYEf4ojRfTdmRgwoH6vEz97xofz5rB0rHnQ+8u57QOvS2FI9399rHgTJORlfMA6vqyRNoq4nwGSgCdvGpJRog3ae+aypaJS1HARTQz3XlRM7gGf0ctro1duQOj232/riZjV+hqlqpqVSfW9gbHK503SoQH1vqGO01HCh2P1Yk270+cgbPKcfkvPVTalvw4bcmdF45yx2r6g0zUSW3kOJmRxjIp6Ja7Ej0WA2U956zo4viRJ2lA6Xbrqtz0rYvXWaEizymHOq+bOT7OFAK136jFXJPVfTLOSMNQ4455R37srDEKpaXJvjtiW8P3YGNFA5Sg1Mz9ox8EYGQ7oucw6AGz+jnnB3Z9+w9+J1+f2bjR62ioHWqMNwttHKzd3VVUylm+snBco2Nf2o2WCcne1c/MrY8oCJn8JxyEi6WfrIfz66pOiT0QIIJnJ6ppg3yow5hR3gWp3wCdsTBengLUMzZkWiy4d2n6midtQPVDQ1GgNXxQY3RG6z1GPd1oJxEtU+qdxhbcrnGOlhThY3+/vFVZddMfrYAACAASURBVG4ZxZrK0SIO4xSpi+BsmdQ+tnT22UgpY+QcRil4Rj/n7Ahnk3pD0eSpqK+0oAOTFIQGl6mlwUQXsfva1McW/dQIILIkMj1GsOv1rnU/9v7siXtGzuB5XvrpOYTaZ4qSQ2E4O/L8nOKxo+LY/9mRxrKp9CXbkCqXvZJCnLMjHI5saMkw48W+x1k6UB2Ngc4TdUEKxO7aHA2nJ5vZeaXwRHYUdHKO8DaNLQVuvGOOzFSLpdcjzOMoSrlMnR5dq71TRIsMU4pAEW345mNFzvHvKXhGP6fsyOkV6/8tZqhD1CvZzE2cV6M4tfDSGDqMH45zNLWkPNi6jX9qKDk5TaP/tJkO3cgZPM9LP0WKPOemxqngdDK77nozkz7YDKoggYfnLGjUUkvosOTAMmVHDkzs2HPhLYu9owUXnqUDRZx+tAnoXT0UKgMh9X3WPhEotUWKRLUsC/ONLSc96cdrtLDllLyYkW/e0nNzfC9w4G3zpseo670XD17hqQGOkXMc/eB5FFaPfg4j4CAizceBmprN5WD1uW0AOsblqeWEa6CqfWyuOFzHrweEjX9sZp73clBzsrdXcqTGVuQMnueln+yI3k+NNlMiw47MkkX1MbVkboxl8sz0PJxayLd19im/mbOjb2pp+Dfv42TsvvasHajTlH33Xvs4UHKz6iF8IXNfBE9WuB6vxdSy6SENdRqdo5pHY+BEKpWn0G5s4boR9nffyDmOU/CMfk7ZkQOQtDr2crQDY0vU/DPa5q8zaGrhtxMpUts5F3HGOSeCbON/3cRNHZjI6TCmuy9yDiMQPI86185DPx0WzHf8qxn9NP/RYGw0QFNL5sa1Sl/maD66UWmCD6+dsaNu6DJ6hFNbZ+lAzY1y2TrCpQNjHwdKkagv14Y2lSMVqXKdOTxzHr9wvnCkMP1UvRL5OW6u0VY95RhhW1Z4iiIhco6bQfCMfk7ZkZQcRmSdc1N25GDl4eDhJPI8tTg5ftRSzo2WEB1UrqD1+zUTN1VQS052/w8T10XO4Hle+ilKpGFqSj+NZxHt1YknADG1DCaWYdF4NTc+ic/gAMKO/nripuiH0C2QYUrO1Y7VWThQHCfV9LzKsSVisHWEy2k4UHKkQvBz3Ts2KvlZNUg4nqY65gw6NLaBU4Y3amrxpFEj6MyZOpGKPCmc45hNvXfkDJ7Rz3GL+4TWwWPjn7IjBytpB7Wat5m5VvTpZq3OYo6ORPRaLQZ7nzq5s3N0KaJbUxG1yBk8D1k/1fKxjW9r0dwpm+M8KX1RrzQ3bxb1j3vhZDSGbWyxI4z+nzVjRwfpQHVCnVa33diHFIFSjL5lcSQVYk45L+6r8t9AYXlahaBTimDT017sNDq1Sbqv/LDcrIK4qY3SyVkxnvlDUytyBs/o57iFLLUj+4IokEOTsRJT9q6I3EmXvTsMTS0PCYccJ+yp1IP3FllAoTBV/Bo5g+eh6yfbkL2RSpuyI4caHIpqoObsSHQY/5trEW+OLe/Nho2bOc0aqIeeRQRqi0Oz5TUcqCGSzi33GnsNvDAOS6HZ/KbC/740NRQ4rqacIu9lQ+f0qIuYKiJXu+G+c7N/ImfwjH6OW/4aO3Lwu2GrbZqydxxxrmXveOKmFhu28Su8ndob1EDh73GwmuO5iZzB85D1ExfUjVpX+pycAg+ehXN2pCbYM1aDxVRqjh05jNgTp2iC1vgK5Lv3WThQS1J4581EvhQ4eCkOR7mgm2CqzsIICF17QvBYVaeWdIsvGZXBVLQKhYJolfefWpEzeEY/xy1kqR25g00ff53XTBWcixSJQilHQJEwtW7XNnSM5ByusaWOUikAtum5k3PkPGJ4D57D2nTe+sk2RJ9kUKbsyN/pMjua6jz1KdEOiVYJnEzR+nhmsiPP2Sm6g6V+wNuuOwsHarVQG1+wTxH5mrdU/4RqXh3DVOGatkneMXqEqeu8t/shEBNinMrl2kh1DlGEuRU5g2f0c9hK1tiRjV/KXAHsFHWIQ43u3DkbJpF6FWUARlFgLh9bHC31IAh+5xyoyBk8D1k/NUJgIWd7U3Ii3FSzhCdyLvAgLceOOM4vmbAjjpYGj2+OAzWO0lk5UL4IDg/mV0NGxxZiM2k5UaUpj9vr5YZxYf3oDC+NYladQ4rn5lbkDJ7Rz2ErWWNHNv57NSLNKa4bnXpOw2pR5giBRbF1BBlFMdVg8jWNfPA7Zgh2fcrIGTwPWT8FCDRLKSSfkhPNgYOIur85O9KYxo6MRpoi8uS0IfE0EmmqRGbumXri72cdgerYyL+uJ8lpdOC53Vk5ULoJnDY5O1Onx+9uhZ8GFM/VQNlQhRl56IaRji33VDsxR4fv9ZEzeEY/hy1pjR3Z+O/R0nhTm7QIlfFUJhRMpfZJpFXb6Vl02qiWsSWKbYzMfRY4UJEzeB6yfupIdxDhRE3JaSyMa0V95+xIhyw7EsmdogpyYHq/qvq+BZHcVU7UWTpQnfOE+bffRYYTRWH2edIYrAENBxQHSpfT2JcGV8pCAYT0p8ZKdM7OV7YveCrva3PWKTg32NQ9I+eRwQTPk9od/VxuR6JKSHZ/coIQF55qzj642f1cug1livl2KEmeO3Ng8me1I3MjKCJn8BwjbD4E/VTEjbeJHY3NdSSn8hTF4YIEc9EidCQiViK5U9QhIk/uLegxZ5trfIG33vSs1iETaa7BQM6VA4XDBW/U0MIr5dQoBCmXOzfAUNGc0KaT7lQRpHsZsiiqNbciZ/CMfg5byRo70jHHNn+1qkyeH1pS9d9YVSa+OxDO2bt0m7om0fcprhvF6xjI2fucAxU5g+ch66forFQ0Oxpj6mdH3UFElHjOjqTl2J3h3FPDjEW91IOanTcXzJh7rh77+1k6UN7Y5oKJVEeLoi+nJiRYGLjnuI3mPthZpfAUayruxAg+pgjSbJwhBeG+tCneC59LyFJ06Ttn0oI8baMiFM3NrcgZPKOfw1ayxo5QlkilPb79DN0Rp5ON34w9Uc85e3fCtj+ojxxzcr0P8j9Rbp28c+3XkTN40tFD1c9rtiyTjnss42NyikBJW8vgzNkRp4wdieROzZt9QEsbsqO5g8jcc/VcHShvLnrT52tSkI2pfN91Vg4Ur1enwFMnNlRDQDlD2pl9uWSbWjhkulqwqW4CrOawMx5mbkXOo7Rw8DypKdHP5XYkLXfndsI1/HdoGZaqrtPByVy2OXv/wJaWE1maGqKuRsqsMXY/50BFzqNIXfA8qaGHoJ+eccafSd8JmIzZkVrCd2+1UnN2hPpHOYtyGs/jseUZLPBgZt6cHc09V8/dgVol4IqLz8qBEjXjQD2v0c0PiYgITMvky1qB29zHcL3QppPuiyYu5r1TmCfP3bBF9yJn8Ix+nkRgjR2pz7Sps+Uxol6RJx1zli68uYUfTmTaz9TgYWUC3tOJfS6dETmD5yHrp0OD6JISFV1zQ4sdKSI3+mVJnS87UlPFQZoaPGwOHlsyp/Jy50B1RJoAE32aa02c23zG/n5WDpTwuxSk4mRfzNCiLEL6WpTHrum/7ipV9ahGTyBkP+Z5Y1LlmE3le7v7Rs4juofgeVJDo5/L7UhdE4oCI1dEiYeWlIOicEWvmkbmloiAug0nZyfjMXt/bHPI1LbMpTMiZ/A8ZP3UBadRCqWPg8PQet9mR9Js0tdz612r6sEtSPGkCRt5dFX9eDuszB1E5t7z2N/PqgZql75A9Karg1ol8MTFZ+VACZWT3Ybqyxta6hEoi4e38Pvc4nEL1SPnfM6EIjy/MaoqJJ9bkTN4Rj+HrWSNHdnUHfzslWophpboj1IE87ichucWe3ew4kSx+7FN3cgme4IOozkHKnIGz0PWT1kWdkT3dcMNLQc71xjhMuYM9l/nXuqB/Ygqj9mRQ4pOWl2Kc3Y0Z7vn4kDtCnUx6qDOyoHyJaPF92WNnTYVhUuf6bDh/c4tm7MCQJ17ikqHFME10gg4ZOaGLHq/yBk8o58nLW+tHZnhxd45KGONLg5MDlUoWpY0eJDBdRpo2P1QYatr7AUiWw5Mc/UgkTN4HrJ+Srexo46PaeiZiCSaHRlnhttpbrER1z2iNXWN2ZHMjTS88phLwoHqAzNGbzAH3u7fz8qBspGaq6MQF/Hl0EJFb1yDzgCe8ZKlsE5oU7h+qNWSt60oXXH41AT37r0iZ/CMfp60vLV2xM7ZsoHfTrFDS12k7mLEpQpVlywnbNEn3bxD87lQoYhGf04j151zoCJn8Dx0/UR66fmFzmBoGZzNjtRJOVwsWRoHRGofM2FHIk/uiwF9zo6WvOfbrjmrFN6uw6TrDg27dVrpvLNyoBS6GXgoRTbGCG5enc/HGfLlLllaLKX7bKpDBGJXbrQIOg+WcFlEzuAZ/TxpeWvtSL3SZzdmf5wzQ8tDweBfKXsb+ZL1Yy1CrfZxaJI8OaUajXiam6Xp/SLnUcQ/eJ7UvkPQT/VKn9Vm0ulqHVpGuHCysPP7HpcsTRt8CJ3pQ8zlV2p2abTZ1Ay+Je914pqzcKC6IvLTdph2P8xZOVA2Kg5U50kPebQo5qXaFLZN8VP0P8NlrRXTBvzGgW+TQyQEqd5iiRcdOY9aW4PnSWWKfi63I9QEn9Y2djO1hmwPEe5nVtUz2wFoyWbcseSjRjBdYHddtbXkG8m0JOIcOY8ae4LnSV06BP3ElWbAvWen+uAhO7phO6xIXf/OEiPqzaT1vQ/ZCTvyHOBAnXoD21k4UArIeZ+nLvwOwGflQFEEw39vUlVm7Awpgg3XuAZtk1MU8/2PgEhTjZMUwJAnLeJFqaQSljhQkTN4Rj9P7sJr7cgJ1uZr01e7OVRDYWySh4Ni1act3PjxxCmWZe+vGXiNg5KHgr1kbhi5l0fO4Hno+imogMrgS0bs6MYtc8OOpnid+uby7W3WLDsaitSyI38TRdbkcarrLByoUxV44mZn5UChm1cTYQAw8ryhDdVgYOMahBWnBif2P455PRwnxaVDG6aIl64djtmSFTmDZ/TzpKVssSNpNJQY6iiGGjxuXlV+RDunRrP0pUEqKE2H/20otaARBceNvYajNbfYe+QMnoesnw4a6rSk6YbkFBm/ZVWh7xibl7drB+iCRF/xPJn8sbvUJ8ruqFseivTO2dXk3+NArYcPZrxZxZ04mYbSbboN8DD9elW9fOFbyAtjYNUGPeQp2xzVtDiRLlmRM3hGP09ayhY7UtOo8PX27bS7e1c26b428Ski3P7rDEF1OkZ++KoBg3ZQcqjynkM1UrsvYe+RM3i+YUCXDkU/Df81406kbEjOT2mlL+xoaeABfQiWc3XVfzpiR5y2joNqybNz8TVxoBZD9bYLYaaeS9GoOTxDqUmtx4q9RZO0Ni9ZFIG3bNDpUCiSw0bxhD+XrMgZPKOfJy1lrR25g8HcBgqbFGC47+5yT86L0Ss6fZYskSXzKrVhD71GSpC9i1QNPWyG3iNyBs9D1k81Tg4FmjGG5FRkrjSGHenEW7I8hz+usZsPBSs4ZexIOcNQacyS9xi9Jg7UNvhu0ULr6OaHTo9SJ4q+KcJQfcPQu9qE1Vq451BI30lUCHSsg2HonpEzeEY/j1vGFjtCSyKFd+8RZ0dhrIcD3q2lnT42dk4SslP0B7vLPaU0nJ6HotxD9h45g+eQM34o+qmsRSRXJmVITrXFolRYyJfakZQfOzIgfGjuqbQ7O/qehZHcVR5BHKhVcL3tYuF6J0jptl1WcJjepdU34HpZ6vXqtrG5I9Mcilo5/WJzvecKkSNn8Ix+HjeYLXakvdqpGU8bqoL+Yu9Ic11zvxV1FlJ0os4/PZL2Q/wnNWE48RBP1NA2EDmD5yHrp7o+A95xN6Ee2LUjJJoOAT+8wo44XCJMDiLoD3aXztlrNs7GpQeRxY/YOFCLoTp2oS+NA4UBdbfYDQEeJVEkR1HevPAtFLfKDztxDuVyOVbaNM30WboiZ/CMfh63li12hOBPVJm973bZKd5Wz/ThbYL8UmdHul5aUKH47sOExLqL7CH+vnQPiZzB85D189otg8KOdrvs2BFH6KOqCsXHGjsSHXYQQTy7u9iYBjOHn6X3XPp8fet8p0tlnVUXHrw+tlEZmKS+OwUa07EWZYWf6pmWUA645zXalyxXOxTelB4wRHjJqIjuO42cR7nv4HnByqOf6+3IqBZT4nG6oSbpLwcmp1z0CKLDS+1djaR0vXQFpuTd9YOtAUWR+RLiXK+PnMHzkPVTJEhklR1h4N+1I0O7OVkCCUvtyBBtESsHDc/H3WW8DX4wdrT0ILLYJ4oDtRiqYxc6PcrX6hTYnXWnk04KT+2TFN9SRTDL6iHtVPrSgddpd1ZwunQ0DIEj59EpP3heUN/o53o7criRbjNKiR32F447TSNmfUm3LbV3NZIORU7GHii7r/N7vG9auodmfA3tXJEzeB6yfprPKlqrLlPX3K4dsbGrV5XDwxo7cgjxrH36wOs4Vk9udrT0ILLYK4gDtRiqYxcaesiBMvRwVxEowNc0JVkTLUI4ah6e6BXHbJdfCqu5ArylxJwEjpzBM/p53Ma32JEBqKhJODJSBf3FcZJ6EImykS9duGts+mo3pF12eXFQoNj8h/429h6RM3gesn56Nqpz6spbdu1ILSEC6AcsNaLGAaXW2Cg0GaFdO0KJABN29M8r7rvo0jhQi2A6cZHwOweKk+Ok2F/ClLdrUQ8O0Zol/KpDQS63rwi+JwR9CMiG6qPG3iNyBs/o5wXr2GpHosNsz4Bu6YL+YmOcK1QDnKE1S8RZakEZQD/KRE6R5ru1wtilE+QjZ/A8ZP102GBH0m6itf3ld8gupdl+do0RtQ48w4dFmnbt6LdbXbHi+qV2tPjt40AthurYhRSBA2XOnzqG/lIEh4n8uVX1+JW3t6H6MYS4H250WjVnS8vmmpE4kTN4Rj8vGOFWO7pKG4SKtwmPTX9dqz0U/qwR564xeQ0hbN0+0e8QIqcaEYXruG2WpjMiZ/A8dP38jMarJtPSX+r3OFfYxEWN1iyDuaXvOEu7dqTERknNUBnHmvcYvDYO1DYI1T1woNQY7dIK6Hzj6PhCl87z6aQQuhRqFInqdwyolxCeRNb3phUiR87gGf28YDBb7agbhGpgMGLLvkODOgAB4PPbBr7CPOtejTDQJt8f3/QebRq9mpA/X+FARc4jAsbgeUELD00/PRsFGIxg6duRwIMJH6gIdgvh52zKHqc+0ffeH9fCjsyTxELugLP0IDL3fm/7exyoxVAdu1AnE28ZsaW6pH66DRkmjhcFoENtlVPvqPsAQysPvD++wcgHymGq9poVOYNn9POCxWy1I/VNyPh0Cem469dSYP9G5CeS5NC0ZjkZu9fuQOFuioF6kTUR58gZPA9ZP1EV4DtUI8yp6duRZ5sJH+xo6UDuzta6Q81v7DCcsyO1UWoUh6Z7rLHVRKD2RuvCDSiC06jp0j+wc3rEKH7d5vm+bOV7Un5pQTngvidtk3Zade81K3IGz+jnBYvZx45Mirfpc0jxsXWLY2WQsJPuEJHflL0ay6RmUrOJSFO3kAnig3NKXzMAlb1HzuB5yPopuIDKAM9Zf+Yrx8rkDHYkmrtmOWhomNIwI9LUtyOpQiS4Q/Nl17xHHKi90bpwA5E7oUjjGLQi95nDFZBrJ1bLtJSOvruzECZFUrfSP3kKefrb1678DJEzeEY/LxjNPnZk3hYuKLYpXdAtNR0i0TrqhsbmTJmsPUTEWvE5rppuGf7qsGT8xOtX2Dx7j5zB85D1Ux2h55hi977Oiz559umoW2tHHC8/xrn05+GJDLMjjVlrDiKLTS4pvMVQnbjwplV1q6r6lar64/ZXJ0BRJLlXG+qazc8teOfCm77wvvN1p6rSooxFee2KnMEz+nlkNfvYkVoSEShdcxpELPYulW8OnkLW/sl/iZ16mEgL6pTsz/HCK4WY0z2XDhLu3i9yBs9D1k+1Tp5xoq5diQs7crhRP3z/DXYkYsuOPHNf1DM8o9FEpjRrLB2ptsRu33ZNHKhVcB272JcmKiRn2xWL43JySrUMEl5T8O01OnqkXHRSoCvoit60fNpgEWmuXZEzeEY/j6xmHzv6sFZL8awema2ibcNK/e1HVwz97WyYkyTKZOSTNuvO3v2u45hbO34icgbPjmz5EPVTtx0yTXbUdamT02g0TVnmSa6dWceOPDM5UJyyzo78DqG11N7aey56zsaBWgTT4EUo52/dumjM9rF0+Til8naFE3dJvebe7arNM8f/YkJ7pwiiXJRjd2zM3P38PXIe8ekEz+jnPnakAB1PjVoK6XkLbYDf4YcSLVpL1IcV/rIWgTKGouOp4VDpRNKIspY9OXIeRaCC52HqJzZyvGmitQq8OztyEFH0LQK1xY6MTZMKR/fTPXfZFmfSz1o7WvJszSy8RSgNX+TL5jX7YjpnSe0TNlU53LVcFt5FFw0GYnUW5mNRBJwwHDQDE7tUzBqxI2fwjH7ub0eaOxyYcKtJCbBNzMpf2BtWusYuXSt1oeZDF95T2l7iNC7SjNXcIOi1h7DIGTwPWT+RvbIjeoq2p7MjDhR7wL6/dnkdp/mRVfW7jYyTHf18O6Cwo7VO2SIZEoFaBNPgRaJN6h/ep31J0nXSZYpKsYY/ccOtKYKTrJQLIj33dKLkWaut6nfqLL195AyeTmLRz/3sSLRJzaO6pfu2lMB12sNA4aruobXL/nvvqtKt+6hWM4mR2QPQRPoXb2BPjpzB85D1890aXYGOWOUqUmuyJJ6lispREaxd7EjdsKHx7AinGjvyLIWFuqhTZyEnZByotV/Vheuv3L50TpOOAgXjOgF05GBEXTOzrrur70OkyZctZYcL6mPa/aUK1haUum/kDJ7Rz/3tCKeajjlRZ3xtunoUvaIzMUJiLXdNZ/PYxjk9impxwKFA4VQZgq27dy35X+QMnoeun+h/PM/UKEnlcabUE7OjteTTnR3p7NO8xY7UD7Ij1D9oR2SE1trRIs8gDtQimAYvgp3WS06TCBEKeozEqv6l4TDiblkUi0IJZSqA8x5aMXFnrA3nd05y5Aye0c/97UiXrA45tqlV3IPA7361RYu22LsHh0OXQtfunjqSPAS3cNfYlyJn8Dxk/fR802zFjkSd0BewAXa0pUyF3Xn23qR1xbsn2/TcxFLuOXpRVhyo7bDCzglUPldHgXC7L1H0yGa49UujXHdu+VtGoPUaJ5Sw/pYwZOQMntHP/e3ITuFUq4NIm7i6CjwzCshFi516tywRbIejx7SHh9ZrdZUmyG9tvY6cwfOQ9fMj20Hkea3YHx+alBs7Wsud2NncxzZ6BNkfKTt8jOqe1BiupRNabMdxoBZDNXih3K15VfK4qv+dJtUzKALdWvWvpkrxG2+cIki/KLbTkbM1DBk5g2f0c387Uo/IwVHHoUZR+k6aXcG3KfJblqJahyOpC7WTGMg1oLj/1ntGzuB5yPqpE48dSbk5OKgbRtfhObdV59kRKpE/aA1YGMg1Zzx2A53QYjuOA7UYqsELkVvqwsFDoU3+w6vqGa2QbeuddQ/YUKUA0dKjqb9HVb1wDwcqcgbP6Of+dqQDT8QZ27caR7w1DjnqLrYuNYoKxkWbpB4cwrRk79M5FDmD56HrpzS1bItuc3aE57CjNdhiS+xIzRNHjB053GieYUdv2XLDJa+JA7UEpfFr8LgoLFUHBUthfCHEtUOE+++gCFSaQEjSohDGcWzpwOvuGzmDZ/Rzfzt611arobYCnmqURIadercu1CXqHpUD+L9uSTWVDlBbUvbkiJzB85D1E+G0midOFDkVkrMjwYeti+0IZrAjz1B2hMyaHW2pHV4kRxyoRTCNXuRL0yWnTokXrQVT6m1L8Wf3Jr4T4xh403il8FsoSl/Lat4XOnIGz+jn/naEZkQ63JBf0+OlSaTb95n0zt59N7qF/GtUDMJPrdhbV+QMnoeunxjz2RGHB2ksZ0cX6tbFjtCKsCM1VuqQOzvaWvoyK0scqFmIZi+QvuP52lhFn/A/7ePseEMpt7tUlVyxsKb6iK2n0e4DRM7gGf3c347YJtI/xd9I+x7dosSzG8XEBQrRzQfzUHEIM3Fgay1I9zaRM3gesn4ioWVH5kjSdySYa8cW7ZqU+mF8iUppfrPdd99n8aRdx4HaZ9s7ei2iSlEozKryuGpN9mU9xQujJVORncJSOd19vejIGTyjn/vbEdu8Xhs7gUBT2/W+NRaK0p3E7SHPb2mH09hDImfwPHT9FCRgR2oJT8OO1FVxpF7Q7ruvHcWB2t9HmryDcLkCNv9SAF/Yvs6Oeykm96+T6GkoQeQMntHP/TcDdnSlVq/ELmG6r707yLJ3qXbdu6e1h0TO4Hno+qleiYyHKufBOlCYdk9zGUJ4KUXUThOb3CsIBIEgEASCQBA4RQTO0+EwrkA4/LQWh+w8P89pfY7cJwgEgSAQBIJAEDhwBC4lh0MY/VL6PAeuOhEvCASBIBAEgsAVF4FLyeGIA3XF1eN88iAQBIJAEAgCZ4pAHKgzhTtvFgSCQBAIAkEgCFwKCJymA4VdFAcDAivMopY5cQaZau3vljEDt2pzaq5VVS89JSATgTolIHObIBAEgkAQCAJBYBqB03SgvBNqdtOU/7CqOEp3aEza2mn9YNf1+89vLL5fXFUPbCSRxo1oZTSc0+oKzBFIIsPSPuyaN4wM6o0DFW0PAkEgCASBIBAEzgSB03agPrSqbtlo2RFB4jbBY+T/L2ujCjBrG5hpfpQIFQeKo2XoH3bfJ7Rp5zdoI1E+varu24bqugZR1oPa3Lk+SHGgzkRl8iZBIAgEgSAQBILAPg7UNauKw/S6xp5rYB8iuDtW1SPagEx06qJSD2/zoqT4ntfo2/1OJIozZOabUSif2ubiYND1dxGruzaWUr8T2bpuVb26jVCIAxUdDgJBIAgEgSAQBM4cgX0cKDVPpn5jzX19j433ps0Z0PtNOwAABABJREFU8mEe0hyqx1bVX1TVN1TV09rYgs6BEnEyE8cQTUN0pfH86zVqqe7WHCfjDn6vqq5aVX8/MMAzEagzV5+8YRAIAkEgCASBKyYC+zhQY4ipcbpfVT2gql7Y0nZSdZyeN7bicVGpX6uq+7ThuzeuqudW1c2q6kkt3Xendv0HVdXdq+orWuTJwM3LksK7YipsPnUQCAJBIAgEgUNA4GI4UKfxuUSxXtkKxm9fVQ9uheRT904E6jSQzz2CwOUTgXtW1fcNiC7C/ROtlOBbTmHi++UTnUgdBILAqSNwqA7U1arqNlUlTfioqnrVgk8eB2oBSLkkCFziCLx3o1L5/qp6+iX+WfPxgkAQOEcEDtWB2gJJHKgtqOU1QeDSQmDIgRLRFskWgVJTiSpFd6+f762qV7QaTA0ut62qlzRI+lEtZQg47rKCQBAIAm9FIA5UFCEIBIFLCYElDhSH6suq6n1aLaZuYRErDpPl//6uJpPT9cHtOk0wiWpdStqSzxIE9kDgYjlQ71xVn1dVj+zVLqEoeHk77Q2JvC9DeSJQeyhCXhoELhEEljhQnZPkWg0p925RJ06T6Qj3b00wT+1FnThXIlWJQl0iipKPEQT2ReBiOVBql4TMf6lXtMmpQnlw5Rb58q/OvI5d3N9c855V1TGU/9OKDxgHagVYuTQIXKIILHGgOkdozoH6uh2MpPtEp7KCQBAIAhcthTfkQN26ql5cVf7F74RD6jqNtuBGVfUzbUZex1DuFPiaFd9RHKgVYOXSIHCJInCaDpQDYFJ2l6ii5GMFgX0ROMsIVOdA3aJ11ok8YRdXf4C9/ClVdfPGQN4xlK/5fHGg1qCVa4PApYnAaThQuzVQDntSdxyqpPAuTb3JpwoCqxE4Dwfqk5qTpIDzI5szFQdq9VeXFwSBIDCAwGk5UG7d78JL+i7qFgSCwDEELqYDdY9W/2Q0ywuq6h1bCm/OgeoYyh9WVc9Y8X0lArUCrFwaBIJAEAgCQSAIbEfgYjlQ2yXa/so4UNuxyyuDQBAIAkEgCASBFQjEgVoBVi4NAkEgCASBIBAEggAE4kBFD4JAEAgCQSAIBIEgsBKBOFArAcvlQSAIBIEgEASCQBC4lByo76iqH8lXGgSCQBAIAkEgCASBi43ApeRAXWyscv8gEASCQBAIAkEgCLwVgThQUYQgEASCQBAIAkEgCKxEIA7USsByeRAIAkEgCASBIBAE4kBFB4JAEAgCQSAIBIEgsBKBOFArAcvlQSAIBIEgEASCQBCIAxUdCAJBIAgEgSAQBILASgTiQK0ELJcHgSAQBIJAEAgCQSAOVHQgCASBIBAEgkAQCAIrEYgDtRKwXB4EgkAQCAJBIAgEgf8Po6YpCaEsvDUAAAAASUVORK5CYII='

# make it pretty:
# sg.theme_previewer()
sg.theme('LightGrey1')

layout_win1 = [
                [sg.Text()],
                [sg.Text('File:'), sg.Push(), sg.Input(key="-FILE-", do_not_clear=True, size=(50,3)), sg.FileBrowse()],
                [sg.Text()],
                [sg.Text('Voltage Limit:'), sg.Push(), sg.Input(key="-VOLT-", do_not_clear=True, size=(50,3))],
                [sg.Text()],
                [sg.Text(size=(70, 5), key='-OUTPUT-', font=('Arial Bold',10,'italic'))],
                [sg.Button('Analyze Voltage Ramp'), sg.Button('Analyze Pulse Burst'), sg.Push(), sg.Button('', image_data=help_button_base64, button_color=(sg.theme_background_color(),sg.theme_background_color()), border_width=0, key='-INFO-'), sg.Button('Exit', button_color='red')]
                ]

win1 = sg.Window(title='Guinness Waveform Analyzer (ST-0001-066-101A)', layout=layout_win1, size = (600,275))
win2_active = False

info_txt_width = 138
info_txt_size = 9

while True:
    try:
        event, value = win1.read(timeout=100)
        
        if event == sg.WIN_CLOSED or event == 'Exit':
            break

        if event == '-INFO-' and win2_active == False:
            win2_active = True
            layout_win2 = [[sg.Text("Instructions For Use", font=('None',12,'bold'))],
                           [sg.Text("Capture the treatment output from the Guinness Generator on an oscilloscope and export the data from the oscilloscope screen as a .csv file. In this application, enter the filepath of the exported .csv file and the voltage limit set during the Guinness Generator treatment output.\n\nThere are several restrictions on the input .csv file to prevent errors and inaccuracies:",size=(info_txt_width, None), font=('None',info_txt_size))],
                           [sg.Text("1.  It must have only 2 columns: the first for timestamps, the second for voltage.\n2.  It must have at least 500 rows of data.\n3.  Its headers (if applicable) must be contained to the first row.",pad=(40,0), size=(info_txt_width,None), font=('None',info_txt_size))],
                           [sg.Text("\nWith the waveform filepath and voltage limit entered, click one of the buttons to analyze the inputs. See the following sections for details surrounding the function of each button.", size=(info_txt_width,None), font=('None',info_txt_size))],
                           [sg.Text("\nAnalyze Voltage Ramp", font=('None',info_txt_size+1,'underline'))],
                           [sg.Text("This button will take the waveform input and look for the lines of best fit of the voltage ramp; this is done piecewise as the Guinness Generator should ramp at a different rate before and after the output voltage reaches 66% of the set voltage limit. For this function to work properly, the input .csv file should capture the voltage ramp of the Guinness Generator from 0V to the set Voltage Limit. For this function to work as intended, the input waveform should resemble the following:", size=(info_txt_width,None), font=('None',info_txt_size))],
                           [sg.Push(), sg.Image(voltage_ramp_example), sg.Push()],
                           [sg.Text("\nAnalyze Pulse Burst", font=('None',info_txt_size+1,'underline'))],
                           [sg.Text("This button will take the waveform input and tranform it into the frequency domain via the Fourier Transform. Frequency will be plotted against amplitude; the more the frequency is present, the higher the plotted amplitude. These frequency amplitudes are used to calculate the Total Harmonic Distortion (THD) per the following equation:", size=(info_txt_width,None), font=('None',info_txt_size))],
                           [sg.Push(), sg.Image(THD_eq), sg.Push()],
                           [sg.Text("For this function to work as intended, the input waveform should be of the pulse burst and resemble the following:", size=(info_txt_width,None), font=('None',info_txt_size))],
                           [sg.Push(), sg.Image(pulse_burst_example), sg.Push()],
                           [sg.Button("Close")]]

            win2 = sg.Window(title="ST-0001-066-101A Information", layout=layout_win2, size=(1000,810))

        if win2_active == True:
            win2_events, win2_values = win2.read(timeout=100)
            if win2_events == sg.WIN_CLOSED or win2_events == 'Close':
                win2_active  = False
                win2.close()

        if event == 'Analyze Voltage Ramp':

            if value['-FILE-'] != '' and value["-VOLT-"] != '':
                
                fileGood = CheckFile(value['-FILE-'])
                voltageGood = VoltageCheck(value['-VOLT-'])

                if fileGood == True and voltageGood == True:
                    csvGood = CheckCSV(value['-FILE-'])

                    if csvGood == True:
                        try:
                            guinnessRampFilter(value['-FILE-'], value["-VOLT-"])
                        except ValueError:
                            value['-OUTPUT-'] = "Error:  Something went wrong. The contents of the input file are incompatible with the requested operation. Review contents of the input waveform .csv and the selected analysis option."
                            win1['-OUTPUT-'].update(value['-OUTPUT-'])

                        except IndexError:
                            value['-OUTPUT-'] = "Error:  Something went wrong. The contents of the input file are incompatible with the requested operation. Review contents of the input waveform .csv and the selected analysis option."
                            win1['-OUTPUT-'].update(value['-OUTPUT-'])
                            
                        except TypeError:
                            value['-OUTPUT-'] = "Error:  Something went wrong. The contents of the input file are incompatible with the requested operation. Review contents of the input waveform .csv and the selected analysis option."
                            win1['-OUTPUT-'].update(value['-OUTPUT-'])
                    else:
                        value['-OUTPUT-'] = "Error:  Input file contains unexpected contents. File must contain only a column of timestamps and a column of corresponding measured voltage."
                        win1['-OUTPUT-'].update(value['-OUTPUT-'])

                elif fileGood == False and voltageGood == True:
                    value['-OUTPUT-'] = "Error:  Invalid filepath or filetype. Input must be a .csv file"
                    win1['-OUTPUT-'].update(value['-OUTPUT-'])

                elif fileGood == True and voltageGood == False:
                    csvGood = CheckCSV(value['-FILE-'])

                    if csvGood == True:
                        value['-OUTPUT-'] = "Error:  Not a valid voltage limit input. Value must be an integer in the range from 0 to 150."
                        win1['-OUTPUT-'].update(value['-OUTPUT-'])
                    else:
                        value['-OUTPUT-'] = "Error:  Not a valid voltage limit input. Value must be an integer in the range from 0 to 150.\n\nError:  Input file contains unexpected contents. File must contain only a column of timestamps and a column of corresponding measured voltage."
                        win1['-OUTPUT-'].update(value['-OUTPUT-'])

                elif fileGood == False and voltageGood == False:
                    value['-OUTPUT-'] = "Error:  Invalid file and voltage limit."
                    win1['-OUTPUT-'].update(value['-OUTPUT-'])
            

            elif value['-FILE-'] == '' or value["-VOLT-"] == '':
                value['-OUTPUT-'] = "Error:  Both the filepath and voltage limit must be entered."
                win1['-OUTPUT-'].update(value['-OUTPUT-'])


        if event == 'Analyze Pulse Burst':
            if value['-FILE-'] != '' and value["-VOLT-"] != '':
                
                fileGood = CheckFile(value['-FILE-'])
                voltageGood = VoltageCheck(value['-VOLT-'])

                if fileGood == True and voltageGood == True:
                    csvGood = CheckCSV(value['-FILE-'])

                    if csvGood == True:
                        try:
                            guinnessTHD(value['-FILE-'], value["-VOLT-"])
                        except ValueError:
                            value['-OUTPUT-'] = "Error:  Something went wrong. The contents of the input file are incompatible with the requested operation. Review contents of the input waveform .csv and the selected analysis option."
                            win1['-OUTPUT-'].update(value['-OUTPUT-'])

                        except IndexError:
                            value['-OUTPUT-'] = "Error:  Something went wrong. The contents of the input file are incompatible with the requested operation. Review contents of the input waveform .csv and the selected analysis option."
                            win1['-OUTPUT-'].update(value['-OUTPUT-'])
                            
                        except TypeError:
                            value['-OUTPUT-'] = "Error:  Something went wrong. The contents of the input file are incompatible with the requested operation. Review contents of the input waveform .csv and the selected analysis option."
                            win1['-OUTPUT-'].update(value['-OUTPUT-'])
                    else:
                        value['-OUTPUT-'] = "Error:  Input file contains unexpected contents. File must contain only a column of timestamps and a column of corresponding measured voltage."
                        win1['-OUTPUT-'].update(value['-OUTPUT-'])

                elif fileGood == False and voltageGood == True:
                    value['-OUTPUT-'] = "Error:  Invalid filepath or filetype. Input must be a .csv file"
                    win1['-OUTPUT-'].update(value['-OUTPUT-'])

                elif fileGood == True and voltageGood == False:
                    csvGood = CheckCSV(value['-FILE-'])

                    if csvGood == True:
                        value['-OUTPUT-'] = "Error:  Not a valid voltage limit input. Value must be an integer in the range from 0 to 150."
                        win1['-OUTPUT-'].update(value['-OUTPUT-'])
                    else:
                        value['-OUTPUT-'] = "Error:  Not a valid voltage limit input. Value must be an integer in the range from 0 to 150.\n\nError:  Input file contains unexpected contents. File must contain only a column of timestamps and a column of corresponding measured voltage."
                        win1['-OUTPUT-'].update(value['-OUTPUT-'])

                elif fileGood == False and voltageGood == False:
                    value['-OUTPUT-'] = "Error:  Invalid file and voltage limit."
                    win1['-OUTPUT-'].update(value['-OUTPUT-'])
            
            elif value['-FILE-'] == '' or value["-VOLT-"] == '':
                value['-OUTPUT-'] = "Error:  Both the filepath and voltage limit must be entered."
                win1['-OUTPUT-'].update(value['-OUTPUT-'])
    
    except ValueError:
        value['-OUTPUT-'] = "Error:  Something went wrong. The contents of the input file are incompatible with the requested operation. Review contents of the input waveform .csv and the selected analysis option."

    except IndexError:
        value['-OUTPUT-'] = "Error:  Something went wrong. The contents of the input file are incompatible with the requested operation. Review contents of the input waveform .csv and the selected analysis option."
        
    except TypeError:
        value['-OUTPUT-'] = "Error:  Something went wrong. The contents of the input file are incompatible with the requested operation. Review contents of the input waveform .csv and the selected analysis option."



'''To create your EXE file from your program that uses PySimpleGUI, my_program.py, enter this command in your Windows command prompt:

pyinstaller my_program.py

You will be left with a single file, my_program.exe, located in a folder named dist under the folder where you executed the pyinstaller command.
'''