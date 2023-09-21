def guinnessFilter():
    import matplotlib.pyplot as plt
    from scipy import signal
    import numpy as np
    from support_functions import CheckFile, VoltageCheck, linearRegression
    import datetime
    import fpdf
    
    # need matplotlib, scipy, numpy, scikit (pip install numpy scikit-learn statsmodels), fpdf

    datetime_rn = datetime.datetime.now()
    str_datetime_rn = datetime_rn.strftime("%d-%b-%Y, %X %Z")

    filepath = input('Enter the name or file path of the oscilloscope .csv output: ')
    filename = CheckFile(filepath)
    
    voltageLimit = VoltageCheck()
    
    headers = ["Time", "Voltage"]
    csvFile = open(filepath)
    csvArray = np.genfromtxt(csvFile, delimiter=",")
    x = csvArray[2:,0]
    y = csvArray[2:,1]
    
    plt.plot(x,y)
    # xaxis = range(int(min(x)), int(max(x)), 2)
    # yaxis = range(int(min(y)), int(max(y)), 5)
    # plt.xticks(xaxis)
    # plt.yticks(yaxis)
    
    # plt.show()
    
    y_peaks_xvalues, ypeak_properties = signal.find_peaks(y, height=5,prominence=15,distance=50)
    y_peaks_yvalues = ypeak_properties["peak_heights"]
    # the distance might have to change when the frequency is fixed to be 2Hz
    
    # plt.scatter(x[y_peaks_xvalues],y[y_peaks_xvalues])
    # plt.plot(x,y)
    # plt.show()
    # this plot won't have to stay here, currently being used to validate settings
    
    cutoff = 0.66*float(voltageLimit)
    ind = np.where(y_peaks_yvalues>=cutoff)[0][0]
    
    fiveVoltRampY = y_peaks_yvalues[:ind]
    fiveVoltRampX = x[y_peaks_xvalues[:ind]]
    
    twoVoltRampY = y_peaks_yvalues[ind:]
    twoVoltRampX = x[y_peaks_xvalues[ind:]]
    
    fiveV_rsq, fiveV_slope, fiveV_intercept, fiveV_fit = linearRegression(fiveVoltRampX, fiveVoltRampY)
    
    twoV_rsq, twoV_slope, twoV_intercept, twoV_fit = linearRegression(twoVoltRampX, twoVoltRampY)
    
    plt.plot(x,y, color = 'blue')
    
    # # plt.axhline(y=cutoff, color = 'k', linestyle = '--', label = "{:.2f}V (66% of Voltage Limit)".format(float(cutoff)))
    
    plt.scatter(fiveVoltRampX,fiveVoltRampY, color = 'green')
    plt.scatter(twoVoltRampX,twoVoltRampY, color = 'orange')
    
    plt.title("Guinness Generator Output, Voltage Limit = {}V\nInput file name: '{}'".format(voltageLimit, filename))
    plt.text(min(x),min(y),"ST-0001-066-101A, {}".format(str_datetime_rn),fontsize="small")
    plt.xlabel(headers[0])
    plt.ylabel(headers[1])
    
    plt.plot(fiveVoltRampX, fiveV_fit, color = 'green', label = "y = {:.2f}x + {:.2f}\n$R^2$ = {:.2f}".format(fiveV_slope[0], fiveV_intercept, fiveV_rsq))
    plt.plot(twoVoltRampX, twoV_fit, color = 'orange', label = "y = {:.2f}x + {:.2f}\n$R^2$ = {:.2f}".format(twoV_slope[0], twoV_intercept, twoV_rsq))
    
    plt.xlim(min(x)-1,max(x)+1)
    plt.ylim(min(y)-3,max(y)+3)
    plt.legend(loc="upper left")
    
    plt.show()
    
guinnessFilter()