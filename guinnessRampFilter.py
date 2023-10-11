def guinnessRampFilter(filepath):
    import numpy as np
    from support_functions import CheckFile, VoltageCheck, plotting_peaks
    import datetime
    
    # need matplotlib, scipy, numpy, scikit (pip install numpy scikit-learn statsmodels), pysimplegui

    datetime_rn = datetime.datetime.now()
    str_datetime_rn = datetime_rn.strftime("%d-%b-%Y, %X %Z")

    # filepath = input('This function is intended to analyze the voltage ramp rate of the Guinness Generator in accordance with its product requirements.\nEnter the name or file path of the oscilloscope .csv output: ')
    filename = CheckFile(filepath)
    
    voltageLimit = input("Enter the voltage limit of the Guinness generator of the captured waveform: ")
    VoltageCheck(voltageLimit)
    
    headers = ["Time", "Voltage"]
    csvFile = open(filepath)
    csvArray = np.genfromtxt(csvFile, delimiter=",")
    x = csvArray[2:,0]
    y = csvArray[2:,1]
    
    plotting_peaks(x, y, voltageLimit, filename, str_datetime_rn, headers)