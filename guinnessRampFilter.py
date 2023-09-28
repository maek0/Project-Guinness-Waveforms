def guinnessRampFilter():
    import numpy as np
    from support_functions import CheckFile, VoltageCheck, plotting_peaks
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
    
    plotting_peaks(x, y, voltageLimit, filename, str_datetime_rn, headers)
    
guinnessRampFilter()