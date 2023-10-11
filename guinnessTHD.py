def guinnessTHD(filepath):
    import numpy as np
    from support_functions import CheckFile, VoltageCheck, THD
    import datetime
    
    # need matplotlib, scipy, numpy, scikit (pip install numpy scikit-learn statsmodels)

    datetime_rn = datetime.datetime.now()
    str_datetime_rn = datetime_rn.strftime("%d-%b-%Y, %X %Z")

    # filepath = input('This function is intended to calculate the total harmonic distortion (THD) of the Guinness Generator and is intended to take in a .csv with only the pulse envelope.\nEnter the name or file path of the oscilloscope .csv output: ')
    filename = CheckFile(filepath)

    voltageLimit = input("Enter the voltage limit of the Guinness generator of the captured waveform: ")
    VoltageCheck(voltageLimit)
    
    headers = ["Time", "Voltage"]
    csvFile = open(filepath)
    csvArray = np.genfromtxt(csvFile, delimiter=",")
    x = csvArray[2:-2,0]
    y = csvArray[2:-2,1]
    
    THD(x, y, voltageLimit, filename, str_datetime_rn, headers)
    
guinnessTHD()