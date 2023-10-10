def guinnessTHD():
    import numpy as np
    import matplotlib.pyplot as plt
    
    from support_functions import CheckFile, VoltageCheck, power_spectral_density
    import datetime
    
    # need matplotlib, scipy, numpy, scikit (pip install numpy scikit-learn statsmodels)

    datetime_rn = datetime.datetime.now()
    str_datetime_rn = datetime_rn.strftime("%d-%b-%Y, %X %Z")

    filepath = input('This function is intended to calculate the total harmonic distortion (THD) of the Guinness Generator.\nEnter the name or file path of the oscilloscope .csv output: ')
    filename = CheckFile(filepath)
    
    voltageLimit = VoltageCheck()
    
    headers = ["Time", "Voltage"]
    csvFile = open(filepath)
    csvArray = np.genfromtxt(csvFile, delimiter=",")
    x = csvArray[2:,0]
    y = csvArray[2:,1]
    
    Pxx = power_spectral_density(x, y, voltageLimit)