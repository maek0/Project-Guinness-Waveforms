import sys
import os
import numpy as np
from sklearn.linear_model import LinearRegression


def CheckFile(filepath):
    if os.path.exists(filepath):
        if filepath[-1] == "/":
            filepath = filepath[:-1]
        elif filepath[-4:] == ".csv":
            print("File found. File is a .csv file.")
            return os.path.basename(filepath)
        else:
            print(
                "File found, but it is not a .csv file. Double check input file type and name."
            )
            sys.exit()
    else:
        print("File was not found. Double check input file path and file name.")
        sys.exit()


def linearRegression(x, y):
    x = np.array(x).reshape((-1, 1))

    model = LinearRegression().fit(x, y)
    r_sq = model.score(x, y)
    slope = model.coef_
    intercept = model.intercept_
    y_predict = model.predict(x)

    return r_sq, slope, intercept, y_predict


def VoltageCheck():
    while True:
        voltageLimit = input(
            "Enter the voltage limit of the Guinness generator of the captured waveform: "
        )
        try:
            voltageLimit = int(voltageLimit)
            if type(voltageLimit) == str:
                raise TypeError
            elif voltageLimit > 150 or voltageLimit < 0:
                raise ValueError
            elif voltageLimit <= 150 and voltageLimit >= 0:
                return voltageLimit
                break
            else:
                print("Invalid input.")
                raise TypeError
        except ValueError:
            print(
                "Not a valid input. Value must be an integer in the range from 0 to 150."
            )
            continue
        except TypeError:
            print(
                "Not a valid input. Value must be an integer in the range from 0 to 150."
            )
            continue
