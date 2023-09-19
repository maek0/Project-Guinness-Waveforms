import sys
import os

def CheckFile(filepath):
    if os.path.exists(filepath):
        if filepath[-4:]==".csv":
            print("File found. File is a .csv file.")
        else:
            print("File found, but it is not a .csv file. Double check input file type and name.")
    else:
        print("File was not found. Double check input file path and file name.")
        sys.exit()