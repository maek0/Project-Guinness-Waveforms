import PySimpleGUI as sg
from guinnessRampFilter import guinnessRampFilter
from guinnessTHD import guinnessTHD
from support_functions import CheckFile, VoltageCheck
import sys
import os

sg.theme('DefaultNoMoreNagging')

# layout = [[sg.Text('Select a file:')],
#           [sg.Input(key='-FILE-', visible=False, enable_events=True), sg.FileBrowse()]]

# event, values = sg.Window('File Compare', layout).read(close=True)

layout = [[sg.Text('Enter a waveform file to evaluate.')],
          [sg.Text('File:', size=(3, 1)), sg.Input(do_not_clear=True, key="-FILE-")],
          [sg.Text('Enter the voltage limit of the waveform.')],
          [sg.Text('Voltage Limit:', size=(11, 1)), sg.Input(key="-VOLT-", do_not_clear=True)],
          [sg.Text(size=(25, 3), key='-OUTPUT-')],
          [sg.Button('Analyze Guinness Voltage Ramp'), sg.Button('Analyze Guinness Pulse Burst'), sg.Button('Exit')]]

win1 = sg.Window('Guinness Waveform Analyzer (ST-0001-066-101A)', layout)

while True:
    event1, values1 = win1.read(timeout=100)
    if event1 == sg.WIN_CLOSED or event1 == 'Exit':
        break

    if event1 == 'Analyze Guinness Voltage Ramp':
        if values1['-FILE-'] != [] and values1["-VOLT-"] != []:

            fileGood = CheckFile(values1['-FILE-'])
            voltageGood = VoltageCheck(values1['-VOLT-'])

            if fileGood and voltageGood:
                guinnessRampFilter(values1['-FILE-'], values1["-VOLT-"])

            elif not fileGood and voltageGood:
                values1['-OUTPUT-'] = "Invalid filepath or filetype. Input must be a .csv file"

            elif fileGood and not voltageGood:
                values1['-OUTPUT-'] = "Not a valid input. Value must be an integer in the range from 0 to 150."

            elif not fileGood and not voltageGood:
                values1['-OUTPUT-'] = "Invalid file and voltage limit."
        

        elif values1['-FILE-'] == [] or values1["-VOLT-"] == []:
            err = "Both the filepath and voltage limit must be entered."
            win1['-OUTPUT-'].update(err)

    if event1 == 'Analyze Guinness Pulse Burst':
        if values1['-FILE-'] != [] and values1["-VOLT-"] != []:
            CheckFile(values1['-VOLT-'])

            if fileGood and voltageGood:
                guinnessTHD(values1['-FILE-'], values1["-VOLT-"])

            elif not fileGood and voltageGood:
                values1['-OUTPUT-'] = "Invalid filepath or filetype. Input must be a .csv file"

            elif fileGood and not voltageGood:
                values1['-OUTPUT-'] = "Not a valid input. Value must be an integer in the range from 0 to 150."

            elif not fileGood and not voltageGood:
                values1['-OUTPUT-'] = "Invalid file and voltage limit."

        elif values1['-FILE-'] == [] or values1["-VOLT-"] == []:
            err = "Both the filepath and voltage limit must be entered."
            win1['-OUTPUT-'].update(err)


'''To create your EXE file from your program that uses PySimpleGUI, my_program.py, enter this command in your Windows command prompt:

pyinstaller -wF my_program.py

You will be left with a single file, my_program.exe, located in a folder named dist under the folder where you executed the pyinstaller command.
'''