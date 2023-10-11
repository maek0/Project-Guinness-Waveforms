import PySimpleGUI as sg
from guinnessRampFilter import guinnessRampFilter
from guinnessTHD import guinnessTHD

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
    # win1['-OUTPUT-'].update(values1[:])
    if event1 == sg.WIN_CLOSED or event1 == 'Exit':
        break

    if event1 == 'Analyze Guinness Voltage Ramp':
        guinnessRampFilter("-FILE-")

    if event1 == 'Analyze Guinness Pulse Burst':
        guinnessTHD("-FILE-")


'''To create your EXE file from your program that uses PySimpleGUI, my_program.py, enter this command in your Windows command prompt:

pyinstaller -wF my_program.py

You will be left with a single file, my_program.exe, located in a folder named dist under the folder where you executed the pyinstaller command.
'''