import PySimpleGUI as sg

sg.theme('DefaultNoMoreNagging')

# layout = [[sg.Text('Select a file:')],
#           [sg.Input(key='-FILE-', visible=False, enable_events=True), sg.FileBrowse()]]

# event, values = sg.Window('File Compare', layout).read(close=True)

layout = [[sg.Text('Guinness Waveform Analyzer (ST-0001-066-101A)'),],
          [sg.Input(do_not_clear=True)],
          [sg.Button('Analyze Guinness Voltage Ramp'),sg.Button('Analyze Guinness Pulse Burst'), sg.Button('Exit')],
          [sg.Input(key='-FILE-', visible=False, enable_events=True), sg.FileBrowse()]]

win1 = sg.Window('Guinness Waveform Analyzer (ST-0001-066-101A)', layout)

win2_active = False
while True:
    event1, values1 = win1.read(timeout=100)
    win1['-OUTPUT-'].update(values1[0])
    if event1 == sg.WIN_CLOSED or event1 == 'Exit':
        break
    elif event1 == 'Analyze Guinness Voltage Ramp':
        break
    elif event1 == 'Analyze Guinness Pulse Burst':
        break

    if not win2_active and event1 == 'Launch 2':
        win2_active = True
        layout2 = [[sg.Text('Window 2')],
                   [sg.Button('Exit')]]

        win2 = sg.Window('Window 2', layout2)

    if win2_active:
        event2, values2 = win2.read(timeout=100)
        if event2 == sg.WIN_CLOSED or event2 == 'Exit':
            win2_active  = False
            win2.close()


'''To create your EXE file from your program that uses PySimpleGUI, my_program.py, enter this command in your Windows command prompt:

pyinstaller -wF my_program.py

You will be left with a single file, my_program.exe, located in a folder named dist under the folder where you executed the pyinstaller command.
'''