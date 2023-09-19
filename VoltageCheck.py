def VoltageCheck():
    while True:
            voltageLimit = input('Enter the voltage limit of the Guinness generator of the captured waveform: ')
            try:
                voltageLimit = int(voltageLimit)
                if type(voltageLimit)==str:
                    raise TypeError
                elif voltageLimit>150 or voltageLimit<0:
                    raise ValueError
                elif voltageLimit<=150 and voltageLimit>=0:
                    break
                else:
                    print("Invalid input.")
                    raise TypeError
            except ValueError:
                print("Not a valid input. Value must be an integer in the range from 0 to 150.")
            except TypeError:
                print("Not a valid input. Value must be an integer in the range from 0 to 150.")