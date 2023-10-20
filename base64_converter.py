import PySimpleGUI as sg
import base64

def convert_file_to_base64(filename):
    try:
        contents = open(filename, 'rb').read()
        encoded = base64.b64encode(contents)
        sg.clipboard_set(encoded)
        # pyperclip.copy(str(encoded))
        sg.popup('Copied to your clipboard!', 'Keep window open until you have pasted the base64 bytestring')
    except Exception as error:
        sg.popup_error('Cancelled - An error occurred', error)


if __name__ == '__main__':
    filename = sg.popup_get_file('Source Image will be encoded and results placed on clipboard', title='Base64 Encoder')

    if filename:
        convert_file_to_base64(filename)
    else:
        sg.popup_cancel('Cancelled - No valid file entered')