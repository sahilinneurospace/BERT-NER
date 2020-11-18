import pandas as pd
import codecs, argparse
import os

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--folder', '-f', help='Folder to read data from', type=str, required=False, default='Events (Week 2)/Data/birthday')
parser.add_argument('--excel-filename', '-xl', help='File name where we write the data', type=str, required=False, default='events.xlsx')
parser.add_argument('--ocr-filename', '-ocr', help='Name of the ocr file', type=str, required=False, default='ocr_log.txt')
parser.add_argument('--intent', '-i', help='Intent of the images associated with the ocr file', type=str, required=False, default='Shopping(Grocery)')

args = parser.parse_args()
folder = args.folder
xl_file = args.excel_filename
ocr_file = args.ocr_filename
intent = args.intent

def ocr_reader(ocr_text):
    ocr_text = ocr_text.split("NEW SAMPLE: ")
    ocr_text = ocr_text[1:]
    images = {}

    for x in ocr_text:
        x = x.split("\n")
        x = [y for y in x if "<<BLOCK>>" not in y]
        x = "\n".join(x)
        x = x.split("<<LINE>>")
        y = x[0].split()
        img_name = y[0] + y[1]
        img_x = float(y[2])
        img_y = float(y[4])
        x = x[1:]
        seq = ""
        coords = []
        for y in x:
            y = y.split("\n")
            y = y[1:]
            for z in y:
                if z:
                    z = z.split()
                    if ">" in z[1]:
                        z = [z[0]] + z[2:]
                    #print(y)
                    seq += z[0][1:-1] + " "
                    coords.append([round(x, 8) for x in [float(z[1])/img_x, float(z[2])/img_y, float(z[3])/img_x, float(z[4])/img_y]])
            seq += "\n"
        images[img_name] = {"text": seq, "coordinates": coords}

    return images

def ocr_to_excel(folder, ocr_file, xl_file, intent):
    with open(os.path.join(folder, ocr_file), 'r', encoding='utf-8') as file:
        f = file.read()
    images = ocr_reader(f)
    df = pd.DataFrame.from_images({"Image Name": list(images.keys()), "Text Recognized": [x["text"] for x in images.values()], "Slot Labels": [""] * len(images.keys()), "Intent": [intent] * len(images.keys()),"Layout Coordinates": [x["coordinates"] for x in images.values()]})
    df.to_excel("{folder}\\{xl_file}")
    
ocr_to_excel(folder, ocr_file, xl_file, intent)