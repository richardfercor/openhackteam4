from flask import Flask
from flask import request
import matplotlib.pyplot as plt
import math
import numpy as np
import os
import PIL
import sys
import cntk as C
from PIL import Image, ImageOps
try:
    from urllib.request import urlopen
except ImportError:
    from urllib import urlopen


def eval(pred_op, image_data):
    labels_master=['axes','boots','carabiners','crampons','gloves','hardshell_jackets','harnesses','helmets','insulated_jackets','pulleys','rope','tents']
    label_lookup = labels_master
    image_mean = 133.0
    image_data -= image_mean
    image_data = np.ascontiguousarray(np.transpose(image_data, (2, 0, 1)))

    result = np.squeeze(pred_op.eval({pred_op.arguments[0]:[image_data]}))

    # Return top 3 results:
    top_count = 3
    result_indices = (-np.array(result)).argsort()[:top_count]
    result_text="Top 3 predictions:\n"
    print("Top 3 predictions:")
    for i in range(top_count):
        result_text=result_text+"\tLabel: {:10s}, confidence: {:.2f}%".format(label_lookup[result_indices[i]], result[result_indices[i]] * 100)
    return result_text
app = Flask(__name__)

@app.route("/")
def hello():
    return "This is the Team 4 Image detection Api-REST"


@app.route("/evalimage/")
def evalimage():
    pred=C.load_model('cntkmodelbasic.dnn')
    url=request.args.get('url')
    original_image=PIL.Image.open(urlopen(url)).convert('RGB')
    myimg = np.array(original_image, dtype=np.float32)
    width,height=original_image.size
    myimg = original_image
    if(width!=height):
        num=max(width, height)
        a4im = Image.new('RGB',
                            (num, num),  
                            (255, 255, 255))  # White
        try:
            a4im.paste(myimg, myimg.getbbox())  # Not centered, top-left corner
        except:
            print("error")
        myimg=a4im
    imgpad=myimg
    imgresized=imgpad.resize((128,128),Image.NEAREST)
    myimg=np.array(imgresized, dtype=np.float32)
    #myimg =imgresized # np.asarray(imgresized)
    return "<img src='" + str(url) + "'></img><br>" + str(eval(pred,myimg))


if __name__ == '__main__':
    app.run(debug=True)
    