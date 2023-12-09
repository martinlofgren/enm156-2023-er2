import os
import tempfile

from flask import Flask, render_template, request, send_file, send_from_directory
from werkzeug.utils import secure_filename

from ultralytics import YOLO
from PIL import Image

# MODEL = 'model/100eXL.pt'
MODEL = 'model/200eNano.pt'

app = Flask(
    __name__,
    static_url_path='',
)

model = YOLO(MODEL)

@app.route('/')
def home():
    return render_template('index.html')

@app.post('/detect')
def detect():
    file = request.files['img']
    _, suffix = os.path.splitext(str(file.filename))
    in_file = tempfile.NamedTemporaryFile(suffix=suffix)
    file.save(in_file)
    print('Saving intermediate file', in_file.name)

    results = model(in_file.name)

    for r in results:
        im_array = r.plot(
            conf=True,
            line_width=2,
            font_size=2,
            font='Arial.ttf',
            pil=False,
            img=None,
            im_gpu=None,
            kpt_radius=5,
            kpt_line=True,
            labels=True,
            boxes=True,
            masks=True,
            probs=True
        )
        im = Image.fromarray(im_array[..., ::-1])
        out_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        im.save(out_file.name)

    return(out_file.name)

@app.get('/get_result')
def get_result():
    f = request.args.get('f', '')
    return send_file(f)
