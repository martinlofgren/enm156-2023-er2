from ultralytics import YOLO
from PIL import Image


image_path1 = "C:/Users/nilsa/Desktop/BiBilder/test"
image_path2 = "c:/Users/nilsa/Pictures/mcbees.jpg"
video_path = "C:/Users/Jakob\OneDrive - Chalmers/ENM156 - Hållbar utveckling och etik/Bees/video_le1/beehive-2020-07-21_09-59-24.mp4"
model = YOLO('100eXL.pt')

results = model(image_path1, conf = 0.5)

for r in results:
    im_array = r.plot(conf=True, line_width=2, font_size=2, font='Arial.ttf', pil=False, img=None,
                        im_gpu=None, kpt_radius=5, kpt_line=True, labels=True, boxes=True, masks=True, probs=True)
    im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
    im.show()  # show image
