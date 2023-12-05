from ultralytics import YOLO
from PIL import Image


image_path1 = "C:/Users/Jakob/OneDrive - Chalmers/ENM156 - Hållbar utveckling och etik/Bees/AWS/Pollen/beehive/images/beehive-2020-08-10_20-10-32.jpg"
video_path = "C:/Users/Jakob\OneDrive - Chalmers/ENM156 - Hållbar utveckling och etik/Bees/video_le1/beehive-2020-07-21_09-59-24.mp4"
model = YOLO('best.pt')

results = model(image_path1)

for r in results:
    im_array = r.plot(conf=True, line_width=2, font_size=2, font='Arial.ttf', pil=False, img=None,
                        im_gpu=None, kpt_radius=5, kpt_line=True, labels=True, boxes=True, masks=True, probs=True)
    im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
    im.show()  # show image
