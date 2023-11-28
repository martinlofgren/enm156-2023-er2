from ultralytics import YOLO
from PIL import Image, ImageDraw
import cv2


image_path = "C:/Users/nilsa/Pictures/street.jpg"
model = YOLO('yolov8n.pt')
font = cv2.FONT_HERSHEY_SIMPLEX

results = model(image_path)
for r in results:
    im_array = r.plot()  # plot a BGR numpy array of predictions
    im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
    im.show()  # show image
    im.save('results.jpg')  # save image


"""
def main():
    results = model(image_path)

    #image = cv2.imread(image_path)

    
    
    for r in results:
        

        namn = r.names
        tens = r.boxes.xywhn
        clas = r.boxes.cls
        (y, x) = r.boxes.orig_shape
        tups = tupList(clas, tens)
        for tup in tups:
            xMul = tup[1][0].item()
            yMul = tup[1][1].item()
            wMul = tup[1][2].item()
            hMul = tup[1][3].item()
            corners = bound(x,y,xMul,yMul,wMul,hMul)
            
            label = namn[int(tup[0].item())]
            
            cv2.rectangle(image, corners[0], corners[1], (0, 0, 255), 2)
            cv2.putText(image, label, corners[0], font, 0.5, (0,0,255), 2)
            
        

    #cv2.imshow("image", image)
    #cv2.waitKey(0)
    cv2.imwrite("resultat.jpg", image)
    

def bound(xOrg, yOrg, xMul, yMul, wMul, hMul):
    xCenter = xOrg * xMul
    yCenter = yOrg * yMul

    width = xOrg * wMul
    height = yOrg * hMul
    up = (int(xCenter-width/2), int(yCenter-height/2))
    low = (int(xCenter+width/2), int(yCenter+height/2))
    return [up, low]

def tupList(cls, tens):
    tuplist = []
    for x in range(len(cls)):
        tuplist.append((cls[x], tens[x]))

    return tuplist

"""
