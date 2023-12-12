import os
import shutil
LABEL_PATH = "C:/Users/Jakob/Downloads/Emmy_annoteringar/obj_train_data"
IMAGE_PATH = "C:/Users/Jakob/Desktop/New"
IMAGE_DESTINATION = "C:/Users/Jakob/Downloads/Emmy_annoteringar/images/"

if __name__ == '__main__':
    for file in os.listdir(LABEL_PATH):
        filename = file.split('.')[0]+".jpeg"
        if filename in os.listdir(IMAGE_PATH):
            shutil.copy(IMAGE_PATH + "/" + filename, IMAGE_DESTINATION+filename)
