import os
import shutil
LABEL_PATH = "C:/Users/nilsa/Desktop/BiBilder/obj_train_data"
IMAGE_PATH = "C:/Users/nilsa/Desktop/BiBilder/Source"
IMAGE_DESTINATION = "C:/Users/nilsa/Desktop/BiBilder/AnotBilder/"

if __name__ == '__main__':
    for file in os.listdir(LABEL_PATH):
        filename = file.split('.')[0]+".jpeg"
        if filename in os.listdir(IMAGE_PATH):
            shutil.copy(IMAGE_PATH + "/" + filename, IMAGE_DESTINATION+filename)
