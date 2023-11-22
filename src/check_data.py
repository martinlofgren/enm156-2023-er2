from PIL import Image
import os

#path to subfolders
folder_paths = [
    "data/PetImages/Cat",
    "data/PetImages/Dog",
    "data/Tests/Cat",
    "data/Tests/Dog"
]

#removes files that are not readable, remove last 2 lines to remove files manually instead
for folder_path in folder_paths:
    for filename in os.listdir(folder_path):
        try:
            image = Image.open(os.path.join(folder_path, filename))
        except Exception as e:
            print(f"Error in file {filename}: {e}")
            # os.remove(os.path.join(folder_path, filename))
            # print(f"Removed file {filename}")

print("\nDone checking files.")