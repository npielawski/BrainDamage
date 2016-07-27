#!/usr/bin/python3
import sys, os
from PIL import Image
import numpy as np

# Author: Nicolas Pielawski
# Creation date: July 26 2016

if len(sys.argv) < 3:
    print("Generate the blob (average picture) of a dataset")
    print("Usage: ./generate_blob.py <dataset folder> <output folder>")
    sys.exit()

folder_path = sys.argv[1]
output_path = sys.argv[2]

shape = None
blob = None
count = 0
file_number = len([name for name in os.listdir(folder_path)])

for subdir, dirs, files in os.walk(folder_path):
    for file in files:
        im = np.array(Image.open(folder_path + "/" + file))
        if shape == None:
            shape = im.shape
            print("Shape is", shape)
            blob = np.zeros(shape)
        if im.shape != shape: print("Image", file, " has a different shape! file skipped!")
        else: count += 1.

        blob += im
        
        percentage = 100. * count / file_number
        print("[{:.2f}%] Processing {}     ".format(percentage, file), end="\r")

print()

blob = blob / count
blobim = Image.fromarray(np.uint8(blob))
blobim.save(output_path + "/blob.png")
print("Image saved in", output_path)
print()
