import csv
import shutil
import os

SRC_DIR = "C:/Users/ziyad/Downloads/celebs/img_celeba"
DST_DIR = "C:/Users/ziyad/Downloads/PICS"

with open("data/filtered_information.csv", 'r') as fd:
    reader = csv.reader(fd)
    next(reader)

    files = [row[0] for row in reader]

for file in files:
    src = os.path.join(SRC_DIR, file)
    dst = os.path.join(DST_DIR, file)

    shutil.copy(src, dst)
