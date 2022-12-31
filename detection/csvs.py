import csv
from collections import defaultdict
from time import perf_counter_ns

from get_image_size import get_image_metadata

IMAGES_DIR = "C:/Users/ziyad/Downloads/celebs/img_celeba"

start = perf_counter_ns()
identity_files = defaultdict(list)
with open("data/information.csv", "r") as fd:
    reader = csv.reader(fd)
    next(reader)

    for row in reader:
        file = row[0]
        x1, y1 = int(row[1]), int(row[2])
        w, h = int(row[3]), int(row[4])

        if w < 75 or h < 75:
            continue

        # get the height and width of the image, if too big skip
        max_size = 96 * 1024  # 96 KB
        size, width, height = get_image_metadata(f"{IMAGES_DIR}/{file}", max_size)
        if (
            size > max_size
            or width > 400
            or height > 400
            or width < 150
            or height < 150
        ):
            continue

        # check window is valid (sanity, doesn't happen)
        if x1 + w > width or y1 + h > height:
            continue

        # try:
        #     io.imread(f"{IMAGES_DIR}/{file}")
        # except:
        #     continue

        identity = row[5]
        identity_files[identity].append((file, x1, y1, w, h, size))
end = perf_counter_ns()
print(f"Time: {(end - start) / 1e9:.2f} seconds")


total_size = sum(size for row in identity_files.values() for _, _, _, _, _, size in row)
print(f"Total size: {total_size / 1024 / 1024:.2f} MB")

file_count = sum(len(v) for v in identity_files.values())
print(f"Total files: {file_count}")

identity_count = len(identity_files)
print(f"Total identities: {identity_count}")

# Write the filtered identities to a file
with open("data/detector_filtered_information.csv", "w") as fd:
    writer = csv.writer(fd, lineterminator="\n")
    writer.writerow(["file", "x1", "y1", "w", "h", "identity"])
    for identity, files in identity_files.items():
        for file, x1, y1, w, h, *_ in files:
            writer.writerow([file, x1, y1, w, h, identity])
