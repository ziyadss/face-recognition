import csv
from collections import defaultdict
from time import perf_counter_ns

from skimage import io

from get_image_size import get_image_metadata

IMAGES_DIR = "C:/Users/ziyad/Downloads/celebs/img_celeba"

# with open("data/boxes.csv", "r") as boxes_file, open(
#     "data/identities.csv", "r"
# ) as identities_file:
#     boxes_reader = csv.reader(boxes_file)
#     identities_reader = csv.reader(identities_file)

#     with open("data/information.csv", "w") as output_file:
#         output_writer = csv.writer(output_file, lineterminator="\n")

#         for (boxes_row, identities_row) in zip(boxes_reader, identities_reader):
#             combined_row = boxes_row + identities_row[1:]
#             output_writer.writerow(combined_row)

start = perf_counter_ns()
identity_files = defaultdict(list)
with open("data/information.csv", "r") as fd:
    reader = csv.reader(fd)
    next(reader)

    for row in reader:
        file = row[0]
        x1, y1 = int(row[1]), int(row[2])
        w, h = int(row[3]), int(row[4])

        if w < 39 or h < 39:
            continue

        # get the height and width of the image, if too big skip
        max_size = 364 * 1024  # 364 KB
        size, width, height = get_image_metadata(f"{IMAGES_DIR}/{file}", max_size)
        if (
            size > max_size
            or width > 1000
            or height > 1000
            or width < 75
            or height < 75
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

print("Before count filter:")
# Print total size
total_size = sum(size for row in identity_files.values() for _, _, _, _, _, size in row)
print(f"Total size: {total_size / 1024 / 1024:.2f} MB")

file_count = sum(len(v) for v in identity_files.values())
print(f"Total files: {file_count}")

identity_count = len(identity_files)
print(f"Total identities: {identity_count}")

# Filter to identities with 30 or more files
identity_files = {k: v for k, v in identity_files.items() if len(v) >= 30}

print("Before most filter:")
# Print total size
total_size = sum(size for row in identity_files.values() for _, _, _, _, _, size in row)
print(f"Total size: {total_size / 1024 / 1024:.2f} MB")

file_count = sum(len(v) for v in identity_files.values())
print(f"Total files: {file_count}")

identity_count = len(identity_files)
print(f"Total identities: {identity_count}")

# get x identities with the most files
identity_files = dict(
    sorted(identity_files.items(), key=lambda x: len(x[1]), reverse=True)[:70]
)

print("After most filter:")
# Print total size
total_size = sum(size for row in identity_files.values() for _, _, _, _, _, size in row)
print(f"Total size: {total_size / 1024 / 1024:.2f} MB")

file_count = sum(len(v) for v in identity_files.values())
print(f"Total files: {file_count}")

identity_count = len(identity_files)
print(f"Total identities: {identity_count}")

# Write the filtered identities to a file
with open("data/filtered_information.csv", "w") as fd:
    writer = csv.writer(fd, lineterminator="\n")
    writer.writerow(["file", "x1", "y1", "w", "h", "identity"])
    for identity, files in identity_files.items():
        for file, x1, y1, w, h, *_ in files:
            writer.writerow([file, x1, y1, w, h, identity])
