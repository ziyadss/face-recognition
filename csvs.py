import csv
from collections import defaultdict

with open("data/boxes.csv", "r") as boxes_file, open(
    "data/identities.csv", "r"
) as identities_file:
    boxes_reader = csv.reader(boxes_file)
    identities_reader = csv.reader(identities_file)

    # Open the output file in write mode
    with open("data/information.csv", "w") as output_file:
        # Create a CSV writer for the output file
        output_writer = csv.writer(output_file, lineterminator="\n")

        # Iterate over the rows in both files in parallel
        for (boxes_row, identities_row) in zip(boxes_reader, identities_reader):
            # Combine the rows into a single list
            combined_row = boxes_row + identities_row[1:]
            # Write the combined row to the output file
            output_writer.writerow(combined_row)

identity_files = defaultdict(list)
with open("data/information.csv", "r") as fd:
    reader = csv.reader(fd)
    next(reader)

    for row in reader:
        file = row[0]
        x1, y1 = int(row[1]), int(row[2])
        w, h = int(row[3]), int(row[4])

        if w == 0 or h == 0:
            continue

        identity = row[5]
        identity_files[identity].append((file, x1, y1, w, h))

# Filter to identities with 30 or more files
identity_files = {k: v for k, v in identity_files.items() if len(v) >= 30}

# Write the filtered identities to a file
with open("data/filtered_information.csv", "w") as fd:
    writer = csv.writer(fd, lineterminator="\n")
    writer.writerow(["file", "x1", "y1", "w", "h", "identity"])
    for identity, files in identity_files.items():
        for file, x1, y1, w, h in files:
            writer.writerow([file, x1, y1, w, h, identity])
