import csv
import os
from collections import OrderedDict

input_file = 'data/itineraries.csv'
output_dir = 'data/legs'

os.makedirs(output_dir, exist_ok=True)

# Parameters to manage open file handles
MAX_OPEN_FILES = 100  # Adjust based on your system's limit
file_handles = OrderedDict()

def get_writer(legId, header):
    if legId in file_handles:
        return file_handles[legId]['writer']
    else:
        if len(file_handles) >= MAX_OPEN_FILES:
            # Close the oldest opened file
            oldest_legId, oldest_file = file_handles.popitem(last=False)
            oldest_file['file'].close()
        file_path = os.path.join(output_dir, f'{legId}.csv')
        f = open(file_path, 'a', newline='', encoding='utf-8')
        writer = csv.writer(f)
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            writer.writerow(header)  # Write header if file is new
        file_handles[legId] = {'file': f, 'writer': writer}
        return writer

with open(input_file, 'r', newline='', encoding='utf-8') as infile:
    reader = csv.reader(infile)
    header = next(reader)  # Read header
    legId_index = header.index('legId')  # Find the index of 'legId'

    for row in reader:
        legId = row[legId_index]
        writer = get_writer(legId, header)
        writer.writerow(row)

# Close any remaining open files
for fh in file_handles.values():
    fh['file'].close()

print("Splitting completed using optimized CSV module.")
