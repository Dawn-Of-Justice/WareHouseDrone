import csv
import json

# Load JSON data
with open('arena_mapper.json', 'r') as json_file:
    arena_data = json.load(json_file)

# Load TXT data
pixel_data = {}
with open('pixel_values.txt', 'r') as txt_file:
    for line in txt_file:
        key, value = line.split(':')
        pixel_data[key.strip()] = value.strip().strip('()').split(',')

# Create CSV and ignore points with A/Q and 1/17
with open('output.csv', 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    # Write header
    writer.writerow(['Point', 'X', 'Y', 'Z', 'PixelX', 'PixelY'])
    
    # Write data rows
    for point, coords in arena_data.items():
        if point.startswith('A') or point.startswith('Q') or point.endswith('1') or point.endswith('17'):
            continue
        pixel = pixel_data.get(point, ["N/A", "N/A"])
        writer.writerow([point, *coords, pixel[0].strip(), pixel[1].strip()])
