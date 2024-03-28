import os
import csv
from pathlib import Path
from PIL import Image
import pytesseract
import shutil
import re

folder_path = r'..\First_Project\screening_data_csv_1'
csv_file_path = 'output.csv'

with open(csv_file_path, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)

    # Write header to CSV file
    csvwriter.writerow(['Column 1', 'Column 2'])

    # Loop through each file in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg'):
            # Open the image file
            image_path = os.path.join(folder_path, filename)
            image = Image.open(image_path)

            # Extract text from the image
            extracted_text = pytesseract.image_to_string(image)

            # Extract two lists of numbers from the extracted text
            numbers = re.findall(r'\d+', extracted_text)
            list_1 = numbers[:4]
            list_2 = numbers[4:]

            # Write the two lists to CSV file
            csvwriter.writerow([', '.join(list_1), ', '.join(list_2)])

print('Text extraction completed. CSV file saved as', csv_file_path)