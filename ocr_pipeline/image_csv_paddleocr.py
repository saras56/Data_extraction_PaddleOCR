from paddleocr import PaddleOCR
from glob import glob
import csv

class OCRProcessor:
    def __init__(self, lang='en'):
        self.ocr_model = PaddleOCR(lang=lang)
        self.final_result = []
        self.tuple_elements = []

    def extract_ocr_results(self, path):
        for f in glob(path):
            result = self.ocr_model.ocr(f)
            self.final_result.append(result)

    def extract_tuple_elements(self):
        for sublist in self.final_result:
            for inner_list in sublist:
                for sub_inner_list in inner_list:
                    for i in range(len(sub_inner_list)):
                        tuple_data = sub_inner_list[i][0]
                        if i % 2 != 0:
                            self.tuple_elements.append(tuple_data)

    def write_to_csv(self, csv_file):
        alternate_elements_1 = self.tuple_elements[::2]
        alternate_elements_2 = self.tuple_elements[1::2]

        with open(csv_file, 'w', newline='') as file:
            writer = csv.writer(file)
            for element1, element2 in zip(alternate_elements_1, alternate_elements_2):
                writer.writerow([element1, element2])

def process_images_and_save_to_csv(path, csv_file):
    processor = OCRProcessor()
    processor.extract_ocr_results(path)
    processor.extract_tuple_elements()
    processor.write_to_csv(csv_file)

if __name__ == "__main__":
    path = './Dataset/*.png'
    csv_file = "bounding_boxes.csv"
    process_images_and_save_to_csv(path, csv_file)