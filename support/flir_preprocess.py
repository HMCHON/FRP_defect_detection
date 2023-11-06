from flir_tool import *

class Preprocess:
    def __init__(self, csv_file, image_size, stride, defect_area):
        self.csv_file_path = csv_file
        self.image_size = int(image_size)
        self.stride = int(stride)
        self.defect_area = float(defect_area)

        self.H = 15 #GFRP height
        self.W = 210 #GFRP width

        self.csv_load()
        separate_frames(self.csv_file_path, self.csv_file, self.stride, self.defect_area, self.image_size, self.image_size, self.W, self.H)

    def csv_load(self):
        self.csv_file = load_csv(self.csv_file_path)
        self.csv_file = extract_frames(self.csv_file)


def set_contact():
    csv_file = file_choose()
    image_size = input("image height:")
    stride = input("Stride:")
    defect_area = input("defect loc:")
    Preprocess(csv_file, image_size, stride, defect_area)


def run():
    set_contact()

if __name__ == "__main__":
    run()



