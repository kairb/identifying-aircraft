from tkinter import *
from data_parser import Data
from sklearn import svm
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
from image_parser import Parser
from drawing import Draw
import numpy as np
from save import Save
from random import randint


class GUI:
    def __init__(self):
        self.root = Tk()
        self.left_frame = Frame(self.root).grid(row=0, column=0)
        self.right_frame = Frame(self.root).grid(row=0, column=10)
        self.classification_method = IntVar()
        self.classification_method.set(2)
        self.x, self.y, self.x_steps, self.y_steps = StringVar(), StringVar(), StringVar(), StringVar()
        self.x.set("250")
        self.y.set("250")
        self.x_steps.set("8")
        self.y_steps.set("8")
        self.root.title("Aircraft Identification")
        self.image_path = "../Airports/12.png"
        self.label_image_path = StringVar()
        self.label_image_path.set(self.image_path[-7:])
        self.save = Save()

    def home(self):
        """
        sets the homepage options
        """
        Label(self.left_frame, text="Select classification method and options").grid(row=0, column=0)
        Radiobutton(self.left_frame, text="Standalone", variable=self.classification_method, value=1).grid(row=1,
                                                                                                           column=0,
                                                                                                           sticky="W")
        Radiobutton(self.left_frame, text="Image search", variable=self.classification_method, value=2).grid(row=2,
                                                                                                             column=0,
                                                                                                             sticky="W")

        Button(self.left_frame, text="Select image", command=self.file_selector).grid(row=3, column=0, sticky="W")
        Label(self.left_frame, textvariable=self.label_image_path).grid(row=3, column=0, sticky="E")

        Label(self.left_frame, text="x").grid(row=4, column=0, sticky="W")
        Entry(self.left_frame, textvariable=self.x).grid(row=4, column=0)

        Label(self.left_frame, text="y").grid(row=5, column=0, sticky="W")
        Entry(self.left_frame, textvariable=self.y).grid(row=5, column=0)

        Label(self.left_frame, text="x steps").grid(row=6, column=0, sticky="W")
        Entry(self.left_frame, textvariable=self.x_steps).grid(row=6, column=0)

        Label(self.left_frame, text="y steps").grid(row=7, column=0, sticky="W")
        Entry(self.left_frame, textvariable=self.y_steps).grid(row=7, column=0)

        Button(self.left_frame, text="Start", command=self.start_classification).grid(row=8)

    def start_classification(self):
        if self.classification_method.get() == 1:
            self.normal_classification()
        elif self.classification_method.get() == 2:
            self.image_search()

    @staticmethod
    def normal_classification():
        """
        starts classification of simple images
        """
        training_set_size = 90
        test_images = []
        for i in range(0, 5):
            test_images.append(randint(90, 100))
            test_images.append(randint(190, 200))

        training_set, training_labels = Data.create_realistic_hog_data_set(training_set_size)
        test_set = Data.create_realistic_hog_test_set(test_images)

        clf = svm.SVC(gamma=0.0001, C=10, probability=True)

        # fit training data
        clf.fit(training_set, training_labels)

        result = clf.predict(test_set)
        probabilities = clf.predict_proba(test_set)

        # GUI
        MAX_COLUMNS = 5
        MAX_ROWS = 2

        for i in range(1, len(test_images) + 1):
            plt.subplot(MAX_ROWS, MAX_COLUMNS, i)
            plt.imshow(Parser.load_full_size_image(test_images[i - 1]), cmap="gray")
            if result[i - 1] == 1:
                plt.title('Aircraft %1.2f' % probabilities[i - 1][1])
            else:
                plt.title('Ground %1.2f' % probabilities[i - 1][0])
            plt.axis('off')

        plt.show(block=False)

    def image_search(self):
        """
        searches images for aircraft
        """

        training_set, training_labels = Data.create_training_data(int(self.x.get()), int(self.y.get()))

        test_images, images = Data.create_airport_hog_data_set(self.image_path, int(self.x_steps.get()),
                                                               int(self.y_steps.get()),
                                                               int(self.x.get()), int(self.y.get()))

        clf = svm.SVC(gamma=0.0001, C=10, probability=True)

        # fit training data
        clf.fit(training_set, training_labels)

        #obtain probabilities
        probabilities = clf.predict_proba(test_images)

        #drawing of results.
        draw = Draw(np.asarray(Parser.load_image_from_path(self.image_path)))

        boxes = draw.draw_boxes(probabilities, int(self.x.get()), int(self.y.get()), int(self.x_steps.get()),
                                int(self.y_steps.get()))
        print(boxes.shape)

        self.save.save_search_results(boxes)

        heat = draw.draw_colour_gradient(probabilities, int(self.x.get()), int(self.y.get()), int(self.x_steps.get()),
                                         int(self.y_steps.get()))
        print(heat.shape)
        self.save.save_heat_map(heat)

        plt.subplot(111)
        plt.title("heatmap")
        plt.imshow(heat, cmap= plt.gray())
        plt.subplot(212)
        plt.title("Search results")
        plt.imshow(boxes)

        plt.show(block=False)

    def file_selector(self):
        """
        sets image_path to users choice using file dialog
        """
        Tk().withdraw()  # stop root window from appearing
        self.image_path = askopenfilename()  # show an "Open" dialog box and return the path to the selected file
        self.label_image_path.set(self.image_path[-7:])

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    gui = GUI()
    gui.home()
    gui.run()
