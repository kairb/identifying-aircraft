from tkinter import *
from data_parser import Data
from sklearn import svm
from matplotlib import pyplot as plt
from tkinter.filedialog import askopenfilename
from image_parser import Parser
from PIL import Image, ImageTk
from pdf import PDF


class GUI:
    def __init__(self):
        self.root = Tk()
        self.left_frame = Frame(self.root).grid(row=0, column=0)
        self.right_frame = Frame(self.root).grid(row=0, column=10)
        self.classification_method = IntVar()
        self.classification_method.set(2)
        self.x, self.y, self.x_step, self.y_step = StringVar(), StringVar(), StringVar(), StringVar()
        self.x.set("250")
        self.y.set("250")
        self.x_step.set("30")
        self.y_step.set("30")
        self.root.title("Aircraft Identification")
        #self.root.state('zoomed')
        self.image_path = "../Airports/3.png"
        self.label_image_path = StringVar()
        self.label_image_path.set(self.image_path[-7:])
        # self.text = Text(self.right_frame, height=30, width=30).pack(anchor=E)

    def home(self):
        """
        sets the homepage options
        """
        Label(self.left_frame, text="Select classification method and options").grid(row=0, column=0)

        Button(self.left_frame, text="Select image", command=self.file_selector).grid(row=1, column=0, sticky="W")
        Label(self.left_frame, textvariable=self.label_image_path).grid(row=1, column=0, sticky="E")

        Radiobutton(self.left_frame, text="Normal", variable=self.classification_method, value=1).grid(row=2, column=0,
                                                                                                       sticky="W")
        Radiobutton(self.left_frame, text="Image search", variable=self.classification_method, value=2).grid(row=3,
                                                                                                             column=0,
                                                                                                             sticky="W")

        Label(self.left_frame, text="x").grid(row=4, column=0, sticky="W")
        Entry(self.left_frame, textvariable=self.x).grid(row=4, column=0)

        Label(self.left_frame, text="y").grid(row=5, column=0, sticky="W")
        Entry(self.left_frame, textvariable=self.y).grid(row=5, column=0)

        Label(self.left_frame, text="x step").grid(row=6, column=0, sticky="W")
        Entry(self.left_frame, textvariable=self.x_step).grid(row=6, column=0)

        Label(self.left_frame, text="y step").grid(row=7, column=0, sticky="W")
        Entry(self.left_frame, textvariable=self.y_step).grid(row=7, column=0)

        Button(self.left_frame, text="Start", command=self.start_classification).grid(row=8)

    def start_classification(self):
        if self.classification_method.get() == 1:
            self.normal_classification()
        elif self.classification_method.get() == 2:
            self.image_search()

    def normal_classification(self):
        TRAINING_SET_SIZE = 90
        TEST_IMAGES = [95, 195, 96, 97, 198, 88, 98, 99, 94, 191]

        # get training data in the form of feature descriptors
        # training_set, training_labels = Data.create_hog_data_set(TRAINING_SET_SIZE)
        # test_images = Data.create_hog_image(TEST_IMAGES)
        training_set, training_labels = Data.create_realistic_hog_data_set(TRAINING_SET_SIZE)
        test_images = Data.create_realistic_hog_test_set(TEST_IMAGES)

        clf = svm.SVC(gamma=0.0001, C=10)

        # fit training data
        clf.fit(training_set, training_labels)

        result = clf.predict(test_images)
        print("Prediction: ", result)

        # GUI
        MAX_COLUMNS = 5
        MAX_ROWS = 2

        for i in range(1, len(test_images) + 1):
            plt.subplot(MAX_ROWS, MAX_COLUMNS, i)
            plt.imshow(Parser.load_full_size_image(TEST_IMAGES[i - 1]), cmap="gray")
            if result[i - 1] == 1:
                plt.title("Aircraft")
            else:
                plt.title("Ground")
            plt.axis('off')

        plt.show()

    def image_search(self):
        """
        searches images for aircraft
        :return:
        """
        # training_set, training_labels = Data.create_resized_hog_data_set(int(self.x.get()), int(self.y.get()))

        training_set, training_labels = Data.create_training_data(int(self.x.get()), int(self.y.get()))

        test_images, images = Data.create_airport_hog_data_set(self.image_path, int(self.x_step.get()),
                                                               int(self.y_step.get()),
                                                               int(self.x.get()), int(self.y.get()))

        clf = svm.SVC(gamma=0.0001, C=10, probability=True)

        # fit training data
        clf.fit(training_set, training_labels)

        result = clf.predict(test_images)

        probability = clf.predict_proba(test_images)

        for i in range(len(result)):
            print(result[i], "Probability  ", probability[i])

        PDF.write_to_folder(images, result, probability)

        MAX_COLUMNS = 5
        MAX_ROWS = 10

        for i in range(1, 50 + 1):
            plt.subplot(MAX_ROWS, MAX_COLUMNS, i)
            plt.imshow(images[i - 1], cmap="gray")
            label = str(result[i - 1]) + str(probability[i - 1])
            plt.title(label)
            plt.axis('off')

        plt.show()
        print("past plt")


    def file_selector(self):
        """
        sets image_path to users choice using file dialog
        """
        Tk().withdraw()  # we don't want a full GUI, so keep the root window from appearing
        self.image_path = askopenfilename()  # show an "Open" dialog box and return the path to the selected file
        self.label_image_path.set(self.image_path[-7:])

    def run(self):
        self.root.mainloop()

    def generate_pdf(self):
        pass


if __name__ == "__main__":
    gui = GUI()
    gui.home()
    gui.run()
