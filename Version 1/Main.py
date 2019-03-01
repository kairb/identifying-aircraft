from tkinter import *
from data_parser import Data
from sklearn import svm
from matplotlib import pyplot as plt
from tkinter.filedialog import askopenfilename
from PIL import Image, ImageTk


class GUI:
    def __init__(self):
        self.root = Tk()
        self.left_frame = Frame(self.root).pack(anchor=NW)
        self.right_frame = Frame(self.root).pack(anchor=NE)
        self.classification_method = IntVar()
        self.classification_method.set(2)
        self.x, self.y, self.x_step, self.y_step = StringVar(), StringVar(), StringVar(), StringVar()
        self.x.set("250")
        self.y.set("250")
        self.x_step.set("50")
        self.y_step.set("50")
        self.root.title("Aircraft Identification")
        #self.root.state('zoomed')
        self.image_path = "../Airports/3.png"
        # self.text = Text(self.right_frame, height=30, width=30).pack(anchor=E)

    def home(self):
        Label(self.left_frame, text="Select classification method and options").pack(anchor=W)

        Button(self.left_frame, text="select image", command=self.file_selector).pack(anchor = W)

        Radiobutton(self.left_frame, text="Normal", variable=self.classification_method, value=1).pack(anchor=W)
        Radiobutton(self.left_frame, text="Image search", variable=self.classification_method, value=2).pack(anchor=W)

        Label(self.left_frame, text="Options")

        Label(self.left_frame, text="x").pack(anchor=W)
        Entry(self.left_frame, textvariable=self.x).pack(anchor=W)

        Label(self.left_frame, text="y").pack(anchor=W)
        Entry(self.left_frame, textvariable=self.y).pack(anchor=W)

        Label(self.left_frame, text="x step").pack(anchor=W)
        Entry(self.left_frame, textvariable=self.x_step).pack(anchor=W)

        Label(self.left_frame, text="y step").pack(anchor=W)
        Entry(self.left_frame, textvariable=self.y_step).pack(anchor=W)

        Button(self.left_frame, text="Start", command=self.start_classification).pack(anchor=W)

    def start_classification(self):
        if self.classification_method.get() == 1:
            self.normal_classification()
        elif self.classification_method.get() == 2:
            self.image_search()

    def normal_classification(self):
        pass

    def image_search(self):
        training_set, training_labels = Data.create_resized_hog_data_set(int(self.x.get()), int(self.y.get()))
        test_images, images = Data.create_airport_hog_data_set(self.image_path, int(self.x_step.get()), int(self.y_step.get()),
                                                               int(self.x.get()), int(self.y.get()))

        clf = svm.SVC(gamma=0.0001, C=10, probability=True)

        # fit training data
        clf.fit(training_set, training_labels)

        result = clf.predict(test_images)

        probability = clf.predict_proba(test_images)

        for i in range(len(result)):
            print(result[i], "Probability  ", probability[i])

        MAX_COLUMNS = 5
        MAX_ROWS = 10

        for i in range(1, 50 + 1):
            plt.subplot(MAX_ROWS, MAX_COLUMNS, i)
            plt.imshow(images[i - 1], cmap="gray")
            label = str(result[i-1]) + str(probability[i-1])
            plt.title(label)
            plt.axis('off')


        plt.show()
        print("images in dataset" , len(images))

    def file_selector(self):
        """
        sets image_path to users choice
        :return:
        """
        Tk().withdraw()  # we don't want a full GUI, so keep the root window from appearing
        self.image_path = askopenfilename()  # show an "Open" dialog box and return the path to the selected file

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    gui = GUI()
    gui.home()
    gui.run()
