import matplotlib.pyplot as plt
import matplotlib.image as mpimg


class Load:
    def __init__(self):
        pass

    image = ''

    def load_image(self):
        self.image = mpimg.imread("1.png")

    def display_image(self):
        plt.imshow(self.image)
        plt.show()


loader = Load()

loader.load_image()
loader.display_image()

