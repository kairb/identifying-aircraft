import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import data, exposure
from image_parser import Parser


class HOG:
    @staticmethod
    # returns feature descriptor of image supplied flattened
    def create_hog_image(image):
        # max_features = 10000
        # cell_size_x = max_features / len(image)
        # cell_size_y = max_features / len(image[0])

        fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                            cells_per_block=(1, 1), visualize=True, multichannel=None, feature_vector=True,
                            block_norm='L2-Hys')
        return fd.ravel()

    @staticmethod
    # displays hog feature descriptor over original image
    def display_hog_image(image):
        max_features = 10000
        cell_size_x = max_features / len(image)
        cell_size_y = max_features / len(image[0])

        print(cell_size_x)
        print(cell_size_y)
        # generate feature descriptor and hog image
        fd, hog_image = hog(image, orientations=8, pixels_per_cell=(cell_size_y, cell_size_x),
                            cells_per_block=(1, 1), visualize=True, multichannel=None, feature_vector=True,
                            block_norm='L2-Hys')
        print(len(fd))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

        ax1.axis('off')
        ax1.imshow(image, cmap=plt.cm.gray)
        ax1.set_title('Input image')

        # Rescale histogram for better display
        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

        ax2.axis('off')
        ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
        ax2.set_title('Histogram of Oriented Gradients')
        plt.show()

# HOG.display_hog_image(Parser.load_image(49))
# HOG.display_hog_image(Parser.load_image(50))
