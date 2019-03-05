from image_parser import Parser
import numpy as np
from PIL import Image


class Partitioner:
    @staticmethod
    def create_subsections(path, x_step, y_step, size_x, size_y):
        image = np.asarray(Parser.load_image_from_path(path))
        images = []
        for x in range(0, len(image[0]) - size_x + 1, x_step):
            for y in range(0, len(image) - size_y + 1, y_step):
                sub_image = image[y:y + size_y, x: x + size_x]
                images.append(sub_image)

        return images

    @staticmethod
    def save_images(images):
        i = 5000
        for image in images:
            im = Image.fromarray(image)
            im.save("../gen/" + str(i) + ".PNG")
            i += 1


out = Partitioner.create_subsections(r'D:\University\Year 3\CE301\capstone_project\Airports\6.PNG', 50, 50, 300, 300)
Partitioner.save_images(out)
