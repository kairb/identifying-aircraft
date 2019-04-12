from PIL import Image, ImageDraw
import uuid


class Save:

    def __init__(self):
        self.RESULT_PATH = "../Results/"
        self.AIRCRAFT_PATH = self.RESULT_PATH + "Aircraft/"
        self.GROUND_PATH = self.RESULT_PATH + "Ground/"
        self.HEAT_MAP_PATH = self.RESULT_PATH + "Heatmaps/"
        self.SEARCH_RESULTS_PATH = self.RESULT_PATH + "Search/"


    def write_to_folder(self, images, probabilities):
        '''
        saves images to results folder with probabilities and predictions
        :param images: images array
        :param probabilities: array of probabilities
        :return: none
        '''

        i = 0
        for image in images:
            img = Image.fromarray(image)
            draw = ImageDraw.Draw(img)
            lowercase_str = uuid.uuid4().hex
            if probabilities[i][1] > 0.4:
                draw.text((0, 0), str(probabilities[i]), (255, 0, 0))
                img.save(self.AIRCRAFT_PATH + lowercase_str + ".png")

            else:
                draw.text((0, 0), str(probabilities[i]), (255, 0, 0))
                img.save(self.GROUND_PATH + lowercase_str + ".png")

            i += 1

    def save_heat_map(self, image):
        """
        Saves heat map to heat map path
        :param image: image
        :return: none
        """
        lowercase_str = uuid.uuid4().hex
        img = Image.fromarray(image)
        img.convert('RGB').save(self.HEAT_MAP_PATH + lowercase_str + ".png")

    def save_search_results(self, image):
        """
        saves search results to search result path
        :param image:image
        :return: none
        """
        lowercase_str = uuid.uuid4().hex
        img = Image.fromarray(image)
        img.convert('RGB').save(self.SEARCH_RESULTS_PATH + lowercase_str + ".png")


