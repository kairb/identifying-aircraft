from PIL import Image, ImageDraw


class PDF:
    @staticmethod
    def write_to_pd(images):
        for image in images:
            im = Image.fromarray(image)

    @staticmethod
    def write_to_folder(images, predictions, probabilities):
        '''
        saves images to results folder with probabilites and predictions
        :param images: images array
        :param predictions: prediction array
        :param probabilities: array of probabilities
        :return: none
        '''
        result_path = "../Results/"
        aircraft_path = result_path + "Aircraft/"
        ground_path = result_path + "Ground/"
        i = 0
        print("image len", len(images))
        for image in images:
            img = Image.fromarray(image)
            draw = ImageDraw.Draw(img)
            if probabilities[i][1] > 0.4:
                draw.text((0, 0), str(probabilities[i]), (255, 0, 0))
            else:
                draw.text((0, 0), str(probabilities[i]), (255, 0, 0))

            img.save(ground_path + str(i) + ".png")

            i += 1
            ##comment
