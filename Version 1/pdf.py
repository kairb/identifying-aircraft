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
        i = 0
        print("image len", len(images))
        for image in images:
            #ImageDraw.Draw(image).text((0, 0), str(predictions[i]) + str(probabilities[i]), (255, 0, 0))
            img = Image.fromarray(image)
            draw = ImageDraw.Draw(img)
            #font = ImageFont.truetype("sans-serif.ttf", 16)
            draw.text((0, 0), str(predictions[i]) + str(probabilities[i]), (255, 0, 0))
            img.save(result_path + str(i) + ".png")
            i += 1