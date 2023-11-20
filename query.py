from __future__ import print_function

import sys
from src.color import Color
from src.daisy import Daisy
from src.DB import Database
from src.edge import Edge
from src.evaluate import infer
from src.gabor import Gabor
from src.HOG import HOG
from src.resnet import ResNetFeat
from src.vggnet import VGGNetFeat
from local_description import ImageSimilarityCalculator

depth = 5
d_type = 'd1'
query_idx = 0

def extract_features_from_image(self, img_path):
        return {
            'img': img_path,
            'cls': None,  # You may set the class to None for the target image
            'hist': self.histogram(img_path),
        }


if __name__ == '__main__':
    db = Database()

    # methods to call
    methods = {
        "color": Color,
        "daisy": Daisy,
        "edge": Edge,
        "hog": HOG,
        "gabor": Gabor,
        "vgg": VGGNetFeat,
        "resnet": ResNetFeat,
        "similarity-matrix": ImageSimilarityCalculator
    }

    try:
        mthd = sys.argv[1].lower()
        #target_img_path = sys.argv[2]  # Path to the target image
    except IndexError:
        print("usage: {} <method> <target_image_path>".format(sys.argv[0]))
        print("supported methods:\ncolor, daisy, edge, gabor, HOG, vgg, resnet, similarity-matrix")
        sys.exit(1)

    # call make_samples(db) accordingly
    samples = getattr(methods[mthd](), "make_samples")(db)

    # query the first img in data.csv
    query = samples[query_idx]

    # query a specific image that doesn't exist in data.csv
    #query = extract_features_from_image(target_img_path)


    _, result = infer(query, samples=samples, depth=depth, d_type=d_type)
    print(result)


