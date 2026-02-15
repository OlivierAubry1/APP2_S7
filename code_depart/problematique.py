
import numpy

import helpers.dataset as dataset


def problematique():
    images = dataset.ImageDataset("data/image_dataset/")

    # TODO Problématique: Générez une représentation des images appropriée
    # pour la classification comme dans le laboratoire 1.
    # -------------------------------------------------------------------------
    features = numpy.zeros((len(images), 3), dtype=numpy.float32) # (représentation fictive, à remplacer)
    representation = dataset.Representation(data=features, labels=images.labels)
    # -------------------------------------------------------------------------

    # TODO: Problématique: Visualisez cette représentation
    # -------------------------------------------------------------------------
    # 
    # -------------------------------------------------------------------------

    # TODO: Problématique: Comparez différents classificateurs sur cette
    # représentation, comme dans le laboratoire 2 et 3.
    # -------------------------------------------------------------------------
    # 
    # -------------------------------------------------------------------------


if __name__ == "__main__":
    problematique()
