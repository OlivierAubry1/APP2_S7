import abc
import os
import pathlib

from typing import List, Tuple

import numpy

# Must be call before any other TensorFlow/Keras import
# Suppress oneDNN custom operations info
# Suppress INFO and WARNING messages from TF (0=all, 1=no INFO, 2=no INFO/WARN, 3=no error)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import keras # pylint: disable=wrong-import-position


class Dataset(abc.ABC):
    """
    Classe abstraite représentant un ensemble de données indexable.

    Attributes:
        path (pathlib.Path): Le chemin vers le dataset.
        labels (numpy.ndarray): Les labels associés aux données.

    Methods:
        sample(n_samples: int, seed=None) -> List[Tuple]: Retourne `n_samples` échantillons aléatoires du dataset.
        get_class(label) -> "Subset": Retourne un sous-ensemble du dataset avec le label donné.
    """
    path: pathlib.Path

    labels: numpy.ndarray

    def __init__(self, path: str):
        """
        Args:
            path (str): Le chemin vers le dataset.
        """
        self.path = pathlib.Path(path)

    @abc.abstractmethod
    def __getitem__(self, index: int) -> Tuple:
        pass

    @abc.abstractmethod
    def __len__(self):
        pass

    def sample(self, n_samples: int, seed=None) -> List[Tuple]:
        """
        Retourne `n_samples` échantillons aléatoires du dataset.

        Args:
            n_samples (int): Le nombre d'échantillons à retourner.
            seed (int, optional): La graine pour le générateur de nombres aléatoires.

        Returns:
            List[Tuple]: Une liste d'échantillons du dataset.
        """
        rng = numpy.random.default_rng(seed)
        samples = rng.choice(len(self), size=n_samples, replace=False)
        return [self[i] for i in samples]

    def get_class(self, label) -> "Subset":
        """
        Retourne un sous-ensemble du dataset avec le label donné.

        Args:
            label: Le label de la classe à extraire.

        Returns:
            Subset: Un sous-ensemble du dataset contenant uniquement les échantillons avec le label donné.
        """
        if label not in self.labels:
            unique_labels = numpy.unique(self.labels)
            raise ValueError(f"Label {label} not found in labels_map. Available labels: {unique_labels}")

        indices = numpy.where(self.labels == label)[0]
        return Subset(self, indices)


class Subset(Dataset):
    """
    Sous-ensemble d'un dataset.

    Attributes:
        dataset (Dataset): Le dataset d'origine.
        indices (List[int]): Les indices des échantillons dans le dataset d'origine.
    """
    def __init__(self, dataset: Dataset, indices: List[int]):
        """
        Args:
            dataset (Dataset): Le dataset d'origine.
            indices (List[int]): Les indices des échantillons dans le dataset d'origine
                pour former le sous-ensemble.
        """
        super().__init__(str(dataset.path))

        self.dataset = dataset
        self.indices = indices
        self.labels = dataset.labels

    def __getitem__(self, index: int) -> Tuple:
        return self.dataset[self.indices[index]]

    def __len__(self):
        return len(self.indices)


class ImageDataset(Dataset):
    """
    Classe représentant un dataset d'images.

    Attributes:
        images (List[str]): Les chemins vers les images.
        labels (numpy.ndarray): Les labels associés aux images.
        unique_labels (numpy.ndarray): Les labels uniques dans le dataset.
    """
    images: List[str]
    labels: numpy.ndarray
    unique_labels: numpy.ndarray

    def __init__(self, path):
        """
        Args:
            path (str): Le chemin vers le dataset.
        """
        super().__init__(path)
        self.images = list(self.path.glob("*.jpg"))

        self.labels = numpy.array([file.name.split("_")[0] for file in self.images])
        self.unique_labels = numpy.unique(self.labels)

    def __getitem__(self, index: int) -> Tuple:
        """
        Returns the image of shape (224, 224, 3) and the label of the image.
        """
        filename = self.images[index]
        img_path = filename

        image = keras.preprocessing.image.load_img(img_path)
        image = keras.preprocessing.image.img_to_array(image)

        return image, self.labels[index]

    def __len__(self):
        return len(self.images)


class MultimodalDataset(Dataset):
    """
    Classe représentant un dataset multimodal (3 distributions gaussiennes en 2D).

    Attributes:
        data (numpy.ndarray): Les données du dataset.
        labels (numpy.ndarray): Les labels associés aux données.
        unique_labels (numpy.ndarray): Les labels uniques dans le dataset.
    """
    data: numpy.ndarray
    labels: numpy.ndarray
    unique_labels: numpy.ndarray

    def __init__(self, path: str):
        """
        Args:
            path (str): Le chemin vers le dataset.
        """
        super().__init__(path)

        self.data = numpy.empty((0, 2))
        self.labels = numpy.empty((0, ))

        for class_name in ["C1", "C2", "C3"]:
            class_path = self.path / f"{class_name}.txt"

            samples = numpy.loadtxt(class_path)

            self.data = numpy.vstack((self.data, samples))
            self.labels = numpy.concatenate((self.labels, numpy.full((len(samples), ), class_name)))

        self.unique_labels = numpy.unique(self.labels)

    def __getitem__(self, index: int) -> Tuple:
        """
        Returns the data sample and its label.
        """
        return self.data[index], self.labels[index]

    def __len__(self):
        return self.data.shape[0]


class Representation(Dataset):
    """
    Classe représentant un ensemble de donnée simplifié en une représentation
    de base dimensionnalité.

    Cette classe permet de charger un ensemble de données de faible dimensionnalité qui est
    stocké en mémoire sous forme de tableaux numpy.

    Attributes:
        dim (int): La dimension des données.
        data (numpy.ndarray): Les données du dataset.
        labels (numpy.ndarray): Les labels associés aux données.
        unique_labels (numpy.ndarray): Les labels uniques dans le dataset.

    Methods:
        get_class(label) -> numpy.ndarray: Retourne tout les données avec le label donné.
    """
    dim: int
    data: numpy.ndarray
    labels: numpy.ndarray
    unique_labels: numpy.ndarray

    def __init__(self, data: numpy.ndarray, labels: numpy.ndarray):
        """
        Args:
            data (numpy.ndarray): Les données du dataset.
            labels (numpy.ndarray): Les labels associés aux données.
        """
        assert data.shape[-1] <= 10, "Vous êtes contraint à 10 dimensions ou moins pour la problématique."
        assert data.shape[0] == labels.shape[0], "Data and labels must have the same number of samples."

        super().__init__(path="")

        self.data = data
        self.labels = labels
        self.unique_labels = numpy.unique(labels)
        self.dim = data.shape[1]

    def __getitem__(self, index: int) -> Tuple:
        """
        Returns the data sample and its label.
        """
        return self.data[index], int(self.labels[index])

    def __len__(self):
        return self.data.shape[0]

    def get_class(self, label: str) -> numpy.ndarray:
        """
        Permet de récupérer toutes les données appartenant à une classe donnée.

        Args:
            label (str): Le label de la classe à extraire.

        Returns:
            numpy.ndarray: Un tableau contenant toutes les données avec le label donné.

        Raises:
            ValueError: Si le label n'existe pas dans les labels du dataset.
        """
        if label not in self.unique_labels:
            raise ValueError(f"Label {label} not found in labels. Available labels: {self.unique_labels}")

        indices = numpy.where(self.labels == label)[0]
        return self.data[indices]


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    dataset = ImageDataset("data/image_dataset/")

    print(f"Number of images: {len(dataset)}")

    img, label = dataset[0]

    plt.figure()
    plt.imshow(img / 255.0)
    plt.title(label)
    plt.show()

    dataset2 = MultimodalDataset("data/data_3classes/")

    print(f"Number of samples: {len(dataset2)}")

    sample, label = dataset2[0]

    print(f"Sample: {sample}, Label: {label}")

    fig = plt.figure()
    plt.scatter(dataset2.data[:, 0], dataset2.data[:, 1], c=dataset2.labels, cmap='viridis', alpha=0.25)
    plt.title("LaboDataset samples")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()
