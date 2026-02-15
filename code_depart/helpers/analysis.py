import abc

from typing import Tuple

import numpy
import sklearn.metrics


def project_onto_new_basis(data: numpy.ndarray, basis: numpy.ndarray) -> numpy.ndarray:
    """
    Projette les données sur une nouvelle base.

    Args:
        data (numpy.ndarray): Les données à projeter, de forme (n_samples, n_features).
        basis (numpy.ndarray): Les nouveaux vecteurs de base, de forme (n_features, n_features).

    Returns:
        numpy.ndarray: Les données projetées, de forme (n_samples, n_features).
    """
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    if data.shape[-1] != basis.shape[0]: # assuming the last dimension of data is features
        raise ValueError("Data and basis dimensions do not match for projection.")

    # L1.E2.5 Complétez cette fonction pour projeter les données sur une nouvelle base
    # -------------------------------------------------------------------------
    return numpy.zeros((data.shape[0], basis.shape[-1]))  # Remplacez cette ligne par le code de projection réel
    # -------------------------------------------------------------------------


def compute_gaussian_model(data: numpy.ndarray):
    """
    Calcule un modèle gaussien à partir des données fournies.

    Args:
        data (numpy.ndarray): Les données d'entrée, de forme (n_samples, n_features).

    Returns:
        Tuple: La moyenne, la matrice de covariance, les valeurs propres et les vecteurs propres.
    """
    # L1.E3.2 Calculer la moyenne et la matrice de covariance des données
    # -------------------------------------------------------------------------
    mean = numpy.zeros(data.shape[1])
    covariance = numpy.eye(data.shape[1])
    # -------------------------------------------------------------------------

    # L1.E3.5 Calculer les valeurs propres et les vecteurs propres de la matrice de covariance
    # -------------------------------------------------------------------------
    eigenvalues, eigenvectors = numpy.zeros(data.shape[1]), numpy.zeros((data.shape[1], data.shape[1]))
    # -------------------------------------------------------------------------

    return mean, covariance, eigenvalues, eigenvectors


def rescale_lab(image: numpy.ndarray, n_bins: int):
    """
    Mise à l'échelle des canaux LAB de l'image dans la plage [0, n_bins - 1].

    Args:
        image (numpy.ndarray): L'image en espace de couleur LAB.
        n_bins (int): Le nombre de bins pour la mise à l'échelle.

    Returns:
        numpy.ndarray: L'image mise à l'échelle dans la plage [0, n_bins - 1].
    """
    # Constantes du domaine LAB
    # a and b are technically unbounded, but in practice they fall within this range
    min_l = 0
    max_l = 100
    min_ab = -128
    max_ab = 127

    # normalize lab to [0, 1] (assuming the channels being the last dimension)
    image[:, :, 0] = (image[:, :, 0] - min_l) / (max_l - min_l)
    image[:, :, 1:3] = (image[:, :, 1:3] - min_ab) / (max_ab - min_ab)

    # Scale to [0, n_bins - 1]
    scaled_lab = numpy.round(image * (n_bins - 1))

    return scaled_lab


def rescale_hsv(image, n_bins: int):
    """
    Mise à l'échelle des canaux HSV de l'image dans la plage [0, n_bins - 1].

    Args:
        image (numpy.ndarray): L'image en espace de couleur HSV.
        n_bins (int): Le nombre de bins pour la mise à l'échelle.

    Returns:
        numpy.ndarray: L'image mise à l'échelle dans la plage [0, n_bins - 1].
    """
    # HSV channels are in [0, 1]
    scaled_hsv = numpy.round(image * (n_bins - 1))
    return scaled_hsv


def generate_histograms(image: numpy.ndarray, n_bins: int):
    """
    Génère des histogrammes pour chaque canal de l'image.

    Args:
        image (numpy.ndarray): L'image d'entrée avec des canaux de couleur.
        n_bins (int): Le nombre de bins pour les histogrammes.

    Returns:
        numpy.ndarray: Un tableau 2D où chaque ligne correspond à l'histogramme d'un canal.
    """
    n_channels = image.shape[-1] # assuming the last dimension is channels

    histogram_counts = numpy.zeros((n_channels, n_bins), dtype=numpy.int64)
    for channel in range(n_channels):
        channel_histogram, _ = numpy.histogram(image[:, :, channel], bins=n_bins, range=(0, n_bins - 1))
        histogram_counts[channel] += channel_histogram

    return histogram_counts


def rescale_data(data: numpy.ndarray, min_range: int = -1, max_range: int = 1) -> numpy.ndarray:
    """
    Mise à l'échelle des données dans une plage spécifiée. (par défaut [-1, 1])

    Args:
        data (numpy.ndarray): Les données à mettre à l'échelle.
        min_range (int): La valeur minimale de la plage cible.
        max_range (int): La valeur maximale de la plage cible.

    Returns:
        numpy.ndarray: Les données mises à l'échelle dans la plage [min_range, max_range].
    """
    min_data = numpy.min(data, axis=0)
    max_data = numpy.max(data, axis=0)

    scaled_data = (max_range - min_range) * (data - min_data) / (max_data - min_data) + min_range
    return scaled_data


def compute_error_rate(targets: numpy.ndarray, predictions: numpy.ndarray) -> Tuple[float, numpy.ndarray]:
    """
    Calcule le taux d'erreur entre les étiquettes cibles et les prédictions.

    Args:
        targets (numpy.ndarray): Les étiquettes cibles réelles.
        predictions (numpy.ndarray): Les étiquettes prédites par le modèle.

    Returns:
        Tuple[float, numpy.ndarray]: Le taux d'erreur et les indices des erreurs.
    """
    error = predictions != targets
    indexes_errors = numpy.where(error)[0]

    error_rate = len(indexes_errors) / len(targets)
    return error_rate, indexes_errors


def compute_confusion_matrix(targets: numpy.ndarray, predictions: numpy.ndarray) -> numpy.ndarray:
    """
    Calcule la matrice de confusion entre les étiquettes cibles et les prédictions.

    Args:
        targets (numpy.ndarray): Les étiquettes cibles réelles.
        predictions (numpy.ndarray): Les étiquettes prédites par le modèle.

    Returns:
        numpy.ndarray: La matrice de confusion.
    """
    confusion_matrix = sklearn.metrics.confusion_matrix(targets, predictions)
    return confusion_matrix


class ProbabilityDensityFunction(abc.ABC):
    """
    Interface pour une fonction de densité de probabilité.
    """
    @abc.abstractmethod
    def compute_probability(self, data: numpy.ndarray) -> numpy.ndarray:
        """
        Calcule la probabilité des données selon la distribution.

        Args:
            data (numpy.ndarray): Les données pour lesquelles calculer la probabilité.

        Returns:
            numpy.ndarray: Les probabilités calculées pour chaque échantillon de données.
        """


class GaussianPDF(ProbabilityDensityFunction):
    """
    Classe représentant une fonction de densité de probabilité gaussienne multivariée.

    Attributes:
        dim (int): La dimension des données.
        mean (numpy.ndarray): Le vecteur moyen de la distribution gaussienne.
        covariance (numpy.ndarray): La matrice de covariance de la distribution gaussienne.
        inv_cov (numpy.ndarray): L'inverse de la matrice de covariance.
        det_cov (float): Le déterminant de la matrice de covariance.

    Methods:
        compute_probability(data): Calcule la probabilité des données selon la
            distribution gaussienne.
    """
    def __init__(self, data: numpy.ndarray):
        """
        Args:
            data (numpy.ndarray): Les données utilisées pour estimer les paramètres
                de la distribution gaussienne.
        """
        self.dim = data.shape[1]

        self.mean, self.covariance, _, _ = compute_gaussian_model(data)

        self.inv_cov = numpy.linalg.inv(self.covariance)
        self.det_cov = numpy.linalg.det(self.covariance)

    def compute_probability(self, data: numpy.ndarray) -> numpy.ndarray:
        """
        Calcule la probabilité de chaque donnée selon la distribution gaussienne.

        Args:
            data (numpy.ndarray): Les données pour lesquelles calculer la probabilité.
        """
        diff = data - self.mean

        mahalanobis_distance = numpy.sum(diff @ self.inv_cov * diff, axis=1) # quadratic form x^T * inv_cov * x
        denominator = numpy.sqrt((2 * numpy.pi) ** self.dim * self.det_cov)
        exponent = numpy.exp(-0.5 * mahalanobis_distance)

        return exponent / denominator


class HistogramPDF(ProbabilityDensityFunction):
    """
    Classe représentant une fonction de densité de probabilité arbitraire basée
    sur un histogramme multidimensionnel.

    Attributes:
        n_bins: le nombre de bins utilisés pour chaque dimension.
        dim (int): La dimension des données.
        histogram (numpy.ndarray): L'histogramme multidimensionnel représentant la densité
            de probabilité.
        bin_edges (List[numpy.ndarray]): Les bords des bins pour chaque dimension.

    Methods:
        compute_probability(data): Calcule la probabilité des données selon la
            distribution basée sur l'histogramme.
    """
    def __init__(self, data: numpy.ndarray, n_bins = 30):
        """
        Args:
            data (numpy.ndarray): Les données utilisées pour estimer les paramètres
                de la distribution basée sur l'histogramme.
            n_bins: Le nombre de bins à utiliser pour chaque dimension.
                (voir la documentation de `numpy.histogramdd` pour plus de détails)
        """
        self.n_bins = n_bins
        self.dim = data.shape[1]

        # L3.S2.1 Construire un modèle empirique de densité de probabilité pour chacune des classes
        # (Utilisez numpy.histogramdd, retirez les tenseur nulles et les 1 suspect)
        # ---------------------------------------------------------------------
        self.histogram, self.bin_edges = numpy.histogramdd(numpy.zeros_like(data), bins=1, density=True)
        # ---------------------------------------------------------------------

    def compute_probability(self, data: numpy.ndarray) -> numpy.ndarray:
        """
        Calcule la probabilité de chaque donnée selon la distribution basée sur l'histogramme.

        Args:
            data (numpy.ndarray): Les données pour lesquelles calculer la probabilité.
        """
        # L3.S2.2 Compléter la méthode pour calculer la probabilité d'appartenir à cette classe
        # ---------------------------------------------------------------------
        return numpy.zeros(data.shape[0])
        # ---------------------------------------------------------------------
