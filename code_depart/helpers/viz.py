"""
Module de visualisation pour supporter la problématique.

Fonctions:
    plot_images(samples: List[Tuple[numpy.ndarray, str]], title: str = "Échantillons d'images"):
        Génère une grille d'images échantillonnées.
    print_gaussian_model(mean: numpy.ndarray, covariance: numpy.ndarray, eigenvalues: numpy.ndarray, eigenvectors: numpy.ndarray):
        Affiche les paramètres d'un modèle gaussien.
    plot_data_distribution(representation: dataset.Representation,
                            title: str = "Distribution des données",
                            xlabel: str = "X",
                            ylabel: str = "Y",
                            zlabel: str = "Z",
                            show_components: bool = False,
                            show_ellipses: bool = False,
                            analytical_boundaries: bool = False,
                            priors: Optional[numpy.ndarray] = None) -> Tuple[plt.Figure, plt.Axes]:
          Affiche la distribution des données en 2D ou 3D.
    add_gaussian_components(ax: plt.Axes, model: GaussianModel, scale: float = 1.0):
        Ajoute les vecteurs propres d'un modèle gaussien à un axe 2D ou 3D existant.
    plot_metric_history(history: keras.callbacks.History):
        Affiche l'historique des métriques d'entraînement et de validation.
    show_confusion_matrix(target: numpy.ndarray, predictions: numpy.ndarray, class_labels: List[str], plot: bool = True):
        Affiche la matrice de confusion entre les étiquettes cibles et les prédictions.
    plot_numerical_decision_regions(model: classifier.Classifier, data: Union[dataset.Representation, numpy.ndarray]):
        Affiche les frontières de décision numérique d'un classificateur sur des données 2D.
    add_analytical_decision_regions(ax: plt.Axes,
                                    representation: dataset.Representation,
                                    priors: Optional[numpy.ndarray] = None):
        Ajoute les frontières de décision analytiques basées sur des modèles gaussiens à un axe 2D existant.
    add_ellipse(ax: plt.Axes, model: GaussianModel):
        Ajoute les ellipses $1 \sigma$ d'un modèle gaussien à un axe 2D existant.
    plot_classification_errors(representation: dataset.Representation,
                               predictions: numpy.ndarray,
                               title: str = "Erreurs de classification",
                               xlabel: str = "X",
                               ylabel: str = "Y",
                               zlabel: str = "Z",):
        Affiche les erreurs de classification sur la distribution des données en 2D ou 3D.
    plot_pdf(representation: dataset.Representation, n_bins: int = 10, title: str = None):
        Affiche les histogrammes de densité de probabilité 1D et 2D pour chaque
        classe dans la représentation.
    plot_data_distribution_with_custom_components(representation: dataset.Representation,
                                                   model: Union[GaussianModel, Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]],
                                                   title: str = "Distribution des points générés"):
        Affiche la distribution des données en 2D ou 3D avec des composantes gaussiennes personnalisées.
    plot_images_histograms(samples: List[Tuple[numpy.ndarray, str]],
                          n_bins: int = 256,
                          title: str = "Histogrammes des canaux par image",
                          x_label: str = None,
                          y_label: str = None,
                          channel_names: Optional[List[str]] = None,
                          colors: Optional[List[str]] = None):
        Affiche les histogrammes des canaux de chaque image échantillonnée.
    plot_features_distribution(representation: dataset.Representation,
                               n_bins: int = 20,
                               title: str = None,
                               xlabel: str = None,
                               ylabel: str = None,
                               features_names: Optional[List[str]] = None):
        Affiche les histogrammes de distribution de chaque classe pour chaque caractéristique dans la représentation.
"""

import dataclasses
import itertools

from typing import List, Optional, Tuple, Union

import keras
import matplotlib.pyplot as plt
import matplotlib.colors
import numpy

from . import (
    analysis,
    classifier,
    dataset
)


DEFAULT_FIGSIZE = (6, 4)
CMAP = matplotlib.colors.ListedColormap([
    "orange",
    "purple",
    "gray"
])


@dataclasses.dataclass
class GaussianModel:
    """
    Représente un modèle gaussien avec sa moyenne, sa matrice de covariance,
    ses valeurs propres et ses vecteurs propres.

    Attributes:
        mean (numpy.ndarray): La moyenne du modèle gaussien.
        covariance (numpy.ndarray): La matrice de covariance du modèle gaussien.
        eigenvalues (numpy.ndarray): Les valeurs propres de la matrice de covariance.
        eigenvectors (numpy.ndarray): Les vecteurs propres de la matrice de covariance.

    Methods:
        from_tuple(cls, data: Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]) -> 'GaussianModel':
            Crée une instance de GaussianModel à partir d'un tuple contenant la moyenne,
            la matrice de covariance, les valeurs propres et les vecteurs propres.
    """
    mean: numpy.ndarray
    covariance: numpy.ndarray
    eigenvalues: numpy.ndarray
    eigenvectors: numpy.ndarray

    @classmethod
    def from_tuple(cls, data: Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]) -> 'GaussianModel':
        """
        Crée une instance de GaussianModel à partir d'un tuple contenant la moyenne,
        la matrice de covariance, les valeurs propres et les vecteurs propres.

        Args:
            data (Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]):
                Un tuple contenant la moyenne, la matrice de covariance,
                les valeurs propres et les vecteurs propres.

        Returns:
            GaussianModel: Une instance de GaussianModel initialisée avec les données fournies.
        """
        mean = numpy.array(data[0])
        covariance = numpy.array(data[1])
        eigenvalues = numpy.array(data[2])
        eigenvectors = numpy.array(data[3])
        return cls(mean, covariance, eigenvalues, eigenvectors)


def _get_axes(ax: Optional[plt.Axes] = None,
              projection: str = None,
              figsize: Tuple[int, int] = DEFAULT_FIGSIZE,
              title: str = None,
              xlabel: str = None,
              ylabel: str = None,
              zlabel: str = None) -> Tuple[plt.Figure, plt.Axes]:
    """
    Crée ou récupère des axes matplotlib en 2D ou 3D et configure les labels et le titre.

    Args:
        ax (Optional[plt.Axes]): Axe matplotlib existant. Si None, un nouvel axe sera créé.
        projection (str): Type de projection ("3d" pour 3D, None pour 2D).
        figsize (Tuple[int, int]): Taille de la figure si un nouvel axe est créé.
        title (str): Titre du graphique.
        xlabel (str): Label de l'axe X.
        ylabel (str): Label de l'axe Y.
        zlabel (str): Label de l'axe Z (si en 3D).

    Returns:
        Tuple[plt.Figure, plt.Axes]: La figure et les axes matplotlib configurés.
    """
    # pylint: disable = using-constant-test, multiple-statements

    fig = None

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1, projection=projection)

    else:
        fig = ax.figure

    if title: ax.set_title(title)
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    if zlabel and projection == "3d": ax.set_zlabel(zlabel)

    return fig, ax


def _create_subplot_grid(n_plots: int, 
                         figsize_per_plot: Tuple[int, int] = DEFAULT_FIGSIZE,
                         ax: Optional[plt.Axes] = None,
                         title: str = None) -> Tuple[plt.Figure, plt.Axes]:
    """
    Crée une grille de sous-graphes pour afficher plusieurs graphiques.

    Args:
        n_plots (int): Nombre total de sous-graphes à créer.
        figsize_per_plot (Tuple[int, int]): Taille de chaque sous-graphe.
        ax (Optional[plt.Axes]): Axe matplotlib existant. Si None, une nouvelle figure sera créée.
        title (str): Titre global de la figure.

    Returns:
        Tuple[plt.Figure, plt.Axes]: La figure et les axes matplotlib configurés
    """
    # Determine grid size
    n_cols = int(numpy.ceil(numpy.sqrt(n_plots)))
    n_rows = int(numpy.ceil(n_plots / n_cols))

    figsize = (figsize_per_plot[0] * n_cols, figsize_per_plot[1] * n_rows)

    if ax is None:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    else:
        fig = ax.figure
        for i in range(n_plots):
            axes = fig.add_subplot(n_rows, n_cols, i + 1)

    if title:
        fig.suptitle(title)

    if n_plots > 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    return fig, axes


def plot_images(samples: List[Tuple[numpy.ndarray, str]], title: str = "Échantillons d'images"):
    """
    Génère une grille d'images échantillonnées.

    Args:
        samples (List[Tuple[numpy.ndarray, str]]): Liste de tuples contenant les images et leurs classes respectives.
        title (str): Titre global de la figure.
    """
    fig, axes = _create_subplot_grid(len(samples), figsize_per_plot=DEFAULT_FIGSIZE, title=title)

    for ax, (image, label) in zip(axes, samples):
        ax.imshow(image / 255.0)
        ax.set_title(label)
        ax.axis('off')

    fig.tight_layout()


def print_gaussian_model(mean: numpy.ndarray, covariance: numpy.ndarray, eigenvalues: numpy.ndarray, eigenvectors: numpy.ndarray):
    """
    Affiche les paramètres d'un modèle gaussien.

    Args:
        mean (numpy.ndarray): La moyenne du modèle gaussien.
        covariance (numpy.ndarray): La matrice de covariance du modèle gaussien.
        eigenvalues (numpy.ndarray): Les valeurs propres de la matrice de covariance.
        eigenvectors (numpy.ndarray): Les vecteurs propres de la matrice de covariance.
    """
    print(f"Moyenne : {mean}")
    print(f"Matrice de covariance : \n{covariance}")
    print(f"Valeurs propres : {eigenvalues}")
    print(f"Vecteurs propres : \n{eigenvectors}")


def plot_data_distribution(representation: dataset.Representation,
                           title: str = "Distribution des données",
                           xlabel: str = "X",
                           ylabel: str = "Y",
                           zlabel: str = "Z",
                           show_components: bool = False,
                           show_ellipses: bool = False,
                           analytical_boundaries: bool = False,
                           priors: Optional[numpy.ndarray] = None) -> Tuple[plt.Figure, plt.Axes]:
    """
    Affiche la distribution des données en 2D ou 3D.

    Args:
        representation (Representation): Représentation des données à visualiser.
        title (str): Titre du graphique.
        xlabel (str): Label de l'axe X.
        ylabel (str): Label de l'axe Y.
        zlabel (str): Label de l'axe Z (si en 3D).
        show_components (bool): Indique si les composantes principales d'une distribution gaussienne doivent être affichées.
        show_ellipses (bool): Indique si les ellipses $1 \sigma$ doivent être affichées.
        analytical_boundaries (bool): Indique si les frontières analytiques doivent être affichées.
        priors (Optional[numpy.ndarray]): Probabilités a priori pour chaque classe (utilisées pour les frontières analytiques).

    Returns:
        Tuple[plt.Figure, plt.Axes]: La figure et les axes matplotlib contenant la visualisation.

    Raises:
        ValueError: Si les données ne sont pas en 2D ou 3D.
    """
    if representation.dim not in [2, 3]:
        raise ValueError("La visualisation n'est supportée que pour des données en 2D ou 3D.")

    projection = None
    if representation.dim == 3:
        projection = "3d"

    fig, ax = _get_axes(projection=projection, title=title, xlabel=xlabel, ylabel=ylabel, zlabel=zlabel)

    unique_labels = representation.unique_labels

    colors = CMAP(numpy.linspace(0, 1, len(unique_labels)))

    for i, label in enumerate(unique_labels):
        data =  representation.get_class(label)

        color = colors[i]

        # plot the scatter points
        if representation.dim == 3:
            ax.scatter(data[:, 0], data[:, 1], data[:, 2], label=label, alpha=0.25, color=color)
        elif representation.dim == 2:
            ax.scatter(data[:, 0], data[:, 1], label=label, alpha=0.25, color=color)

        # plot the gaussian components if requested
        if show_components:
            gaussian_model = analysis.compute_gaussian_model(data)
            model = GaussianModel.from_tuple(gaussian_model)
            add_gaussian_components(ax, model)

        # plot the ellipses if requested
        if show_ellipses:
            gaussian_model = analysis.compute_gaussian_model(data)
            model = GaussianModel.from_tuple(gaussian_model)
            add_ellipse(ax, model)
    
    if analytical_boundaries:
        add_analytical_decision_regions(ax, representation, priors=priors)

    ax_handles, ax_labels = ax.get_legend_handles_labels()
    fig.legend(ax_handles, ax_labels, loc="upper right", bbox_to_anchor=(1, 0.925))

    fig.gca().set_aspect("equal", adjustable="datalim")

    fig.tight_layout()

    return fig, ax


def add_gaussian_components(ax: plt.Axes, model: GaussianModel, scale: float = 1.0):
    """
    Ajoute les vecteurs propres d'un modèle gaussien à un axe 2D ou 3D existant.

    Args:
        ax (plt.Axes): Axe 2D ou 3D sur lequel ajouter les vecteurs propres.
        model (GaussianModel): Modèle gaussien contenant les vecteurs propres.
        scale (float): Facteur d'échelle pour la longueur des vecteurs.
    """
    color = ["r", "g", "b"]

    dim = model.mean.shape[0]
    origin = model.mean

    if dim not in [2, 3]:
        raise ValueError("La visualisation n'est supportée que pour des données en 2D ou 3D.")

    for i in range(dim):
        vector = model.eigenvectors[:, i] * numpy.sqrt(model.eigenvalues[i]) * scale
        
        if dim == 2:
            ax.quiver(*origin, *vector, color=color[i], alpha=1, angles="xy", scale_units="xy", scale=1, width=0.005, headwidth=2, headlength=4)
        elif dim == 3:
            ax.quiver(*origin, *vector, color=color[i], alpha=1)


def plot_metric_history(history: keras.callbacks.History):
    """
    Affiche l'historique des métriques d'entraînement et de validation.

    Args:
        history (keras.callbacks.History): Objet History retourné par l'entraînement d'un modèle Keras.
    """
    metrics = [key for key in history.history.keys() if not key.startswith("val_")]
    n_metrics = len(metrics)

    fig, axes = _create_subplot_grid(n_metrics, figsize_per_plot=DEFAULT_FIGSIZE, title="Historique des métriques durant l'entraînement")

    for ax, metric in zip(axes, metrics):
        ax.plot(history.history[metric], label="training")

        val_metric = f"val_{metric}"
        if val_metric in history.history:
            ax.plot(history.history[val_metric], label="validation")

        ax.set_title(metric)
        ax.set_xlabel("Epochs")
        ax.grid()

    ax_handles, ax_labels = ax.get_legend_handles_labels()
    fig.legend(ax_handles, ax_labels, loc="upper right", bbox_to_anchor=(1, 0.95))

    fig.tight_layout()

def show_confusion_matrix(target: numpy.ndarray, predictions: numpy.ndarray, class_labels: List[str], plot: bool = True):
    """
    Affiche la matrice de confusion entre les étiquettes cibles et les prédictions.

    Args:
        targets (numpy.ndarray): Les étiquettes cibles réelles.
        predictions (numpy.ndarray): Les étiquettes prédites par le modèle.
        class_labels (List[str]): Liste des noms des classes pour l'axe des labels.
        plot (bool): Indique si la matrice de confusion doit être affichée graphiquement.
    """
    confusion_matrix = analysis.compute_confusion_matrix(target, predictions)

    print("Matrice de confusion:")
    print(confusion_matrix)

    if not plot:
        return

    fig, ax = _get_axes(figsize=DEFAULT_FIGSIZE, title="Matrice de confusion", xlabel="Prédictions", ylabel="Cibles")

    ax.imshow(confusion_matrix, cmap="Blues")
    fig.colorbar(ax.images[0], ax=ax)
    ax.set_xticks(numpy.arange(len(class_labels)), labels=class_labels, rotation=45)
    ax.set_yticks(numpy.arange(len(class_labels)), labels=class_labels)

    fig.tight_layout()


def plot_numerical_decision_regions(model: classifier.Classifier, data: Union[dataset.Representation, numpy.ndarray]):
    """
    Affiche les frontières de décision d'un classificateur numérique sur des données 2D,
    en utilisant une grille de points uniformément distribués à prédire dans l'espace des caractéristiques.

    Args:
        model (classifier.Classifier): Le classificateur à utiliser pour prédire les classes des points de la grille.
        data (Union[Representation, numpy.ndarray]): Les données utilisées pour déterminer les limites de la grille.

    Raises:
        ValueError: Si les données ne sont pas en 2D.
    """
    data = data.data if isinstance(data, dataset.Representation) else data

    if data.shape[1] != 2:
        raise ValueError("La visualisation des frontières de décision n'est supportée que pour des données en 2D.")

    margins = (data.max(axis=0) - data.min(axis=0)) * 0.1
    min_features = data.min(axis=0) - margins
    max_features = data.max(axis=0) + margins

    # Create a grid of samples over the feature space
    n_samples_per_interval = 10 * (max_features - min_features)
    x = numpy.linspace(min_features[0], max_features[0], int(n_samples_per_interval[0]))
    y = numpy.linspace(min_features[1], max_features[1], int(n_samples_per_interval[1]))
    mesh = numpy.meshgrid(x, y)
    points = numpy.vstack([axis.flatten() for axis in mesh]).T

    # Predict classes for each sample in the grid
    grid_predictions = model.predict(points)

    if len(grid_predictions.shape) > 1 and grid_predictions.shape[1] > 1: # logits
        prediction = numpy.argmax(grid_predictions, axis=-1)
    else:
        prediction = grid_predictions

    # Transform non-numerical labels to numerical if needed
    if not numpy.issubdtype(prediction.dtype, numpy.number):
        _, numerical_labels = numpy.unique(prediction, return_inverse=True)
        prediction = numerical_labels

    prediction = prediction.reshape(mesh[0].shape)

    # Plot decision regions
    fig, ax = _get_axes(figsize=DEFAULT_FIGSIZE, title="Frontières de décision numérique", xlabel="X", ylabel="Y")

    ax.contourf(*mesh, prediction, cmap=CMAP, alpha=1)

    ax.set_xlim(min_features[0], max_features[0])
    ax.set_ylim(min_features[1], max_features[1])

    fig.tight_layout()


def add_analytical_decision_regions(ax: plt.Axes,
                                    representation: dataset.Representation,
                                    priors: Optional[numpy.ndarray] = None):
    """
    Ajoute les frontières de décision analytiques basées sur des modèles gaussiens
    à un axe 2D existant.

    Args:
        ax (plt.Axes): Axe 2D sur lequel ajouter les frontières de décision.
        representation (Representation): Représentation des données utilisées pour calculer les modèles gaussiens.
        priors (Optional[numpy.ndarray]): Probabilités a priori pour chaque classe (utilisées pour les frontières analytiques).

    Raises:
        ValueError: Si les données ne sont pas en 2D.
    """
    if representation.dim != 2:
        raise ValueError("La visualisation des frontières de décision n'est supportée que pour des données en 2D.")

    margins = (representation.data.max(axis=0) - representation.data.min(axis=0)) * 0.1
    min_features = representation.data.min(axis=0) - margins
    max_features = representation.data.max(axis=0) + margins

    # Create a grid of samples over the feature space
    n_samples_per_interval = 10 * (max_features - min_features)
    x = numpy.linspace(min_features[0], max_features[0], int(n_samples_per_interval[0]))
    y = numpy.linspace(min_features[1], max_features[1], int(n_samples_per_interval[1]))
    mesh = numpy.meshgrid(x, y)
    points = numpy.vstack([axis.flatten() for axis in mesh]).T

    if priors is None:
        priors = numpy.array([1 / len(representation.unique_labels)] * len(representation.unique_labels))

    label_to_index = {label: index for index, label in enumerate(representation.unique_labels)}

    for pair in itertools.combinations(representation.unique_labels, 2):
        mean1, cov1, _, _ = analysis.compute_gaussian_model(representation.get_class(pair[0]))
        mean2, cov2, _, _ = analysis.compute_gaussian_model(representation.get_class(pair[1]))

        inv_cov1 = numpy.linalg.inv(cov1)
        inv_cov2 = numpy.linalg.inv(cov2)

        idx1 = label_to_index[pair[0]]
        idx2 = label_to_index[pair[1]]

        quadratic_discriminant = inv_cov1 - inv_cov2
        linear_discriminant = 2 * (inv_cov2 @ mean2 - inv_cov1 @ mean1)
        bias_term = (
            mean1.T @ inv_cov1 @ mean1
            - mean2.T @ inv_cov2 @ mean2
            + numpy.log(numpy.linalg.det(cov1) / numpy.linalg.det(cov2))
            - 2 * numpy.log(priors[idx1] / priors[idx2])
        )

        Z = numpy.sum(points @ quadratic_discriminant * points, axis=1) + points @ linear_discriminant + bias_term
        Z = Z.reshape(mesh[0].shape)

        ax.contour(*mesh, Z, levels=[0], colors='k', linewidths=2)


def add_ellipse(ax: plt.Axes, model: GaussianModel):
    """
    Ajoute les ellipses $1 \sigma$ d'un modèle gaussien à un axe 2D existant.

    Args:
        ax (plt.Axes): Axe 2D sur lequel ajouter les ellipses.
        model (GaussianModel): Modèle gaussien contenant les paramètres pour dessiner les ellipses.
    """
    dim = model.mean.shape[0]

    theta = numpy.linspace(0, 2 * numpy.pi, 300)

    ax.scatter(*model.mean, color="black", s=10, linewidths=3)

    for plane in itertools.combinations(range(dim), 2):
        # L3.E1.1 Remplacer les valeurs bidons par les bons paramètres
        # à partir des métriques accessible (les `1` sont suspect)
        # ---------------------------------------------------------------------
        mean1, mean2 = model.mean[list(plane)]
        cov1, cov2 = model.covariance[list(plane)]
        val1, val2 = model.eigenvalues[list(plane)]
        vec1, vec2 = model.eigenvectors[:, list(plane)].T

        ellipse = numpy.vstack([
            numpy.sqrt(1) * numpy.cos(theta),
            numpy.sqrt(1) * numpy.sin(theta)
        ])
        # ---------------------------------------------------------------------

        basis = model.eigenvectors[:, plane]

        projected_ellipse = numpy.matmul(basis, ellipse).T + model.mean

        ax.plot(*projected_ellipse.T, color="black", linewidth=2, alpha=0.8)


def plot_classification_errors(representation: dataset.Representation,
                               predictions: numpy.ndarray,
                               title: str = "Erreurs de classification",
                               xlabel: str = "X",
                               ylabel: str = "Y",
                               zlabel: str = "Z",):
    """
    Affiche les erreurs de classification sur la distribution des données en 2D ou 3D.

    Args:
        representation (Representation): Représentation des données à visualiser.
        predictions (numpy.ndarray): Les étiquettes prédites par le modèle.
        title (str): Titre du graphique.
        xlabel (str): Label de l'axe X.
        ylabel (str): Label de l'axe Y.
        zlabel (str): Label de l'axe Z (si en 3D).

    Raises:
        ValueError: Si les données ne sont pas en 2D ou 3D.
    """
    if representation.dim not in [2, 3]:
        raise ValueError("La visualisation n'est supportée que pour des données en 2D ou 3D.")

    projection = None
    if representation.dim == 3:
        projection = "3d"

    fig, ax = _get_axes(projection=projection, title=title, xlabel=xlabel, ylabel=ylabel, zlabel=zlabel)

    unique_labels = representation.unique_labels

    colors = CMAP(numpy.linspace(0, 1, len(unique_labels)))

    for i, label in enumerate(unique_labels):
        data =  representation.get_class(label)

        color = colors[i]

        # plot the scatter points
        if representation.dim == 3:
            ax.scatter(data[:, 0], data[:, 1], data[:, 2], label=label, alpha=0.25, color=color)
        elif representation.dim == 2:
            ax.scatter(data[:, 0], data[:, 1], label=label, alpha=0.25, color=color)

    error_mask = predictions != representation.labels
    error_indices = numpy.where(error_mask)[0]

    ax.scatter(representation.data[error_indices, 0],
               representation.data[error_indices, 1],
               representation.data[error_indices, 2] if representation.dim == 3 else None,
               label="Erreurs",
               color="red")

    ax_handles, ax_labels = ax.get_legend_handles_labels()
    fig.legend(ax_handles, ax_labels, loc="upper right", bbox_to_anchor=(1, 0.925))

    fig.tight_layout()


def plot_pdf(representation: dataset.Representation, n_bins: int = 10, title: str = None):
    """
    Affiche les histogrammes de densité de probabilité 1D et 2D pour chaque
    classe dans la représentation.

    Args:
        representation (Representation): Représentation des données à visualiser.
        n_bins: Nombre de bacs à utiliser pour les histogrammes.
            (voir analysis.HistogramPDF pour plus de détails)
        title (str): Titre du graphique.

    Raises:
        ValueError: Si les données ne sont pas en 1D ou 2D.
    """
    if representation.dim not in [1, 2]:
        raise ValueError("La visualisation des histogrammes n'est supportée que pour des données en 1D ou 2D.")

    dim = representation.dim

    projection = "3d" if dim == 2 else None

    unique_labels = representation.unique_labels
    colors = CMAP(numpy.linspace(0, 1, len(unique_labels)))

    for i, label in enumerate(unique_labels):
        class_data = representation.get_class(label)
        color = colors[i]

        histogram_pdf = analysis.HistogramPDF(class_data, n_bins=n_bins)

        histogram = histogram_pdf.histogram
        bin_edges = histogram_pdf.bin_edges

        bin_edges = numpy.array(bin_edges)
        bin_centers = (bin_edges[:, :-1] + bin_edges[:, 1:]) / 2
        bin_widths = numpy.diff(bin_edges, axis=1)

        if title is None:
            title = f"Histogramme densité de probabilité - Classe {label}"

        if dim == 1:
            _, axes = _get_axes(projection=projection, title=title, xlabel="X", ylabel="Densité de probabilité")

            axes.bar(bin_centers[0], histogram, width=bin_widths[0], color=color)

        elif dim == 2:
            _, axes = _get_axes(projection=projection, title=title, xlabel="X", ylabel="Y", zlabel="Densité de probabilité")

            xpos, ypos = numpy.meshgrid(bin_centers[0], bin_centers[1], indexing="ij")
            xwidth, ywidth = numpy.meshgrid(bin_widths[0], bin_widths[1], indexing="ij")
            dz = histogram.ravel()

            max_height = numpy.max(dz)
            min_height = numpy.min(dz)

            cmap = plt.get_cmap('viridis')
            colors_2d = cmap((dz - min_height) / (max_height - min_height + 1e-9))

            axes.bar3d(xpos.ravel(),
                       ypos.ravel(),
                       numpy.zeros_like(dz),
                       dx=xwidth.ravel(),
                       dy=ywidth.ravel(),
                       dz=dz,
                       zsort='average',
                       alpha=0.5,
                       color=colors_2d)


def plot_data_distribution_with_custom_components(representation: dataset.Representation,
                                                   model: Union[GaussianModel, Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]],
                                                   title: str = "Distribution des points générés"):
    """
    Affiche la distribution des données en 2D ou 3D avec des composantes gaussiennes personnalisées.

    Args:
        representation (Representation): Représentation des données à visualiser.
        model (Union[GaussianModel, Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]]):
            Modèle gaussien ou tuple contenant la moyenne, la matrice de covariance,
            les valeurs propres et les vecteurs propres.
        title (str): Titre du graphique.
    """
    if isinstance(model, tuple):
        model = GaussianModel.from_tuple(model)

    _, ax = plot_data_distribution(representation, title=title)
    add_gaussian_components(ax, model=model)

    # Fix quiver scale
    limit = numpy.max(numpy.abs(representation.data))
    if limit != -limit:
        ax.set_xlim([-limit, limit])
        ax.set_ylim([-limit, limit])
        ax.set_zlim([-limit, limit])


def plot_images_histograms(samples: List[Tuple[numpy.ndarray, str]],
                          n_bins: int = 256,
                          title: str = "Histogrammes des canaux par image",
                          x_label: str = None,
                          y_label: str = None,
                          channel_names: Optional[List[str]] = None,
                          colors: Optional[List[str]] = None):
    """
    Affiche les histogrammes des canaux de chaque image échantillonnée.

    Args:
        samples (List[Tuple[numpy.ndarray, str]]): Liste de tuples contenant les images et leurs classes respectives.
        n_bins (int): Nombre de bacs à utiliser pour les histogrammes.
        title (str): Titre global de la figure.
        x_label (str): Label de l'axe X.
        y_label (str): Label de l'axe Y.
        channel_names (Optional[List[str]]): Noms des canaux à utiliser dans la légende.
        colors (Optional[List[str]]): Couleurs à utiliser pour chaque canal.
    """
    n_channels = samples[0][0].shape[-1]

    fig, axes = _create_subplot_grid(len(samples), figsize_per_plot=DEFAULT_FIGSIZE, title=title)

    for ax, (image, label) in zip(axes, samples):
        for channel in range(n_channels):
            channel_label = channel_names[channel] if channel_names else f"Canal {channel}"
            color = colors[channel] if colors else None

            histogram, bin_edges = numpy.histogram(image[:, :, channel], bins=n_bins, range=(0, n_bins - 1))

            ax.plot(bin_edges[:-1], histogram, label=channel_label, color=color)

        ax.set_title(label)

        if x_label:
            ax.set_xlabel(x_label)

        if y_label:
            ax.set_ylabel(y_label)

        ax.label_outer()

    ax_handles, ax_labels = ax.get_legend_handles_labels()
    fig.legend(ax_handles, ax_labels, loc="upper right", bbox_to_anchor=(1, 0.925))

    fig.tight_layout()


def plot_features_distribution(representation: dataset.Representation,
                               n_bins: int = 20,
                               title: str = None,
                               xlabel: str = None,
                               ylabel: str = None,
                               features_names: Optional[List[str]] = None):
    """
    Affiche les histogrammes de distribution de chaque classe pour chaque
    caractéristique dans la représentation.

    Args:
        representation (Representation): Représentation des données à visualiser.
        n_bins (int): Nombre de bacs à utiliser pour les histogrammes.
        title (str): Titre du graphique.
        xlabel (str): Label de l'axe X.
        ylabel (str): Label de l'axe Y.
        features_names (Optional[List[str]]): Noms des caractéristiques à utiliser dans les titres des sous-graphes.
    """
    n_features = representation.dim

    features_min = numpy.min(representation.data)
    features_max = numpy.max(representation.data)

    bin_edges = numpy.linspace(features_min, features_max, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_widths = numpy.diff(bin_edges)

    colors = CMAP(numpy.linspace(0, 1, len(representation.unique_labels)))

    if features_names is None:
        features_names = [f"Feature {i}" for i in range(n_features)]

    fig, axes = _create_subplot_grid(n_features, figsize_per_plot=DEFAULT_FIGSIZE, title=title)

    for ax, i in zip(axes, range(n_features)):
        for j, label in enumerate(representation.unique_labels):
            data = representation.get_class(label)

            histogram, _ = numpy.histogram(data[:, i], bins=bin_edges, range=(features_min, features_max))

            ax.bar(bin_centers, histogram, width=bin_widths, alpha=0.5, label=label, color=colors[j])

        if xlabel:
            ax.set_xlabel(xlabel)

        if ylabel:
            ax.set_ylabel(ylabel)

        ax.label_outer()

        ax.set_title(features_names[i])

    ax_handles, ax_labels = ax.get_legend_handles_labels()
    fig.legend(ax_handles, ax_labels, loc="upper right", bbox_to_anchor=(1, 0.925))

    fig.tight_layout()
