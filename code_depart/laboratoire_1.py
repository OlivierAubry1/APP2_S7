# pylint: disable = missing-function-docstring, missing-module-docstring, wrong-import-position
import os
import pathlib

import matplotlib.pyplot as plt
import numpy
import skimage

# Must be call before any other TensorFlow/Keras import
# Suppress oneDNN custom operations info
# Suppress INFO and WARNING messages from TF (0=all, 1=no INFO, 2=no INFO/WARN, 3=no error)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import helpers.analysis as analysis
import helpers.dataset as dataset
import helpers.viz as viz


def exercice_2_decorrelation():
    mean = [0, 0, 0]
    covariance = numpy.array([
        [2, 1, 0],
        [1, 2, 0],
        [0, 0, 7]
    ])

    # L1.E2.1 Compléter le code ci-dessus pour calculer les valeurs propres et vecteurs propres de la matrice de covariance
    # -------------------------------------------------------------------------
    # Utilisez la fonction appropriée pour calculer les valeurs propres et vecteurs propres
    # À la place des vecteurs et valeurs propres nulles ci-dessous
    eigenvalues, eigenvectors = numpy.zeros(3), numpy.zeros((3, 3))

    print("Exercice 2.1: Calcul des valeurs propres et vecteurs propres")
    viz.print_gaussian_model(mean, covariance, eigenvalues, eigenvectors)
    print("\n")
    # -------------------------------------------------------------------------

    # L1.E2.3 Visualisez les valeurs et vecteurs propres obtenus ainsi que la distribution des points
    # -------------------------------------------------------------------------
    samples = numpy.random.multivariate_normal(mean, covariance, 200)

    representation = dataset.Representation(data=samples, labels=numpy.array(["Data"] * samples.shape[0]))
    gaussian_model = (mean, covariance, eigenvalues, eigenvectors)
    viz.plot_data_distribution_with_custom_components(representation, model=gaussian_model, title="Données échantillonnées")
    # -------------------------------------------------------------------------

    # L1.E2.5 Projetez la représentation des données sur la première composante principale
    # -------------------------------------------------------------------------
    first_principal_component = numpy.zeros((3, 1))                                             # Sélectionnez la première composante principale
    decorrelated_samples = analysis.project_onto_new_basis(samples, first_principal_component)  # Complétez la fonction project_onto_new_basis dans analysis.py

    representation = dataset.Representation(data=decorrelated_samples, labels=numpy.array(["Data"] * decorrelated_samples.shape[0]))
    viz.plot_pdf(representation, n_bins=10, title="Projection des données sur la 1er composante")
    # -------------------------------------------------------------------------

    # L1.E2.6 Projetez la représentation des données sur les 2e et 3e composantes principales
    # -------------------------------------------------------------------------
    e23 = numpy.zeros((3, 2))                                       # Sélectionnez la 2e et 3e composante principale
    reduced_samples = analysis.project_onto_new_basis(samples, e23) # Projetez les données sur les 2e et 3e composantes principales

    projected_covariance = numpy.zeros((2,2))                                           # Utilisez la fonction appropriée pour calculer la matrice de covariance des données projetées
    projected_eigenvalues, projected_eigenvectors = numpy.zeros(2), numpy.zeros((2,2))  # Utilisez la fonction appropriée pour calculer les valeurs propres et vecteurs propres des données projetées

    print("Exercice 2.6: Calcul de la matrice de covariance, vecteurs et valeurs propres projetées")
    viz.print_gaussian_model(mean[1:3], projected_covariance, projected_eigenvalues, projected_eigenvectors)
    print("\n")

    # On reprojettent les données dans l'espace original pour visualiser l'effet de la réduction de dimensionnalité
    reconstruction = analysis.project_onto_new_basis(reduced_samples, e23.T)

    representation = dataset.Representation(data=reconstruction, labels=numpy.array(["Data"] * reconstruction.shape[0]))
    viz.plot_data_distribution_with_custom_components(representation, model=gaussian_model, title="Données projetées sur les 2e et 3e composantes principales")
    # -------------------------------------------------------------------------

    plt.show()


def exercice_3_visualisation_representation():
    # L1.E3.1 Visualiser la distribution des points pour les 3 classes.
    # -------------------------------------------------------------------------
    data3classes = dataset.MultimodalDataset(pathlib.Path(__file__).parent / "data/data_3classes/")
    reprensentation = dataset.Representation(data=data3classes.data, labels=data3classes.labels)

    viz.plot_data_distribution(reprensentation, title="Représentation de MultimodalDataset", show_components=True)
    # -------------------------------------------------------------------------

    # L1.E3.2 et L1.E3.5 (complétez la fonction compute_gaussian_model dans analysis.py)
    # -------------------------------------------------------------------------
    print("Exercice 3.2: Modèles gaussiens pour chaque classe")
    for class_name in reprensentation.unique_labels:
        class_data = reprensentation.get_class(class_name)

        # Completez la fonction compute_gaussian_model dans analysis.py
        mean, covariance, eigenvalues, eigenvectors = analysis.compute_gaussian_model(class_data)

        print(f"Classe {class_name}")
        print("-------------------------------")
        viz.print_gaussian_model(mean, covariance, eigenvalues, eigenvectors)
        print("-------------------------------\n")
    # -------------------------------------------------------------------------

    # L1.E3.4 Calculer les variances sur chaque dimension pour la classe C1 ainsi que leur corrélations
    # -------------------------------------------------------------------------
    data_C1 = reprensentation.get_class("C1")
    variances = numpy.zeros(data_C1.shape[1])                           # Utilisez la fonction appropriée pour calculer les variances
    correlations = numpy.zeros((data_C1.shape[1], data_C1.shape[1]))    # Utilisez la fonction appropriée pour calculer les corrélations
    print("Exercice 3.4: Variances et corrélations pour la classe C1")
    print(f"Variances : {variances}")
    print(f"Corrélations : \n{correlations}")
    # -------------------------------------------------------------------------

    # L1.E3.6 Décorrélez les données basé sur les composantes principales de la classe C1
    # -------------------------------------------------------------------------
    _, _, _, eigenvectors_C1 = analysis.compute_gaussian_model(data_C1)

    # Utilisez la fonction appropriée pour projeter les données sur la nouvelle base
    # Indice: Utilisez la fonction project_onto_new_basis définie précédement pour créer une nouvelle représentation des données
    decorrelated_data = numpy.zeros_like(data3classes.data)
    decorrelated_representation = dataset.Representation(data=decorrelated_data, labels=data3classes.labels)

    print("\nExercice 3.6: Données décorrelées de la classe C1")
    decorrelated_data_C1 = decorrelated_representation.get_class("C1")
    _, covariance_decorrelated, _, _ = analysis.compute_gaussian_model(decorrelated_data_C1)
    print(f"Matrice de covariance des données décorrelées : \n{covariance_decorrelated}")
    # -------------------------------------------------------------------------

    # L1.E3.7 Est-ce que la décorrélation serait applicable à l'ensemble des classes?
    # -------------------------------------------------------------------------
    viz.plot_data_distribution(decorrelated_representation, title="Représentation décorrelée de MultimodalDataset", show_components=True)
    # -------------------------------------------------------------------------

    plt.show()


def exercice_4_choix_representation():
    images = dataset.ImageDataset(pathlib.Path(__file__).parent / "data/image_dataset/")

    # L1.E4.1 Visualiser quelques images du dataset
    # ---------------------------------------------------------------------
    N = 6
    samples = images.sample(N)
    viz.plot_images(samples, title="Exemples d'images du dataset")
    # -------------------------------------------------------------------------

    # L1.E4.3 Observer l'histograme de couleur d'une image
    # -------------------------------------------------------------------------
    viz.plot_images_histograms(samples, n_bins=256, 
                              title="Histogrammes des intensités de pixels RGB",
                              x_label="Valeur",
                              y_label="Nombre de pixels",
                              channel_names=['Red', 'Green', 'Blue'],
                              colors=['r', 'g', 'b'])
    # -------------------------------------------------------------------------

    # L1.E4.5 Utilisez `scikit-image.color` pour explorer d'autres espaces de couleur (LAB et HSV)
    # -------------------------------------------------------------------------
    samples_lab = []
    samples_hsv = []
    for image, label in samples:
        image_lab = skimage.color.rgb2lab(image / 255.0)
        scaled_lab = analysis.rescale_lab(image_lab, n_bins=256)
        samples_lab.append((scaled_lab, label))

        image_hsv = skimage.color.rgb2hsv(image / 255.0)
        scaled_hsv = analysis.rescale_hsv(image_hsv, n_bins=256)
        samples_hsv.append((scaled_hsv, label))

    # Visualiez les histogrammes des images dans les différents espaces de couleur
    # Indice: vous pouvez réutiliser la fonction viz.plot_images_histograms

    # -------------------------------------------------------------------------

    # L1.E4.6 Calculer la moyen de chaque canal R, G et B pour chaque classe du dataset
    # =========================================================================
    features = numpy.zeros((len(images), 6)) # 3 moyennes + 3 écarts-types
    for i, (image, _) in enumerate(images):
        channels_mean = numpy.zeros(3)  # Calculer la moyenne de chaque canal R, G et B

        # L1.E4.7 Répéter pour une autre métrique de votre choix
        # ---------------------------------------------------------------------
        other_feature = numpy.zeros(3)  # Calculer une autre métrique de votre choix
        # ---------------------------------------------------------------------

        features[i] = numpy.concatenate((channels_mean, other_feature))

    features = numpy.array(features)
    # =========================================================================

    # L1.E4.8 Étudier si les quelques métriques obtenu sont corrélées, discriminantes, etc.
    # -------------------------------------------------------------------------
    representation_mean = dataset.Representation(data=features[:, :3], labels=images.labels)
    viz.plot_data_distribution(representation_mean,
                               title="Distribution des images basée sur les moyennes des canaux RGB",
                               xlabel="Rouge",
                               ylabel="Verte",
                               zlabel="Bleue")

    viz.plot_features_distribution(representation_mean, n_bins=32,
                                  title="Histogrammes des moyennes des canaux RGB",
                                  features_names=["Rouge", "Vert", "Bleu"],
                                  xlabel="Valeur moyenne",
                                  ylabel="Nombre d'images")

    # Complétez l'affichage pour la métrique au choix
    representation_other_feature = dataset.Representation(data=features[:, 3:], labels=images.labels)
    viz.plot_data_distribution(representation_other_feature,
                               title="Distribution des images basée sur la métrique au choix",
                               xlabel="Rouge",
                               ylabel="Verte",
                               zlabel="Bleue")

    viz.plot_features_distribution(representation_other_feature, n_bins=32,
                                  title="Histogrammes de la métrique au choix",
                                  features_names=["Rouge", "Vert", "Bleu"],
                                  xlabel="Valeur",
                                  ylabel="Nombre d'images")

    # Étude de la corrélations
    representation = dataset.Representation(data=features, labels=images.labels)

    coast_data = representation.get_class("coast")
    variances = numpy.var(coast_data, axis=0)
    correlations = numpy.corrcoef(coast_data, rowvar=False)
    print("Variances et corrélations pour la classe coast")
    print(f"Variances : {variances}")
    print(f"Corrélations : \n{correlations}")
    # -------------------------------------------------------------------------

    plt.show()


def main():
    # pylint: disable = using-constant-test, multiple-statements

    if True: exercice_2_decorrelation()
    if True: exercice_3_visualisation_representation()
    if True: exercice_4_choix_representation()


if __name__ == "__main__":
    main()
