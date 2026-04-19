# pylint: disable = missing-function-docstring, missing-module-docstring, wrong-import-position
import os
import pathlib

import matplotlib.pyplot as plt
import numpy

# Must be call before any other TensorFlow/Keras import
# Suppress oneDNN custom operations info
# Suppress INFO and WARNING messages from TF (0=all, 1=no INFO, 2=no INFO/WARN, 3=no error)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import helpers.analysis as analysis
import helpers.classifier as classifier
import helpers.dataset as dataset
import helpers.viz as viz


def exercice_1_modele_gaussiens():
    data3classes = dataset.MultimodalDataset(pathlib.Path(__file__).parent / "data/data_3classes/")
    representation = dataset.Representation(data=data3classes.data, labels=data3classes.labels)

    # L3.E1.1 Obtenir les valeurs et vecteurs propres de ces classes, puis superposer sur le graphique les ellipses à 1 \sigma de ces classes.
    # (Complétez la fonction helpers.viz.add_ellipse utilisée ici par viz.plot_data_distribution)
    # =========================================================================
    modes = {}
    for label in representation.unique_labels:
        class_data = representation.get_class(label)

        gaussian_model = analysis.compute_gaussian_model(class_data)

        print(f"\n----- Classe {label} -----")
        viz.print_gaussian_model(*gaussian_model)

        modes[label] = class_data, gaussian_model

    # L3.E1.3 Superposer sur le graphique des classes les frontières calculées dans l'exercice préparatoire.
    # -------------------------------------------------------------------------
    viz.plot_data_distribution(representation, show_ellipses=False, analytical_boundaries=False)
    # -------------------------------------------------------------------------
    # =========================================================================

    plt.show()


def exercice_2_classificateur_ppv():
    data3classes = dataset.MultimodalDataset(pathlib.Path(__file__).parent / "data/data_3classes/")
    representation = dataset.Representation(data=data3classes.data, labels=data3classes.labels)

    # L3.E2.1 et L3.E2.2 Comparez 1-PPV et 5-PPV avec les données fournies comme représentant.
    # (Complétez la classe helpers.classifier.KNNClassifier)
    # -------------------------------------------------------------------------
    knn_classifier = classifier.KNNClassifier(n_neighbors=1)
    knn_classifier.fit(representation)
    predictions = knn_classifier.predict(representation.data)

    error_rate, error_indices = analysis.compute_error_rate(representation.labels, predictions)
    print(f"\n\n{len(error_indices)} erreur de classification sur {len(representation.labels)} échantillons ({error_rate * 100:.2f} %)")

    viz.show_confusion_matrix(representation.labels, predictions, representation.unique_labels, plot=False)

    viz.plot_classification_errors(representation, predictions)
    viz.plot_numerical_decision_regions(knn_classifier, representation)
    # -------------------------------------------------------------------------

    # L3.E2.3 et L3.E2.4 Générer un seul représentant pour chaque classe au moyen de l'algorithme des k-moyennes
    # (Complétez la classe helpers.classifier.KNNClassifier)
    # -------------------------------------------------------------------------
    knn_classifier = classifier.KNNClassifier(n_neighbors=1, use_kmeans=True, n_representatives=1)
    knn_classifier.fit(representation)
    predictions = knn_classifier.predict(representation.data)

    error_rate, error_indices = analysis.compute_error_rate(representation.labels, predictions)
    print(f"\n\n{len(error_indices)} erreur de classification sur {len(representation.labels)} échantillons ({error_rate * 100:.2f} %)")

    viz.show_confusion_matrix(representation.labels, predictions, representation.unique_labels, plot=False)
    viz.plot_classification_errors(representation, predictions)
    viz.plot_numerical_decision_regions(knn_classifier, representation)
    # -------------------------------------------------------------------------

    plt.show()


def exercice_3_classificateur_bayesien():
    data3classes = dataset.MultimodalDataset(pathlib.Path(__file__).parent / "data/data_3classes/")
    representation = dataset.Representation(data=data3classes.data, labels=data3classes.labels)

    # L3.E3.2 Compléter le classificateur Bayesian
    # (Complétez la classe helpers.classifier.BayesClassifier)
    # -------------------------------------------------------------------------
    bayes_classifier = classifier.BayesClassifier()
    bayes_classifier.fit(representation)
    predictions = bayes_classifier.predict(representation.data)

    predictions = numpy.array([representation.unique_labels[p] for p in predictions])

    error_rate, error_indices = analysis.compute_error_rate(representation.labels, predictions)
    print(f"\n\n{len(error_indices)} erreur de classification sur {len(representation.labels)} échantillons ({error_rate * 100:.2f} %)")

    viz.show_confusion_matrix(representation.labels, predictions, representation.unique_labels, plot=False)
    viz.plot_classification_errors(representation, predictions)
    viz.plot_numerical_decision_regions(bayes_classifier, representation)
    # -------------------------------------------------------------------------

    plt.show()


def exercice_s1_classificateur_bayesien_complet():
    data3classes = dataset.MultimodalDataset(pathlib.Path(__file__).parent / "data/data_3classes/")
    representation = dataset.Representation(data=data3classes.data, labels=data3classes.labels)

    # L3.S1.2 Instancier un classificateur de Bayes qui prend en compte des aprioris et une matrice de coût
    # (Complétez la classe helpers.classifier.BayesClassifier)
    # -------------------------------------------------------------------------
    aprioris = numpy.array([1 / len(representation.unique_labels)] * len(representation.unique_labels))
    cost_matrix = numpy.ones((len(representation.unique_labels), len(representation.unique_labels))) - numpy.eye(len(representation.unique_labels))

    bayes_classifier_hist = classifier.BayesClassifier(aprioris=aprioris, cost_matrix=cost_matrix, density_function=analysis.GaussianPDF)
    bayes_classifier_hist.fit(representation)
    predictions = bayes_classifier_hist.predict(representation.data)
    predictions = numpy.array([representation.unique_labels[p] for p in predictions])

    error_rate, error_indices = analysis.compute_error_rate(representation.labels, predictions)
    print(f"\n\n{len(error_indices)} erreur de classification sur {len(representation.labels)} échantillons ({error_rate * 100:.2f} %)")

    viz.show_confusion_matrix(representation.labels, predictions, representation.unique_labels, plot=False)
    viz.plot_classification_errors(representation, predictions)
    viz.plot_numerical_decision_regions(bayes_classifier_hist, representation)
    # -------------------------------------------------------------------------

    plt.show()


def exercice_s2_pdf_arbitraire():
    data3classes = dataset.MultimodalDataset(pathlib.Path(__file__).parent / "data/data_3classes/")
    representation = dataset.Representation(data=data3classes.data, labels=data3classes.labels)

    # L3.S2.1 Construire un modèle de densité de probabilité empirique pour chacune des classes
    # (Complétez le constructeur de la classe helpers.analysis.HistogramPDF utilisée par helpers.viz.plot_pdf)
    # -------------------------------------------------------------------------
    viz.plot_pdf(representation, n_bins=30)
    # -------------------------------------------------------------------------

    # L3.S2.3 Instancier un classificateur de Bayes qui utilise une densité de probabilité empirique
    # (Complétez la méthode compute_probability de la classe helpers.analysis.HistogramPDF)
    # -------------------------------------------------------------------------
    aprioris = numpy.array([1 / len(representation.unique_labels)] * len(representation.unique_labels))
    cost_matrix = numpy.ones((len(representation.unique_labels), len(representation.unique_labels))) - numpy.eye(len(representation.unique_labels))

    bayes_classifier_hist = classifier.BayesClassifier(aprioris=aprioris, cost_matrix=cost_matrix, density_function=analysis.HistogramPDF)
    bayes_classifier_hist.fit(representation)
    predictions = bayes_classifier_hist.predict(representation.data)
    predictions = numpy.array([representation.unique_labels[p] for p in predictions])

    error_rate, error_indices = analysis.compute_error_rate(representation.labels, predictions)
    print(f"\n\n{len(error_indices)} erreur de classification sur {len(representation.labels)} échantillons ({error_rate * 100:.2f} %)")

    viz.show_confusion_matrix(representation.labels, predictions, representation.unique_labels, plot=False)
    viz.plot_classification_errors(representation, predictions)
    viz.plot_numerical_decision_regions(bayes_classifier_hist, representation)
    # -------------------------------------------------------------------------

    plt.show()


def main():
    # pylint: disable = using-constant-test, multiple-statements

    if True: exercice_1_modele_gaussiens()
    if True: exercice_2_classificateur_ppv()
    if True: exercice_3_classificateur_bayesien()
    if True: exercice_s1_classificateur_bayesien_complet()
    if True: exercice_s2_pdf_arbitraire()


if __name__ == "__main__":
    main()
