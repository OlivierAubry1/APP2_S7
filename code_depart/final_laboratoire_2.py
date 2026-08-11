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


def main():
    data3classes = dataset.MultimodalDataset(pathlib.Path(__file__).parent / "data/data_3classes/")
    representation = dataset.Representation(data=data3classes.data, labels=data3classes.labels)

    # L2.E4.1, L2.E4.2, L2.E4.3 et L2.E4.4
    # Complétez la classe NeuralNetworkClassifier dans helpers/classifier.py
    # -------------------------------------------------------------------------
    nn_classifier = classifier.NeuralNetworkClassifier(input_dim=representation.data.shape[1],
                                                       output_dim=len(representation.unique_labels),
                                                       n_hidden=1,
                                                       n_neurons=2,
                                                       lr=0.01,
                                                       n_epochs=10,
                                                       batch_size=16)
    # -------------------------------------------------------------------------
    nn_classifier.fit(representation)

    # Save the model
    nn_classifier.save(pathlib.Path(__file__).parent / "saves/multimodal_classifier.keras")

    # Plot training metrics
    viz.plot_metric_history(nn_classifier.history)

    # Evaluate the model
    # Load the trained model
    nn_classifier.load(pathlib.Path(__file__).parent / "saves/multimodal_classifier.keras")

    # Generate a uniform distribution of samples over the minmax domain of the data
    viz.plot_numerical_decision_regions(nn_classifier, representation)

    # Predict the classes over the whole dataset
    predictions = nn_classifier.predict(representation.data)
    predictions = numpy.array([representation.unique_labels[i] for i in predictions])

    # L2.E4.5 Calculez et commentez les performance au moyen du taux de classification de données
    # et la matrice de confusion du modèle sur l'ensemble des données après l'entraînement.
    # -------------------------------------------------------------------------
    error_rate, indexes_errors = analysis.compute_error_rate(representation.labels, predictions)
    print(f"\n{len(indexes_errors)} erreurs de classification sur {len(representation.labels)} échantillons ({error_rate * 100:.2f}%).")

    viz.show_confusion_matrix(representation.labels, predictions, representation.unique_labels, plot=True)
    # -------------------------------------------------------------------------

    viz.plot_classification_errors(representation, predictions)

    plt.show()


if __name__ == "__main__":
    # Décommenter ceci pour rendre le code déterministe et pouvoir déverminer.
    # classifier.set_deterministic(seed=42)

    main()
