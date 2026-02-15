# Copyright (c) 2018, Simon Brodeur
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
#  - Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#  - Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#  - Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
# NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
# OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

# Author: Simon Brodeur <simon.brodeur@usherbrooke.ca>
# Université de Sherbrooke, APP3 S8GIA, A2018

# Update log:
# 2026-01-20: Gabriel Lauzier, Mise à jour vers Keras 3.x

"""
Laboratoire 2 - IA probabiliste et bioinspirée, Reconnaissance des formes, Apprentissage machine

Objectif: Concevoir un classificateur RN à partir d'une représentation existante

L2.E3  - Réseau de neurones pour classifier des fleurs
"""
# pylint: disable = missing-function-docstring, missing-module-docstring, ungrouped-imports, wrong-import-position
import os
import pathlib

import matplotlib.pyplot as plt
import numpy
import scipy
import sklearn.decomposition
import sklearn.model_selection

# Must be call before any other TensorFlow/Keras import
# Suppress oneDNN custom operations info
# Suppress INFO and WARNING messages from TF (0=all, 1=no INFO, 2=no INFO/WARN, 3=no error)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import keras

import helpers.analysis as analysis
import helpers.dataset as dataset
import helpers.classifier as classifier
import helpers.viz as viz


def main():
    # Load the iris dataset
    data_matrix = scipy.io.loadmat(pathlib.Path(__file__).parent / "data/iris.mat")
    data = numpy.array(data_matrix["data"], dtype=numpy.float32)
    labels_one_hot = numpy.array(data_matrix["target"], dtype=numpy.int32)

    # L2.E3.1 Étudiez l'espace de la représentation
    # =========================================================================
    labels = numpy.argmax(labels_one_hot, axis=-1) # labels one-hot encoding to integer

    C1 = data[numpy.where(labels == 0)]
    C2 = data[numpy.where(labels == 1)]
    C3 = data[numpy.where(labels == 2)]

    print("\n----- Classe 1 -----")
    mean1, cov1, eigvals1, eigvecs1 = analysis.compute_gaussian_model(C1)
    viz.print_gaussian_model(mean1, cov1, eigvals1, eigvecs1)

    print("\n----- Classe 2 -----")
    mean2, cov2, eigvals2, eigvecs2 = analysis.compute_gaussian_model(C2)
    viz.print_gaussian_model(mean2, cov2, eigvals2, eigvecs2)

    print("\n----- Classe 3 -----")
    mean3, cov3, eigvals3, eigvecs3 = analysis.compute_gaussian_model(C3)
    viz.print_gaussian_model(mean3, cov3, eigvals3, eigvecs3)

    # Plot 3D representation of the dataset

    # L2.E3.1 Observez si différentes combinaisons de dimensions sont discriminantes.
    # -------------------------------------------------------------------------
    # dimension 0, 1, 2
    representation = dataset.Representation(data=data[:, [0, 1, 2]], labels=labels)
    viz.plot_data_distribution(representation, title="Représentation 3D des fleurs d'iris (dim 1, 2, 3)", xlabel="Caractéristique 1", ylabel="Caractéristique 2", zlabel="Caractéristique 3")

    # dimension i, j, k

    # -----------------------------------------------------------------

    plt.show()

    # Décorélation
    # Ici, nous utilisons une PCA (analyse par composante principale)
    # au lieu de la fonction `analysis.project_onto_new_basis` avec les vecteurs propres.
    # Une PCA calcul les vecteurs et valeurs propres des données et les trie en fonction
    # de leurs valeurs propres (importance). Elle projete ensuite les données dans cet espace.
    # Ici, nous demandons une projection dans l'espace des 3 composantes avec la plus grande variance.
    # TODO: Dans la problématique on demande d'utiliser les techniques vue au laboratoire 1
    pca_3_components = sklearn.decomposition.PCA(n_components=3)
    pca_3_components.fit(data)

    # Projection des données dans l'espace PCA à 3 composantes
    data_projected = pca_3_components.transform(data)
    C1_projected = pca_3_components.transform(C1)
    C2_projected = pca_3_components.transform(C2)
    C3_projected = pca_3_components.transform(C3)

    print("\n----- Classe 1 projetée -----")
    mean1_p, cov1_p, eigvals1_p, eigvecs1_p = analysis.compute_gaussian_model(C1_projected)
    viz.print_gaussian_model(mean1_p, cov1_p, eigvals1_p, eigvecs1_p)

    print("\n----- Classe 2 projetée -----")
    mean2_p, cov2_p, eigvals2_p, eigvecs2_p = analysis.compute_gaussian_model(C2_projected)
    viz.print_gaussian_model(mean2_p, cov2_p, eigvals2_p, eigvecs2_p)

    print("\n----- Classe 3 projetée -----")
    mean3_p, cov3_p, eigvals3_p, eigvecs3_p = analysis.compute_gaussian_model(C3_projected)
    viz.print_gaussian_model(mean3_p, cov3_p, eigvals3_p, eigvecs3_p)

    representation = dataset.Representation(data=data_projected, labels=labels)
    viz.plot_data_distribution(representation, title="Représentation projetées sur les 3 premières composantes principales", xlabel="PC 1", ylabel="PC 2", zlabel="PC 3")

    plt.show()
    # =========================================================================

    # L2.E3.1 et L2.E3.2 Conservez les dimensions qui vous semblent appropriées et décorrélées les au besoin.
    # (e.g., normalisation, centrage, filtrage, réduction de dimensionnalité, etc.)
    # -------------------------------------------------------------------------
    # normalized in the range [-1, 1]
    scaled_data = analysis.rescale_data(data_projected)
    # -------------------------------------------------------------------------

    # L2.E3.3 Créez un ensemble d'entraînement et de validation à partir des données préparées.
    # -------------------------------------------------------------------------
    train_data = scaled_data
    val_data = []
    train_labels = labels_one_hot
    val_labels = []
    # -------------------------------------------------------------------------

    # L2.E3.4 Testez plusieurs configurations de réseaux de neurones et de fonction d'activation.
    # -------------------------------------------------------------------------
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=(scaled_data.shape[-1],)))
    model.add(keras.layers.Dense(units=3, activation="linear"))
    model.add(keras.layers.Dense(units=labels_one_hot.shape[-1], activation="linear"))
    print(model.summary())
    # -------------------------------------------------------------------------

    # L2.E3.4 Testez plusieurs configurations d'optimisateur, de taux d'apprentissage et de fonction de coût.
    # -------------------------------------------------------------------------
    model.compile(
        optimizer=keras.optimizers.SGD(learning_rate=0.001, momentum=0.01),
        loss=keras.losses.MeanSquaredError(),
        metrics=["accuracy"]
    )
    # -------------------------------------------------------------------------

    # Entrainement du modèle
    callbacks=[]

    # L2.E3.3 Ajoutez les arguments pour l'ensemble de validation.
    # L2.E3.4 Testez plusieurs configurations de nombre d'epochs et de taille de batch.
    # -------------------------------------------------------------------------
    history: keras.callbacks.History = model.fit(
        train_data, train_labels,
        batch_size=16,
        shuffle=True,
        epochs=10,
        callbacks=callbacks,
        verbose=True
    )
    # -------------------------------------------------------------------------

    # Sauvegarde du modèle
    model.save(pathlib.Path(__file__).parent / "saves/iris_classifier.keras")

    # Affichage des métriques d'entraînement
    viz.plot_metric_history(history)

    # Évaluation du modèle
    loaded_model = keras.models.load_model(pathlib.Path(__file__).parent / "saves/iris_classifier.keras")
    prediction = numpy.argmax(loaded_model.predict(scaled_data), axis=-1)

    error_rate, indexes_errors = analysis.compute_error_rate(labels, prediction)
    print(f"\n\n{len(indexes_errors)} erreurs de classification sur {len(labels)} échantillons ({error_rate * 100:.2f}%).")

    viz.show_confusion_matrix(labels, prediction, representation.unique_labels, plot=True)

    plt.show()


if __name__ == "__main__":
    # Décommenter ceci pour rendre le code déterministe et pouvoir déverminer.
    # classifier.set_deterministic(seed=42)
    main()
