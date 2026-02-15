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
# 2022-09-01: Jean-Baptiste Michaud, Mise à jours vers TensorFlow 2.x
# 2026-01-20: Gabriel Lauzier, Mise à jour vers Keras 3.x

"""
Laboratoire 2 - IA probabiliste et bioinspirée, Reconnaissance des formes, Apprentissage machine

Objectif: Réaliser un classificateur élémentaire

L2.E2  - OU exclusif et réseaux de neurones artificiels
"""
# pylint: disable = missing-function-docstring, missing-module-docstring, ungrouped-imports, wrong-import-position
import os
import pathlib

import matplotlib.pyplot as plt
import numpy

# Must be call before any other TensorFlow/Keras import
# Suppress oneDNN custom operations info
# Suppress INFO and WARNING messages from TF (0=all, 1=no INFO, 2=no INFO/WARN, 3=no error)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import keras

import helpers.viz as viz


def main():
    # XOR dataset
    data = numpy.array([[0, 0],
                        [0, 1],
                        [1, 0],
                        [1, 1]])

    labels = numpy.array([[0],
                          [1],
                          [1],
                          [0]])

    # Plotting the XOR dataset
    plt.figure(figsize=(4, 4))
    scatter = plt.scatter(data[:, 0], data[:, 1], c=labels.flatten(), s=200, cmap="bwr", edgecolors="k")
    plt.title("XOR Dataset Visualization")
    plt.xlabel("Input 1")
    plt.ylabel("Input 2")
    plt.legend(*scatter.legend_elements(), title="Classes")

    # Create neural network model
    # TODO: Expérimentez avec différentes architectures de réseaux de neurones
    # (nombre de couches, nombre de neurones par couche, fonctions d'activation, etc.)
    # -------------------------------------------------------------------------
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(shape=(2,)))
    model.add(keras.layers.Dense(units=1, activation="sigmoid"))
    model.add(keras.layers.Dense(units=1, activation="linear"))
    print(model.summary())
    # -------------------------------------------------------------------------

    # Define loss function and optimizer
    # TODO: Expérimentez avec différent hyperparamètres d'entraînement
    # (fonction de perte, optimiseur, taux d'apprentissage, etc.)
    # -------------------------------------------------------------------------
    model.compile(
        optimizer=keras.optimizers.SGD(learning_rate=0.5, momentum=0.9),
        loss=keras.losses.MeanSquaredError(),
        metrics=None
    )
    # -------------------------------------------------------------------------

    # Train the model
    history: keras.callbacks.History = model.fit(
        data, labels,
        batch_size=len(data),
        shuffle=True,
        epochs=1000,
        callbacks=None,
        verbose=True
    )

    # Plot metrics
    viz.plot_metric_history(history)

    # Save the trained model
    saves_directory = pathlib.Path(__file__).parent / "saves"
    saves_directory.mkdir(exist_ok=True)
    model.save(saves_directory / "xor_model.keras")

    # Load the trained model
    loaded_model: keras.models.Model = keras.models.load_model(saves_directory / "xor_model.keras")

    # Evaluate the loaded model
    prediction = loaded_model.predict(data)

    n_errors = numpy.sum(numpy.round(prediction) != labels)
    accuracy = (len(data) - n_errors) / len(data)
    print(f"Model accuracy on XOR dataset: {accuracy * 100:.2f}%")

    plt.show()


if __name__ == "__main__":
    # Décommenter ceci pour rendre le code déterministe et pouvoir déverminer.
    # import helpers.classifier as classifier
    # classifier.set_deterministic(seed=42)

    main()
