import abc

from typing import List, Optional, Type

import keras
import numpy
import sklearn.cluster
import sklearn.model_selection
import sklearn.neighbors
import sklearn.preprocessing

from . import (
    analysis,
    dataset
)


class Classifier(abc.ABC):
    """
    Interface pour les classificateurs.

    Methods
    -------
    predict(data)
        Prédit les étiquettes de classe pour les données fournies.
    fit(representation)
        Entraîne le classificateur en utilisant la représentation fournie.
    """
    @abc.abstractmethod
    def predict(self, data):
        """
        Prédit les étiquettes de classe pour les données fournies.

        Args:
            data: Données à classer.

        Returns:
            Étiquettes de classe prédites.
        """

    @abc.abstractmethod
    def fit(self, representation: dataset.Representation):
        """
        Entraîne le classificateur en utilisant la représentation fournie.

        Args:
            representation: Représentation des données d'entraînement.
        """


class BayesClassifier(Classifier):
    """
    Classificateur Bayésien.

    Attributes
    ----------
    aprioris : numpy.ndarray
        Probabilités a priori pour chaque classe.
    cost_matrix : numpy.ndarray
        Matrice de coût pour la minimisation du risque.
    density_function : Type[analysis.ProbabilityDensityFunction]
        Type de fonction de densité de probabilité à utiliser pour modéliser chaque classe.
    densities : List[analysis.ProbabilityDensityFunction]
        Liste des fonctions de densité de probabilité pour chaque classe.

    Methods
    -------
    fit(representation)
        Entraîne le classificateur en utilisant la représentation fournie.
    predict(data)
        Prédit les étiquettes de classe pour les données fournies.
    """

    def __init__(self, aprioris: numpy.ndarray = None, cost_matrix: numpy.ndarray = None, density_function: Type[analysis.ProbabilityDensityFunction] = analysis.GaussianPDF):
        """
        Args:
            aprioris: Probabilités a priori pour chaque classe.
            cost_matrix: Matrice de coût pour la minimisation du risque.
            density_function: Type de fonction de densité de probabilité à utiliser pour
                modéliser chaque classe.
        """
        self.aprioris = aprioris
        self.cost_matrix = cost_matrix
        self.density_function = density_function

        self.densities: List[analysis.ProbabilityDensityFunction] = []

    def fit(self, representation: dataset.Representation):
        """
        Entraîne le classificateur en utilisant la représentation fournie
        en modélisant chaque classe avec la fonction de densité de probabilité `density_function`.

        Args:
            representation: Représentation des données d'entraînement.
        """
        if self.aprioris is None:
            self.aprioris = numpy.array([1 / len(representation.unique_labels)] * len(representation.unique_labels))

        if self.cost_matrix is None:
            self.cost_matrix = numpy.ones((len(representation.unique_labels), len(representation.unique_labels))) - numpy.eye(len(representation.unique_labels))

        for label in representation.unique_labels:
            class_data = representation.get_class(label)
            self.densities.append(self.density_function(class_data))

    def predict(self, data):
        """
        Prédit les étiquettes de classe pour les données fournies en utilisant la
        minimisation du risque.

        Args:
            data: Données à classer.

        Returns:
            Étiquettes de classe prédites.
        """
        class_probabilities = []
        for density in self.densities:
            probability = density.compute_probability(data) # P(x|C_i) for all x in the dataset
            class_probabilities.append(probability)
        class_probabilities = numpy.array(class_probabilities)

        # Minimize the risk
        # L3.E3.2 Compléter cette fonction pour déployer le classificateur en assumant des classe équiprobables à coût unitaire
        # L3.S1 Modifier cette partie pour prendre en compte la matrice de coût et des classes non équiprobables
        # ---------------------------------------------------------------------
        risks = numpy.zeros((data.shape[0], len(self.densities)))

        # Ici, argmax de la probabilité assume des coûts unitaires et des aprioris égaux
        # Dans le cadre de la problématique on cherche à minimiser le risque
        predictions = numpy.argmax(class_probabilities.T, axis=1)
        # ---------------------------------------------------------------------

        return predictions


class KNNClassifier(Classifier):
    """
    Classificateur K-Plus Proches Voisins (K-PPV).

    Attributes
    ----------
    n_neighbors : int
        Nombre de voisins à considérer lors de la classification.
    use_kmeans : bool
        Indique si les K-moyennes doivent être utilisés pour trouver des représentants de classe.
    n_representatives : int
        Nombre de représentants par classe à utiliser si use_kmeans est True.
    metric : str
        Métrique de distance à utiliser pour le calcul des voisins les plus proches.

    Methods
    -------
    fit(representation)
        Entraîne le classificateur en utilisant la représentation fournie.
    predict(data)
        Prédit les étiquettes de classe pour les données fournies.
    """
    def __init__(self, n_neighbors: int, use_kmeans: bool = False, n_representatives: int = 1, metric: str = "minkowski"):
        """
        Args:
            n_neighbors: Nombre de voisins à considérer lors de la classification.
            use_kmeans: Indique si les K-moyennes doivent être utilisés pour trouver des
                représentants de classe.
            n_representatives: Nombre de représentants par classe à utiliser si use_kmeans est True.
            metric: Métrique de distance à utiliser pour le calcul des voisins les plus proches.

        Raises:
            ValueError: Si n_representatives < 1 lorsque use_kmeans est True.
            ValueError: Si n_representatives < n_neighbors lorsque use_kmeans est True.
        """
        if use_kmeans and n_representatives < 1:
            raise ValueError("n_representatives must be at least 1 when using KMeans")

        if use_kmeans and n_representatives < n_neighbors:
            raise ValueError("n_representatives must be at least equal to n_neighbors when using KMeans")

        self.n_neighbors = n_neighbors

        self.use_kmeans = use_kmeans
        self.n_representatives = n_representatives

        if use_kmeans:
            # L3.E2.3 Compléter l'utilisation de KMeans
            # à partir des arguments fournis au constructeur de KNNClassifier
            # -----------------------------------------------------------------
            self.kmeans = sklearn.cluster.KMeans(1, n_init="auto")
            # -----------------------------------------------------------------

        # L3.E2.1 Complétez l'utilisation de KNeighborsClassifier
        # à partire des arguments fournis au constructeur de KNNClassifier
        # ---------------------------------------------------------------------
        self.knn = sklearn.neighbors.KNeighborsClassifier(1)
        # ---------------------------------------------------------------------

    def fit(self, representation: dataset.Representation):
        """
        Entraîne le classificateur en utilisant la représentation fournie.

        Si use_kmeans est True, utilise K-moyennes pour trouver un nombre `n_representatives`
        de représentants pour chaque classe avant d'entraîner le classificateur K-PPV.

        Args:
            representation: Représentation des données d'entraînement.
        """
        representatives = []
        representatives_labels = []

        if self.use_kmeans:
            for label in representation.unique_labels:
                class_data = representation.get_class(label)

                # la fonction fit écrase l'ancien estimateur KMeans entrainée.
                # Puisque seulement les représentants sont nécessaires, cela ne
                # pose pas de problème car ils sont stockés dans la liste
                # de représentants.
                self.kmeans.fit(class_data)

                representatives.append(self.kmeans.cluster_centers_)
                representatives_labels.extend([label] * self.n_representatives)

            representatives = numpy.vstack(representatives)
            representatives_labels = numpy.array(representatives_labels)

        else:
            representatives = representation.data
            representatives_labels = representation.labels

        self.knn.fit(representatives, representatives_labels)

    def predict(self, data):
        """
        Prédit les étiquettes de classe pour les données fournies en appliquant l'algorithme K-PPV.

        Args:
            data: Données à classer.
        """
        return self.knn.predict(data)


class NeuralNetworkClassifier(Classifier):
    """
    Classificateur par Réseau de Neurones à couches denses.

    Attributes
    ----------
    n_hidden : int
        Nombre de couches cachées dans le réseau de neurones.
    n_neurons : int
        Nombre de neurones par couche cachée.
    lr : float
        Taux d'apprentissage pour l'optimiseur.
    n_epochs : int
        Nombre d'époques pour l'entraînement.
    batch_size : int
        Taille des lots pour l'entraînement.
    model : Optional[keras.models.Model]
        Modèle de réseau de neurones Keras.
    history : Optional[keras.callbacks.History]
        Historique de l'entraînement du modèle.

    Methods
    -------
    fit(representation)
        Entraîne le classificateur en utilisant la représentation fournie.
    predict(data)
        Prédit les étiquettes de classe pour les données fournies.
    save(path)
        Sauvegarde le modèle entraîné à l'emplacement spécifié.
    load(path)
        Charge un modèle entraîné depuis l'emplacement spécifié. Ce dernier remplace et écrase le modèle actuel.
    """
    model: Optional[keras.models.Model]
    history: Optional[keras.callbacks.History]

    def __init__(self, input_dim: int, output_dim: int, n_hidden:int=2, n_neurons:int=2, lr:float=0.01, n_epochs:int=1000, batch_size:int=16):
        """
        Args:
            input_dim: Dimension des données d'entrée.
            output_dim: Nombre de classes de sortie.
            n_hidden: Nombre de couches cachées dans le réseau de neurones.
            n_neurons: Nombre de neurones par couche cachée.
            lr: Taux d'apprentissage pour l'optimiseur.
            n_epochs: Nombre d'époques pour l'entraînement.
            batch_size: Taille des lots pour l'entraînement.
        """
        self.n_hidden = n_hidden
        self.n_neurons = n_neurons

        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        self.history = None

        # Define the model's architecture

        # L2.E4.3 Complétez le code fourni pour déployer un classificateur par RN sur les 3 classes fournies.
        # (configurer le nombre de neurones par couches et le nombre de couches cachées via le constructeur (NeuralNetworkClassifier(...)))
        # (configurer les fonctions d'activation appropriées pour chaque couche directement ici)
        # -------------------------------------------------------------------------
        self.model = keras.models.Sequential()
        self.model.add(keras.layers.InputLayer(shape=(input_dim,)))
        for _ in range(self.n_hidden - 1):
            self.model.add(keras.layers.Dense(units=self.n_neurons, activation="tanh"))
        self.model.add(keras.layers.Dense(units=output_dim, activation="softmax"))
        # -------------------------------------------------------------------------

        print(self.model.summary())

        # L2.E4.1 Utilisez une loss plus appropriée que MSE pour l'entraînement d'un classificateur.
        # L2.E4.5 Calculez la précision du modèle à chaque epoch.
        # (choisir la métrique appropriée)
        # -------------------------------------------------------------------------
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.lr),
            loss=keras.losses.MeanSquaredError(),
            metrics=[]
        )
        # -------------------------------------------------------------------------

    def preprocess_data(self, data: numpy.ndarray) -> numpy.ndarray:
        """
        Prétraite les données avant l'entraînement ou la prédiction.

        Args:
            data: Données à prétraiter.

        Returns:
            Données prétraitées.
        """

        # TODO Problematique: voir si d'autres prétraitements sont nécessaires (l'appliquer également sur les données de validation/test)
        # -------------------------------------------------------------------------
        data = analysis.rescale_data(data)
        # -------------------------------------------------------------------------

        return data

    def prepare_datasets(self, representation: dataset.Representation):
        """
        Prépare les ensembles d'entraînement et de validation à partir de la représentation fournie.

        Args:
            representation: Représentation des données.

        Returns:
            Tuple contenant les données d'entraînement, les données de validation,
        """

        # L2.E4.1 Convertissez les étiquettes de classe en un format qui permet d'utiliser une loss plus approprié que MSE
        # pour l'entraînement d'un classificateur.
        # -------------------------------------------------------------------------
        # Utiliser OneHotEncoder de sklearn à la place de cette ligne
        one_hot_labels = numpy.zeros((representation.labels.shape[0], len(representation.unique_labels)))
        # -------------------------------------------------------------------------

        # L2.E4.2 Partitionnez les données en sous-ensemble d'entraînement et de validation.
        # -------------------------------------------------------------------------
        # Prepare datasets
        train_data = representation.data
        val_data = []
        train_labels = one_hot_labels
        val_labels = []
        # -------------------------------------------------------------------------

        return train_data, val_data, train_labels, val_labels

    def fit(self, representation: dataset.Representation):
        """
        Entraîne le classificateur en utilisant la représentation fournie.

        Suite à l'entraînement, l'historique de l'entraînement est stocké
        dans l'attribut `history`.

        Args:
            representation: Représentation des données d'entraînement.
        """
        representation.data = self.preprocess_data(representation.data)
        train_data, val_data, train_labels, val_labels = self.prepare_datasets(representation)

        # L2.E4.4 Utilisez un callback pour visualiser la performance de l'entraînement tout les 25 epochs.
        # et un autre pour arrêter l'entraînement lorsque la généralisation se dégrade.
        # -------------------------------------------------------------------------
        callbacks=[]

        self.history = self.model.fit(
            train_data, train_labels,
            # validation_data=(val_data, val_labels), # TODO: Décommenter si un ensemble de validation est utilisé
            batch_size=self.batch_size,
            epochs=self.n_epochs,
            callbacks=callbacks,
            verbose=False
        )
        # -------------------------------------------------------------------------

    def predict(self, data):
        """
        Prédit les étiquettes de classe pour les données fournies.

        Args:
            data: Données à classer.

        Returns:
            Étiquettes de classe prédites.

        Raises:
            ValueError: Si le modèle n'a pas encore été entraîné.
        """
        if self.model is None:
            raise ValueError("The model has not been trained yet. Call fit() before predict() or load an existing model using load().")

        data = self.preprocess_data(data)

        predictions = self.model.predict(data)
        predicted_classes = numpy.argmax(predictions, axis=-1)
        return predicted_classes

    def save(self, path: str):
        """
        Sauvegarde le modèle entraîné à l'emplacement spécifié.

        Args:
            path: Chemin où sauvegarder le modèle.

        Raises:
            ValueError: Si le modèle n'a pas encore été entraîné.
        """
        if self.model is None:
            raise ValueError("The model has not been trained yet. Call fit() before save() or load an existing model using load().")

        self.model.save(path)

    def load(self, path: str) -> keras.models.Model:
        """
        Charge un modèle entraîné depuis l'emplacement spécifié.
        Ce dernier remplace et écrase le modèle actuel.
        """
        self.model = keras.models.load_model(path)
        return self.model


class PrintEveryNEpochs(keras.callbacks.Callback):
    """
    Classe de callback Keras qui affiche la performance de l'entraînement
    tous les n_epochs spécifiés.

    Attributes:
        n_epochs (int): le nombre d'époques entre chaque affichage.

    Methods:
        on_epoch_end(epoch, logs): méthode appelée à la fin de chaque époque pour
            afficher les métriques si l'époque est un multiple de n_epochs.
    """
    def __init__(self, n_epochs: int):
        """
        Args:
            n_epochs (int): le nombre d'époques entre chaque affichage.
        """
        super().__init__()
        self.n_epochs = n_epochs

    def on_epoch_end(self, epoch, logs=None):
        """
        Méthode appelée à la fin de chaque époque pour afficher les métriques
        si l'époque est un multiple de n_epochs.

        Args:
            epoch (int): L'indice de l'époque actuelle.
            logs (dict): Un dictionnaire contenant les métriques de l'époque.
        """
        # L2.E2.4 Visualiser la performance de l'entraînement d'une manière plus ergonomique
        # que l'affichage par défaut, par exemple à chaque multiple de n_epochs.
        if (epoch + 1) % self.n_epochs == 0:
            loss = logs["loss"]
            val_loss = logs["val_loss"]

            accuracy = ""
            val_accuracy = ""
            if "accuracy" in logs:
                accuracy = logs["accuracy"]
                val_accuracy = logs["val_accuracy"]

            print(f"Epoch {epoch + 1:>3}: loss = {loss:.4f}, val_loss = {val_loss:.4f}, accuracy = {accuracy:.4f}, val_accuracy = {val_accuracy:.4f}")


def set_deterministic(seed: int = 0):
    """
    Définit un seed pour rendre les opérations aléatoires déterministes.
    À utiliser quand vous voulez déverminer vos expériences et obtenir des résultats reproductibles.

    Args:
        seed (int): la valeur du seed à utiliser.
    """
    # Lazy import as they may not be needed by the user otherwise
    import random
    import tensorflow as tf

    random.seed(seed)
    numpy.random.seed(seed)
    tf.random.set_seed(seed)
