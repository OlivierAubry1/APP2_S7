import pathlib

import keras
import numpy as np
import sklearn
from keras.src.utils.module_utils import scipy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import helpers.dataset as dataset
import helpers.viz as viz
import matplotlib.pyplot as plt
from skimage import filters, color

from helpers import analysis, classifier


def to_grayscale(img):
    if img.ndim == 3 and img.shape[2] == 3:
        return np.dot(img[...,:3], [0.299, 0.587, 0.114])
    return img


def extract_rgb_features(dataset):
    num_samples = len(dataset)
    features = np.zeros((num_samples, 3), dtype=np.float32)

    for i in range(num_samples):
        img_array, _ = dataset[i]
        # Red channel mean
        features[i, 0] = np.mean(img_array[:, :, 0])
        # Green channel mean
        features[i, 1] = np.mean(img_array[:, :, 1])
        # Blue channel mean
        features[i, 2] = np.mean(img_array[:, :, 2])

    return features


def extract_bw_features(dataset):
    num_samples = len(dataset)
    features = np.zeros((num_samples, 3), dtype=np.float32)

    for i in range(num_samples):
        img_rgb, _ = dataset[i]

        # Convert to grayscale
        bw_img = np.dot(img_rgb[..., :3], [0.299, 0.587, 0.114])

        # Feature 1: Average Brightness
        features[i, 0] = np.mean(bw_img)
        # Feature 2: Contrast (Standard Deviation)
        features[i, 1] = np.std(bw_img)

    return features


def extract_sobel_features(dataset):
    num_samples = len(dataset)
    features = np.zeros((num_samples, 3), dtype=np.float32)

    for i in range(num_samples):
        img_rgb, _ = dataset[i]

        img_gray = color.rgb2gray(img_rgb)

        edge_h = np.abs(filters.sobel_h(img_gray))
        edge_v = np.abs(filters.sobel_v(img_gray))

        sum_h = np.sum(edge_h)
        sum_v = np.sum(edge_v)

        h, w = edge_h.shape

        # Horizontal-ness of the middle
        middle_strip = edge_h[int(h * 0.30): int(h * 0.60), :]
        features[i, 0] = np.mean(middle_strip)

        # Standard Deviation of the horizontal edge map
        features[i, 1] = np.std(edge_h)

        # ratio of Vertical to Horizontal
        features[i, 2] = sum_v / (sum_h + 1e-6)

    return features

def extract_hsv_features(dataset):
    num_samples = len(dataset)
    features = np.zeros((num_samples, 3), dtype=np.float32)

    for i in range(num_samples):
        img_rgb, _ = dataset[i]
        img_hsv = color.rgb2hsv(img_rgb)

        features[i, 0] = np.mean(img_hsv[:, :, 0])

        # Average Saturation
        features[i, 1] = np.mean(img_hsv[:, :, 1])

        # Average Value (Brightness)
        features[i, 2] = np.mean(img_hsv[:, :, 2])

    return features

def extract_grayness(dataset):
    features = np.zeros((len(dataset), 1))
    for i in range(len(dataset)):
        img_rgb, _ = dataset[i]

        diff = np.std(img_rgb, axis=2)

        gray_pixels = np.sum(diff < 10) / (img_rgb.shape[0] * img_rgb.shape[1])
        features[i, 0] = gray_pixels
    return features

def extract_perspective_convergence(dataset):
    features = np.zeros((len(dataset), 1), dtype=np.float32)

    for i in range(len(dataset)):
        img_rgb, _ = dataset[i]
        img_gray = color.rgb2gray(img_rgb)

        # 1. Get both H and V gradients
        gh = filters.sobel_h(img_gray)
        gv = filters.sobel_v(img_gray)

        # 2. Calculate the Angle of every edge (in radians)
        # This tells us if a line is horizontal, vertical, or diagonal
        angles = np.arctan2(gv, gh)

        # 3. Split the image into Left and Right halves
        h, w = angles.shape
        left_half = angles[:, :w // 2]
        right_half = angles[:, w // 2:]

        # 4. Check for "Street Geometry":
        # We want to see if the left side has / slopes and right has \ slopes
        # A simple way: are the average angles mirrored?
        left_diag = np.mean(np.abs(left_half))
        right_diag = np.mean(np.abs(right_half))

        # The feature: how much "Diagonal Energy" is balanced across the center
        features[i, 0] = (left_diag + right_diag) / 2
    return features


def train_eval_nn(X_train, X_val, y_train, y_val, class_labels):
    """
    Configure, entraîne et évalue le réseau de neurones avec des données pré-séparées.
    """
    # L2.E3.4 Network Configuration
    print("\n" + "="*50)
    print("--- Entraînement du Réseau de Neurones ---")
    model = keras.models.Sequential([
        keras.layers.InputLayer(shape=(X_train.shape[-1],)),
        keras.layers.Dense(units=32, activation="relu"),
        keras.layers.Dense(units=16, activation="relu"),
        keras.layers.Dense(units=y_train.shape[-1], activation="linear")
    ])

    model.compile(
        optimizer=keras.optimizers.SGD(learning_rate=0.08, momentum=0.03),
        loss="mean_squared_error",
        metrics=["accuracy"]
    )

    history = model.fit(
        X_train, y_train,
        batch_size=len(X_train),
        epochs=750,
        validation_data=(X_val, y_val),
        verbose=True
    )

    # Save and Evaluate
    model_save_path = pathlib.Path(__file__).parent / "saves/image_classifier.keras"
    # S'assurer que le dossier saves existe
    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_save_path)
    viz.plot_metric_history(history)

    y_val_integers = np.argmax(y_val, axis=-1)
    prediction = np.argmax(model.predict(X_val), axis=-1)

    error_rate, indexes_errors = analysis.compute_error_rate(y_val_integers, prediction)

    print(f"\n\n{len(indexes_errors)} erreurs sur {len(y_val_integers)} échantillons ({error_rate * 100:.2f}%).")
    viz.show_confusion_matrix(y_val_integers, prediction, class_labels, plot=True)
    plt.show()

def train_eval_bayes(X_train, X_val, y_train, y_val, class_labels):
    """
    Configure, entraîne et évalue le classificateur Bayésien
    """
    print("\n" + "="*50)
    print("--- Entraînement du Classificateur Bayésien ---")
    
    # 1. Reconvertir les étiquettes one-hot en entiers (0, 1, 2)
    y_train_integers = np.argmax(y_train, axis=-1)
    y_val_integers = np.argmax(y_val, axis=-1)
    
    # 2. Créer l'objet Representation attendu par la méthode fit()
    train_representation = dataset.Representation(data=X_train, labels=y_train_integers)
    
    # 3. Définir les aprioris et la matrice de coûts (équiprobables et coûts unitaires par défaut)
    n_classes = len(class_labels)
    aprioris = np.array([1 / n_classes] * n_classes)
    cost_matrix = np.ones((n_classes, n_classes)) - np.eye(n_classes)
    
    bayes_classifier = classifier.BayesClassifier(
        aprioris=aprioris, 
        cost_matrix=cost_matrix, 
        density_function=analysis.GaussianPDF
    )
    
    bayes_classifier.fit(train_representation)

    prediction = bayes_classifier.predict(X_val)
    
    # 7. Évaluer et afficher les performances
    error_rate, indexes_errors = analysis.compute_error_rate(y_val_integers, prediction)
    
    print(f"\nPerformances Bayésien : {len(indexes_errors)} erreurs sur {len(y_val_integers)} échantillons ({error_rate * 100:.2f}% d'erreur).")
    
    # Affichage de la matrice de confusion
    viz.show_confusion_matrix(y_val_integers, prediction, class_labels, plot=True)
    plt.show()

def knn(n_neighbors, representation, use_kmeans, n_representatives):
    knn_classifier = classifier.KNNClassifier(n_neighbors=n_neighbors, use_kmeans=use_kmeans, n_representatives=n_representatives)
    knn_classifier.fit(representation)
    predictions = knn_classifier.predict(representation.data)

    error_rate, error_indices = analysis.compute_error_rate(representation.labels, predictions)
    print(
        f"\n\n{len(error_indices)} erreur de classification sur {len(representation.labels)} échantillons ({error_rate * 100:.2f} %)")


def problematique():
    dataset_path = pathlib.Path(__file__).parent / "data" / "image_dataset"
    images = dataset.ImageDataset(dataset_path)

    # 1. Feature Extraction
    features_sobel = extract_sobel_features(images)
    features_perspective = extract_perspective_convergence(images)
    features_hsv = extract_hsv_features(images)
    features_grayness = extract_grayness(images)

    # 2. Stack specific features as requested
    # [Sobel Std, HSV Value, V/H Ratio, Perspective]
    features = np.column_stack((
        features_sobel[:, 1],
        features_hsv[:, 2],
        features_sobel[:, 2],
        features_perspective[:, 0],
        features_grayness[:, 0]
    ))

    # 3. Standardization (Centering and Scaling)
    means = np.mean(features, axis=0)
    stds = np.std(features, axis=0)
    features_centered = (features - means) / (stds + 1e-6)

    # 4. Map to variable names used in the original main()
    data = features_centered.astype(np.float32)

    # Process Labels
    encodeur = LabelEncoder()
    labels = encodeur.fit_transform(images.labels)  # Integer labels [0, 1, 2]
    labels_one_hot = keras.utils.to_categorical(labels).astype(np.int32)

    # L2.E3.1 Étudiez l'espace de la représentation
    # =========================================================================
    C1 = data[np.where(labels == 0)]
    C2 = data[np.where(labels == 1)]
    C3 = data[np.where(labels == 2)]

    print("\n----- Classe 1 -----")
    mean1, cov1, eigvals1, eigvecs1 = analysis.compute_gaussian_model(C1)
    viz.print_gaussian_model(mean1, cov1, eigvals1, eigvecs1)

    print("\n----- Classe 2 -----")
    mean2, cov2, eigvals2, eigvecs2 = analysis.compute_gaussian_model(C2)
    viz.print_gaussian_model(mean2, cov2, eigvals2, eigvecs2)

    print("\n----- Classe 3 -----")
    mean3, cov3, eigvals3, eigvecs3 = analysis.compute_gaussian_model(C3)
    viz.print_gaussian_model(mean3, cov3, eigvals3, eigvecs3)

    # Plot 3D representations using different combinations of your 4 features
    representation1 = dataset.Representation(data=data[:, [0, 1, 2]], labels=labels)
    viz.plot_data_distribution(representation1, title="Représentation 3D (Sobel Std, HSV Val, V/H Ratio)",
                               xlabel="Sobel Std", ylabel="HSV Value", zlabel="V/H Ratio")

    representation2 = dataset.Representation(data=data[:, [0, 2, 3]], labels=labels)
    viz.plot_data_distribution(representation2, title="Représentation 3D (Sobel Std, V/H Ratio, Perspective)",
                               xlabel="Sobel Std", ylabel="V/H Ratio", zlabel="Perspective")

    plt.show()
    
    # 1. Séparation (Split)
    X_train_brut, X_val_brut, y_train, y_val = train_test_split(
        data, labels_one_hot, test_size=0.3, random_state=42
    )

    # 2. Standardisation (Centrage et Réduction)
    # On "apprend" la moyenne et l'écart-type
    scaler = sklearn.preprocessing.StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_brut)
    # On applique la même transformation à la validation
    X_val_scaled = scaler.transform(X_val_brut)

    # 3. PCA Decorrelation
    # On "apprend" les composantes principales
    pca_5_components = sklearn.decomposition.PCA(n_components=5)
    X_train_pca = pca_5_components.fit_transform(X_train_scaled)
    # On projette la validation sur les mêmes axes
    X_val_pca = pca_5_components.transform(X_val_scaled)

    representation_pca = dataset.Representation(data=np.vstack((X_train_pca, X_val_pca)), 
                                                labels=np.concatenate((np.argmax(y_train, axis=1), np.argmax(y_val, axis=1))))
    # viz.plot_data_distribution(representation_pca, title="Données projetées sur les 3 composantes PCA",
    #                            xlabel="PC 1", ylabel="PC 2", zlabel="PC 3")
    # plt.show()
    
    #train_eval_nn(X_train_pca, X_val_pca, y_train, y_val, encodeur.classes_)
    #train_eval_bayes(X_train_pca, X_val_pca, y_train, y_val, encodeur.classes_)
    print("\n" + "="*50)
    print("--- Entraînement du Classificateur KNN ---")
    
    # On crée l'objet Representation avec les données d'entraînement projetées
    # (Puisque ta fonction s'évalue elle-même sur representation.data)
    y_train_integers = np.argmax(y_train, axis=-1)
    knn_representation = dataset.Representation(data=X_train_pca, labels=y_train_integers)
    

    knn(5, knn_representation, use_kmeans=False, n_representatives=5)
    knn(12, knn_representation, use_kmeans=True, n_representatives=24)

if __name__ == "__main__":
    problematique()