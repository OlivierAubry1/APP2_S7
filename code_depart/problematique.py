import os
import pathlib

# Doit être appelé avant toute importation de TensorFlow/Keras
# Supprime les infos d'opérations personnalisées oneDNN et les messages INFO/WARNING de TF
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import matplotlib.pyplot as plt
import skimage
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import keras
from scipy.stats import multivariate_normal

# Importation de tes modules d'aide
import helpers.analysis as analysis
import helpers.dataset as dataset
import helpers.viz as viz


def problematique(run_nn=True, run_bayes=True):
    img_path = pathlib.Path(__file__).parent / "data" / "image_dataset"
    images = dataset.ImageDataset(str(img_path))

    # =========================================================================
    # 1. REPRÉSENTATION : Extraction des caractéristiques
    # =========================================================================
    print("Extraction des caractéristiques en cours...")
    features = np.zeros((len(images), 8))
    
    for i, (image, label) in enumerate(images):
        image_norm = image / 255.0
        hsv_image = skimage.color.rgb2hsv(image_norm)
        gray_image = skimage.color.rgb2gray(image_norm)

        height = image.shape[0]
        top_part = image[:height//3, :, :]

        edges = np.abs(skimage.filters.sobel(gray_image))
        edge_h = skimage.filters.sobel_h(gray_image)
        edge_v = skimage.filters.sobel_v(gray_image)

        sum_h = np.sum(np.abs(edge_h))
        sum_v = np.sum(np.abs(edge_v))

        # Calcul des 8 caractéristiques
        feat_saturation = np.mean(hsv_image[:, :, 1])
        feat_hue = np.mean(hsv_image[:, :, 0]) 
        feat_texture = np.std(hsv_image[:, :, 2])       
        feat_top_blue = np.mean(top_part[:, :, 2])
        feat_green = np.mean(image[:, :, 1])
        feat_edges = np.mean(edges)
        feat_texture_sobel = np.std(edges)
        feat_ratio = sum_v / (sum_h + sum_v + 1e-6)

        features[i] = [
            feat_saturation, feat_hue, feat_texture, feat_top_blue, 
            feat_green, feat_edges, feat_texture_sobel, feat_ratio
        ]

    # Normalisation
    features_mean = np.mean(features, axis=0)
    features_std = np.std(features, axis=0) + 1e-8
    features_normalized = (features - features_mean) / features_std
    
    # =========================================================================
    # 2. VISUALISATION DES CARACTÉRISTIQUES (Optionnel, tu peux commenter plt.show())
    # =========================================================================
    feature_names = ["Saturation", "Teinte", "Texture", "Bleu_Haut", "Vert_Global", "Edges Sobel", "Texture Sobel", "Ratio Sobel"]
    representation_raw = dataset.Representation(data=features_normalized, labels=images.labels)
    
    viz.plot_features_distribution(
        representation_raw, n_bins=32,
        title="Distribution des features normalisées",
        features_names=feature_names,
        xlabel="Valeur", ylabel="Nombre d'images"
    )

    # Matrice de dispersion 2D
    n_features = features_normalized.shape[1]
    fig, axes = plt.subplots(n_features, n_features, figsize=(15, 15))
    fig.canvas.manager.set_window_title('Matrice de dispersion 2D (Pairplot)')

    labels_uniques = np.unique(images.labels)
    couleurs = ['#FFA500', '#800080', '#808080']

    for i in range(n_features):
        for j in range(n_features):
            ax = axes[i, j]
            for k, label in enumerate(labels_uniques):
                indices = np.where(images.labels == label)[0]
                if i == j:
                    ax.hist(features_normalized[indices, i], bins=20, color=couleurs[k], alpha=0.5, density=True)
                else:
                    ax.scatter(features_normalized[indices, j], features_normalized[indices, i], 
                               c=couleurs[k], label=label if (i==0 and j==1) else "", 
                               alpha=0.4, s=5, edgecolors='none')
            
            if i == n_features - 1:
                ax.set_xlabel(feature_names[j], fontsize=8)
            else:
                ax.set_xticks([])
                
            if j == 0:
                ax.set_ylabel(feature_names[i], fontsize=8)
            else:
                ax.set_yticks([])

    fig.legend(loc='upper right', bbox_to_anchor=(0.95, 0.95))
    plt.tight_layout()

    # =========================================================================
    # 3. PRÉTRAITEMENT : PCA
    # =========================================================================
    covariance_matrix = np.cov(features_normalized, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]
    
    decorrelated_data = analysis.project_onto_new_basis(features_normalized, eigenvectors)
    representation_pca = dataset.Representation(data=decorrelated_data, labels=images.labels)
    
    viz.plot_features_distribution(
        representation_pca, n_bins=32,
        title="Composantes Principales (PCA)",
        features_names=["PC1", "PC2", "PC3", "PC4", "PC5", "PC6", "PC7", "PC8"],
        xlabel="Valeur projetée", ylabel="Nombre d'images"
    )
                                  
    plt.show() # Affiche les 3 graphiques générés jusqu'ici

    # =========================================================================
    # 4. PRÉPARATION COMMUNE POUR LES CLASSIFICATEURS
    # =========================================================================
    encodeur = LabelEncoder()
    y_encode = encodeur.fit_transform(images.labels) # Format Entier (ex: 0, 1, 2)
    y_one_hot = keras.utils.to_categorical(y_encode) # Format One-Hot (ex: [1, 0, 0])

    # On utilise les 7 premières composantes principales
    X_pca_final = decorrelated_data[:, :7]

    # Séparation commune pour comparer équitablement
    X_train, X_val, y_train_hot, y_val_hot, y_train_int, y_val_int = train_test_split(
        X_pca_final, y_one_hot, y_encode, test_size=0.3, random_state=42
    )
    noms_classes = encodeur.classes_

    # =========================================================================
    # CLASSIFICATEUR 1 : RÉSEAU DE NEURONES
    # =========================================================================
    if run_nn:
        print("\n" + "="*50)
        print("--- Entraînement du Réseau de Neurones ---")
        print("="*50)
        
        modele = keras.models.Sequential([
            keras.layers.Input(shape=(X_train.shape[1],)),
            keras.layers.Dense(units=32, activation='relu'),
            keras.layers.Dense(units=16, activation='relu'),
            keras.layers.Dense(units=8, activation='relu'),
            keras.layers.Dense(units=y_one_hot.shape[1], activation='linear')
        ])

        descente_gradient = keras.optimizers.SGD(learning_rate=0.08, momentum=0.01)

        modele.compile(
            optimizer=descente_gradient,
            loss='mean_squared_error',
            metrics=['accuracy']
        )

        historique = modele.fit(
            X_train, y_train_hot,
            epochs=750,
            batch_size=len(X_train),
            validation_data=(X_val, y_val_hot),
            verbose=0 # Met à 1 si tu veux voir la progression des epochs
        )
        
        # Sauvegarde sécuritaire du modèle
        save_dir = pathlib.Path(__file__).parent / "saves"
        save_dir.mkdir(exist_ok=True)
        model_save_path = save_dir / "NN_class_prob.keras"
        modele.save(model_save_path)

        viz.plot_metric_history(historique)

        predictions_prob = modele.predict(X_val)
        predictions_classes_nn = np.argmax(predictions_prob, axis=-1)

        error_rate_nn, indexes_errors_nn = analysis.compute_error_rate(y_val_int, predictions_classes_nn)
        print(f"\n[Réseau de Neurones] {len(indexes_errors_nn)} erreurs sur {len(y_val_int)} échantillons ({error_rate_nn * 100:.2f}%).")

        viz.show_confusion_matrix(y_val_int, predictions_classes_nn, noms_classes, plot=True)
        plt.show()

    # =========================================================================
    # CLASSIFICATEUR 2 : BAYÉSIEN (MODÈLE GAUSSIEN)
    # =========================================================================
    if run_bayes:
        print("\n" + "="*50)
        print("--- Entraînement du Classificateur Bayésien ---")
        print("="*50)

        classes = np.unique(y_train_int)
        modeles_gaussiens = {}
        priors = {}

        # 1. Apprentissage (Calcul des priors, moyennes et covariances)
        for c in classes:
            X_c = X_train[y_train_int == c]
            priors[c] = len(X_c) / len(X_train)
            mean_c = np.mean(X_c, axis=0)
            cov_c = np.cov(X_c, rowvar=False) + np.eye(X_train.shape[1]) * 1e-6 
            modeles_gaussiens[c] = multivariate_normal(mean=mean_c, cov=cov_c)

        # 2. Matrice des coûts (0 sur la diagonale, 1 ailleurs)
        # Tu pourras ajuster ça plus tard selon l'énoncé si une erreur est plus grave qu'une autre.
        matrice_couts = np.ones((len(classes), len(classes))) - np.eye(len(classes))

        # 3. Prédiction (Minimisation du risque de Bayes)
        predictions_bayes = []
        for x in X_val:
            # Calcul des probabilités pour chaque classe
            vraisemblances = np.array([modeles_gaussiens[c].pdf(x) * priors[c] for c in classes])
            
            # Application des coûts : Risque = Matrice * Vraisemblances
            risques = np.dot(matrice_couts, vraisemblances) 
            
            # Règle de décision : Choisir l'action avec le risque minimal
            predictions_bayes.append(np.argmin(risques))

        predictions_bayes = np.array(predictions_bayes)

        # 4. Évaluation
        error_rate_bayes, indexes_errors_bayes = analysis.compute_error_rate(y_val_int, predictions_bayes)
        print(f"\n[Classificateur Bayésien] {len(indexes_errors_bayes)} erreurs sur {len(y_val_int)} échantillons ({error_rate_bayes * 100:.2f}%).")

        viz.show_confusion_matrix(y_val_int, predictions_bayes, noms_classes, plot=True)
        plt.show()


# =========================================================================
# POINT D'ENTRÉE PRINCIPAL
# =========================================================================
if __name__ == "__main__":
    # Tu peux choisir d'exécuter l'un, l'autre, ou les deux!
    problematique(run_nn=True, run_bayes=True)