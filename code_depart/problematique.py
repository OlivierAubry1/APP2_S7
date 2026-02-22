import os
import itertools

import matplotlib.pyplot as plt
import numpy as np
import skimage
import pathlib

# Must be call before any other TensorFlow/Keras import
# Suppress oneDNN custom operations info
# Suppress INFO and WARNING messages from TF (0=all, 1=no INFO, 2=no INFO/WARN, 3=no error)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import keras

import helpers.analysis as analysis
import helpers.dataset as dataset
import helpers.viz as viz


def problematique():
    img_path = pathlib.Path(__file__).parent / "data/image_dataset/"
    images = dataset.ImageDataset(img_path)

    # TODO Problématique: Générez une représentation des images appropriée
    # pour la classification comme dans le laboratoire 1.
    # -------------------------------------------------------------------------

    features = np.zeros((len(images), 6))
    
    for i, (image, label) in enumerate(images):
        hsv_image = skimage.color.rgb2hsv(image / 255.0) #transformer img en hsv
        height = image.shape[0]
        top_part = image[:height//3, :, :]  #couper le haut pour mesurer le ciel
        
        #-----------features------------------

        feat_saturation = np.mean(hsv_image[:, :, 1])   #moy du canal saturation s dans hsv
        feat_hue = np.mean(hsv_image[:, :, 0])          #moy du canal hue h dans 
        #important: écart type de luminosité (V) dans hsv. on mesure à quelle point les pixels sont constants dans leur luminosité.
        # Ex: ciel bleu a une texture basse(en théorie), pcq tous les pixels on un V (luminosité similaires.) 
        # Ex: Forêt avec des ombres et des feuilles texture hautes pcq la luminosité des pixels alterne bcp plus 
        feat_texture = np.std(hsv_image[:, :, 2])       
        feat_top_blue = np.mean(top_part[:, :, 2])  #mesure la moyenne des pixels bleu dans le haut de l'image
        feat_green = np.mean(image[:, :, 1])        #vert global
        feat_red = np.mean(image[:, :, 0])          #rouge global
        
        features[i] = [feat_saturation, feat_hue, feat_texture, feat_top_blue, feat_green, feat_red]

    features_mean = np.mean(features, axis=0)
    features_std = np.std(features, axis=0) + 1e-8
    features_normalized = (features - features_mean) / features_std
    
    # -------------------------------------------------------------------------

    # TODO: Problématique: Visualisez cette représentation
    # -------------------------------------------------------------------------
    representation_raw = dataset.Representation(data=features_normalized, labels=images.labels)
    
    viz.plot_features_distribution(representation_raw, n_bins=32,
                                  title="Distribution des features normalisées",
                                  features_names=["Saturation", "Teinte", "Texture", "Bleu_Haut", "Vert_Global", "Rouge_Global"],
                                  xlabel="Valeur", ylabel="Nombre d'images")

    feature_names = ["Saturation", "Teinte", "Texture", "Bleu_Haut", "Vert_Global", "Rouge_Global"]
    
    n_features = features_normalized.shape[1]
    fig, axes = plt.subplots(n_features, n_features, figsize=(15, 15))
    fig.canvas.manager.set_window_title('Matrice de dispersion 2D (Pairplot)')

    labels_uniques = np.unique(images.labels)
    couleurs = ['#FFA500', '#800080', '#808080']

    for i in range(n_features):
        for j in range(n_features):
            ax = axes[i, j]
            
            if i == j:
                for k, label in enumerate(labels_uniques):
                    indices = np.where(images.labels == label)[0]
                    ax.hist(features_normalized[indices, i], bins=20, color=couleurs[k], alpha=0.5, density=True)
            else:
                for k, label in enumerate(labels_uniques):
                    indices = np.where(images.labels == label)[0]
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

    covariance_matrix = np.cov(features_normalized, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    decorrelated_data = analysis.project_onto_new_basis(features_normalized, eigenvectors)
    representation_pca = dataset.Representation(data=decorrelated_data, labels=images.labels)
    
    viz.plot_features_distribution(representation_pca, n_bins=32,
                                  title="Composantes Principales (PCA)",
                                  features_names=["PC1", "PC2", "PC3", "PC4", "PC5", "PC6"],
                                  xlabel="Valeur projetée", ylabel="Nombre d'images")
                                  
    plt.show()
    # -------------------------------------------------------------------------

    # TODO: Problématique: Comparez différents classificateurs sur cette
    # représentation, comme dans le laboratoire 2 et 3.
    # -------------------------------------------------------------------------
    # ---------------NN--------------------------------------------------------
    encodeur = LabelEncoder()
    y_encode = encodeur.fit_transform(images.labels)
    y_one_hot = keras.utils.to_categorical(y_encode)

    X_pca_final = decorrelated_data[:, :6]

    X_train, X_val, y_train, y_val = train_test_split(
        X_pca_final, y_one_hot, test_size=0.3, random_state=42
    )

    modele = keras.models.Sequential()
    modele.add(keras.layers.Dense(units=32, activation='relu', input_shape=(X_train.shape[1],)))
    modele.add(keras.layers.Dense(units=16, activation='relu'))
    modele.add(keras.layers.Dense(units=3, activation='softmax'))

    descente_gradient = keras.optimizers.SGD(learning_rate=0.1, momentum = 0.9)

    modele.compile(
        optimizer=descente_gradient,
        loss='mean_squared_error',
        metrics=['accuracy']
    )

    historique = modele.fit(
        X_train, y_train,
        epochs=750,
        batch_size=len(X_train),
        validation_data=(X_val, y_val),
        verbose=1
    )
    model_save_path = pathlib.Path(__file__).parent / "saves" / "NN_class_prob.keras"
    modele.save(model_save_path)

    viz.plot_metric_history(historique)

    loaded_model = keras.models.load_model(model_save_path)


    predictions_prob = loaded_model.predict(X_val)
    predictions_classes = np.argmax(predictions_prob, axis=-1)
    y_val_classes = np.argmax(y_val, axis=-1)

    noms_classes = encodeur.classes_

    error_rate, indexes_errors = analysis.compute_error_rate(y_val_classes, predictions_classes)
    print(f"\n\n{len(indexes_errors)} erreurs de classification sur {len(y_val)} échantillons de validation ({error_rate * 100:.2f}%).")

    viz.show_confusion_matrix(y_val_classes, predictions_classes, noms_classes, plot=True)
    plt.show()
    # -------------------------------------------------------------------------


if __name__ == "__main__":
    problematique()
