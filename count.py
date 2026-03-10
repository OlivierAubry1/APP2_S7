from pathlib import Path
from collections import Counter

def count_dataset_classes(folder_path="."):
    path = Path(folder_path)
    class_counts = Counter()
    
    # Extensions d'images courantes pour filtrer
    extensions = {'.jpg', '.jpeg', '.png'}
    
    for file in path.iterdir():
        if file.is_file() and file.suffix.lower() in extensions:
            # Extrait la classe avant le premier '_' 
            class_name = file.name.split('_')[0]
            class_counts[class_name] += 1
            
    # Affichage des résultats
    print("=== Décompte par classe ===")
    for class_name, count in sorted(class_counts.items()):
        print(f"- {class_name.capitalize()} : {count} images")
        
    print("-" * 25)
    print(f"Total d'images : {sum(class_counts.values())}")

if __name__ == "__main__":
    # Remplace "." par le chemin de ton dossier si le script est ailleurs
    count_dataset_classes("code_depart\data\image_dataset")