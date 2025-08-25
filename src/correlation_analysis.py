"""
Script per l'analisi di correlazione.
Genera una heatmap di correlazione per tutte le variabili numeriche nel dataset.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def plot_correlation_heatmap():
    """
    Carica i dati, calcola la matrice di correlazione e la visualizza come heatmap.
    """
    print("Caricamento del dataset 'mxmh_final.csv'...")
    try:
        df = pd.read_csv('data/mxmh_final.csv')
    except FileNotFoundError:
        print("Errore: 'data/mxmh_final.csv' non trovato. Assicurati che il file esista.")
        return

    print("Selezione delle colonne numeriche per l'analisi di correlazione...")
    numeric_df = df.select_dtypes(include=np.number)

    # Rimuoviamo colonne che non sono feature dirette o che sono ridondanti
    # 'Cluster' è un output del modello, non una feature di input originale.
    if 'Cluster' in numeric_df.columns:
        numeric_df = numeric_df.drop(columns=['Cluster'])
    
    print(f"Calcolo della matrice di correlazione su {len(numeric_df.columns)} variabili.")
    
    # Calcolo della matrice di correlazione
    corr_matrix = numeric_df.corr()

    # Creazione della heatmap
    # Aumento la dimensione per rendere i numeri più leggibili
    plt.figure(figsize=(24, 20))
    # Aggiungo annot=True per mostrare i valori numerici e riduco la dimensione del font
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", annot_kws={"size": 8})
    plt.title('Heatmap di Correlazione tra le Variabili Numeriche del Dataset', fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout() # Aggiusta il layout per evitare sovrapposizioni

    # Salva il grafico
    output_path = 'correlation_heatmap.png'
    plt.savefig(output_path)
    print(f"\nGrafico di correlazione salvato come '{output_path}'")
    
    # Mostra il grafico
    plt.show()

if __name__ == "__main__":
    plot_correlation_heatmap()
