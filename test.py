#!/usr/bin/env python3
print("Inizio test...")

try:
    import pandas as pd
    print("✓ Pandas importato")
    
    import matplotlib.pyplot as plt
    print("✓ Matplotlib importato")
    
    import seaborn as sns
    print("✓ Seaborn importato")
    
    # Test caricamento dataset
    df = pd.read_csv("data/mxmh_clean.csv")
    print(f"✓ Dataset caricato: {df.shape}")
    
    # Test creazione grafico semplice
    plt.figure(figsize=(8, 6))
    plt.hist(df["Age"].dropna(), bins=20)
    plt.title("Test Age Distribution")
    plt.savefig("test_plot.png")
    plt.close()
    print("✓ Grafico test salvato come test_plot.png")
    
except Exception as e:
    print(f"❌ Errore: {e}")
    import traceback
    traceback.print_exc()

print("Test completato!")
