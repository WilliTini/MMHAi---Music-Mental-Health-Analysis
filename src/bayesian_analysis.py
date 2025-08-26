import pandas as pd
import bnlearn as bn
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
import numpy as np
from matplotlib.patches import Rectangle
import os

def ensure_images_directory():
    """Crea la cartella images se non esiste."""
    if not os.path.exists('images'):
        os.makedirs('images')
        print("Cartella 'images/' creata.")

def preprocess_for_bayesian(df):
    """Prepara il DataFrame per l'analisi bayesiana."""
    print("1. Preprocessing dei dati per la rete bayesiana...")
    
    # Selezioniamo un sottoinsieme di colonne per rendere il grafo leggibile
    # CORREZIONE: Usare i nomi delle colonne con spazi, come nel file CSV
    cols_to_keep = [
        'Age', 'Hours per day', 'Fav genre', 
        'Music effects', 'Anxiety', 'Depression', 'Insomnia', 'OCD'
    ]
    df_bayes = df[cols_to_keep].copy()

    # Discretizzazione delle variabili numeriche
    # Age
    df_bayes['Age_group'] = pd.cut(df_bayes['Age'], 
                                   bins=[0, 18, 30, 50, 100], 
                                   labels=['Teen', 'Young_Adult', 'Adult', 'Senior'])
    
    # Hours_per_day
    # CORREZIONE: Usare il nome corretto della colonna 'Hours per day'
    df_bayes['Listening_hours'] = pd.cut(df_bayes['Hours per day'], 
                                         bins=[-1, 1, 4, 8, 25], 
                                         labels=['Low', 'Medium', 'High', 'Very_High'])
    
    # Mental health scores (0-3: Low, 4-7: Medium, 8-10: High)
    for col in ['Anxiety', 'Depression', 'Insomnia', 'OCD']:
        df_bayes[f'{col}_level'] = pd.cut(df_bayes[col], 
                                          bins=[-1, 3, 7, 10], 
                                          labels=['Low', 'Medium', 'High'])

    # Rinominiamo Music_effects per chiarezza
    # CORREZIONE: Usare il nome corretto della colonna 'Music effects'
    df_bayes.rename(columns={'Music effects': 'Music_effect_on_mood'}, inplace=True)

    # Selezioniamo le colonne finali (solo quelle discrete)
    # CORREZIONE: Usare il nome corretto della colonna 'Fav genre'
    final_cols = [
        'Age_group', 'Listening_hours', 'Fav genre', 
        'Music_effect_on_mood', 'Anxiety_level', 'Depression_level', 
        'Insomnia_level', 'OCD_level'
    ]
    df_final = df_bayes[final_cols].dropna()
    
    print("Preprocessing completato. Colonne finali:", final_cols)
    return df_final

def learn_and_plot_structure(df):
    """Impara la struttura della rete dai dati e la visualizza."""
    print("\n2. Apprendimento della struttura del grafo (potrebbe richiedere un po' di tempo)...")
    
    # Usiamo un algoritmo di structure learning (Hill Climb Search)
    model_results = bn.structure_learning.fit(df, methodtype='hc', scoretype='bic')
    
    print("Apprendimento della struttura completato.")
    
    # CORREZIONE: Estrarre il DAG e creare un BayesianNetwork vero
    learned_dag = model_results['model']
    edges = learned_dag.edges()
    
    print("Archi trovati nel grafo:")
    print(edges)
    
    # CORREZIONE: Creare un BayesianNetwork di pgmpy usando gli edges appena scoperti
    bayesian_model = BayesianNetwork(edges)

    # Visualizzazione del grafo
    print("\n3. Visualizzazione del grafo...")
    plt.figure(figsize=(15, 10))
    G = nx.DiGraph()
    G.add_edges_from(edges)
    
    pos = nx.kamada_kawai_layout(G) # Layout più gradevole
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color='skyblue', 
            font_size=10, font_weight='bold', arrows=True, arrowsize=20)
    plt.title('Struttura della Rete Bayesiana (appresa dai dati)', size=18)
    ensure_images_directory()
    plt.savefig("images/bayesian_network_structure.png", dpi=300, bbox_inches='tight')
    print("Grafo salvato come 'images/bayesian_network_structure.png'")
    plt.show()
    
    return bayesian_model

def plot_probability_distribution(prob_result, title, save_name):
    """Crea un grafico a barre per una distribuzione di probabilità."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    if hasattr(prob_result, 'values'):
        # Distribuzione marginale (una sola variabile)
        variable_name = prob_result.variables[0]
        states = prob_result.state_names[variable_name]
        probabilities = prob_result.values
        
        bars = ax.bar(states, probabilities, color='lightblue', edgecolor='navy', alpha=0.7)
        ax.set_xlabel(variable_name, fontsize=12)
        ax.set_ylabel('Probabilità', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Aggiungi i valori sopra le barre
        for bar, prob in zip(bars, probabilities):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{prob:.3f}', ha='center', va='bottom', fontsize=10)
        
        ax.set_ylim(0, max(probabilities) * 1.15)
        
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'images/{save_name}.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_probability_table(prob_result, title):
    """Crea una tabella formattata per le probabilità."""
    if hasattr(prob_result, 'values'):
        variable_name = prob_result.variables[0]
        states = prob_result.state_names[variable_name]
        probabilities = prob_result.values
        
        df_table = pd.DataFrame({
            'Stato': states,
            'Probabilità': probabilities,
            'Percentuale': [f'{p*100:.2f}%' for p in probabilities]
        })
        
        return df_table
    return None

def plot_conditional_probability(prob_result, evidence_var, evidence_value, title, save_name):
    """Crea un grafico per probabilità condizionali multiple."""
    if len(prob_result.variables) == 1:
        # Caso semplice: una sola variabile
        plot_probability_distribution(prob_result, title, save_name)
        return
    
    # Caso complesso: multiple variabili
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    variables = prob_result.variables
    
    for i, var in enumerate(variables[:4]):  # Massimo 4 subplot
        if i >= len(axes):
            break
            
        ax = axes[i]
        
        # Ottieni le probabilità marginali per questa variabile
        states = prob_result.state_names[var]
        
        # Calcola le probabilità marginali
        var_index = variables.index(var)
        other_indices = [j for j, v in enumerate(variables) if v != var]
        
        marginal_probs = np.sum(prob_result.values, axis=tuple(other_indices))
        
        bars = ax.bar(states, marginal_probs, color='lightcoral', edgecolor='darkred', alpha=0.7)
        ax.set_xlabel(var, fontsize=10)
        ax.set_ylabel('Probabilità', fontsize=10)
        ax.set_title(f'{var} | {evidence_var}={evidence_value}', fontsize=11, fontweight='bold')
        
        # Aggiungi i valori sopra le barre
        for bar, prob in zip(bars, marginal_probs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                   f'{prob:.3f}', ha='center', va='bottom', fontsize=8)
        
        ax.tick_params(axis='x', rotation=45)
    
    # Nascondi subplot vuoti
    for j in range(len(variables), len(axes)):
        axes[j].set_visible(False)
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'images/{save_name}.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_summary_table(mental_health_vars, other_vars, model_info):
    """Crea una tabella riassuntiva del modello."""
    
    # Crea una figura per la tabella
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Dati per la tabella
    table_data = [
        ['Categoria', 'Variabili', 'Descrizione'],
        ['Salute Mentale', ', '.join(mental_health_vars), 'Variabili target di salute mentale'],
        ['Fattori Esterni', ', '.join(other_vars), 'Variabili di influenza esterna'],
        ['Totale Variabili', str(len(mental_health_vars) + len(other_vars)), 'Numero totale di nodi nel grafo'],
        ['Archi nel Grafo', str(model_info.get('num_edges', 'N/A')), 'Connessioni causali apprese'],
        ['Algoritmo', 'Hill Climb Search + BIC', 'Metodo di structure learning'],
        ['Inferenza', 'Variable Elimination', 'Metodo di inferenza probabilistica']
    ]
    
    # Crea la tabella
    table = ax.table(cellText=table_data[1:], colLabels=table_data[0], 
                    cellLoc='left', loc='center', bbox=[0, 0, 1, 1])
    
    # Stilizza la tabella
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Colora l'header
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Colora le righe alternativamente
    for i in range(1, len(table_data)):
        for j in range(len(table_data[0])):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#F0F0F0')
    
    plt.title('Riassunto del Modello Bayesiano', fontsize=16, fontweight='bold', pad=20)
    plt.savefig('images/bayesian_model_summary.png', dpi=300, bbox_inches='tight')
    plt.show()
    
def fit_model_and_run_inference(pgmpy_model, df):
    """Addestra il modello e esegue query di inferenza con visualizzazioni."""
    print("\n4. Addestramento del modello (apprendimento dei parametri)...")
    
    # Usiamo un Bayesian Estimator per calcolare le CPT (Tabelle di Probabilità Condizionata)
    pgmpy_model.fit(df, estimator=BayesianEstimator, prior_type="BDeu")
    print("Addestramento completato.")

    # Creazione dell'oggetto per l'inferenza
    inference = VariableElimination(pgmpy_model)

    print("\n5. Esecuzione di query di inferenza e creazione visualizzazioni...")
    
    # Otteniamo tutte le variabili disponibili nel modello
    available_nodes = list(pgmpy_model.nodes())
    print(f"Variabili nel modello: {available_nodes}")
    
    if len(available_nodes) == 0:
        print("ATTENZIONE: Il modello non ha variabili.")
        return
    
    # Identifichiamo le variabili di salute mentale
    mental_health_vars = [var for var in available_nodes if '_level' in var]
    other_vars = [var for var in available_nodes if '_level' not in var]
    
    print(f"Variabili di salute mentale: {mental_health_vars}")
    print(f"Altre variabili: {other_vars}")
    
    # Crea tabella riassuntiva del modello
    model_info = {'num_edges': len(pgmpy_model.edges())}
    summary_data = create_summary_table(mental_health_vars, other_vars, model_info)
    
    # Query 1: Distribuzione marginale di tutte le variabili di salute mentale
    if mental_health_vars:
        print("\n--- Creazione grafici per distribuzioni marginali ---")
        
        # Crea una figura con subplot per tutte le variabili di salute mentale
        n_vars = len(mental_health_vars)
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        tables_data = []
        
        for i, var in enumerate(mental_health_vars[:4]):  # Massimo 4
            try:
                prob = inference.query(variables=[var])
                
                # Crea il subplot
                if i < len(axes):
                    ax = axes[i]
                    states = prob.state_names[var]
                    probabilities = prob.values
                    
                    bars = ax.bar(states, probabilities, color='lightblue', edgecolor='navy', alpha=0.7)
                    ax.set_xlabel(var, fontsize=10)
                    ax.set_ylabel('Probabilità', fontsize=10)
                    ax.set_title(f'Distribuzione di {var}', fontsize=11, fontweight='bold')
                    
                    # Aggiungi valori sopra le barre
                    for bar, prob_val in zip(bars, probabilities):
                        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                               f'{prob_val:.3f}', ha='center', va='bottom', fontsize=9)
                    
                    ax.tick_params(axis='x', rotation=45)
                
                # Crea tabella dati
                table_data = create_probability_table(prob, f'Distribuzione {var}')
                if table_data is not None:
                    tables_data.append((var, table_data))
                    
            except Exception as e:
                print(f"Errore per {var}: {e}")
        
        # Nascondi subplot vuoti
        for j in range(n_vars, len(axes)):
            axes[j].set_visible(False)
        
        plt.suptitle('Distribuzioni Marginali - Variabili di Salute Mentale', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('images/marginal_distributions.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Mostra le tabelle
        print("\nTabelle delle probabilità marginali:")
        for var_name, table in tables_data:
            print(f"\n{var_name}:")
            print(table.to_string(index=False))
    
    # Query 2: Inferenza condizionata
    if len(mental_health_vars) >= 2:
        print("\n--- Creazione grafici per inferenza condizionata ---")
        evidence_var = mental_health_vars[0]
        query_vars = mental_health_vars[1:2]  # Prendiamo solo una variabile per semplicità
        
        # Crea grafici per diversi livelli di evidenza
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for idx, level in enumerate(['Low', 'Medium', 'High']):
            try:
                prob = inference.query(
                    variables=query_vars,
                    evidence={evidence_var: level}
                )
                
                ax = axes[idx]
                var_name = query_vars[0]
                states = prob.state_names[var_name]
                probabilities = prob.values
                
                bars = ax.bar(states, probabilities, color='lightcoral', edgecolor='darkred', alpha=0.7)
                ax.set_xlabel(var_name, fontsize=10)
                ax.set_ylabel('Probabilità', fontsize=10)
                ax.set_title(f'{var_name} | {evidence_var}={level}', fontsize=11, fontweight='bold')
                
                # Aggiungi valori sopra le barre
                for bar, prob_val in zip(bars, probabilities):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{prob_val:.3f}', ha='center', va='bottom', fontsize=9)
                
                ax.tick_params(axis='x', rotation=45)
                ax.set_ylim(0, max(probabilities) * 1.15)
                
            except Exception as e:
                print(f"Errore per {evidence_var}={level}: {e}")
                continue
        
        plt.suptitle(f'Inferenza Condizionata: {query_vars[0]} dato {evidence_var}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('images/conditional_inference.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # Query 3: Analisi di correlazione tra variabili di salute mentale
    if len(mental_health_vars) >= 2:
        print("\n--- Creazione matrice di correlazione bayesiana ---")
        
        # Calcoliamo le probabilità condizionali tra coppie di variabili
        correlation_matrix = np.zeros((len(mental_health_vars), len(mental_health_vars)))
        
        for i, var1 in enumerate(mental_health_vars):
            for j, var2 in enumerate(mental_health_vars):
                if i == j:
                    correlation_matrix[i, j] = 1.0
                else:
                    try:
                        # Calcoliamo P(var2=High | var1=High)
                        prob = inference.query(variables=[var2], evidence={var1: 'High'})
                        high_prob = prob.values[list(prob.state_names[var2]).index('High')]
                        correlation_matrix[i, j] = high_prob
                    except:
                        correlation_matrix[i, j] = 0.0
        
        # Crea heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, 
                   xticklabels=mental_health_vars, 
                   yticklabels=mental_health_vars,
                   annot=True, fmt='.3f', cmap='RdYlBu_r',
                   square=True, linewidths=0.5)
        plt.title('Matrice di Probabilità Condizionali\nP(Variabile Colonna = High | Variabile Riga = High)', 
                 fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('images/bayesian_correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Crea tabella della matrice di correlazione
        correlation_df = pd.DataFrame(correlation_matrix, 
                                    index=mental_health_vars, 
                                    columns=mental_health_vars)
        print("\nMatrice di probabilità condizionali:")
        print(correlation_df.round(3).to_string())
    
    # Query 4: Influenza delle altre variabili
    if other_vars and mental_health_vars:
        print(f"\n--- Analisi influenza di {other_vars[0]} ---")
        influence_var = other_vars[0]
        
        # Crea un grafico che mostra l'influenza su tutte le variabili di salute mentale
        n_mental = len(mental_health_vars)
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, target_var in enumerate(mental_health_vars[:4]):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            try:
                # Ottieni valori possibili per la variabile di influenza
                marginal = inference.query(variables=[influence_var])
                possible_values = list(marginal.state_names[influence_var])
                
                # Calcola probabilità di 'High' per ogni valore dell'influenza
                high_probs = []
                labels = []
                
                for value in possible_values[:5]:  # Massimo 5 valori
                    try:
                        prob = inference.query(variables=[target_var], evidence={influence_var: value})
                        states = list(prob.state_names[target_var])
                        if 'High' in states:
                            high_prob = prob.values[states.index('High')]
                            high_probs.append(high_prob)
                            labels.append(value)
                    except:
                        continue
                
                if high_probs:
                    bars = ax.bar(labels, high_probs, color='orange', edgecolor='darkorange', alpha=0.7)
                    ax.set_xlabel(influence_var, fontsize=10)
                    ax.set_ylabel(f'P({target_var}=High)', fontsize=10)
                    ax.set_title(f'Influenza su {target_var}', fontsize=11, fontweight='bold')
                    
                    # Aggiungi valori sopra le barre
                    for bar, prob_val in zip(bars, high_probs):
                        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                               f'{prob_val:.3f}', ha='center', va='bottom', fontsize=9)
                    
                    ax.tick_params(axis='x', rotation=45)
                
            except Exception as e:
                print(f"Errore nell'analisi di influenza per {target_var}: {e}")
        
        # Nascondi subplot vuoti
        for j in range(n_mental, len(axes)):
            axes[j].set_visible(False)
        
        plt.suptitle(f'Influenza di {influence_var} sulla Salute Mentale', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('images/influence_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    print("\nTutte le visualizzazioni sono state salvate nella cartella 'images/'")
    print("Le tabelle con i dati numerici sono state mostrate sopra")

def main():
    """Funzione principale per l'analisi con rete bayesiana."""
    print("Inizio analisi con rete bayesiana")
    print("="*50)
    
    # Assicuriamoci che la cartella images esista
    ensure_images_directory()
    
    # Caricamento dati
    try:
        df = pd.read_csv('data/mxmh_final.csv')
        print(f"Dataset caricato: {df.shape[0]} righe.")
    except FileNotFoundError:
        print("Errore: file 'data/mxmh_final.csv' non trovato.")
        return

    # 1. Preprocessing
    df_processed = preprocess_for_bayesian(df)
    
    # 2. Apprendimento e visualizzazione della struttura
    pgmpy_model = learn_and_plot_structure(df_processed)
    
    # 3. Addestramento del modello ed esecuzione inferenze
    fit_model_and_run_inference(pgmpy_model, df_processed)
    
    print("\n"+"="*50)
    print("Analisi bayesiana completata")
    print("Tutti i grafici sono stati salvati nella cartella 'images/'")
    
    # Lista dei file creati
    image_files = [
        'bayesian_network_structure.png',
        'bayesian_model_summary.png', 
        'marginal_distributions.png',
        'conditional_inference.png',
        'bayesian_correlation_matrix.png',
        'influence_analysis.png'
    ]
    
    print("\nFile grafici creati:")
    for img_file in image_files:
        if os.path.exists(f'images/{img_file}'):
            print(f"   - {img_file}")
        else:
            print(f"   - {img_file} (non creato)")

if __name__ == '__main__':
    main()
