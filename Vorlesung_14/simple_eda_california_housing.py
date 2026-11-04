"""
EDA (Exploratory Data Analysis) auf California Housing Dataset

Einfache Datenanalyse mit grundlegenden Funktionen für Studierende.
Zeigt die wichtigsten EDA-Schritte mit einfachen Python-Befehlen.

Eine Zeile ist eine Zensus-Blockgruppe (Zensus 1990). target ist der
Median-Hauswert (100.000 USD), MedInc das Median-Haushaltseinkommen (10.000 USD).
Hauswerte sind bei 5.00001 nach oben gekappt; der tatsächliche höchste
Median-Hauswert ist daher nicht bestimmbar. Beim ersten Laden wird Internet benötigt.

"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

def load_data():
    """Lädt den California Housing Datensatz."""
    print("Lade California Housing Datensatz...")
    housing = fetch_california_housing()
    df = pd.DataFrame(housing.data, columns=housing.feature_names)
    df['target'] = housing.target
    print(f"Datensatz geladen: {df.shape[0]} Zeilen, {df.shape[1]} Spalten")
    return df

def basic_info(df):
    """Zeigt grundlegende Informationen über den Datensatz."""
    print("\n=== GRUNDLEGENDE INFORMATIONEN ===")
    print("Spalten im Datensatz:")
    print(df.columns.tolist())
    
    print(f"\nAnzahl Zeilen: {len(df)}")
    print(f"Anzahl Spalten: {len(df.columns)}")
    
    print("\nDatentypen:")
    print(df.dtypes)

def show_statistics(df):
    """Zeigt deskriptive Statistiken."""
    print("\n=== DESKRIPTIVE STATISTIKEN ===")
    print(df.describe().round(2))

def check_missing_values(df):
    """Prüft auf fehlende Werte."""
    print("\n=== FEHLENDE WERTE ===")
    missing = df.isnull().sum()
    print(missing)
    if missing.sum() == 0:
        print("✓ Keine fehlenden Werte gefunden!")

def simple_plots(df):
    """Erstellt einfache Plots zur Datenvisualisierung."""
    print("\n=== EINFACHE VISUALISIERUNGEN ===")
    
    # 2x2 Subplot erstellen
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Histogramm der Immobilienpreise
    axes[0, 0].hist(df['target'], bins=30, alpha=0.7, color='blue')
    axes[0, 0].set_title('Verteilung der Immobilienpreise')
    axes[0, 0].set_xlabel('Preis (100k USD)')
    axes[0, 0].set_ylabel('Häufigkeit')
    
    # Plot 2: Histogramm des Einkommens
    axes[0, 1].hist(df['MedInc'], bins=30, alpha=0.7, color='green')
    axes[0, 1].set_title('Verteilung des mittleren Einkommens')
    axes[0, 1].set_xlabel('Einkommen (10k USD)')
    axes[0, 1].set_ylabel('Häufigkeit')
    
    # Plot 3: Scatter Plot Einkommen vs. Preis
    axes[1, 0].scatter(df['MedInc'], df['target'], alpha=0.5, s=10)
    axes[1, 0].set_title('Einkommen vs. Immobilienpreise')
    axes[1, 0].set_xlabel('Mittleres Einkommen (10k USD)')
    axes[1, 0].set_ylabel('Immobilienpreis (100k USD)')
    
    # Plot 4: Histogramm der Zimmeranzahl
    axes[1, 1].hist(df['AveRooms'], bins=30, alpha=0.7, color='red')
    axes[1, 1].set_title('Durchschnittliche Zimmeranzahl')
    axes[1, 1].set_xlabel('Anzahl Zimmer')
    axes[1, 1].set_ylabel('Häufigkeit')
    
    plt.tight_layout()
    plt.show()

def find_extremes(df):
    """Findet extreme Werte im Datensatz."""
    print("\n=== EXTREME WERTE ===")
    
    # Teuerste Region
    max_price_idx = df['target'].idxmax()
    print(f"Eine Region mit höchstem gespeichertem Median-Hauswert: ${df.loc[max_price_idx, 'target']*100:.0f}k (gekappt)")
    print(f"  Mittleres Einkommen: ${df.loc[max_price_idx, 'MedInc']*10:.0f}k")
    print(f"  Durchschnittliche Zimmeranzahl: {df.loc[max_price_idx, 'AveRooms']:.1f}")
    
    # Billigste Region
    min_price_idx = df['target'].idxmin()
    print(f"\nEine Region mit niedrigstem Median-Hauswert: ${df.loc[min_price_idx, 'target']*100:.0f}k")
    print(f"  Mittleres Einkommen: ${df.loc[min_price_idx, 'MedInc']*10:.0f}k")
    print(f"  Durchschnittliche Zimmeranzahl: {df.loc[min_price_idx, 'AveRooms']:.1f}")

def simple_correlation(df):
    """Berechnet einfache Korrelationen."""
    print("\n=== KORRELATIONEN MIT IMMOBILIENPREISEN ===")
    correlations = df.corr()['target'].drop('target').sort_values(ascending=False)
    
    print("Stärkste positive Korrelationen:")
    for feature, corr in correlations[correlations > 0].head(3).items():
        print(f"  {feature}: {corr:.3f}")
    
    print("\nStärkste negative Korrelationen:")
    for feature, corr in correlations[correlations < 0].sort_values().head(3).items():
        print(f"  {feature}: {corr:.3f}")

def main():
    """Hauptfunktion - führt die EDA durch."""
    print("=== CALIFORNIA HOUSING EDA ===")
    print("Einfache explorative Datenanalyse\n")
    
    # Schritt 1: Daten laden
    df = load_data()
    
    # Schritt 2: Grundlegende Informationen
    basic_info(df)
    
    # Schritt 3: Deskriptive Statistiken
    show_statistics(df)
    
    # Schritt 4: Fehlende Werte prüfen
    check_missing_values(df)
    
    # Schritt 5: Einfache Visualisierungen
    simple_plots(df)
    
    # Schritt 6: Extreme Werte finden
    find_extremes(df)
    
    # Schritt 7: Korrelationen berechnen
    simple_correlation(df)
    
    print("\n=== EDA ABGESCHLOSSEN ===")
    print("Grundlegende Analyse des California Housing Datensatzes beendet.")
    print("Siehe Aufgaben für Studenten am Ende der Datei.")

# =============================================================================
# ÜBUNGEN
# =============================================================================
# Erweitern Sie das Skript um die folgenden 5 Funktionen.
# Implementieren Sie jede Funktion einzeln und testen Sie sie.
# Rufen Sie die neuen Funktionen am Ende der main() Funktion auf.

# -----------------------------------------------------------------------------
# AUFGABE 1: Altersanalyse
# -----------------------------------------------------------------------------
# Erstellen Sie eine Funktion 'analyze_house_age(df)':
# - Histogramm des HouseAge Features erstellen
# - Mittelwert und Maximum des regionalen Median-Gebäudealters finden
#   (HouseAge ist ein Median pro Region und bei 52 gekappt)
# - Korrelation zwischen Gebäudealter und Preis berechnen
#
# Nützliche Funktionen:
# - plt.hist(df['HouseAge'], bins=30)
# - df['HouseAge'].mean()
# - df['HouseAge'].max()
# - df.corr()['target']['HouseAge']

# -----------------------------------------------------------------------------
# AUFGABE 2: Geografische Verteilung
# -----------------------------------------------------------------------------
# Erstellen Sie eine Funktion 'geographic_analysis(df)':
# - Scatter Plot von Longitude vs. Latitude erstellen
# - Punkte nach Immobilienpreisen einfärben (teuer = rot, billig = blau)
# - Nördlichste und südlichste Region finden
#
# Nützliche Funktionen:
# - plt.scatter(df['Longitude'], df['Latitude'], c=df['target'], cmap='RdBu_r')
# - plt.colorbar()
# - df['Latitude'].max() und df['Latitude'].min()
# - df.loc[df['Latitude'].idxmax()]

# -----------------------------------------------------------------------------
# AUFGABE 3: Bevölkerungszahl
# -----------------------------------------------------------------------------
# Erstellen Sie eine Funktion 'population_analysis(df)':
# - Korrelation zwischen Population und Immobilienpreisen berechnen
# - Histogramm der Population zeigen
# - Die 5 bevölkerungsreichsten Gebiete identifizieren
#
# Nützliche Funktionen:
# - df.corr()['target']['Population']
# - plt.hist(df['Population'], bins=50)
# - df.nlargest(5, 'Population')
# - df.sort_values('Population', ascending=False).head(5)

# -----------------------------------------------------------------------------
# AUFGABE 4: Preiskategorien
# -----------------------------------------------------------------------------
# Erstellen Sie eine Funktion 'price_categories(df)':
# - Regionen in 3 Preiskategorien einteilen: 'günstig', 'mittel', 'teuer'
# - Anzahl Regionen pro Kategorie zählen
# - Balkendiagramm der Kategorien erstellen
#
# Nützliche Funktionen:
# - pd.qcut(df['target'], q=3, labels=['günstig', 'mittel', 'teuer'])
#   für drei annähernd gleich große Gruppen (Quantile)
# - pd.cut(..., bins=3) erzeugt stattdessen drei gleich breite Preisintervalle
# - categories.value_counts()
# - plt.bar(categories.index, categories.values)

# -----------------------------------------------------------------------------
# AUFGABE 5: Feature-Vergleich
# -----------------------------------------------------------------------------
# Erstellen Sie eine Funktion 'compare_features(df)':
# - Die beiden Features mit höchster Korrelation zum Preis nehmen
# - 2D-Scatter Plot dieser Features erstellen
# - Punkte nach Immobilienpreisen einfärben
#
# Nützliche Funktionen:
# - df.corr()['target'].abs().sort_values(ascending=False)
# - correlations.drop('target').head(2) für beste Features
# - plt.scatter(df[feature1], df[feature2], c=df['target'], cmap='viridis')
# - plt.xlabel() und plt.ylabel() für Achsenbeschriftung

if __name__ == "__main__":
    main()