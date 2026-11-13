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

    analyze_house_age(df)
    geographic_analysis(df)
    population_analysis(df)
    price_categories(df)
    compare_features(df)
    
    print("\n=== EDA ABGESCHLOSSEN ===")
    print("Grundlegende Analyse des California Housing Datensatzes beendet.")
    print("Die Lösungen zu den Übungen wurden ebenfalls ausgeführt.")

# =============================================================================
# ÜBUNGEN
# =============================================================================
# Erweitern Sie das Skript um die folgenden 5 Funktionen.
# Implementieren Sie jede Funktion einzeln und testen Sie sie.
# Rufen Sie die neuen Funktionen am Ende der main() Funktion auf.
# Die Musterlösungen stehen jeweils direkt unter der Aufgabenbeschreibung.

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

def analyze_house_age(df):
    """Untersucht das regionale Median-Gebäudealter."""
    print("\n=== ALTERSANALYSE ===")
    print(f"Mittelwert der regionalen Median-Gebäudealter: {df['HouseAge'].mean():.2f} Jahre")
    print(f"Höchster gespeicherter Wert: {df['HouseAge'].max():.0f} Jahre (bei 52 gekappt)")
    print(f"Korrelation mit dem Hauswert: {df.corr()['target']['HouseAge']:.3f}")

    plt.figure(figsize=(8, 5))
    plt.hist(df['HouseAge'], bins=30)
    plt.title('Verteilung des regionalen Median-Gebäudealters')
    plt.xlabel('Median-Gebäudealter (Jahre)')
    plt.ylabel('Anzahl Regionen')
    plt.tight_layout()
    plt.show()

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

def geographic_analysis(df):
    """Zeigt die geografische Verteilung und die äußersten Breitengrade."""
    print("\n=== GEOGRAFISCHE VERTEILUNG ===")
    print("Eine nördlichste Region:")
    print(df.loc[df['Latitude'].idxmax(), ['Longitude', 'Latitude', 'target']])
    print("\nEine südlichste Region:")
    print(df.loc[df['Latitude'].idxmin(), ['Longitude', 'Latitude', 'target']])

    plt.figure(figsize=(8, 6))
    plt.scatter(df['Longitude'], df['Latitude'], c=df['target'],
                cmap='RdBu_r', s=10, alpha=0.5)
    plt.colorbar(label='Median-Hauswert (100.000 USD)')
    plt.title('Geografische Verteilung der Hauswerte')
    plt.xlabel('Längengrad')
    plt.ylabel('Breitengrad')
    plt.tight_layout()
    plt.show()

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

def population_analysis(df):
    """Untersucht die Bevölkerungszahl der Regionen."""
    print("\n=== BEVÖLKERUNGSZAHL ===")
    print(f"Korrelation mit dem Hauswert: {df.corr()['target']['Population']:.3f}")
    print("\nDie fünf bevölkerungsreichsten Gebiete:")
    print(df.nlargest(5, 'Population')[['Population', 'Longitude', 'Latitude', 'target']])

    plt.figure(figsize=(8, 5))
    plt.hist(df['Population'], bins=50)
    plt.title('Verteilung der Bevölkerungszahl')
    plt.xlabel('Bevölkerungszahl pro Region')
    plt.ylabel('Anzahl Regionen')
    plt.tight_layout()
    plt.show()

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

def price_categories(df):
    """Teilt Hauswerte in drei annähernd gleich große Gruppen ein."""
    print("\n=== PREISKATEGORIEN ===")
    categories = pd.qcut(df['target'], q=3, labels=['günstig', 'mittel', 'teuer'])
    counts = categories.value_counts(sort=False)
    print(counts)

    plt.figure(figsize=(8, 5))
    plt.bar(counts.index, counts.values)
    plt.title('Regionen nach Preiskategorie (Quantile)')
    plt.xlabel('Preiskategorie')
    plt.ylabel('Anzahl Regionen')
    plt.tight_layout()
    plt.show()

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

def compare_features(df):
    """Vergleicht die zwei am stärksten mit dem Hauswert korrelierten Merkmale."""
    print("\n=== FEATURE-VERGLEICH ===")
    correlations = df.corr()['target'].drop('target')
    strongest = correlations.abs().nlargest(2)
    feature1, feature2 = strongest.index
    print("Auswahl nach dem Betrag der Korrelation:")
    print(correlations.loc[strongest.index])

    plt.figure(figsize=(8, 6))
    plt.scatter(df[feature1], df[feature2], c=df['target'],
                cmap='viridis', s=10, alpha=0.5)
    plt.colorbar(label='Median-Hauswert (100.000 USD)')
    plt.title('Die zwei am stärksten korrelierten Merkmale')
    plt.xlabel(feature1)
    plt.ylabel(feature2)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()