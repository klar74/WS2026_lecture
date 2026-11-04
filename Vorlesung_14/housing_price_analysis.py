#!/usr/bin/env python3
"""
California Housing: Preisverteilung und Standardabweichung

Kurze Analyse der Hauspreise und des mittleren Einkommens in Bins
- Histogramm der Häufigkeiten
- Balkendiagramm der Standardabweichungen pro Bin

Eine Zeile ist eine Zensus-Blockgruppe, kein einzelnes Haus. Die Zielvariable
ist der Median-Hauswert (100.000 USD), MedInc das Median-Haushaltseinkommen
(10.000 USD), Zensus 1990. Beim ersten Laden ist Internetzugriff erforderlich.
Die Auswahl ohne gekappte Werte verändert die Population (Selektionsbias).
Die Standardabweichung innerhalb eines Bins ist durch die Binbreite begrenzt
und misst nicht die Gesamtstreuung. Pandas verwendet ddof=1; bei nur einer
Beobachtung ist die Standardabweichung undefiniert (NaN).
MedInc ist ebenfalls oben gekappt (15.0001); die Streuung 0 im obersten
belegten Bin folgt aus identischen gekappten Werten, nicht gleichen Einkommen.
Leere Einkommens-Bins werden nicht dargestellt.
Quelle: https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

def load_and_clean_data():
    """Lädt und bereinigt die California Housing Daten."""
    print("Bibliotheken geladen!")
    
    # California Housing Daten laden
    housing = fetch_california_housing()
    housing_df = pd.DataFrame(housing.data, columns=housing.feature_names)
    housing_df['target'] = housing.target

    # Teilmengenauswahl: Nur Werte an der gespeicherten Kappungsgrenze entfernen.
    housing_clean = housing_df[housing_df['target'] < housing_df['target'].max()].copy()

    # Konvertiere zu vollständigen USD
    housing_clean['price_usd'] = housing_clean['target'] * 100000

    print(f"Daten geladen: {len(housing_clean)} Regionen")
    print(f"Preisspanne: ${housing_clean['price_usd'].min():,.0f} - ${housing_clean['price_usd'].max():,.0f}")
    
    return housing_clean

def analyze_house_prices(housing_clean):
    """Analysiert die Hauspreise in 50k USD Bins."""
    print("\n" + "="*60)
    print("ANALYSE DER HAUSPREISE")
    print("="*60)
    
    # Erstelle 50k USD Bins
    bin_size = 50000
    min_price = 0
    max_price = 500000

    bins = np.arange(min_price, max_price + bin_size, bin_size)
    housing_clean['price_bin'] = pd.cut(housing_clean['price_usd'], bins=bins, include_lowest=True)

    # Berechne Statistiken pro Bin
    bin_stats = housing_clean.groupby('price_bin', observed=False).agg({
        'price_usd': ['count', 'std']
    }).round(0)

    bin_stats.columns = ['count', 'std']
    bin_stats = bin_stats.reset_index()

    # Erstelle Bin-Labels
    bin_stats['bin_label'] = bin_stats['price_bin'].apply(
        lambda x: f"${x.left/1000:.0f}k-${x.right/1000:.0f}k"
    )

    print(f"Bins erstellt: {len(bin_stats)} Bins")
    print(bin_stats[['bin_label', 'count', 'std']])
    
    return bin_stats

def plot_house_prices(bin_stats):
    """Erstellt Plots für Hauspreise."""
    # Plots erstellen
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Häufigkeit als Histogramm
    ax1.bar(range(len(bin_stats)), bin_stats['count'], 
            alpha=0.7, color='skyblue', edgecolor='navy')
    ax1.set_xlabel('Preisbins')
    ax1.set_ylabel('Anzahl Regionen')
    ax1.set_title('Häufigkeitsverteilung der Hauspreise\n(50k USD Bins)')
    ax1.set_xticks(range(len(bin_stats)))
    ax1.set_xticklabels(bin_stats['bin_label'], rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Standardabweichung als Balkendiagramm
    ax2.bar(range(len(bin_stats)), bin_stats['std'], 
            alpha=0.7, color='lightcoral', edgecolor='darkred')
    ax2.set_xlabel('Preisbins')
    ax2.set_ylabel('Standardabweichung (USD)')
    ax2.set_title('Standardabweichung der Hauspreise\n(50k USD Bins)')
    ax2.set_xticks(range(len(bin_stats)))
    ax2.set_xticklabels(bin_stats['bin_label'], rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Zusammenfassung
    print("\nZUSAMMENFASSUNG HAUSPREISE:")
    print(f"Höchste Häufigkeit: {bin_stats['count'].max():,.0f} Regionen im Bin {bin_stats.loc[bin_stats['count'].idxmax(), 'bin_label']}")
    print(f"Höchste Standardabweichung: ${bin_stats['std'].max():,.0f} im Bin {bin_stats.loc[bin_stats['std'].idxmax(), 'bin_label']}")
    print(f"Niedrigste Standardabweichung: ${bin_stats['std'].min():,.0f} im Bin {bin_stats.loc[bin_stats['std'].idxmin(), 'bin_label']}")

def analyze_income(housing_clean):
    """Analysiert das mittlere Einkommen in 10k USD Bins."""
    print("\n" + "="*60)
    print("ANALYSE DES MITTLEREN EINKOMMENS")
    print("="*60)
    
    # Erstelle Einkommens-Bins (MedInc ist in 10k USD, also 1.0 = 10k USD)
    # Eine Einheit MedInc entspricht 10k USD.
    income_bin_size = 1.0  # entspricht 10k USD
    min_income = 0
    max_income = np.ceil(housing_clean['MedInc'].max() / income_bin_size) * income_bin_size

    income_bins = np.arange(min_income, max_income + income_bin_size, income_bin_size)
    housing_clean['income_bin'] = pd.cut(housing_clean['MedInc'], bins=income_bins, include_lowest=True)

    # Berechne Statistiken pro Einkommens-Bin
    income_bin_stats = housing_clean.groupby('income_bin', observed=False).agg({
        'MedInc': ['count', 'std']
    }).round(3)

    income_bin_stats.columns = ['count', 'std']
    income_bin_stats = income_bin_stats.reset_index()

    # Entferne Bins ohne Daten
    income_bin_stats = income_bin_stats[income_bin_stats['count'] > 0].copy()

    # Erstelle Bin-Labels (konvertiere zu 10k USD)
    income_bin_stats['bin_label'] = income_bin_stats['income_bin'].apply(
        lambda x: f"${x.left*10:.0f}k-${x.right*10:.0f}k"
    )

    print(f"Einkommen-Bins erstellt: {len(income_bin_stats)} Bins")
    print(f"Einkommensspanne: {housing_clean['MedInc'].min():.1f} - {housing_clean['MedInc'].max():.1f} (in 10k USD)")
    print(income_bin_stats[['bin_label', 'count', 'std']])
    
    return income_bin_stats

def plot_income(income_bin_stats):
    """Erstellt Plots für Einkommen."""
    # Plots für Einkommen erstellen
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Häufigkeit als Histogramm
    ax1.bar(range(len(income_bin_stats)), income_bin_stats['count'], 
            alpha=0.7, color='lightgreen', edgecolor='darkgreen')
    ax1.set_xlabel('Einkommensbins')
    ax1.set_ylabel('Anzahl Regionen')
    ax1.set_title('Häufigkeitsverteilung des mittleren Einkommens\n(10k USD Bins)')
    ax1.set_xticks(range(len(income_bin_stats)))
    ax1.set_xticklabels(income_bin_stats['bin_label'], rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Standardabweichung als Balkendiagramm
    ax2.bar(range(len(income_bin_stats)), income_bin_stats['std'], 
            alpha=0.7, color='orange', edgecolor='darkorange')
    ax2.set_xlabel('Einkommensbins')
    ax2.set_ylabel('Standardabweichung (10k USD)')
    ax2.set_title('Standardabweichung des mittleren Einkommens\n(10k USD Bins)')
    ax2.set_xticks(range(len(income_bin_stats)))
    ax2.set_xticklabels(income_bin_stats['bin_label'], rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Zusammenfassung für Einkommen
    print("\nZUSAMMENFASSUNG EINKOMMEN:")
    print(f"Höchste Häufigkeit: {income_bin_stats['count'].max():,.0f} Regionen im Bin {income_bin_stats.loc[income_bin_stats['count'].idxmax(), 'bin_label']}")
    print(f"Höchste Standardabweichung: {income_bin_stats['std'].max():.3f} im Bin {income_bin_stats.loc[income_bin_stats['std'].idxmax(), 'bin_label']}")
    print(f"Niedrigste Standardabweichung: {income_bin_stats['std'].min():.3f} im Bin {income_bin_stats.loc[income_bin_stats['std'].idxmin(), 'bin_label']}")

def main():
    """Hauptfunktion - führt die komplette Analyse durch."""
    print("California Housing: Preisverteilung und Standardabweichung")
    print("="*60)
    
    # Daten laden und bereinigen
    housing_clean = load_and_clean_data()
    
    # Hauspreise analysieren
    bin_stats = analyze_house_prices(housing_clean)
    plot_house_prices(bin_stats)
    
    # Einkommen analysieren
    income_bin_stats = analyze_income(housing_clean)
    plot_income(income_bin_stats)
    
    print("\n" + "="*60)
    print("ANALYSE ABGESCHLOSSEN")
    print("="*60)
    print("Zwei Variablen derselben Teilmenge analysiert:")
    print("1. Hauspreise in 50k USD Bins")
    print("2. Mittleres Einkommen in 10k USD Bins")
    print("\nBeide zeigen Häufigkeitsverteilung und Standardabweichung pro Bin.")

if __name__ == "__main__":
    main()