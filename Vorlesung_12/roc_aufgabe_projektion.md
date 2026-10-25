# ROC-Übung ohne Programmieren

Diese Aufgabe eignet sich zum gemeinsamen Rechnen an der Tafel oder auf Papier. Die Lösungen stehen am Ende in einem aufklappbaren Abschnitt. Zur Kontrolle kann anschließend das [Notebook zur manuellen ROC-Berechnung](roc_kurve_manuell.ipynb) verwendet werden.

## Aufgabe: ROC-Kurve von Hand erstellen

**Gegeben sind 10 geprüfte Bauteile mit folgenden Ergebnissen:**

### Daten

| Bauteil | Wahres Label | Defektwahrscheinlichkeit |
|--------|--------------|-------------------------|
| 1      | 0 (OK)       | 0.2                     |
| 2      | 0 (OK)       | 0.3                     |
| 3      | 0 (OK)       | 0.1                     |
| 4      | 0 (OK)       | 0.6                     |
| 5      | 0 (OK)       | 0.7                     |
| 6      | 1 (NOK)      | 0.8                     |
| 7      | 1 (NOK)      | 0.9                     |
| 8      | 1 (NOK)      | 0.4                     |
| 9      | 1 (NOK)      | 0.85                    |
| 10     | 1 (NOK)      | 0.75                    |

**Erklärung:**
- **Label 0**: Teil ist in Ordnung (OK)
- **Label 1**: Teil ist defekt (NOK)
- **Defektwahrscheinlichkeit**: Eine erfundene Vorhersage für diese Übung. Je höher der Wert, desto eher meldet das System einen Defekt. In der Praxis müsste man zusätzlich prüfen, wie zuverlässig solche Prozentangaben sind.

**Entscheidungsregel für alle Aufgaben:**
- Vorhersagewert **≥ Schwelle** → als defekt (NOK) einstufen
- Vorhersagewert **< Schwelle** → als in Ordnung (OK) einstufen

Beispiel: Bei Schwelle 0.5 wird ein Teil mit Vorhersagewert 0.6 als defekt eingestuft. Ob diese Entscheidung richtig ist, sehen wir am wahren Label.

---

## Aufgaben

### 1. Richtige und falsche Entscheidungen zählen
Verwenden Sie die Schwelle 0.5 und zählen Sie:
- **True Positives (TP)**: Korrekt als defekt erkannt
- **True Negatives (TN)**: Korrekt als OK erkannt  
- **False Positives (FP)**: Fälschlicherweise als defekt erkannt
- **False Negatives (FN)**: Übersehene Defekte

Eine Tabelle mit diesen vier Anzahlen nennt man **Konfusionsmatrix**.

### 2. Trefferquote und Fehlalarmrate berechnen
Berechnen Sie:
- **TPR (True Positive Rate, Trefferquote)** = TP / (TP + FN): Welcher Anteil der defekten Teile wird erkannt?
- **FPR (False Positive Rate, Fehlalarmrate)** = FP / (FP + TN): Welcher Anteil der OK-Teile löst fälschlich einen Alarm aus?

### 3. ROC-Kurve erstellen
- Wählen Sie mindestens 5 verschiedene Schwellenwerte
- Berechnen Sie für jeden Schwellenwert TPR und FPR
- Zeichnen Sie die ROC-Kurve (FPR auf x-Achse, TPR auf y-Achse)
- Ergänzen Sie die Endpunkte (0, 0): kein Teil als defekt eingestuft, und (1, 1): alle Teile als defekt eingestuft.

**Vorschlag:** Beginnen Sie bei Schwelle 1.1 und senken Sie diese auf 0.9, 0.7, 0.5, 0.3 und 0.0.

Die **Schwelle 1.1 ist keine Wahrscheinlichkeit**, sondern nur eine Vergleichsgrenze oberhalb aller Vorhersagewerte. Kein Wert erreicht diese Grenze, deshalb werden alle Teile als OK eingestuft.

### 4. AUC schätzen
Schätzen Sie die **AUC (Area Under Curve)**, also die Fläche unter Ihrer ROC-Kurve. Wenige ausgewählte Schwellen liefern nur eine grobe Schätzung. Für die vollständige Kurve verwenden Sie alle unterschiedlichen Vorhersagewerte sowie eine Schwelle oberhalb des größten Werts.

### 5. Gemeinsam besprechen
1. Was passiert bei einer sehr niedrigen Schwelle, zum Beispiel 0.1?
2. Was passiert bei einer sehr hohen Schwelle, zum Beispiel 0.9?
3. Wann sind zusätzliche Fehlalarme eher akzeptabel, wann problematisch?

**Freiwillige Zusatzfrage:** Wir erhöhen alle Vorhersagewerte um 0.1, ohne sie anderweitig zu verändern. In diesem Beispiel bleiben sie damit zwischen 0 und 1. Bleibt die vollständige ROC-Kurve gleich? Und bleiben auch die Entscheidungen bei der festen Schwelle 0.5 gleich?

---

<details>
<summary><strong>Lösungen anzeigen – erst nach dem eigenen Versuch öffnen</strong></summary>

## Lösungen

### 1. Entscheidungen bei Schwelle 0.5

**Vorhersagen:**
- Bauteile 1–3: OK (richtig) → TN = 3
- Bauteile 4–5: NOK (falsch) → FP = 2
- Bauteile 6–7 und 9–10: NOK (richtig) → TP = 4
- Bauteil 8: OK (falsch) → FN = 1

| Tatsächlich | Als OK eingestuft | Als NOK eingestuft |
|-------------|------------------|-------------------|
| OK          | TN = 3           | FP = 2            |
| NOK         | FN = 1           | TP = 4            |

### 2. Trefferquote und Fehlalarmrate

- TPR = 4/(4+1) = **0.8**: 80 % der defekten Teile werden erkannt.
- FPR = 2/(2+3) = **0.4**: 40 % der OK-Teile lösen fälschlich einen Alarm aus.

### 3. Vollständige Tabelle für die ROC-Kurve

Die Schwelle wird von oben nach unten gesenkt. Tragen Sie jeweils **(FPR, TPR)** ein und verbinden Sie die Punkte in dieser Reihenfolge.

| Schwelle | TP | TN | FP | FN | FPR | TPR |
|----------|----|----|----|----|-----|-----|
| 1.1      | 0  | 5  | 0  | 5  | 0.0 | 0.0 |
| 0.9      | 1  | 5  | 0  | 4  | 0.0 | 0.2 |
| 0.85     | 2  | 5  | 0  | 3  | 0.0 | 0.4 |
| 0.8      | 3  | 5  | 0  | 2  | 0.0 | 0.6 |
| 0.75     | 4  | 5  | 0  | 1  | 0.0 | 0.8 |
| 0.7      | 4  | 4  | 1  | 1  | 0.2 | 0.8 |
| 0.6      | 4  | 3  | 2  | 1  | 0.4 | 0.8 |
| 0.5      | 4  | 3  | 2  | 1  | 0.4 | 0.8 |
| 0.4      | 5  | 3  | 2  | 0  | 0.4 | 1.0 |
| 0.3      | 5  | 2  | 3  | 0  | 0.6 | 1.0 |
| 0.2      | 5  | 1  | 4  | 0  | 0.8 | 1.0 |
| 0.1      | 5  | 0  | 5  | 0  | 1.0 | 1.0 |
| 0.0      | 5  | 0  | 5  | 0  | 1.0 | 1.0 |

**In diesem Beispiel** verläuft die vollständige Kurve zwischen den Endpunkten oberhalb der Diagonalen. Bei anderen Daten muss das nicht überall so sein.

### 4. Fläche unter der Kurve

Mit allen Vorhersagewerten ergibt sich **AUC = 0.92**. Hier kann man die Fläche besonders einfach aus zwei Rechtecken zusammensetzen:

- Von FPR 0.0 bis 0.4 hat die Kurve die Höhe 0.8: Fläche = 0.4 × 0.8 = 0.32.
- Von FPR 0.4 bis 1.0 hat sie die Höhe 1.0: Fläche = 0.6 × 1.0 = 0.60.
- Zusammen: **0.32 + 0.60 = 0.92**.

Wer nur die vorgeschlagenen sechs Schwellen verwendet und die Punkte geradlinig verbindet, erhält **0.84**. Das ist eine gröbere Schätzung: Einige Ecken der vollständigen Kurve fehlen.

**AUC-Interpretation:**
- AUC = 0.5: Beim Ordnen der Teile nach dem Vorhersagewert liegt das Ergebnis auf Zufallsniveau.
- AUC = 1.0: Alle NOK-Teile haben höhere Vorhersagewerte als alle OK-Teile. Eine passende Schwelle kann sie vollständig trennen.
- Je näher die AUC an 1 liegt, desto besser werden NOK-Teile vor OK-Teilen eingeordnet. Welche Fehler im Betrieb akzeptabel sind, sagt sie allein nicht.

### 5. Antworten auf die Diskussionsfragen

1. **Schwelle 0.1:** Alle zehn Teile werden als defekt eingestuft. Kein Defekt wird übersehen, aber alle fünf OK-Teile lösen einen Fehlalarm aus. TPR = 1.0, FPR = 1.0.
2. **Schwelle 0.9:** Nur Bauteil 7 wird als defekt eingestuft. Es gibt keinen Fehlalarm, aber vier Defekte werden übersehen. TPR = 0.2, FPR = 0.0.
3. **Fehlalarme abwägen:** Bei einem sicherheitskritischen Bauteil kann eine zusätzliche Prüfung besser sein als ein übersehener Defekt. Wenn jeder Alarm eine ganze Produktionslinie stoppt, sind Fehlalarme dagegen teuer. Man muss immer beide Fehlerarten betrachten.

### Freiwillige Zusatzfrage: Alle Werte um 0.1 erhöhen

**Die vollständige ROC-Kurve und die AUC bleiben gleich.** Die Reihenfolge der Teile ändert sich nicht: Ein zuvor höherer Wert ist auch danach höher. Dieselben Entscheidungen entstehen jetzt jeweils bei einer um 0.1 höheren Schwelle.

**Bei der festen Schwelle 0.5 ändern sich die Entscheidungen trotzdem:** Bauteil 8 steigt von 0.4 auf 0.5 und wird nun als defekt erkannt. Damit gilt TP = 5, TN = 3, FP = 2, FN = 0, also TPR = 1.0 und FPR = 0.4.

Merksatz: **Die Reihenfolge kann gleich bleiben, obwohl sich die Entscheidung an einer festen Grenze ändert.**

</details>