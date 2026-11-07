# ROC-Kennzahlen Übung: Defekterkennung

## 📊 Aufgabenstellung

**Szenario:** Automatische Qualitätskontrolle in der Produktion

### Gegeben:
- **Testdatensatz:** 100 Bauteile
- **Tatsächliche Verteilung:**
  - 90 Bauteile sind **OK** (negativ)
  - 10 Bauteile sind **NOK** (positiv/defekt)
- **Modell-Vorhersagen:**
  - 10 False Positives (FP)
  - 5 False Negatives (FN)

---

## 🎯 Aufgabe 1: Konfusionsmatrix erstellen

Vervollständigen Sie die Konfusionsmatrix:

```
                             Vorhersage
                        OK (neg)  NOK (pos)
Tatsächlich  OK (neg)   ???       10
            NOK (pos)   5         ???
```

**Hinweise:**
- FP = 10 (OK-Teile fälschlich als NOK klassifiziert)
- FN = 5 (NOK-Teile fälschlich als OK klassifiziert)
- Gesamt: 90 OK-Teile, 10 NOK-Teile

---

## 📐 Aufgabe 2: Kennzahlen berechnen

### Gegeben: Formeln

**True Positive Rate (TPR):**
```
TPR = TP / (TP + FN)
```

**False Positive Rate (FPR):**
```
FPR = FP / (FP + TN)
```

**True Negative Rate (TNR):**
```
TNR = TN / (TN + FP)
```

**False Negative Rate (FNR):**
```
FNR = FN / (FN + TP)
```

**Accuracy:**
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

**Precision:**
```
Precision = TP / (TP + FP)
```

### Berechnen Sie:
1. **TPR (Sensitivität)** = ?
2. **FPR** = ?
3. **TNR (Spezifität)** = ?
4. **FNR** = ?
5. **Accuracy** = ?
6. **Precision** = ?

---

## 🎯 Aufgabe 3: Interpretation

**Beantworten Sie:**

1. **Wie viel Prozent aller defekten Teile werden erkannt?**
   → TPR = ?

2. **Wie viel Prozent aller OK-Teile werden fälschlich als defekt eingestuft?**
   → FPR = ?

3. **Ist dieses Modell für die Qualitätskontrolle geeignet?**
   → Begründung basierend auf den Kennzahlen

---

## 📈 Aufgabe 4: ROC-Kurve verstehen

**Gegeben:** Der aktuelle Klassifikations-Schwellwert ergibt die obigen Werte.

**Was passiert, wenn wir den Schwellwert ändern?**

### Niedrigerer Schwellwert (mehr NOK-Meldungen möglich):
- TPR: ⬆️ **kann steigen oder gleich bleiben** (es können weniger defekte Teile übersehen werden)
- FPR: ⬆️ **kann steigen oder gleich bleiben** (es können mehr OK-Teile fälschlich als defekt gelten)

### Höherer Schwellwert (weniger NOK-Meldungen möglich):
- TPR: ⬇️ **kann sinken oder gleich bleiben** (es können mehr defekte Teile übersehen werden)
- FPR: ⬇️ **kann sinken oder gleich bleiben** (es können weniger OK-Teile fälschlich als defekt gelten)

**Frage:** Wann wäre eine niedrigere Schwelle sinnvoll? Vergleichen Sie die Folgen von übersehenen Defekten und Fehlalarmen.

**Überlege und erkläre dann:** Warum entspricht "Würfeln" der Diagonalen im ROC-Plot?

---

## 🔧 Vorgehensweise: ROC-Kurve erstellen

**Hinweis:** Die gegebenen Anzahlen liefern nur einen Punkt im ROC-Plot. Für eine ganze Kurve brauchen wir die Vorhersagewerte der einzelnen Teile und ihre wahren Labels, um weitere Schwellen auszuprobieren. Die folgenden Schritte beschreiben dieses Vorgehen.

### Schritt 1: Wahrscheinlichkeiten sammeln
- Für jedes Bauteil: Wahrscheinlichkeit "defekt"
- Beispiel: [0.1, 0.3, 0.7, 0.9, ...]

### Schritt 2: Verschiedene Schwellwerte testen
- Schwellwerte: 0.1, 0.2, 0.3, ..., 0.9
- Für jeden Schwellwert:
  - Wenn P(defekt) ≥ Schwellwert → Vorhersage: NOK
  - Sonst → Vorhersage: OK

### Schritt 3: Für jeden Schwellwert berechnen
- Neue Konfusionsmatrix
- TPR und FPR berechnen
- Punkt (FPR, TPR) notieren

### Schritt 4: Kurve zeichnen
- X-Achse: FPR (False Positive Rate)
- Y-Achse: TPR (True Positive Rate)
- Punkte verbinden → ROC-Kurve

### Schritt 5: AUC berechnen
- AUC = Fläche unter der ROC-Kurve
- Perfect: AUC = 1.0
- Zufall: AUC = 0.5

---

## 💡 Praxis-Tipp: Schwellwert-Optimierung

**Für Qualitätskontrolle wichtig:**
- **Hohe TPR** (wenig defekte Teile übersehen)
- **Akzeptable FPR** (nicht zu viele gute Teile verwerfen)

**Kostenabwägung:**
- Kosten FN: Defektes Teil gelangt zum Kunden
- Kosten FP: Gutes Teil wird unnötig verworfen

➤ **Optimaler Schwellwert minimiert Gesamtkosten**

---

## 🎯 Lernziele erreicht?

Nach dieser Übung können Sie:
- ✅ Konfusionsmatrix aus gegebenen Werten erstellen
- ✅ TPR, FPR, TNR, FNR, Accuracy, Precision berechnen
- ✅ Kennzahlen interpretieren und bewerten
- ✅ ROC-Kurven-Erstellung nachvollziehen
- ✅ Schwellwert-Optimierung verstehen