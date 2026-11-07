# ROC-Kennzahlen Übung: Lösungen

## ✅ Lösung Aufgabe 1: Konfusionsmatrix

```
                    Vorhersage
                 OK (neg)  NOK (pos)  Total
Tatsächlich  OK     80       10        90
            NOK      5        5        10
            Total   85       15       100
```

**Herleitung:**
- TN = 90 - 10 = 80 (OK-Teile korrekt als OK erkannt)
- TP = 10 - 5 = 5 (NOK-Teile korrekt als NOK erkannt)
- FP = 10 (gegeben)
- FN = 5 (gegeben)

---

## ✅ Lösung Aufgabe 2: Kennzahlen

**Gegeben:**
- TP = 5, TN = 80, FP = 10, FN = 5

### Berechnungen:

1. **TPR = TP / (TP + FN) = 5 / (5 + 5) = 5/10 = 0.5 = 50%**

2. **FPR = FP / (FP + TN) = 10 / (10 + 80) = 10/90 = 0.111 = 11.1%**

3. **TNR = TN / (TN + FP) = 80 / (80 + 10) = 80/90 = 0.889 = 88.9%**

4. **FNR = FN / (FN + TP) = 5 / (5 + 5) = 5/10 = 0.5 = 50%**

5. **Accuracy = (TP + TN) / Gesamt = (5 + 80) / 100 = 85/100 = 0.85 = 85%**

6. **Precision = TP / (TP + FP) = 5 / (5 + 10) = 5/15 = 0.333 = 33.3%**

---

## ✅ Lösung Aufgabe 3: Interpretation

1. **50% aller defekten Teile werden erkannt** (TPR = 50%)
   - Das bedeutet: Jedes zweite defekte Teil wird übersehen! 😟

2. **11.1% aller OK-Teile werden fälschlich als defekt eingestuft** (FPR = 11.1%)
   - Das bedeutet: Etwa jedes 9. gute Teil wird unnötig verworfen

3. **Bewertung für Qualitätskontrolle:**
   - ❌ **TPR = 50% ist zu niedrig** → Zu viele defekte Teile gelangen zum Kunden
   - ⚠️ **FPR = 11.1% ist akzeptabel** → Verschwendung, aber nicht kritisch
   - ⚠️ **Precision = 33.3% ist niedrig** → Nur jede 3. "NOK"-Meldung ist korrekt
   
   **Fazit:** Das Modell muss verbessert werden! Schwellwert senken für höhere Sensitivität.

---

## 🎯 Qualitätskontrolle-spezifische Bewertung

**Kritisch für Qualitätskontrolle:**
- **FN = 5 defekte Teile übersehen** → Reklamationen, Imageschaden
- **FP = 10 gute Teile verworfen** → Materialverschwendung (weniger kritisch)

**Empfehlung:** Schwellwert senken
- Mehr Teile als "NOK" klassifizieren
- TPR erhöhen (weniger übersehen)
- FPR steigt (mehr Verschwendung, aber akzeptabel)

---

## 📊 Prüfung der Zusammenhänge

**Kontrolle der Formeln:**
- TPR + FNR = 0.5 + 0.5 = 1.0 ✅
- TNR + FPR = 0.889 + 0.111 = 1.0 ✅
- Gesamt: 80 + 10 + 5 + 5 = 100 ✅

**Alle Werte sind konsistent!**

---

## ✅ Lösung Aufgabe 4: ROC-Kurve verstehen

### Optimaler Schwellwert für Qualitätskontrolle:

**Antwort:** Ein **niedrigerer Schwellwert** ist optimal für die Qualitätskontrolle.

**Begründung:**
- In der Qualitätskontrolle sind **False Negatives kritischer** als False Positives
- Ein übersehenes defektes Teil (FN) → Kunde erhält defekte Ware → Reklamation, Imageschaden, Folgeschäden ggf. schwer!
- Ein verworfenes gutes Teil (FP) → Materialverschwendung → weniger kritisch

**Konkret:**
- Niedrigerer Schwellwert → Mehr Teile werden als "NOK" klassifiziert
- TPR steigt → Weniger defekte Teile übersehen ✅
- FPR steigt → Mehr gute Teile verworfen ⚠️ (aber akzeptabel)

### Warum entspricht "Würfeln" der Diagonalen?

**Schritt-für-Schritt Erklärung:**

#### 🎲 Situation: Zufälliger Klassifikator
Stellen Sie sich vor, wir haben **kein Modell** und entscheiden **per Münzwurf**:
- Kopf → "Das Teil ist NOK" 
- Zahl → "Das Teil ist OK"

#### 📊 Konkrete Berechnung mit unserem Beispiel:
**Gegeben:** 90 OK-Teile + 10 NOK-Teile = 100 Teile insgesamt

**Bei 50:50 Münzwurf-Entscheidungen:**

**1. Was passiert mit den 10 defekten Teilen?**
- 50% werden richtig als "NOK" erkannt → TP = 5
- 50% werden fälschlich als "OK" erkannt → FN = 5
- **TPR = TP/(TP+FN) = 5/(5+5) = 50%**

**2. Was passiert mit den 90 OK-Teilen?**  
- 50% werden richtig als "OK" erkannt → TN = 45
- 50% werden fälschlich als "NOK" erkannt → FP = 45
- **FPR = FP/(FP+TN) = 45/(45+45) = 50%**

**Ergebnis:** TPR = FPR = 50% → Punkt (0.5, 0.5) auf der Diagonalen

#### 🎯 Warum die GANZE Diagonale? 
**Der Trick: Zufällige Wahrscheinlichkeiten + verschiedene Schwellwerte**

**Stellen Sie sich vor: Ein "Zufalls-Modell"**
- Für jedes Teil gibt es eine **zufällige Wahrscheinlichkeit** zwischen 0 und 1
- Beispiel: Teil 1 → 0.23, Teil 2 → 0.67, Teil 3 → 0.15, etc.
- Diese Zahlen haben **keine Bedeutung** - sie sind völlig zufällig!

**Jetzt verschiedene Schwellwerte testen:**

**Schwellwert 0.9 (sehr hoch):**
- Nur Teile mit P > 0.9 werden als "NOK" klassifiziert
- Das sind etwa 10% aller Teile - **egal ob wirklich defekt oder nicht!**
- Von 10 defekten Teilen: ~1 erkannt → TPR = 10%
- Von 90 OK-Teilen: ~9 fälschlich als NOK → FPR = 10%
- **Punkt: (0.1, 0.1)**

**Schwellwert 0.2 (sehr niedrig):**
- Teile mit P > 0.2 werden als "NOK" klassifiziert  
- Das sind etwa 80% aller Teile - **wieder egal ob wirklich defekt oder nicht!**
- Von 10 defekten Teilen: ~8 erkannt → TPR = 80%
- Von 90 OK-Teilen: ~72 fälschlich als NOK → FPR = 80%
- **Punkt: (0.8, 0.8)**

#### 💡 Die Kernidee:
**Zufällige Wahrscheinlichkeiten sind "klassenblind":**
- Ein zufälliges Modell kann nicht zwischen defekt und OK unterscheiden
- Daher trifft jeder Schwellwert **beide Klassen gleich**
- Bei gleichmäßig verteilten Zufallswerten: Schwellwert x → etwa (1 − x) · 100% aller defekten UND aller OK-Teile als "NOK"
- **Resultat: TPR ≈ FPR ≈ 1 − x** (z. B. Schwellwert 0.9 → etwa 0.1 = 10%)

**Das ist wie ein blindes Modell mit verschiedenen "Risikobereitschaften"!**

**Interpretation:**
- Oberhalb der Diagonalen = besser als Zufall ✅
- Auf der Diagonalen = wie Zufall (nutzlos) ⚠️
- Unterhalb der Diagonalen = schlechter als Zufall ❌ (Vorhersagen umkehren würde helfen!)