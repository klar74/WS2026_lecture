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

**Zum ROC-Plot:** Diese Werte liefern den Punkt **(FPR, TPR) ≈ (0.111, 0.5)**. Für eine ganze Kurve brauchen wir Ergebnisse bei weiteren Schwellen.

**Achtung bei Accuracy:** Würden wir immer „OK“ vorhersagen, wären bereits 90 von 100 Entscheidungen richtig: **90% Accuracy**, aber kein Defekt würde erkannt. Die 85% allein sagen deshalb wenig darüber aus, wie gut das Modell Defekte erkennt.

---

## ✅ Lösung Aufgabe 3: Interpretation

1. **50% aller defekten Teile werden erkannt** (TPR = 50%)
   - Das bedeutet: Jedes zweite defekte Teil wird übersehen! 😟

2. **11.1% aller OK-Teile werden fälschlich als defekt eingestuft** (FPR = 11.1%)
   - Das bedeutet: Etwa jedes 9. gute Teil wird unnötig verworfen

3. **Bewertung für Qualitätskontrolle:**
   - **TPR = 50%** → Die Hälfte der Defekte wird übersehen. Für eine Kontrolle, die fast alle Defekte erkennen soll, ist das zu wenig.
   - **FPR = 11.1%** → Ob diese Fehlalarme akzeptabel sind, hängt von ihren Kosten ab.
   - **Precision = 33.3%** → Nur jede 3. "NOK"-Meldung ist korrekt.
   
   **Fazit:** Die Eignung hängt von den Anforderungen ab. Wenn übersehene Defekte besonders teuer sind, kann eine niedrigere Schwelle sinnvoll sein.

---

## 🎯 Qualitätskontrolle-spezifische Bewertung

**Folgen der Fehler abwägen:**
- **FN = 5 defekte Teile übersehen** → Reklamationen, Imageschaden
- **FP = 10 gute Teile verworfen** → Materialverschwendung

**Mögliche Maßnahme:** Schwellwert senken, wenn übersehene Defekte besonders teuer sind.
- Es können mehr Teile als "NOK" klassifiziert werden.
- Die TPR kann steigen: weniger Defekte werden übersehen.
- Die FPR kann ebenfalls steigen: mehr gute Teile lösen einen Alarm aus. Diese zusätzlichen Kosten müssen wir mit abwägen.

---

## 📊 Prüfung der Zusammenhänge

**Kontrolle der Formeln:**
- TPR + FNR = 0.5 + 0.5 = 1.0 ✅
- TNR + FPR = 0.889 + 0.111 = 1.0 ✅
- Gesamt: 80 + 10 + 5 + 5 = 100 ✅

**Alle Werte sind konsistent!**

---

## ✅ Lösung Aufgabe 4: ROC-Kurve verstehen

### Wann wäre eine niedrigere Schwelle sinnvoll?

**Antwort:** Wenn übersehene Defekte besonders teuer sind, kann eine **niedrigere Schwelle** sinnvoll sein.

**Begründung:**
- Ein übersehenes defektes Teil (FN) → Kunde erhält defekte Ware → Reklamation, Imageschaden, Folgeschäden ggf. schwer!
- Ein verworfenes gutes Teil (FP) → Materialverschwendung und zusätzliche Kosten
- Welche Fehler schwerer wiegen, hängt vom Produkt und den Kosten ab.

**Konkret:**
- Niedrigerer Schwellwert → Es können mehr Teile als "NOK" klassifiziert werden
- TPR kann steigen → Weniger defekte Teile übersehen ✅
- FPR kann steigen → Mehr gute Teile verworfen ⚠️

**Eine optimale Schwelle können wir aus den gegebenen Zahlen nicht bestimmen.** Dafür brauchen wir die Fehlerkosten und Ergebnisse bei weiteren Schwellen.

### Warum entspricht "Würfeln" der Diagonalen?

**Schritt-für-Schritt Erklärung:**

#### 🎲 Situation: Zufälliger Klassifikator
Stellen Sie sich vor, wir haben **kein Modell** und entscheiden **per Münzwurf**:
- Kopf → "Das Teil ist NOK" 
- Zahl → "Das Teil ist OK"

#### 📊 Konkrete Berechnung mit unserem Beispiel:
**Gegeben:** 90 OK-Teile + 10 NOK-Teile = 100 Teile insgesamt

**Bei 50:50 Münzwurf-Entscheidungen:**

Im Durchschnitt erwarten wir die folgenden Anzahlen. Bei einem einzelnen Versuch können die Ergebnisse abweichen.

**1. Was passiert mit den 10 defekten Teilen?**
- 50% werden richtig als "NOK" erkannt → TP = 5
- 50% werden fälschlich als "OK" erkannt → FN = 5
- **TPR = TP/(TP+FN) = 5/(5+5) = 50%**

**2. Was passiert mit den 90 OK-Teilen?**  
- 50% werden richtig als "OK" erkannt → TN = 45
- 50% werden fälschlich als "NOK" erkannt → FP = 45
- **FPR = FP/(FP+TN) = 45/(45+45) = 50%**

**Im Durchschnitt:** TPR = FPR = 50% → Punkt (0.5, 0.5) auf der Diagonalen

#### 🎯 Warum die GANZE Diagonale? 
**Der Trick: Zufällige Wahrscheinlichkeiten + verschiedene Schwellwerte**

**Stellen Sie sich vor: Ein "Zufalls-Modell"**
- Für jedes Teil ziehen wir einen **Zufallswert**, gleichmäßig verteilt zwischen 0 und 1
- Ob das Teil tatsächlich defekt ist, spielt beim Ziehen keine Rolle.
- Beispiel: Teil 1 → 0.23, Teil 2 → 0.67, Teil 3 → 0.15, etc.
- Diese Zahlen haben **keine Bedeutung** - sie sind völlig zufällig!

**Jetzt verschiedene Schwellwerte testen:**

**Schwellwert 0.9 (sehr hoch):**
- Nur Teile mit P ≥ 0.9 werden als "NOK" klassifiziert
- Das sind etwa 10% aller Teile - **egal ob wirklich defekt oder nicht!**
- Von 10 defekten Teilen: ~1 erkannt → TPR = 10%
- Von 90 OK-Teilen: ~9 fälschlich als NOK → FPR = 10%
- **Punkt: (0.1, 0.1)**

**Schwellwert 0.2 (sehr niedrig):**
- Teile mit P ≥ 0.2 werden als "NOK" klassifiziert
- Das sind etwa 80% aller Teile - **wieder egal ob wirklich defekt oder nicht!**
- Von 10 defekten Teilen: ~8 erkannt → TPR = 80%
- Von 90 OK-Teilen: ~72 fälschlich als NOK → FPR = 80%
- **Punkt: (0.8, 0.8)**

#### 💡 Die Kernidee:
**Zufällige Wahrscheinlichkeiten sind "klassenblind":**
- Ein zufälliges Modell kann nicht zwischen defekt und OK unterscheiden
- Daher trifft jeder Schwellwert **im Durchschnitt denselben Anteil beider Klassen**
- Bei gleichmäßig verteilten Zufallswerten: Schwellwert x → etwa (1 − x) · 100% aller defekten UND aller OK-Teile als "NOK"
- **Resultat: TPR ≈ FPR ≈ 1 − x** (z. B. Schwellwert 0.9 → etwa 0.1 = 10%)

**Das ist wie ein blindes Modell mit verschiedenen "Risikobereitschaften"!**

**Interpretation:**
- Oberhalb der Diagonalen = besser als Zufall ✅
- Auf der Diagonalen = wie Zufall (nutzlos) ⚠️
- Unterhalb der Diagonalen = schlechter als Zufall ❌ (Vorhersagen umkehren würde helfen!)