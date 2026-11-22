# Lösung: Ein Neuron von Hand berechnen

1. Für `x1 = 2` und `x2 = 1`:

   `z = 2 * 0,5 + 1 * (-1) + 0,25 = 0,25`

   Da `z >= 0`, ist die Ausgabe **1**.

2. Für `x1 = 1` und `x2 = 2`:

   `z = 1 * 0,5 + 2 * (-1) + 0,25 = -1,25`

   Da `z < 0`, ist die Ausgabe **0**.

3. Mit dem geänderten Bias `b = 1,5`:

   `z = 1 * 0,5 + 2 * (-1) + 1,5 = 0`

   Da `z >= 0`, ist die Ausgabe nun **1**.

**Merke:** Eingaben, Gewichte und Bias beeinflussen gemeinsam die Entscheidung eines Neurons.
