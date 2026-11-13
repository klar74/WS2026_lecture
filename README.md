# WS2026_lecture
Jupyter Notebooks for Education.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/klar74/WS2026_lecture/HEAD)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/klar74/WS2026_lecture)

## Python-Umgebung mit Conda einrichten

Voraussetzung: Conda ist installiert, zum Beispiel über Miniconda.
Führe die folgenden Befehle im Repository-Ordner aus, zum Beispiel in der
Anaconda Prompt.

### Variante 1: Direkt aus der YAML-Datei (kurzer Weg)

Erstelle die Umgebung einschließlich Python und aller Abhängigkeiten direkt
aus [environment.yml](environment.yml):

```shell
conda env create -n lecture_env -f environment.yml
conda activate lecture_env
```

Mit `-n lecture_env` wird der Name ausdrücklich festgelegt. Ohne diese Option
wird der Name aus der YAML-Datei verwendet.

### Variante 2: Umgebung zuerst erstellen, dann Abhängigkeiten installieren

Erstelle zuerst die Umgebung mit Python 3.11:

```shell
conda create -n lecture_env python=3.11
```

Installiere anschließend die Abhängigkeiten aus [environment.yml](environment.yml)
und aktiviere die Umgebung:

```shell
conda env update -n lecture_env -f environment.yml
conda activate lecture_env
```

Die YAML-Datei legt Python auf `3.11.13` fest. Beim Update wird daher diese
Version installiert. Mit `-n lecture_env` wird die Zielumgebung ausdrücklich
gewählt, auch wenn in der YAML-Datei ein anderer Name steht.

Wähle in VS Code für die Python-Dateien und als Kernel der Jupyter Notebooks
die Umgebung `lecture_env` aus.
