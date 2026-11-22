# Snack Detector

Snacks erkennen mit Python, OpenCV und neuronalen Netzen.

## Ordner

```text
Snack_Detector_DSAI/
├── opencv_webcam.ipynb          # Farben, Graustufen, Kanten
├── YOLO_Bild_Vortrainiert.ipynb # Allgemeines YOLO ausprobieren
├── Snack_Detection.ipynb       # Snacks mit Boxen erkennen
├── Snack_Classification.ipynb  # Ganzes Bild klassifizieren
├── Snack_Segmentation.ipynb    # Snacks mit Masken erkennen
├── camera.py                  # Eigenes Kamerafenster
├── requirements.txt           # Python-Pakete
├── models/                    # Fertige Modelle
├── data/                      # Beispielbilder und Datensätze
└── runs/                      # Gespeicherte Ergebnisse
```

## Voraussetzungen

- Python 3.12
- Git oder Download als ZIP
- Internet für die erste Installation
- Webcam für den Live-Test

Eine NVIDIA-Grafikkarte ist zum Ausprobieren nicht nötig.

Fehlt Python, zuerst Python 3.12 installieren:
https://www.python.org/downloads/

## 1. Projekt herunterladen

```bash
git clone https://github.com/DanDerMan17/Snack_Detector_DSAI.git
cd Snack_Detector_DSAI
```

Alternativ: Auf GitHub „Code → Download ZIP“, entpacken und
ein Terminal im Projektordner öffnen.

## 2. Einrichten und starten

Die Befehle erstellen eine eigene `.venv`.
Eine Aktivierung ist nicht nötig.

### Windows PowerShell

```powershell
py -3.12 -m venv .venv

.\.venv\Scripts\python.exe -m pip install --upgrade pip

.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

.\.venv\Scripts\python.exe -m pip install -r requirements.txt

.\.venv\Scripts\python.exe -m jupyterlab
```

### Linux

```bash
python3.12 -m venv .venv

.venv/bin/python -m pip install --upgrade pip

.venv/bin/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

.venv/bin/python -m pip install -r requirements.txt

.venv/bin/python -m jupyterlab
```

### macOS

```bash
python3.12 -m venv .venv

.venv/bin/python -m pip install --upgrade pip

.venv/bin/python -m pip install -r requirements.txt

.venv/bin/python -m jupyterlab
```

Die erste Installation lädt größere Pakete herunter.
Falls sie im Unterricht länger dauert: an der gemeinsamen Demo
mitmachen und die Einrichtung später abschließen.

`requirements.lock.txt` enthält Windows-Pakete.
Für diese Anleitung verwenden wir `requirements.txt`.

## 3. Notebooks ausprobieren

Jupyter öffnet sich im Browser.

Notebook öffnen und Codezellen einzeln mit **Shift + Enter** ausführen.

Reihenfolge:

1. `opencv_webcam.ipynb`
   - Zellen ausführen
   - Start drücken
   - Filter wechseln
   - Mit Stopp die Kamera freigeben

2. `YOLO_Bild_Vortrainiert.ipynb`
   - Codezellen ausführen
   - Allgemeine Objekte auf Beispielbildern erkennen

3. `Snack_Detection.ipynb`
   - Zellen nacheinander ausführen
   - `TRAIN = False` lassen
   - Letzte Zelle startet die Snack-Erkennung

Optional:

- `Snack_Classification.ipynb`: Bounty oder Snickers
- `Snack_Segmentation.ipynb`: Masken für Bounty und Snickers

Auch dort `TRAIN = False` lassen.
Die Modelle sind bereits trainiert.

Die zusätzlichen School-Supplies-Notebooks sind optionale Experimente
und gehören nicht zum Unterrichtsablauf.

## 4. Kamerafenster

- **Q oder Esc:** beenden
- **S:** Originalfoto speichern
- Immer nur ein Kameraprogramm gleichzeitig öffnen

Die Snack-Detektion erkennt:

- Bounty
- Mars
- MilkyWay
- Skittles
- Snickers
- Twix

## 5. Mitmachaufgabe

1. Einen bekannten Snack zeigen.
2. Abstand verändern.
3. Verpackung teilweise verdecken.
4. Eine unbekannte Verpackung zeigen.
5. Konfidenzschwelle verändern.

Notieren:

- Was wurde erkannt?
- Was wurde übersehen?
- Welche Klasse war falsch?
- Woran könnte es liegen?

Die Konfidenz ist ein Modellscore, keine garantierte Wahrscheinlichkeit.

## Probleme

**Kamera besetzt:** Andere Kamera-Apps schließen.
Im bisherigen Fenster Q drücken oder den OpenCV-Stream stoppen.

**Falsche Kamera:** `CAMERA = 1` beziehungsweise `CAM_INDEX = 1` testen.

**Modell oder Datensatz fehlt:** Prüfen, ob `models/` und `data/`
vollständig heruntergeladen wurden.

**Kein Trainingsdatensatz:** Für die fertige Snack-Demo nicht nötig.
`TRAIN = False` lassen.

**Keine Kamera:** OpenCV bietet ein Testbild.
Die Snack-Detektion kann das mitgelieferte Foto verwenden.

Windows:

```powershell
.\.venv\Scripts\python.exe camera.py --image data/raw/beispiel.jpeg --output runs/test.jpg
```

Linux/macOS:

```bash
.venv/bin/python camera.py --image data/raw/beispiel.jpeg --output runs/test.jpg
```

Das Ergebnis liegt anschließend unter `runs/test.jpg`.

## Datenschutz

Kamera auf die Snacks richten.
Keine Personenaufnahmen ohne Einwilligung speichern oder veröffentlichen.