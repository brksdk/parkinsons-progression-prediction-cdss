# Reproduzierbarer Code zur Masterarbeit

Dieses Verzeichnis enthält den vollständigen Python-Code zur Reproduktion der in der Masterarbeit berichteten Ergebnisse sowie zusätzliche Skripte für das klinische Dashboard 
und die biomarkerbasierte Risikostratifizierung. Der Code wurde ursprünglich in Google Colab entwickelt, ist für die Abgabe jedoch so angepasst, dass er lokal über 
relative Pfade ausgeführt werden kann. Die Rohdaten werden ausschließlich aus dem Ordner raw_data/ gelesen und nicht manuell verändert.

## Titel der Arbeit

Personalisiertes System für Fortschrittsvorhersage und Behandlungsempfehlung bei Parkinson: Klinisches Dashboard zur Entscheidungsunterstützung mit Multiomics-Daten, 
maschinellem Lernen und erklärbarer Künstlicher Intelligenz



## Ordnerstruktur der Abgabe

abgabe/
│
├── thesis/
│   └── Sadikoglu_Berk_3021716.pdf
│
├── literature/
│   └── autor_jahr.pdf
│
├── raw_data/
│   ├── train_clinical_data.csv
│   ├── supplemental_clinical_data.csv
│   ├── train_proteins.csv
│   ├── train_peptides.csv
│   └── README_raw_data.txt
│
└── code/
    ├── parkinson_thesis_reproducible.ipynb
    ├── app_full.py
    ├── risikostratifizierung_outputs.py
    ├── requirements.txt
    ├── README.md
    │
    ├── dashboard_assets_full/
    │   ├── xgb_model_full.pkl
    │   ├── all_features.pkl
    │   ├── test_data_full.pkl
    │   ├── y_scaler.pkl
    │   └── peptide_map.pkl
    │
    ├── outputs/
    │   ├── figures/
    │   ├── tables/
    │   └── models/
    │
    └── thesis_outputs_risikostratifizierung/
        ├── cluster_zusammenfassung.csv
        ├── top_biomarker_clustervergleich.csv
        ├── cluster_validierung_metriken.csv
        ├── boxplot_updrs3_24_nach_risikoprofil.png
        ├── balkendiagramm_top10_biomarker_clustervergleich.png
        └── pca_biomarker_cluster.png




## Inhalt des Code-Verzeichnisses


### parkinson_thesis_reproducible.ipynb

Das Notebook ist der zentrale Bestandteil der Reproduktion. Es führt die vollständige Analyse von den Rohdaten bis zu den Ergebnissen aus.


Es umfasst insbesondere:

1. Laden der unveränderten Rohdaten aus raw_data/
2. Zusammenführung klinischer, proteomischer und peptidomischer Daten
3. Pivotierung der Protein- und Peptiddaten
4. Feature Engineering ohne zukünftige Informationsleckage
5. Patientenspezifische Trainings- und Testaufteilung mittels `GroupKFold`
6. Training mehrerer Machine-Learning- und Deep-Learning-Modelle
7. Berechnung des gewichteten Ensemble-Modells
8. Berechnung der Evaluationsmetriken
9. Erstellung von Tabellen und Abbildungen
10. SHAP-basierte Explainable-AI-Analyse
11. Export der Modellartefakte für das klinische Dashboard



### app_full.py

Dieses Skript enthält das Streamlit-basierte klinische Entscheidungsunterstützungssystem. Es nutzt das Full-Feature-XGBoost-Modell und die exportierten 
Dashboard-Artefakte aus dashboard_assets_full/.

Das Dashboard enthält unter anderem:

- patientenspezifische Fortschrittsprognosen,
- biomarkerbasierte Risikostratifizierung,
- lokale SHAP-basierte Erklärung ausgewählter Modellprognosen,
- modellbasierte klinische Handlungshinweise,
- eine Berichtsfunktion,
- eine Modellkarte mit methodischen Grenzen,
- optional einen erklärenden CDSS-Assistenten über eine externe Gemini-Schnittstelle.

Das Dashboard ist ein Forschungsprototyp und ersetzt keine ärztliche Diagnose oder klinische Entscheidung.




### risikostratifizierung_outputs.py

Dieses Skript erzeugt zusätzliche thesis-taugliche Tabellen und Abbildungen zur biomarkerbasierten Risikostratifizierung. Es verändert das Dashboard nicht, sondern liest 
die bereits exportierten Artefakte aus dashboard_assets_full/ und speichert separate Ergebnisse in thesis_outputs_risikostratifizierung/.

Es erzeugt unter anderem:

- Cluster-Zusammenfassung,
- Patienten-/Beobachtungstabelle mit Risikoprofilen,
- Top-Biomarker-Differenzen zwischen Risikoprofilen,
- PCA-Projektion der biomarkerbasierten Cluster,
- Validierungsmetriken,
- CSV-, LaTeX-, Excel-, PNG- und ZIP-Ausgaben.





### requirements.txt

Diese Datei enthält die benötigten Python-Bibliotheken zur lokalen Ausführung des Notebooks, des Dashboards und der Zusatzskripte.




## Rohdaten

Die Rohdaten müssen unverändert im Ordner raw_data/ liegen. Erwartet werden folgende Dateien:

raw_data/train_clinical_data.csv
raw_data/supplemental_clinical_data.csv
raw_data/train_proteins.csv
raw_data/train_peptides.csv

Die Rohdaten werden im Notebook ausschließlich gelesen und nicht verändert. Alle Verarbeitungsschritte, einschließlich Zusammenführung, Pivotierung, Feature Engineering, 
Modelltraining und Ergebnisgenerierung, sind im Notebook implementiert. 



## Installation der Python-Umgebung

Es wird empfohlen, für die Ausführung eine separate virtuelle Python-Umgebung zu verwenden.

### 1. Virtuelle Umgebung erstellen

Im Hauptordner der Abgabe:

*bash
py -3.11 -m venv .venv


Falls py -3.11 nicht verfügbar ist:

*bash
python -m venv .venv


### 2. Virtuelle Umgebung aktivieren

Unter Windows:

*bash
.venv\Scripts\activate


Unter macOS/Linux:

*bash
source .venv/bin/activate


### 3. Benötigte Bibliotheken installieren

Aus dem Ordner code/:

*bash
cd code
python -m pip install -r requirements.txt





## Ausführungsreihenfolge

Für eine vollständige Reproduktion wird folgende Reihenfolge empfohlen:

### Schritt 1: Notebook ausführen: parkinson_thesis_reproducible.ipynb

Das Notebook erzeugt die zentralen Ergebnisse der Arbeit, darunter Metriken, Heatmaps, SHAP-Tabellen, Abbildungen und Modellartefakte.

Wichtig: Das Notebook muss vor dem Dashboard und vor dem Risikostratifizierungs-Skript ausgeführt werden, da beide auf die exportierten Modellartefakte angewiesen sind.

### Schritt 2: Dashboard-Artefakte prüfen

Nach Ausführung des Notebooks sollte folgender Ordner vorhanden sein:
code/dashboard_assets_full/


Er muss mindestens folgende Dateien enthalten:
xgb_model_full.pkl
all_features.pkl
test_data_full.pkl
y_scaler.pkl
peptide_map.pkl

### Schritt 3: Risikostratifizierungs-Outputs erzeugen

Aus dem Ordner code/:
python risikostratifizierung_outputs.py

Die erzeugten Tabellen und Abbildungen werden im Ordner `thesis_outputs_risikostratifizierung/` gespeichert.

### Schritt 4: Streamlit-Dashboard starten

Aus dem Ordner code/:
*bash
streamlit run app_full.py

## Optional: API-Schlüssel für den erklärenden CDSS-Assistenten

Das Dashboard enthält optional einen erklärenden CDSS-Assistenten, der über eine externe Gemini-Schnittstelle angesprochen werden kann. Dieser Teil ist nicht erforderlich, 
um die im Notebook berichteten Modellmetriken, Tabellen, Abbildungen oder Dashboard-Visualisierungen zu reproduzieren.

Für eine lokale Ausführung des Chatbot-Moduls muss ein eigener gültiger API-Schlüssel in einer `.env`-Datei hinterlegt werden:
GOOGLE_API_KEY=Ihr_eigener_API_Schluessel




