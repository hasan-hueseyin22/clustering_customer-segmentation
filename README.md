# Customer Segmentation using Clustering Algorithms

## 📝 Projektbeschreibung

Dieses Projekt zielt darauf ab, Kundengruppen in einem Einkaufszentrum zu identifizieren. Durch die Anwendung von Clustering-Algorithmen auf Kundendaten (Jahreseinkommen und Ausgabepunktzahl) werden verschiedene Segmente aufgedeckt. Diese Segmentierung kann für gezielte Marketingkampagnen und personalisierte Angebote genutzt werden.

Der Fokus liegt auf dem **Vergleich verschiedener Clustering-Verfahren**, um deren Effektivität für diesen Anwendungsfall zu bewerten.

**Verglichene Algorithmen:**
1.  **K-Means:** Ein zentroidbasiertes Verfahren, das Datenpunkte dem nächstgelegenen Cluster-Mittelpunkt zuordnet.
2.  **DBSCAN:** Ein dichtebasiertes Verfahren, das Cluster als Bereiche hoher Dichte identifiziert und gut mit Rauschen umgehen kann.
3.  **Agglomerative Clustering:** Ein hierarchisches Verfahren, das eine Baumstruktur von Clustern aufbaut.

**Datensatz:** [Mall Customer Segmentation Data](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python) von Kaggle.

## 🛠️ Tech Stack

-   **Python**
-   **Pandas** für Datenmanipulation
-   **Scikit-learn** für die Implementierung der Clustering-Algorithmen und Metriken
-   **Matplotlib** & **Seaborn** für die Visualisierung der Ergebnisse

## 🚀 Installation und Ausführung

1.  **Repository klonen:**
    ```bash
    git clone [https://github.com/hasan-hueseyin22/clustering_customer-segmentation.git](https://github.com/hasan-hueseyin22/clustering_customer-segmentation.git)
    cd customer-segmentation-clustering
    ```

2.  **Kaggle API einrichten und Daten herunterladen:**
    -   Installiere die Kaggle-Bibliothek: `pip install kaggle`
    -   Lade deine `kaggle.json` API-Datei von deinem Kaggle-Account herunter (`"My Account"` -> `"Create New API Token"`).
    -   Platziere die `kaggle.json` Datei im `~/.kaggle/` Verzeichnis.
    -   Lade den Datensatz herunter und platziere ihn im richtigen Ordner:
    ```bash
    kaggle datasets download -d vjchoudhary7/customer-segmentation-tutorial-in-python -p data/raw/ --unzip
    ```

3.  **Virtuelle Umgebung erstellen und aktivieren (empfohlen):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Auf Windows: venv\Scripts\activate
    ```

4.  **Abhängigkeiten installieren:**
    ```bash
    pip install -r requirements.txt
    ```

5.  **Analyse ausführen:**
    ```bash
    python src/analysis.py
    ```


## 📊 Ergebnisse

Das Skript führt die folgenden Aktionen aus:
1.  Es wendet drei verschiedene Clustering-Algorithmen auf die skalierten Daten an.
2.  Es berechnet den **Silhouette Score** und den **Davies-Bouldin Index** für jeden Algorithmus, um die Qualität der Cluster objektiv zu vergleichen. Die Ergebnisse werden in der Konsole ausgegeben und in `results/comparison_metrics.csv` gespeichert.
3.  Es generiert und speichert 2D-Streudiagramme für jeden Algorithmus im Ordner `visualizations/`, die die gefundenen Kundensegmente farblich hervorheben.


## 📂 Repository-Struktur
```
customer-segmentation-clustering/
├── data/
├── results/
├── src/
│   ├── config.py
│   ├── data_preprocessing.py
│   └── analysis.py
├── visualizations/
├── .gitignore
├── README.md
└── requirements.txt
```