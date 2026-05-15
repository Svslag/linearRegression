# Music Listener Segmentation Project

This is an **Unsupervised Learning** project.  
The model divides music listeners into groups using their listening habits.

## Topic
Music listener segmentation.

## Algorithm
KMeans Clustering.

## Dataset columns
- daily_minutes
- rock_percent
- pop_percent
- hiphop_percent
- playlist_count

## Project files
- `music_listeners.csv` — dataset
- `train_model.py` — trains the clustering model
- `test_model.py` — tests cluster prediction from terminal
- `app.py` — Flask web app
- `templates/index.html` — HTML page
- `static/styles.css` — CSS styles
- `requirements.txt` — needed libraries
- `clustered_music_listeners.csv` — dataset with cluster labels after training
- `music_cluster_model.joblib` — saved KMeans model after training
- `scaler.joblib` — saved scaler after training

## How to run in VS Code

### 1. Open the folder in VS Code
Open the folder `Music-Listener-Segmentation`.

### 2. Create virtual environment
```bash
python -m venv .venv
```

### 3. Activate virtual environment
Windows:
```bash
.venv\Scripts\activate
```

Mac/Linux:
```bash
source .venv/bin/activate
```

### 4. Install libraries
```bash
pip install -r requirements.txt
```

### 5. Train the model
```bash
python train_model.py
```

### 6. Test in terminal
```bash
python test_model.py
```

Example input:
```text
daily_minutes: 90
rock_percent: 20
pop_percent: 60
hiphop_percent: 10
playlist_count: 8
```

### 7. Run website
```bash
python app.py
```

Then open the local link from the terminal, usually:
```text
http://127.0.0.1:5000
```
