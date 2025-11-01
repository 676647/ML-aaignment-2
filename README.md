# ML-aaignment-2
ML aaignment 2

A) This project can be run in two distinct modes: 1. Dataset Lookup and 2. Real-Time MP3 Analysis.

B) Data and Model Details:

Total songs in the dataset: 114000 songs

Songs used for training: 35000 

Test set Split: 20% of the 35 000 songs were reserved for testing

Final Model: RFC

Achieved Accuracy: on validation: 61% + 90% ROC AUC. on test: 58% + 89% ROC AUC


C) How to run:
Method 1:
1. Run app.py
- Song choice from the dataset

Method 2: 
1. Run GenreDetectorV2 chapters 1,2,3,4 and 6 to ensure the final model is saved as model.pkl
2. Run RuccoBeatsAPI
- Mainly for demonstration.
- API Constraint: This method uses the Reccobeats API (Spotify no longer allows developers to use their Web API) and is limited to MP3 files under 5MB. The Reccobeats API also truncates audio analysis to the first 30 seconds, leading to a significant drop in accuracy because the features are not consistent with the full-song features used for training. For instance a song can be pop or rap but if there is a guitar played only in the first 30 seconds it will result in the model to label the song as Country. 
