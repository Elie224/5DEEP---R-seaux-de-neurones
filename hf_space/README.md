---
title: Intel Scene Classifier
emoji: 🖼️
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
---

# Intel Scene Classifier

Cette Space deploye ton meilleur modele CNN exporte depuis le notebook.

## Classes predites
- buildings
- forest
- glacier
- mountain
- sea
- street

## Fichiers inclus
- model_package.pt : package PyTorch exporte (state_dict + metadonnees)
- app.py : interface Gradio + inference
- requirements.txt : dependances

## Notes
Le modele est charge sur CPU et effectue une prediction multi-classes avec probabilites.
