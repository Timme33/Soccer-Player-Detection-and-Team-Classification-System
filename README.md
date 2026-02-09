# Soccer Player Detection and Team Classification (Backend)

This repository contains the **backend computer vision pipeline** for a real-time soccer player detection and team classification system developed during **HackUMass XIII**.

The backend performs player detection and separates players into teams using visual features, producing annotated outputs for each team.

---

## Overview

The goal of this project is to process soccer match imagery and:
1. Detect all players in a frame
2. Classify detected players into two teams
3. Output separate visualizations for each team

This repository contains **only the backend code**, which implements the core computer vision logic.  
The frontend is not included.

---

## Example Output

**Input Frame**

![Input frame](assets/CityVSLiv.jpg)

**Detected Players by Team**

| Team A | Team B |
|---|---|
| ![Team A output](assets/teamA_boxes_lines.jpg) | ![Team B output](assets/teamB_boxes_lines.jpg) |

Each output image shows bounding boxes and visual indicators drawn around the players belonging to a single team.

---

## How It Works

The backend implements a real-time computer vision pipeline with the following components:

- **Player Detection**  
  Players are detected using a YOLOv8 object detection model.

- **Team Classification**  
  Detected player regions are classified into teams using **color-based clustering**, leveraging visual differences in team kits.

- **Visualization Output**  
  The system generates separate annotated images for each team, drawing bounding
  boxes around detected players and connecting lines between players belonging
  to the same team.

---

## Backend Architecture

- **Language:** Python  
- **API Framework:** FastAPI  
- **Responsibilities:**
  - Model inference
  - Image preprocessing and postprocessing
  - Team classification logic
  - Output generation

The backend is designed to be **modular and extensible**, with future support in mind for:
- Player tracking across frames
- Higher-level analysis such as formation inference

---

## Notes

- This repository focuses on the **computer vision backend**, which constitutes the core technical contribution.
- The frontend implementation is intentionally excluded.

---