# MaterialMatch

AI waste-to-feedstock matching for Bengaluru. A waste generator uploads a photo, the model identifies
the material, and the app suggests nearby recyclers and estimates the CO₂e saved by recycling it.
Manufacturers can search listed waste in plain words.

**[Live demo](https://materialmatch.streamlit.app/)**

> **Demo data.** All 205 waste listings and the 10 recycler profiles are synthetic samples, not real
> businesses or partnerships. Most carbon factors are proxies (EPA WARM) and are flagged in the app.

## Features

- **I have waste**: photo classification (9 waste types, 27 sub-types), "unknown material" detection,
  recycler matching, CO₂e estimate, publish a listing.
- **I need feedstock**: semantic search over listings, ranked by relevance, distance and quality, with
  cost and CO₂e for the quantity needed.
- **Impact**: potential CO₂e of the listings by zone and by sub-type (proxy-based estimates hatched), the
  biggest lever in the current selection, and the carbon factor table with sources.
- **Feedback loop**: users confirm or correct predictions; confirmed photos adjust similar future
  predictions right away and can be used for a gated retrain.

## Architecture

```mermaid
flowchart LR
    U[User] --> FE[Streamlit app<br/>frontend/]
    C[API client] --> BE[FastAPI<br/>backend/]

    FE --> ML
    BE --> ML

    subgraph ML[ml/]
        CLS[Photo classifier<br/>YOLOv8s-cls + kNN unknown check]
        MEM[Feedback memory]
        EMB[Listing search<br/>MiniLM embeddings]
        MAT[Matcher<br/>distance, capacity, specialism]
        CO2[Carbon calculator<br/>EPA WARM factors]
    end

    CLS --- MEM
    ML --> CSV[(data/raw CSVs<br/>listings, recyclers, factors)]
    FE --> FB[(data/feedback<br/>user corrections)]
    BE --> FB
    FB -. optional sync .-> HF[(Private Hugging Face<br/>dataset)]

    subgraph Training[Offline training]
        DS[Image datasets] --> TR[train_yolo.py] --> W[weights/best.pt]
        TR -. optional .-> KG[Kaggle GPU]
    end
    W --> CLS
```

## Quick start

Requires Python 3.11.

```bash
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt    # CPU PyTorch index is set inside the file

streamlit run frontend/app.py      # web app
uvicorn backend.main:app --reload  # API, docs at http://127.0.0.1:8000/docs
```

The trained model (`ml/classifier/weights/best.pt`) is included. The MiniLM text model downloads on
first use.

**Security.** The API has no user accounts. Before exposing it beyond `127.0.0.1`, set `MM_API_KEY`;
`POST /listings` and `POST /feedback` then require the header `X-API-Key: <key>`. Uploads are limited
to 10 MB per photo and 80 megapixels, and only real JPG, PNG, WEBP or HEIC photos are stored.

**Deploying (Streamlit Community Cloud).** Main file `frontend/app.py`, Python 3.11. The host wipes its
disk on every reboot, so to keep user feedback (and the feedback memory) add two secrets:
`HF_TOKEN` (a Hugging Face write token) and `MM_FEEDBACK_REPO = "<hf-user>/materialmatch-feedback"`.
`data/feedback/` is then restored at startup and every answer is uploaded to that dataset, which is
created private; a public repo is refused. Without the secrets feedback stays on the local disk. Stored
feedback photos are capped at 1 GB (`MM_FEEDBACK_MAX_MB`), and users are told before answering that the photo
is saved.

## Project layout

```
frontend/   Streamlit pages (I have waste, I need feedstock, Impact) and visuals.py
backend/    FastAPI routes: listings, match, classify, feedback
ml/         taxonomy, carbon, embeddings, matcher, evaluation
ml/classifier/   prediction, unknown detection, feedback memory, shipped weights
ml/classifier/tools/   dataset build, training, calibration, evaluation (dev only)
data/raw/   synthetic listings, sample recyclers, carbon factors
```

## Model

YOLOv8s-cls trained on clean item photos plus conveyor-belt crops, with blur, noise, JPEG, low-light
and heavy-crop augmentation so it copes with poor phone photos.

| Test | Result |
|---|---|
| Held-out test split (4,944 images) | 94.5% waste type, 91.7% sub-type |
| Conveyor-belt objects never seen in training (ZeroWaste test split, 5,074) | 87.3% |
| Corrupted photos (blur, noise, JPEG, low light) | 91.6%; 82.1% on corruption types not used in training |

Unfamiliar photos are flagged as "Please confirm" or "Other / unknown", and every prediction is
confirmed by the user before publishing. Bales and bulk loads (for example textile bales) are still
the weakest case.

Retrain (`pip install -r requirements-dev.txt`): `python ml/classifier/tools/train_yolo.py --auto` (CPU)
or `python -m ml.classifier.tools.kaggle_runner train` (Kaggle GPU, needs your own Kaggle API token).

## Datasets and licences

- CODD: Demetriou et al., Construction and Demolition Waste Object Detection Dataset, Mendeley Data,
  doi:10.17632/wds85kt64j.3 (CC BY 4.0)
- Garbage Dataset v2: Suman Kunwar, Kaggle (MIT)
- E Waste Image Dataset: Akshat Tamrakar, Kaggle (Apache 2.0)
- ZeroWaste: Bashkirova et al., ZeroWaste Dataset: Towards Deformable Object Segmentation in Cluttered
  Scenes, CVPR 2022 (conveyor-belt crops from its train and val splits; licence per the dataset page)
- Carbon factors: US EPA WARM v13 (recycling and composting chapters), Cotton Incorporated LCA

Image datasets are not included in this repository.

## Limitations

- Listings and recycler profiles are synthetic; nothing here represents real trade data.
- Carbon factors are US-based estimates, not India-specific LCA results.
- The classifier has no polymer labels (PET, HDPE, PP) and is still weak on bales and bulk loads.
