# Wildlife Re-Identification

Embedding-based individual identification for wildlife imagery. Given a query
photo of an animal, the system encodes it into a fixed-length embedding, searches
a [FAISS](https://github.com/facebookresearch/faiss) vector index of known
individuals, and returns the top-k most similar matches with similarity scores.

It is an **open-set** problem solved with **metric learning + vector retrieval**
rather than classification: new individuals can be enrolled by adding their
embedding to the gallery, with no retraining. The reference dataset is the public
[SeaTurtleID2022](https://www.kaggle.com/datasets/wildlifedatasets/seaturtleid2022)
benchmark, and the same code supports any dataset described by a metadata CSV or a
folder-per-individual layout.

## Why retrieval instead of classification

Wildlife monitoring is inherently open-set:

- new individuals appear continuously,
- most individuals have only a handful of images,
- a per-individual classifier head does not scale and cannot recognize an
  individual it never trained on.

The system instead learns an embedding space where images of the same individual
are close and different individuals are far apart, then identifies by
nearest-neighbor search. This is the same `embed → vector search → top-k` pattern
used in retrieval-augmented generation (RAG), applied to images.

## Architecture

```text
Query image
     │
     ▼
Embedding encoder  (EfficientNetV2-M backbone + projection head → 256-d, L2-normalized)
     │
     ▼
FAISS vector index  (gallery embeddings, cosine similarity)
     │
     ▼
Top-k matches + scores
     │
     ▼
FastAPI service  (/health, /search)
```

Any artifact path (gallery index, model checkpoint) may be a **local path** or a
`gs://` Cloud Storage URI, so the same code runs on a laptop and on Google Cloud.

## Project layout

```text
configs/              YAML configs (dataset, model, index, training, paths)
src/wildlife_reid/    Core library
  ├─ config.py          typed configuration
  ├─ storage.py         local / Cloud Storage path abstraction
  ├─ transforms.py      preprocessing + augmentation
  ├─ data/              dataset cataloguing, triplet mining, datasets
  ├─ models/            embedding encoder
  ├─ embedding.py       model loading + inference service
  ├─ index.py           FAISS vector index
  ├─ retrieval.py       build-gallery / search orchestration
  ├─ evaluation.py      top-1 / top-5 retrieval accuracy
  └─ training/          triplet-loss fine-tuning
scripts/              CLI entry points (build_index, search, evaluate, train_triplet)
app/                  FastAPI service
deploy/               Cloud Run deploy + artifact upload scripts
tests/               unit tests (network-free, GPU-free)
Dockerfile           container image for serving
```

## Installation

```bash
pip install -e .

# Optional: Google Cloud Storage support for gs:// artifact paths
pip install -e ".[gcp]"

# Optional: development tooling (ruff, pytest)
pip install -e ".[dev]"
```

Requires Python 3.10+.

## Dataset

Point `dataset.root` in `configs/sea_turtle.yaml` at your dataset directory. The
expected SeaTurtleID2022 layout is:

```text
<dataset.root>/
  metadata_splits.csv        # columns: file_name, identity, split_open
  images/<identity>/<image>.jpg
```

`file_name` entries in the CSV are paths relative to `dataset.root` (e.g.
`images/t001/abc.JPG`). The default config uses a relative `data/` location; edit
it to match where your dataset lives.

For datasets organized as one folder per individual, use the
`folder_per_identity` layout (see `configs/example_folder_dataset.yaml`) — the
commands are identical, only the config changes.

## Usage

### Build the gallery index

```bash
python scripts/build_index.py --config configs/sea_turtle.yaml \
  --output artifacts/sea_turtle/index
```

Writes `index.faiss`, `metadata.json`, and a `manifest.json` capturing the build
provenance (backbone, embedding dim, image size, checkpoint, counts, timestamp).
`--output` accepts a local path or a `gs://` URI.

### Search

```bash
python scripts/search.py --config configs/sea_turtle.yaml \
  --query path/to/query.jpg --top-k 5
```

### Evaluate

```bash
python scripts/evaluate.py --config configs/sea_turtle.yaml
```

Reports top-1 and top-5 retrieval accuracy on the held-out query split.

### Run the API

```bash
uvicorn app.api:app --reload --port 8080
```

```bash
curl http://localhost:8080/health
curl -X POST "http://localhost:8080/search?top_k=5" -F "file=@path/to/query.jpg"
```

### Fine-tune the encoder (optional)

```bash
python scripts/train_triplet.py --config configs/sea_turtle.yaml
```

Trains the embedding model with triplet loss and saves the best checkpoint. Set
`model.checkpoint` in the config to the resulting `best.pt` to use it for indexing
and serving.

## Design notes

- **Embeddings, not classification** — open-set identification by nearest-neighbor
  search; enroll new individuals without retraining.
- **EfficientNetV2-M backbone** with a projection head into a 256-d embedding;
  the backbone is a registry lookup, so swapping it is a one-line config change.
- **Cosine similarity on L2-normalized vectors** — inner product equals cosine
  similarity, giving interpretable scores in `[-1, 1]`.
- **FAISS flat (exact) index** — exact search with full recall at low-thousands
  gallery sizes; the index layer is isolated so an approximate (IVF/PQ/HNSW) or
  managed vector search can be substituted as the gallery grows.
- **Local / Cloud Storage parity** — a single storage abstraction lets every
  artifact path be local or `gs://`, so dev and cloud share one codebase.
- **Portable checkpoints** — saved as weights plus primitive hyperparameters
  (not a pickled config object), so they load across operating systems and survive
  refactors.

## Deployment (Google Cloud)

The service runs on **Cloud Run** with artifacts in **Cloud Storage**:

```bash
# Build the gallery index and upload it to Cloud Storage
pip install -e ".[gcp]"
./deploy/upload_artifacts.sh

# Build the image and deploy to Cloud Run (scales to zero)
./deploy/deploy.sh
```

The container bakes the ImageNet backbone weights into the image (so cold starts
skip the download) and loads the gallery index from Cloud Storage at startup.
Project, bucket, and region defaults live in `configs/gcp.env` and the `deploy/`
scripts; override them via environment variables to target your own project.

## Development

```bash
pip install -e ".[dev]"

ruff check .           # lint
ruff format --check .  # formatting
pytest                 # unit tests
```

Continuous integration (`.github/workflows/ci.yml`) runs lint, formatting, the
test suite, and a Docker build on every push and pull request. A pre-commit
config is provided (`pre-commit install`) to run the same checks locally.

## License

MIT
