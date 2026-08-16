# Recommendation Engine

A two-stage recommendation system that retrieves relevant users using interests and geographic proximity, then ranks those candidates with a neural two-tower model.

The system is designed around a simple principle: **do not run the expensive ranking model against the entire user population**. Instead, Stage A narrows the search space to a small candidate set, and Stage B performs the more expressive ranking.

---

## 1. Overview

The recommendation pipeline has two main stages:

### Stage A — Candidate Retrieval

The retrieval layer reduces the full user base to approximately 100 candidates using:

* An inverted index over user interests
* Geographic bucketing using 5° latitude/longitude cells
* Jaccard similarity between interest sets
* Haversine geographic distance
* A weighted retrieval score

### Stage B — Neural Ranking

The retrieved candidates are passed to a two-tower neural network that:

* Encodes the target user and candidate into the same embedding space
* L2-normalizes the embeddings
* Calculates their dot-product similarity
* Applies a learned temperature/logit scale
* Produces the final affinity score

The final API returns the highest-ranked candidates.

---

## 2. Architecture

```text
                    User Profile
                         │
                         ▼
              ┌─────────────────────┐
              │ Stage A: Retrieval  │
              ├─────────────────────┤
              │ Interest Index      │
              │ Geo Bucketing       │
              │ Candidate Filtering │
              │ Jaccard + Geo Score │
              └──────────┬──────────┘
                         │
                    Top 100
                         │
                         ▼
              ┌─────────────────────┐
              │ Stage B: Ranking    │
              ├─────────────────────┤
              │ Feature Encoding    │
              │ Two-Tower MLP       │
              │ Normalized Embeds   │
              │ Dot Product         │
              │ Learned Scale       │
              └──────────┬──────────┘
                         │
                         ▼
                 Top-K Recommendations
                         │
                         ▼
                    FastAPI API
```

The same `RecommendationService` is used by both the API and the offline test script. This avoids having two separate recommendation implementations that could behave differently.

---

# 3. Dataset Processing

The raw dataset contains user information such as:

* User ID
* Name
* Gender
* Date of birth
* Interests
* City
* Country

The raw dataset does not directly provide latitude and longitude.

Before retrieval or ranking, the data goes through a preprocessing pipeline.

### Interest parsing

Interests arrive as string representations rather than Python lists.

For example:

```text
'Gaming', 'Travel', 'Music'
```

These are converted into a list:

```python
["Gaming", "Travel", "Music"]
```

Duplicate interests are removed while preserving their original order.

### Age calculation

Age is derived from the DOB using a fixed reference date:

```text
2026-08-16
```

Using a fixed reference date keeps the preprocessing deterministic. Running the same pipeline later will not silently change someone's age simply because the system date changed.

### Geographic coordinates

The dataset contains city/country text rather than coordinates.

I resolve coordinates offline using `geonamescache`.

This has two advantages:

1. No external geocoding API is required.
2. The pipeline remains reproducible without network access.

If an exact city match cannot be found, the implementation falls back to the largest known city for that country from the GeoNames data.

This is a practical fallback rather than an exact geographic representation, so it can introduce some noise for smaller or unusual locations.

Rows that cannot be assigned coordinates, have no usable interests, or have an invalid DOB are removed because geographic and interest information are required by the recommendation pipeline.

---

# 4. Train / Validation / Test Split

The processed users are divided into:

```text
80% → Training
10% → Validation
10% → Test
```

The split is deterministic.

Instead of using `random.sample`, the pipeline hashes each `user_id` using MD5 and maps the resulting value into one of the three buckets.

This means the same user always goes into the same split.

It also means I don't need to store a separate split file to reproduce the experiment.

---

# 5. Stage A — Candidate Retrieval

The retrieval implementation is in:

```text
app/retrieval/candidate_generator.py
```

The objective is to reduce the full user population to a small number of potentially relevant users before invoking the neural ranker.

## 5.1 Interest inverted index

I create an inverted index:

```text
interest → set of user IDs
```

For example:

```text
Travel   → {12, 45, 91, 104, ...}
Gaming   → {8, 12, 37, 91, ...}
Music    → {4, 12, 18, 73, ...}
```

For a target user, I can therefore immediately retrieve users who share at least one interest.

This avoids calculating interest similarity against every user in the dataset.

The main reason for using this approach is that the number of interest categories is small and individual users generally have only a few interests. As a result, the interest overlap is sparse.

---

## 5.2 Geographic bucketing

Users are also placed into geographic cells.

The cell size is:

```text
5° latitude × 5° longitude
```

For a target user, the retrieval system checks the user's cell and its eight neighboring cells.

This provides a cheap geographic filter before calculating exact Haversine distance.

The idea is that someone geographically very far away is unlikely to be an immediately useful candidate when the system is specifically incorporating geographic proximity into its ranking.

---

## 5.3 Candidate pool logic

The retrieval stage first calculates:

```text
interest candidates
geo candidates
```

It then prefers candidates that satisfy **both** conditions:

```text
interest_pool ∩ geo_pool
```

If that intersection contains fewer than the requested number of candidates, the implementation falls back to:

```text
interest_pool ∪ geo_pool
```

If there still aren't enough candidates, it finally falls back to the complete user population.

This makes the retrieval stage robust to sparse regions or unusual interest combinations.

The target user itself is always removed from the candidate pool.

---

# 6. Retrieval Scoring

Once the candidate pool has been created, each candidate is scored using:

```text
retrieval_score =
    0.6 × interest_similarity
    + 0.4 × geographic_similarity
```

### Interest similarity

Interest similarity is calculated using Jaccard similarity:

```text
J(A, B) = |A ∩ B| / |A ∪ B|
```

For example:

```text
User A = {music, travel, gaming}
User B = {music, travel, fitness}
```

Then:

```text
intersection = {music, travel}
union        = {music, travel, gaming, fitness}

Jaccard = 2 / 4 = 0.5
```

### Geographic similarity

The actual geographic distance is calculated using the Haversine formula.

The distance is converted into a smoothly decaying score:

```text
geo_score = 1 / (1 + distance_km / 500)
```

At approximately 500 km, the geographic score is around 0.5.

The final retrieval score is therefore:

```text
0.6 × Jaccard
+
0.4 × geographic score
```

The highest-scoring candidates are returned.

The default candidate pool is:

```text
100 users
```

---

# 7. Why This Retrieval Approach?

The dataset is only around 25,000 users.

At this size, I felt that introducing a vector database or approximate nearest-neighbor system at the retrieval stage would add infrastructure without providing much practical benefit.

The inverted interest index and geographic buckets reduce the candidate set first, after which scoring the remaining candidates is inexpensive.

This gives a relatively simple implementation while still avoiding a full brute-force comparison.

For millions of users, I would change this architecture. That is discussed later in the scaling section.

---

# 8. Stage B — Neural Ranking

The neural ranking implementation is located under:

```text
app/ranking/
```

The model uses a two-tower architecture.

```text
Target User
     │
     ▼
 User Tower
     │
     ▼
 User Embedding
     │
     ├──── Dot Product ────► Affinity Score
     │
     ▲
Candidate Embedding
     │
     ▲
Candidate Tower
     │
     ▲
Candidate User
```

The two towers share weights.

This is possible because both inputs represent the same type of object: a user.

There isn't a fundamental difference between the target-user feature space and candidate-user feature space.

---

# 9. User Feature Representation

Each user is converted into a fixed-size vector.

The features are:

### Interests

Interests are represented using a multi-hot vector.

For example, if the vocabulary contains:

```text
Music
Travel
Gaming
Fitness
Finance
```

a user interested in Music and Gaming could be represented as:

```text
[1, 0, 1, 0, 0]
```

This works well here because the number of interest categories is relatively small.

### Gender

Gender is represented using a one-hot vector.

### Age

Age is normalized to the range `[0, 1]` using the configured range:

```text
13 to 90
```

### Location

Latitude and longitude are normalized to approximately:

```text
latitude / 90
longitude / 180
```

The resulting vector contains:

```text
[interest features]
[gender features]
[normalized age]
[normalized latitude]
[normalized longitude]
```

---

# 10. Two-Tower Model

Each user is passed through an MLP:

```text
Input Features
      │
      ▼
Linear Layer
      │
      ▼
ReLU
      │
      ▼
Dropout
      │
      ▼
Linear Layer
      │
      ▼
32-dimensional embedding
      │
      ▼
L2 normalization
```

The hidden dimension is 64 and the final embedding dimension is 32.

The target and candidate embeddings are compared using a dot product.

A learned `logit_scale` parameter acts as the temperature/scale applied to the similarity.

The model outputs raw logits, which are converted into probabilities using a sigmoid when producing recommendations.

---

# 11. Training Objective

The model is trained using:

```text
BCEWithLogitsLoss
```

However, there is an important limitation here.

The supplied dataset contains **no actual recommendation interaction labels**.

There is no information about:

* who followed whom
* who clicked whom
* profile views
* messages
* dwell time
* accepted connections
* explicit dislikes

Therefore, the model cannot currently learn from real user behavior.

---

# 12. Proxy Label

To make it possible to train the ranking model, I use a proxy affinity label:

```text
proxy_label =
    0.7 × interest_Jaccard
    + 0.3 × geographic_similarity
```

This produces a continuous target between 0 and 1.

Training pairs are generated from the users in each split, with a deterministic random seed.

The important limitation is that this proxy is based on the same underlying concepts already used during retrieval.

So the neural ranker is currently learning a smoother, nonlinear representation of those signals rather than discovering actual behavioral preferences.

I consider this the biggest limitation of the current implementation.

It is important to be explicit about this because a low validation loss on this proxy objective should not be interpreted as proof of production recommendation quality.

---

# 13. What I Would Use in Production

With actual product interaction data, I would replace the proxy target with behavioral signals.

For example:

```text
Profile impression  → weak signal
Profile click       → positive signal
Follow              → stronger positive signal
Message             → strong positive signal
Long dwell time     → positive signal
Explicit rejection  → negative signal
```

I would also construct negatives from users who were actually shown to the target user but were not interacted with.

That would make the training data much closer to the real recommendation problem.

The two-tower architecture could remain largely unchanged. The main improvement would be replacing the synthetic/proxy supervision with actual behavioral supervision.

---

# 14. Retrieval vs Ranking

The two stages intentionally have different responsibilities.

### Retrieval

Optimized for:

* speed
* high recall
* inexpensive filtering
* reducing the search space

It uses:

```text
Interests
+
Geography
```

### Ranking

Optimized for:

* finer-grained scoring
* learning nonlinear relationships
* future personalization

It uses:

```text
Interests
+
Gender
+
Age
+
Location
```

This separation makes it possible to improve the ranker without completely rewriting the retrieval system.

---

# 15. Recommendation Service

The central integration is:

```text
app/pipeline_service.py
```

`RecommendationService` loads:

* all processed users
* the retrieval indexes
* the feature vocabulary
* the trained two-tower checkpoint

These are loaded once when the service starts.

For every recommendation request:

```text
Target user
    │
    ▼
CandidateGenerator
    │
    ▼
100 candidates
    │
    ▼
Feature encoding
    │
    ▼
Two-tower ranking
    │
    ▼
Sigmoid affinity scores
    │
    ▼
Top-K results
```

The default output contains the top 10 recommendations.

Each result contains:

```text
user_id
name
city
country
shared_interests
affinity_score
```

---

# 16. Test User

The offline test user is defined in:

```text
scripts/run_test_user.py
```

The current example profile is:

```text
Name: Test Candidate
Gender: Male
Age: 24
Interests: Technology, Gaming, Music, Travel
City: Delhi
Country: India
Latitude: 28.6139
Longitude: 77.2090
```

Before final submission, I would update this profile to the actual profile I want to use as the test user and regenerate the sample results.

This ensures that the README and `sample_results.csv` describe the same input.

---

# 17. Setup

## Requirements

The project requires Python and the dependencies listed in:

```text
requirements.txt
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it.

### Linux / macOS

```bash
source .venv/bin/activate
```

### Windows

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# 18. Run the Full Training Pipeline

From the project root, run:

```bash
python -m app.ranking.train --csv data/Assessment_TwitterDataset.csv --epochs 8
```

This command performs the complete preprocessing and training flow.

It:

1. Loads the raw CSV.
2. Parses and cleans interests.
3. Calculates age.
4. Resolves geographic coordinates.
5. Removes unusable rows.
6. Creates the deterministic 80/10/10 split.
7. Writes processed Parquet files.
8. Builds the feature vocabulary.
9. Generates training pairs.
10. Trains the two-tower model.
11. Evaluates validation loss after each epoch.
12. Saves the best validation checkpoint.

The processed files are written to:

```text
artifacts/processed/
```

The feature vocabulary is saved to:

```text
artifacts/vocab.json
```

The best model checkpoint is saved to:

```text
artifacts/two_tower.pt
```

The training script chooses the checkpoint with the lowest validation loss.

---

# 19. Run the Offline Recommendation Pipeline

After training, run:

```bash
PYTHONPATH=. python scripts/run_test_user.py
```

This uses the exact same `RecommendationService` used by the API.

It runs:

```text
Test User
   ↓
Candidate Retrieval
   ↓
Top 100 Candidates
   ↓
Neural Ranking
   ↓
Top 10 Recommendations
```

The results are written to:

```text
sample_results.csv
```

The script also prints the test-user profile and the location of the generated results file.

---

# 20. Run the API

Start FastAPI with:

```bash
uvicorn app.api.main:app --reload
```

The API will run at:

```text
http://localhost:8000
```

---

# 21. API Endpoints

## Health Check

```http
GET /health
```

Example:

```bash
curl http://localhost:8000/health
```

The response contains the service status and number of users loaded.

---

## Recommend for Existing User

```http
GET /recommend/{user_id}
```

Example:

```bash
curl "http://localhost:8000/recommend/123?top_k=10"
```

The service finds the user in the processed user directory and generates recommendations.

---

## Recommend for a New User

```http
POST /recommend
```

The request accepts:

```json
{
  "name": "Rohan Verma",
  "gender": "Male",
  "age": 23,
  "interests": [
    "Technology",
    "Gaming",
    "Music",
    "Fitness"
  ],
  "city": "Delhi",
  "country": "India",
  "latitude": 28.6139,
  "longitude": 77.2090,
  "top_k": 10
}
```

This endpoint is useful for recommending users who are not already present in the dataset.

---

# 22. Run Tests

Run the automated tests with:

```bash
pytest tests/ -q
```

The tests cover important parts of the system including:

* Haversine distance
* Candidate generation
* Candidate filtering
* Shared-interest retrieval
* Feature encoding
* Two-tower tensor shapes

---

# 23. Project Structure

```text
reco-engine/
│
├── app/
│   ├── api/
│   │   └── main.py
│   │
│   ├── data/
│   │   ├── geocode.py
│   │   ├── pipeline.py
│   │   └── schema.py
│   │
│   ├── ranking/
│   │   ├── dataset.py
│   │   ├── features.py
│   │   ├── model.py
│   │   └── train.py
│   │
│   ├── retrieval/
│   │   └── candidate_generator.py
│   │
│   └── pipeline_service.py
│
├── scripts/
│   └── run_test_user.py
│
├── tests/
│   └── test_pipeline.py
│
├── data/
│   └── Assessment_TwitterDataset.csv
│
├── artifacts/
│   ├── processed/
│   ├── vocab.json
│   └── two_tower.pt
│
├── sample_results.csv
├── requirements.txt
└── README.md
```

---

# 24. Main Design Decisions

## Why an inverted index?

Interests are sparse.

If a user has only a few interests, there is no reason to calculate detailed similarity against every user.

The inverted index lets me jump directly to users who share at least one interest.

---

## Why geographic buckets?

Calculating exact geographic distance for the entire population would be unnecessary work.

The 5° geographic buckets provide a cheap first-stage locality filter.

Exact Haversine distance is then calculated only for the reduced candidate pool.

---

## Why 100 retrieval candidates?

The goal of Stage A is recall rather than final ranking.

I want enough candidates for Stage B to have a reasonable selection to rank, without sending the entire user base through the neural model.

100 is a simple and reasonable operating point for this dataset and can be tuned using offline recall/quality measurements.

---

## Why a two-tower architecture?

A concatenated MLP could also learn a pairwise score.

I chose a two-tower model because it gives a better path to production scaling.

Candidate embeddings can eventually be precomputed and stored in an ANN index.

At serving time, the target user can be encoded once and compared against candidate embeddings efficiently.

---

# 25. Limitations

There are three limitations I would specifically call out.

### 1. No real interaction labels

This is the largest limitation.

The ranking model currently learns from a proxy similarity label rather than real user behavior.

The model therefore demonstrates the ranking architecture, but its current validation results should not be interpreted as evidence that it predicts actual follows or user engagement.

### 2. Geographic fallback

Some cities may not match the local GeoNames data exactly.

The fallback uses the largest known city for the corresponding country, which can introduce significant distance error for smaller towns.

### 3. No ANN or caching layer

The current system keeps the entire retrieval structure in memory.

That is reasonable for approximately 25,000 users but would not be the architecture I would deploy for tens or hundreds of millions of users.

---

# 26. Scaling to Millions of Users

At a much larger scale, I would keep the overall two-stage design but replace the individual components with scalable equivalents.

A possible architecture would be:

```text
                    User Request
                         │
                         ▼
                 Feature Service
                         │
                         ▼
               Geo-aware Retrieval
                         │
                         ▼
                  ANN Index
                FAISS / ScaNN
                         │
                         ▼
                    Top-K
                         │
                         ▼
                Ranking Service
                         │
                         ▼
                  Final Results
```

### Candidate embeddings

Candidate embeddings could be generated periodically and stored in an ANN index.

They could be:

* rebuilt nightly
* updated incrementally
* refreshed through a streaming pipeline

### Geographic sharding

The candidate index could also be divided by geographic region.

A request would first identify the relevant region and then search only the appropriate shards.

### Stateless ranking service

The ranking model could be deployed as a stateless service behind a load balancer.

This would allow horizontal scaling as recommendation traffic increases.

### Caching

Frequently requested candidate embeddings and user features could also be cached.

---

# 27. Production Improvements

If I were taking this beyond the assignment, I would prioritize improvements in roughly this order:

### 1. Collect real behavioral labels

This would provide the biggest improvement in recommendation quality.

### 2. Improve negative sampling

Instead of randomly selecting negatives, I would use impression-based negatives: users that were actually shown but ignored.

### 3. Add proper ranking metrics

I would evaluate:

```text
Recall@K
Precision@K
NDCG@K
MRR
```

and compare the neural ranker against the Stage A heuristic baseline.

### 4. Precompute candidate embeddings

This would make the two-tower design more useful at serving time.

### 5. Introduce ANN retrieval

FAISS or ScaNN would become appropriate once the candidate population becomes sufficiently large.

### 6. Improve geocoding

A production geocoding source with better city-level coverage would reduce geographic noise.

### 7. Add monitoring

I would monitor:

* retrieval latency
* ranking latency
* candidate recall
* recommendation diversity
* engagement metrics
* model drift
* feature distribution changes

### 8. Online experimentation

Once behavioral labels are available, I would use A/B testing to measure whether ranking changes actually improve user outcomes.

---

# 28. AI Assistance

I used Claude as a coding assistant during the initial scaffolding and implementation of this assignment.

It assisted with parts of the implementation including:

* retrieval scaffolding
* two-tower model implementation
* data pipeline structure
* FastAPI wiring
* supporting code

I reviewed and verified the implementation rather than treating generated code as automatically correct.

The architectural decisions I focused on were:

* separating retrieval from ranking
* using an inverted interest index
* using geographic bucketing
* keeping the retrieval implementation lightweight for the current dataset size
* using a two-tower architecture for future scalability
* making the train/validation/test split deterministic
* explicitly documenting the lack of real interaction labels

The proxy-label limitation is particularly important. In a real recommendation system, I would replace the synthetic similarity target with actual behavioral signals before considering the model production-ready.

---

# 29. Reproducibility

The pipeline is designed to be reproducible.

Important sources of determinism include:

* Fixed reference date for age calculation
* MD5-based train/validation/test split
* Fixed random seeds for training-pair generation
* Offline geocoding using `geonamescache`
* Saved feature vocabulary
* Saved best model checkpoint

This means the same source data and configuration should produce consistent preprocessing and dataset splits.

---

# 30. Quick Start

For someone evaluating the assignment, the shortest path is:

```bash
# Create environment
python -m venv .venv

# Activate environment
# Linux/macOS:
source .venv/bin/activate

# Windows:
# .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Build data + train ranker
python -m app.ranking.train --csv data/Assessment_TwitterDataset.csv --epochs 8

# Generate recommendations for the test user
PYTHONPATH=. python scripts/run_test_user.py

# Run tests
pytest tests/ -q

# Start API
uvicorn app.api.main:app --reload
```

After running the pipeline, the main generated artifacts are:

```text
artifacts/processed/
artifacts/vocab.json
artifacts/two_tower.pt
sample_results.csv
```

---

# 31. Final Takeaway

The main design choice in this project is the separation between **candidate retrieval** and **neural ranking**.

For the current dataset size, a lightweight inverted index plus geographic bucketing provides an efficient way to reduce the search space without introducing unnecessary ANN infrastructure.

The two-tower ranker then provides a more flexible scoring layer and, importantly, gives the system a natural path toward large-scale ANN-based serving.

The biggest limitation is the lack of real interaction data. The current model is trained using a proxy affinity target based on interest overlap and geographic proximity. I would consider replacing that proxy with actual behavioral data the first major step toward making this a production recommendation system.
