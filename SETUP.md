# Setup

```bash
python -m venv .venv && source .venv/bin/activate (macos)
pip install -r requirements.txt

# 1. Build features + train the ranker (also runs the data pipeline/split)
python -m app.ranking.train --csv data/Assessment_TwitterDataset.csv --epochs 8

# 2. Run the offline verification script -> writes sample_results.csv
PYTHONPATH=. python scripts/run_test_user.py

# 3. Run the API
uvicorn app.api.main:app --reload
# then: GET  http://localhost:8000/recommend/{user_id}
#       POST http://localhost:8000/recommend   (body = new user profile)

# 4. Run tests
pytest tests/ -q
```

Fold the relevant bits of this into your README's "Setup Instructions" section
in your own words.
