import torch

from app.data.schema import UserRecord
from app.retrieval.candidate_generator import CandidateGenerator, _haversine_km
from app.ranking.features import FeatureVocab, encode_user, feature_dim
from app.ranking.model import TwoTowerModel


def _mock_users(n=50):
    interests_pool = ["Music", "Travel", "Gaming", "Fitness", "Art"]
    users = []
    for i in range(n):
        users.append(UserRecord(
            user_id=i, name=f"user{i}", gender="Male" if i % 2 == 0 else "Female",
            age=20 + (i % 40), interests=[interests_pool[i % len(interests_pool)],
                                           interests_pool[(i + 1) % len(interests_pool)]],
            city=f"city{i % 5}", country="TestLand",
            latitude=10.0 + (i % 5), longitude=20.0 + (i % 5),
        ))
    return users


def test_haversine_zero_distance():
    assert _haversine_km(10.0, 20.0, 10.0, 20.0) == 0.0


def test_candidate_generator_returns_k_or_fewer():
    users = _mock_users()
    gen = CandidateGenerator(users)
    target = users[0]
    candidates = gen.generate(target, k=10)
    assert len(candidates) <= 10
    assert target.user_id not in {c.user_id for c in candidates}


def test_candidate_generator_prefers_shared_interests():
    users = _mock_users()
    gen = CandidateGenerator(users)
    target = users[0]
    candidates = gen.generate(target, k=10)
    target_interests = {i.lower() for i in target.interests}
    top_candidate_interests = {i.lower() for i in candidates[0].interests}
    assert len(target_interests & top_candidate_interests) > 0


def test_feature_encoding_shape():
    users = _mock_users()
    vocab = FeatureVocab.from_records(users)
    vec = encode_user(users[0], vocab)
    assert vec.shape[0] == feature_dim(vocab)


def test_two_tower_forward_shapes():
    users = _mock_users()
    vocab = FeatureVocab.from_records(users)
    dim = feature_dim(vocab)
    model = TwoTowerModel(input_dim=dim)

    batch = torch.stack([encode_user(u, vocab) for u in users[:8]])
    logits = model(batch, batch)
    assert logits.shape == (8,)
