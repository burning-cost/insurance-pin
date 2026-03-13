"""
Tests for insurance_pin.networks module.
"""
import pytest
import torch
import numpy as np
from insurance_pin.networks import (
    centered_hard_sigmoid,
    ContinuousEmbedding,
    CategoricalEmbedding,
    FeatureEmbeddings,
    SharedInteractionNet,
    InteractionTokens,
)


# ─────────────────────────────────────────────
# centered_hard_sigmoid
# ─────────────────────────────────────────────

class TestCenteredHardSigmoid:
    def test_zero_maps_to_half(self):
        x = torch.tensor([0.0])
        out = centered_hard_sigmoid(x)
        assert torch.isclose(out, torch.tensor([0.5])), f"Expected 0.5 but got {out}"

    def test_minus_one_maps_to_zero(self):
        x = torch.tensor([-1.0])
        out = centered_hard_sigmoid(x)
        assert torch.isclose(out, torch.tensor([0.0]))

    def test_plus_one_maps_to_one(self):
        x = torch.tensor([1.0])
        out = centered_hard_sigmoid(x)
        assert torch.isclose(out, torch.tensor([1.0]))

    def test_large_negative_saturates_at_zero(self):
        x = torch.tensor([-100.0])
        out = centered_hard_sigmoid(x)
        assert out.item() == 0.0

    def test_large_positive_saturates_at_one(self):
        x = torch.tensor([100.0])
        out = centered_hard_sigmoid(x)
        assert out.item() == 1.0

    def test_output_range(self):
        x = torch.randn(1000)
        out = centered_hard_sigmoid(x)
        assert (out >= 0.0).all()
        assert (out <= 1.0).all()

    def test_not_torch_hardsigmoid(self):
        """Verify PIN formula differs from nn.Hardsigmoid."""
        x = torch.tensor([1.0])
        pin_val = centered_hard_sigmoid(x)
        hs = torch.nn.Hardsigmoid()
        torch_val = hs(x)
        # PIN: (1+1)/2 = 1.0; torch: (1+3)/6 = 0.667
        assert not torch.isclose(pin_val, torch_val, atol=1e-3)

    def test_gradient_flows(self):
        x = torch.tensor([0.0], requires_grad=True)
        out = centered_hard_sigmoid(x)
        out.backward()
        # At x=0, derivative of clamp((1+x)/2, 0, 1) is 0.5
        assert x.grad is not None
        assert torch.isclose(x.grad, torch.tensor([0.5]))

    def test_batch_input(self):
        x = torch.linspace(-2, 2, 100)
        out = centered_hard_sigmoid(x)
        assert out.shape == (100,)

    def test_monotone(self):
        x = torch.linspace(-5, 5, 200)
        out = centered_hard_sigmoid(x)
        diffs = out[1:] - out[:-1]
        assert (diffs >= 0).all()

    def test_midpoint_symmetry(self):
        """f(x) + f(-x) == 1 for all x (symmetry around (0, 0.5))."""
        x = torch.linspace(-3, 3, 50)
        f = centered_hard_sigmoid(x)
        f_neg = centered_hard_sigmoid(-x)
        assert torch.allclose(f + f_neg, torch.ones_like(f), atol=1e-6)


# ─────────────────────────────────────────────
# ContinuousEmbedding
# ─────────────────────────────────────────────

class TestContinuousEmbedding:
    def test_output_shape(self):
        emb = ContinuousEmbedding(embedding_dim=10, hidden_dim=20)
        x = torch.randn(32)
        out = emb(x)
        assert out.shape == (32, 10)

    def test_output_shape_2d_input(self):
        emb = ContinuousEmbedding(embedding_dim=10, hidden_dim=20)
        x = torch.randn(32, 1)
        out = emb(x)
        assert out.shape == (32, 10)

    def test_single_sample(self):
        emb = ContinuousEmbedding(embedding_dim=5, hidden_dim=8)
        x = torch.randn(1)
        out = emb(x)
        assert out.shape == (1, 5)

    def test_gradient_flows(self):
        emb = ContinuousEmbedding(embedding_dim=10, hidden_dim=20)
        x = torch.randn(8, requires_grad=True)
        out = emb(x).sum()
        out.backward()
        assert x.grad is not None

    def test_different_dims(self):
        for d in [4, 8, 16, 32]:
            emb = ContinuousEmbedding(embedding_dim=d, hidden_dim=d * 2)
            x = torch.randn(16)
            out = emb(x)
            assert out.shape == (16, d)


# ─────────────────────────────────────────────
# CategoricalEmbedding
# ─────────────────────────────────────────────

class TestCategoricalEmbedding:
    def test_output_shape(self):
        emb = CategoricalEmbedding(num_categories=10, embedding_dim=8)
        x = torch.randint(0, 10, (32,))
        out = emb(x)
        assert out.shape == (32, 8)

    def test_different_categories(self):
        emb = CategoricalEmbedding(num_categories=5, embedding_dim=6)
        x = torch.tensor([0, 1, 2, 3, 4])
        out = emb(x)
        # All 5 categories should produce different embeddings
        assert out.shape == (5, 6)
        # At least some should differ
        assert not torch.allclose(out[0], out[1])

    def test_gradient_flows(self):
        emb = CategoricalEmbedding(num_categories=5, embedding_dim=4)
        x = torch.tensor([0, 1, 2])
        out = emb(x).sum()
        out.backward()
        assert emb.embedding.weight.grad is not None


# ─────────────────────────────────────────────
# FeatureEmbeddings
# ─────────────────────────────────────────────

class TestFeatureEmbeddings:
    @pytest.fixture
    def features(self):
        return {
            "age": "continuous",
            "bm": "continuous",
            "area": 5,  # 5 categories
        }

    @pytest.fixture
    def embs(self, features):
        return FeatureEmbeddings(features=features, embedding_dim=10, hidden_dim=20)

    def test_embed_all_shapes(self, embs):
        x_dict = {
            "age": torch.randn(32),
            "bm": torch.randn(32),
            "area": torch.randint(0, 5, (32,)),
        }
        out = embs.embed_all(x_dict)
        assert set(out.keys()) == {"age", "bm", "area"}
        for name, tensor in out.items():
            assert tensor.shape == (32, 10), f"{name}: {tensor.shape}"

    def test_embed_feature_by_name(self, embs):
        x = torch.randn(16)
        out = embs.embed_feature("age", x)
        assert out.shape == (16, 10)

    def test_invalid_feature_spec(self):
        with pytest.raises(ValueError, match="spec must be"):
            FeatureEmbeddings(
                features={"x": "invalid_type"},
                embedding_dim=10,
                hidden_dim=20,
            )

    def test_feature_names_order(self, features, embs):
        assert embs.feature_names == ["age", "bm", "area"]

    def test_parameters_registered(self, embs):
        params = list(embs.parameters())
        assert len(params) > 0


# ─────────────────────────────────────────────
# SharedInteractionNet
# ─────────────────────────────────────────────

class TestSharedInteractionNet:
    @pytest.fixture
    def net(self):
        return SharedInteractionNet(embedding_dim=10, token_dim=8, layer1_dim=30, layer2_dim=20)

    def test_output_shape(self, net):
        phi_j = torch.randn(32, 10)
        phi_k = torch.randn(32, 10)
        e = torch.randn(32, 8)
        out = net(phi_j, phi_k, e)
        assert out.shape == (32, 1)

    def test_token_broadcast(self, net):
        phi_j = torch.randn(32, 10)
        phi_k = torch.randn(32, 10)
        e = torch.randn(8)  # 1D token, should broadcast
        out = net(phi_j, phi_k, e)
        assert out.shape == (32, 1)

    def test_gradient_flows(self, net):
        phi_j = torch.randn(32, 10, requires_grad=True)
        phi_k = torch.randn(32, 10, requires_grad=True)
        e = torch.randn(32, 8)
        out = net(phi_j, phi_k, e).sum()
        out.backward()
        assert phi_j.grad is not None
        assert phi_k.grad is not None

    def test_shared_weights(self, net):
        """Same network handles different pairs."""
        phi_j = torch.randn(4, 10)
        phi_k = torch.randn(4, 10)
        e1 = torch.randn(4, 8)
        e2 = torch.randn(4, 8)
        out1 = net(phi_j, phi_k, e1)
        out2 = net(phi_j, phi_k, e2)
        # Same inputs, different tokens -> different outputs
        assert not torch.allclose(out1, out2)

    def test_different_embedding_dims(self):
        net = SharedInteractionNet(embedding_dim=16, token_dim=12, layer1_dim=64, layer2_dim=32)
        phi_j = torch.randn(8, 16)
        phi_k = torch.randn(8, 16)
        e = torch.randn(8, 12)
        out = net(phi_j, phi_k, e)
        assert out.shape == (8, 1)

    def test_parameter_count(self):
        """Reference config: d=10, d0=10, d1=30, d2=20."""
        net = SharedInteractionNet(embedding_dim=10, token_dim=10, layer1_dim=30, layer2_dim=20)
        n_params = sum(p.numel() for p in net.parameters())
        # Layer sizes: (30, 20+10) + (20, 30) + (1, 20)
        expected = (30 * 20 + 30) + (20 * 30 + 20) + (1 * 20 + 1)
        assert n_params == expected, f"Expected {expected}, got {n_params}"


# ─────────────────────────────────────────────
# InteractionTokens
# ─────────────────────────────────────────────

class TestInteractionTokens:
    def test_n_pairs(self):
        tokens = InteractionTokens(n_features=4, token_dim=8)
        # 4*(4+1)/2 = 10 pairs
        assert tokens.tokens.shape == (10, 8)

    def test_9_features_gives_45_pairs(self):
        tokens = InteractionTokens(n_features=9, token_dim=10)
        assert tokens.tokens.shape == (45, 10)

    def test_get_token_shape(self):
        tokens = InteractionTokens(n_features=4, token_dim=8)
        t = tokens.get_token(0, 1)
        assert t.shape == (8,)

    def test_get_token_symmetry(self):
        """Token for (j,k) == token for (k,j)."""
        tokens = InteractionTokens(n_features=4, token_dim=8)
        t01 = tokens.get_token(0, 1)
        t10 = tokens.get_token(1, 0)
        assert torch.allclose(t01, t10)

    def test_diagonal_tokens(self):
        tokens = InteractionTokens(n_features=3, token_dim=5)
        for i in range(3):
            t = tokens.get_token(i, i)
            assert t.shape == (5,)

    def test_pair_indices_all_upper_triangle(self):
        tokens = InteractionTokens(n_features=4, token_dim=8)
        pairs = tokens.pair_indices()
        assert len(pairs) == 10
        for j, k in pairs:
            assert j <= k

    def test_different_pairs_have_independent_tokens(self):
        tokens = InteractionTokens(n_features=3, token_dim=8)
        t01 = tokens.get_token(0, 1)
        t02 = tokens.get_token(0, 2)
        # They should not be identically zero (random init)
        # and should be different
        assert not torch.allclose(t01, t02)

    def test_all_tokens_are_learnable(self):
        tokens = InteractionTokens(n_features=4, token_dim=8)
        # tokens.tokens should be nn.Parameter
        assert isinstance(tokens.tokens, torch.nn.Parameter)

    def test_gradient_through_tokens(self):
        tokens = InteractionTokens(n_features=3, token_dim=5)
        t = tokens.get_token(0, 1)
        loss = t.sum()
        loss.backward()
        assert tokens.tokens.grad is not None
