from types import SimpleNamespace

import pytest
import torch

from slime.utils import megatron_bridge_utils


NUM_GPUS = 0


def make_model(tied, config_has_sharing):
    model = torch.nn.Module()
    model.backbone = torch.nn.Linear(3, 3)
    model.output_layer = torch.nn.Linear(3, 1)
    model.config = SimpleNamespace()
    model.share_embeddings_and_output_weights = tied
    if config_has_sharing:
        model.config.share_embeddings_and_output_weights = tied
    return model


@pytest.mark.parametrize("tied", [False, True])
@pytest.mark.parametrize("config_has_sharing", [False, True])
@pytest.mark.parametrize("loading_fails", [False, True])
def test_hf_critic_load_preserves_value_head(monkeypatch, tied, config_has_sharing, loading_fails):
    models = [make_model(tied, config_has_sharing) for _ in range(2)]
    monkeypatch.setattr(megatron_bridge_utils, "unwrap_model", lambda value: value)
    heads = [model.output_layer for model in models]
    initial_weights = [head.weight.detach().clone() for head in heads]
    optimizer = torch.optim.SGD([p for model in models for p in model.parameters()], lr=0.1)
    original_order = [name for name, _ in models[0].named_parameters()]

    def load_backbone():
        with megatron_bridge_utils.patch_megatron_model(models, skip_output_layer=True):
            assert not models[0].config.share_embeddings_and_output_weights
            for model in models:
                parameters = dict(model.named_parameters())
                assert "backbone.weight" in parameters
                assert "output_layer.weight" not in parameters
                assert "output_layer.bias" not in parameters
                with torch.no_grad():
                    parameters["backbone.weight"].fill_(7)
            if loading_fails:
                raise RuntimeError("HF load failed")

    if loading_fails:
        with pytest.raises(RuntimeError, match="HF load failed"):
            load_backbone()
    else:
        load_backbone()

    for model, head, initial in zip(models, heads, initial_weights, strict=True):
        assert model.output_layer is head
        assert torch.equal(head.weight, initial)
        assert torch.all(model.backbone.weight == 7)
        assert any(p is head.weight for p in optimizer.param_groups[0]["params"])
    assert [name for name, _ in models[0].named_parameters()] == original_order
    assert hasattr(models[0].config, "share_embeddings_and_output_weights") == config_has_sharing
    if config_has_sharing:
        assert models[0].config.share_embeddings_and_output_weights == tied


def test_actor_hf_load_keeps_lm_head_visible(monkeypatch):
    model = make_model(False, False)
    monkeypatch.setattr(megatron_bridge_utils, "unwrap_model", lambda value: value)
    with megatron_bridge_utils.patch_megatron_model([model]):
        assert "output_layer.weight" in dict(model.named_parameters())
        assert not model.config.share_embeddings_and_output_weights
    assert not hasattr(model.config, "share_embeddings_and_output_weights")
