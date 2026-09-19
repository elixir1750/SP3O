from contextlib import contextmanager

try:
    from megatron.core.utils import unwrap_model
except ImportError:
    unwrap_model = None


@contextmanager
def patch_megatron_model(model, *, skip_output_layer=False):
    unwrapped_models = unwrap_model(model)
    unwrapped_model = unwrapped_models[0]
    model_config = unwrapped_model.config
    attribute_was_added = False
    if not hasattr(model_config, "share_embeddings_and_output_weights"):
        model_config.share_embeddings_and_output_weights = unwrapped_model.share_embeddings_and_output_weights
        attribute_was_added = True

    original_sharing = model_config.share_embeddings_and_output_weights
    output_layers = []
    try:
        if skip_output_layer:
            # A critic's scalar head has no corresponding HF LM-head weights.
            # Hide it from Bridge's parameter enumeration while retaining the
            # same Parameter objects in the optimizer and restoring them below.
            for chunk in unwrapped_models:
                if getattr(chunk, "output_layer", None) is not None:
                    output_layers.append((chunk, chunk.output_layer))
                    chunk.output_layer = None
            # Never broadcast token embeddings into the scalar value head.
            model_config.share_embeddings_and_output_weights = False
        yield
    finally:
        for chunk, output_layer in output_layers:
            chunk.output_layer = output_layer
        model_config.share_embeddings_and_output_weights = original_sharing
        if attribute_was_added:
            delattr(model_config, "share_embeddings_and_output_weights")
