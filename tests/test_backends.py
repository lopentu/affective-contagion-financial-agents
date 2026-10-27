"""Transformers-API contract tests for the Hugging Face backend.

The backend is exercised through stub modules so the contract stays testable without downloading
weights. These tests pin two call conventions that changed silently across Transformers major
versions and that type checking cannot catch, because the backend imports Transformers lazily.
"""

from __future__ import annotations

import os
from collections import abc
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pytest

from affective_contagion.backends import huggingface as backend_module
from affective_contagion.generation import ChatMessage
from affective_contagion.roster import ModelSpec

HIDDEN_SIZE = 4
PROMPT_LENGTH = 3
CONTINUATION_IDS = (101, 102)
DECODER_DEPTH = 4


class StubTensor:
    """Minimal tensor stand-in supporting exactly the indexing the backend performs."""

    def __init__(self, values: list[list[int]]) -> None:
        """Store nested token ids and expose a two-dimensional shape."""
        self.values = values
        self.shape = (len(values), len(values[0]))
        self.device = "cpu"

    def to(self, device: str) -> StubTensor:
        """Record the requested device, mirroring `torch.Tensor.to`."""
        self.device = device
        return self

    def __getitem__(self, key: tuple[int, slice]) -> list[int]:
        """Return one row slice, mirroring `tensor[row, start:]`."""
        row, columns = key
        return self.values[row][columns]


class StubHidden:
    """Residual-stream stand-in for the forward hook."""

    def __init__(self, sequence_length: int) -> None:
        """Create a batch-of-one hidden state with a known fill value."""
        self.shape = (1, sequence_length, HIDDEN_SIZE)

    def __getitem__(self, key: tuple[int, int, slice]) -> StubHidden:
        """Return self so chained tensor-style access stays inspectable."""
        del key
        return self

    def detach(self) -> StubHidden:
        """Mirror `torch.Tensor.detach`."""
        return self

    def float(self) -> StubHidden:
        """Mirror `torch.Tensor.float`."""
        return self

    def cpu(self) -> StubHidden:
        """Mirror `torch.Tensor.cpu`."""
        return self

    def numpy(self) -> np.ndarray:
        """Return a deterministic vector the assertions can check."""
        return np.arange(HIDDEN_SIZE, dtype=np.float64)


class StubHandle:
    """Forward-hook handle recording removal."""

    def __init__(self) -> None:
        """Start attached."""
        self.removed = False

    def remove(self) -> None:
        """Detach the hook."""
        self.removed = True


class StubLayer:
    """Decoder block that fires its registered hook on a simulated prefill."""

    def __init__(self) -> None:
        """Start with no registered hook."""
        self.hook: Callable[[object, object, tuple[StubHidden, ...]], None] | None = None
        self.handle = StubHandle()

    def register_forward_hook(
        self,
        hook: Callable[[object, object, tuple[StubHidden, ...]], None],
    ) -> StubHandle:
        """Record the hook and return a removable handle."""
        self.hook = hook
        return self.handle

    def fire_prefill(self) -> None:
        """Invoke the hook, if one is attached, with a multi-token hidden state."""
        if self.hook is not None:
            self.hook(self, None, (StubHidden(PROMPT_LENGTH),))


class StubInner:
    """Container exposing the decoder-layer list the backend searches for."""

    def __init__(self, layers: list[StubLayer]) -> None:
        """Store decoder blocks."""
        self.layers = layers


class StubModel:
    """Causal-LM stand-in recording the keyword arguments generation receives."""

    def __init__(self, layers: list[StubLayer]) -> None:
        """Create a model with a known device and decoder depth."""
        self.model = StubInner(layers)
        self.device = "cpu"
        self.generate_kwargs: dict[str, Any] | None = None
        self.forward_calls = 0

    def eval(self) -> StubModel:
        """Mirror the Transformers eval-mode call."""
        return self

    def generate(self, ids: StubTensor, **kwargs: object) -> StubTensor:
        """Fire the activation hook and return prompt ids plus a fixed continuation."""
        self.generate_kwargs = dict(kwargs)
        self._fire_prefill()
        return StubTensor([[*ids.values[0], *CONTINUATION_IDS]])

    def __call__(self, ids: StubTensor) -> None:
        """Run a forward-only prefill."""
        del ids
        self.forward_calls += 1
        self._fire_prefill()

    def _fire_prefill(self) -> None:
        """Fire whichever decoder block currently carries the capture hook."""
        for layer in self.model.layers:
            layer.fire_prefill()


class StubTokenizer:
    """Tokenizer stand-in recording chat-template keyword arguments."""

    eos_token_id = 7

    def __init__(self) -> None:
        """Start with no recorded template call."""
        self.template_kwargs: dict[str, Any] | None = None

    def apply_chat_template(self, conversation: object, **kwargs: object) -> StubTensor:
        """Record kwargs and return a bare tensor, as `return_dict=False` requires."""
        del conversation
        self.template_kwargs = dict(kwargs)
        if kwargs.get("return_dict") is not False:
            msg = "transformers 5 returns a BatchEncoding unless return_dict=False"
            raise TypeError(msg)
        return StubTensor([list(range(PROMPT_LENGTH))])

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool) -> str:
        """Render the decoded ids so tests can assert which span was decoded."""
        if not skip_special_tokens:
            msg = "backend should skip special tokens"
            raise AssertionError(msg)
        return f"decoded:{list(token_ids)}"


class _StubMatmul:
    """Stand-in for `torch.backends.cuda.matmul`, whose TF32 flag the backend clears."""

    allow_tf32 = True


class _StubCuda:
    """Stand-in for `torch.backends.cuda`."""

    matmul = _StubMatmul()


class _StubCudnn:
    """Stand-in for `torch.backends.cudnn`, whose autotuner the backend disables."""

    benchmark = True
    allow_tf32 = True


class _StubBackends:
    """Stand-in for `torch.backends`."""

    cuda = _StubCuda()
    cudnn = _StubCudnn()


class StubTorch:
    """Torch stand-in recording global reseeding and rejecting per-call generators."""

    bfloat16 = "bfloat16-dtype"
    float32 = "float32-dtype"

    def __init__(self) -> None:
        """Start with no recorded seeds."""
        self.seeds: list[int] = []
        # Mirrors the real API surface the backend touches, so a determinism request is observable
        # rather than something the stub silently swallows.
        self.deterministic_calls: list[tuple[bool, bool]] = []
        self.backends = _StubBackends()

    def use_deterministic_algorithms(
        self,
        mode: bool,  # noqa: FBT001  # positional, matching the real torch signature
        *,
        warn_only: bool = False,
    ) -> None:
        """Record a determinism request the way the real API receives it."""
        self.deterministic_calls.append((mode, warn_only))

    def manual_seed(self, seed: int) -> None:
        """Record a global reseed."""
        self.seeds.append(seed)

    # `abc.Generator`, because this class defines its own `Generator` method.
    @contextmanager
    def inference_mode(self) -> abc.Generator[None]:
        """Provide a no-op inference context."""
        yield

    def Generator(self, *args: object, **kwargs: object) -> object:  # noqa: N802
        """Fail loudly, because `generate` supports no per-call generator."""
        del args, kwargs
        msg = "per-call torch.Generator is not supported by transformers generate"
        raise AssertionError(msg)


class StubTransformers:
    """Transformers stand-in returning the stub tokenizer and model."""

    def __init__(self, tokenizer: StubTokenizer, model: StubModel) -> None:
        """Wire the auto classes to the supplied stubs and record load kwargs."""
        self.tokenizer_kwargs: dict[str, Any] | None = None
        self.model_kwargs: dict[str, Any] | None = None
        recorder = self

        class AutoTokenizer:
            """Stub auto tokenizer factory."""

            @staticmethod
            def from_pretrained(repository: str, **kwargs: object) -> StubTokenizer:
                """Record the load call and return the stub tokenizer."""
                recorder.tokenizer_kwargs = {"repository": repository, **kwargs}
                return tokenizer

        class AutoModelForCausalLM:
            """Stub auto causal-LM factory."""

            @staticmethod
            def from_pretrained(repository: str, **kwargs: object) -> StubModel:
                """Record the load call and return the stub model."""
                recorder.model_kwargs = {"repository": repository, **kwargs}
                return model

        self.AutoTokenizer = AutoTokenizer
        self.AutoModelForCausalLM = AutoModelForCausalLM


@dataclass(frozen=True)
class StubBundle:
    """Backend under test together with the stub modules it was wired to."""

    backend: backend_module.HuggingFaceCausalLMBackend
    torch: StubTorch
    tokenizer: StubTokenizer
    model: StubModel
    transformers: StubTransformers


@pytest.fixture
def stubbed_backend(monkeypatch: pytest.MonkeyPatch) -> StubBundle:
    """Build a backend whose lazy imports resolve to stub modules."""
    torch = StubTorch()
    tokenizer = StubTokenizer()
    model = StubModel([StubLayer() for _ in range(DECODER_DEPTH)])
    transformers = StubTransformers(tokenizer, model)
    modules: dict[str, object] = {"torch": torch, "transformers": transformers}

    def fake_import(name: str) -> object:
        return modules[name]

    monkeypatch.setattr(backend_module.importlib, "import_module", fake_import)
    spec = ModelSpec(
        key="stub",
        repository="stub/model",
        revision="deadbeef",
        activation_layer=3,
        dtype="bfloat16",
        device_map="cpu",
        max_new_tokens=8,
        temperature=0.7,
    )
    return StubBundle(
        backend=backend_module.HuggingFaceCausalLMBackend(spec),
        torch=torch,
        tokenizer=tokenizer,
        model=model,
        transformers=transformers,
    )


def test_generate_requests_a_bare_tensor_from_the_chat_template(
    stubbed_backend: StubBundle,
) -> None:
    """The backend must force tensor output instead of a Transformers 5 BatchEncoding."""
    stubbed_backend.backend.generate((ChatMessage(role="user", content="hello"),), seed=11)
    template_kwargs = stubbed_backend.tokenizer.template_kwargs
    assert template_kwargs is not None
    assert template_kwargs["return_dict"] is False
    assert template_kwargs["return_tensors"] == "pt"
    assert template_kwargs["add_generation_prompt"] is True


def test_generate_seeds_globally_and_passes_no_generator(
    stubbed_backend: StubBundle,
) -> None:
    """Per-generation determinism must use global reseeding, not an unsupported kwarg."""
    stubbed_backend.backend.generate((ChatMessage(role="user", content="hello"),), seed=1234)
    assert stubbed_backend.torch.seeds == [1234]
    generate_kwargs = stubbed_backend.model.generate_kwargs
    assert generate_kwargs is not None
    assert "generator" not in generate_kwargs
    assert generate_kwargs["max_new_tokens"] == 8
    assert generate_kwargs["do_sample"] is True
    assert generate_kwargs["pad_token_id"] == StubTokenizer.eos_token_id


def test_zero_temperature_disables_sampling(monkeypatch: pytest.MonkeyPatch) -> None:
    """A zero-temperature spec must request greedy decoding."""
    torch = StubTorch()
    model = StubModel([StubLayer() for _ in range(DECODER_DEPTH)])
    transformers = StubTransformers(StubTokenizer(), model)
    modules: dict[str, object] = {"torch": torch, "transformers": transformers}
    monkeypatch.setattr(backend_module.importlib, "import_module", lambda name: modules[name])
    spec = ModelSpec(key="stub", repository="stub/model", temperature=0.0)
    backend = backend_module.HuggingFaceCausalLMBackend(spec)
    backend.generate((ChatMessage(role="user", content="hello"),), seed=3)
    generate_kwargs = model.generate_kwargs
    assert generate_kwargs is not None
    assert generate_kwargs["do_sample"] is False


def test_generate_returns_decoded_continuation_and_activation(
    stubbed_backend: StubBundle,
) -> None:
    """Only continuation tokens are decoded, and the configured layer is reported."""
    generation = stubbed_backend.backend.generate(
        (ChatMessage(role="user", content="hello"),), seed=5
    )
    assert generation.text == f"decoded:{list(CONTINUATION_IDS)}"
    assert generation.activation_layer == 3
    assert generation.activation is not None
    np.testing.assert_allclose(generation.activation, np.arange(HIDDEN_SIZE, dtype=np.float64))


def test_load_forwards_pinned_revision_and_dtype(stubbed_backend: StubBundle) -> None:
    """Pinned revisions must reach both auto classes and the dtype must reach the model."""
    stubbed_backend.backend.generate((ChatMessage(role="user", content="hello"),), seed=1)
    assert stubbed_backend.transformers.tokenizer_kwargs == {
        "repository": "stub/model",
        "revision": "deadbeef",
        "trust_remote_code": False,
    }
    assert stubbed_backend.transformers.model_kwargs == {
        "repository": "stub/model",
        "revision": "deadbeef",
        "device_map": "cpu",
        "torch_dtype": StubTorch.bfloat16,
        "trust_remote_code": False,
    }


def test_read_activation_uses_forward_only_prefill(stubbed_backend: StubBundle) -> None:
    """Probing must capture the prefill activation without generating or reseeding."""
    activation = stubbed_backend.backend.read_activation(
        (ChatMessage(role="user", content="hi"),), layer=3
    )
    np.testing.assert_allclose(activation, np.arange(HIDDEN_SIZE, dtype=np.float64))
    assert stubbed_backend.model.forward_calls == 1
    assert stubbed_backend.model.generate_kwargs is None
    assert stubbed_backend.torch.seeds == []


def test_activation_layer_outside_decoder_is_rejected(stubbed_backend: StubBundle) -> None:
    """An out-of-range layer must fail loudly rather than silently capture nothing."""
    with pytest.raises(IndexError, match="outside decoder"):
        stubbed_backend.backend.read_activation((ChatMessage(role="user", content="hi"),), layer=99)


def test_hook_is_removed_after_generation(stubbed_backend: StubBundle) -> None:
    """The capture hook must not leak across generations."""
    stubbed_backend.backend.generate((ChatMessage(role="user", content="hello"),), seed=2)
    assert stubbed_backend.model.model.layers[3].handle.removed is True


class StubProcessor:
    """Logits processor stand-in that counts resets."""

    def __init__(self) -> None:
        """Start unreset."""
        self.resets = 0

    def reset(self) -> None:
        """Record a reset."""
        self.resets += 1


class StubOutlinesBackends:
    """Stand-in for `outlines.backends`, counting expensive index compilations."""

    def __init__(self) -> None:
        """Start with no compilations."""
        self.builds: list[str] = []
        self.processor = StubProcessor()

    def get_regex_logits_processor(
        self, backend_name: object, model: object, regex: str
    ) -> StubProcessor:
        """Record one compilation of the regex-to-vocabulary index."""
        del backend_name, model
        self.builds.append(regex)
        return self.processor


class StubOutlines:
    """Stand-in for the `outlines` top-level module."""

    def __init__(self, wrapped: object) -> None:
        """Store what `from_transformers` should hand back."""
        self._wrapped = wrapped
        self.wraps = 0

    def from_transformers(self, model: object, tokenizer: object) -> object:
        """Count how often the model is re-wrapped."""
        del model, tokenizer
        self.wraps += 1
        return self._wrapped


def test_constraint_index_is_compiled_once_and_reset_per_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The regex logits processor is built once per grammar and reset between generations.

    Outlines does no caching of its own. `get_regex_logits_processor` rebuilds
    `Index(regex, vocabulary)` on every call, roughly 3s against a 151k-token vocabulary against
    roughly 0.8s of sampling. `outlines.Generator` likewise builds once and resets.
    """
    torch = StubTorch()
    tokenizer = StubTokenizer()
    model = StubModel([StubLayer() for _ in range(DECODER_DEPTH)])
    transformers = StubTransformers(tokenizer, model)
    outlines_backends = StubOutlinesBackends()
    outlines = StubOutlines(object())
    modules: dict[str, object] = {
        "torch": torch,
        "transformers": transformers,
        "outlines": outlines,
        "outlines.backends": outlines_backends,
    }
    monkeypatch.setattr(backend_module.importlib, "import_module", lambda name: modules[name])
    spec = ModelSpec(key="stub", repository="stub/model", activation_layer=3)
    backend = backend_module.HuggingFaceCausalLMBackend(spec)

    for index in range(3):
        backend.generate(
            (ChatMessage(role="user", content="hello"),), seed=index, output_regex="GRAMMAR"
        )

    assert outlines_backends.builds == ["GRAMMAR"], "index must compile exactly once"
    assert outlines.wraps == 1, "the model must be wrapped once, not per generation"
    # First generation builds, the next two reuse and must reset so no automaton state leaks.
    assert outlines_backends.processor.resets == 2
    generate_kwargs = model.generate_kwargs
    assert generate_kwargs is not None
    assert generate_kwargs["logits_processor"] == [outlines_backends.processor]


def test_a_second_grammar_gets_its_own_compiled_index(monkeypatch: pytest.MonkeyPatch) -> None:
    """Caching is per grammar, so a different regex must not silently reuse the wrong index."""
    torch = StubTorch()
    model = StubModel([StubLayer() for _ in range(DECODER_DEPTH)])
    outlines_backends = StubOutlinesBackends()
    modules: dict[str, object] = {
        "torch": torch,
        "transformers": StubTransformers(StubTokenizer(), model),
        "outlines": StubOutlines(object()),
        "outlines.backends": outlines_backends,
    }
    monkeypatch.setattr(backend_module.importlib, "import_module", lambda name: modules[name])
    backend = backend_module.HuggingFaceCausalLMBackend(
        ModelSpec(key="stub", repository="stub/model")
    )
    message = (ChatMessage(role="user", content="hello"),)
    backend.generate(message, seed=1, output_regex="FIRST")
    backend.generate(message, seed=2, output_regex="SECOND")
    assert outlines_backends.builds == ["FIRST", "SECOND"]


def test_remote_code_stays_off_unless_the_spec_opts_in(monkeypatch: pytest.MonkeyPatch) -> None:
    """Executing a checkpoint's own Python must be explicit, and must reach both auto classes.

    Some architectures cannot load without it. Defaulting it on would silently execute third-party
    code for every model, so the flag is per-spec and off unless a roster sets it.
    """
    model = StubModel([StubLayer() for _ in range(DECODER_DEPTH)])
    transformers = StubTransformers(StubTokenizer(), model)
    modules: dict[str, object] = {"torch": StubTorch(), "transformers": transformers}
    monkeypatch.setattr(backend_module.importlib, "import_module", lambda name: modules[name])
    spec = ModelSpec(key="stub", repository="stub/model", trust_remote_code=True)
    backend_module.HuggingFaceCausalLMBackend(spec).generate(
        (ChatMessage(role="user", content="hello"),), seed=1
    )
    assert transformers.tokenizer_kwargs is not None
    assert transformers.model_kwargs is not None
    assert transformers.tokenizer_kwargs["trust_remote_code"] is True
    assert transformers.model_kwargs["trust_remote_code"] is True


def _run_with_stub_torch(
    monkeypatch: pytest.MonkeyPatch, *, deterministic: bool = True
) -> StubTorch:
    """Generate once against stub modules and return the torch stand-in for inspection."""
    torch = StubTorch()
    model = StubModel([StubLayer() for _ in range(DECODER_DEPTH)])
    modules: dict[str, object] = {
        "torch": torch,
        "transformers": StubTransformers(StubTokenizer(), model),
    }
    monkeypatch.setattr(backend_module.importlib, "import_module", lambda name: modules[name])
    spec = ModelSpec(key="stub", repository="stub/model", deterministic=deterministic)
    backend_module.HuggingFaceCausalLMBackend(spec).generate(
        (ChatMessage(role="user", content="hello"),), seed=1
    )
    return torch


def test_determinism_is_requested_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Determinism controls are requested by default.

    Without them, one seed does not reproduce its generations bitwise. Because one agent's message
    enters every peer's next inbox, a single flipped token desynchronises the run and destroys the
    byte-identical pre-treatment matching the paired design depends on.
    """
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    torch = _run_with_stub_torch(monkeypatch)
    assert torch.deterministic_calls == [(True, True)]
    assert torch.backends.cudnn.benchmark is False
    assert torch.backends.cuda.matmul.allow_tf32 is False
    assert torch.backends.cudnn.allow_tf32 is False
    # cuBLAS needs this set before the first handle is created, so the backend sets it at load.
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"


def test_determinism_can_be_turned_off_to_measure_its_cost(monkeypatch: pytest.MonkeyPatch) -> None:
    """Opting out must be explicit, and must actually skip the calls."""
    torch = _run_with_stub_torch(monkeypatch, deterministic=False)
    assert torch.deterministic_calls == []


class _NestedTextModel:
    """Multimodal wrapper whose decoder layers live under `model.language_model`."""

    def __init__(self, layers: list[StubLayer]) -> None:
        """Nest the layers two levels down, as Gemma 4 and Qwen3.5 do."""
        self.model = type("_Inner", (), {"language_model": type("_LM", (), {"layers": layers})()})()
        self.device = "cpu"
        self.generated: list[object] = []

    def eval(self) -> None:
        """Match the Transformers API."""

    def generate(self, ids: object, **kwargs: object) -> object:
        """Return a plausible completion."""
        del kwargs
        self.generated.append(ids)
        return StubTensor([[1, 2, 3, 4]])

    def __call__(self, ids: object) -> object:
        """Match the forward-pass API used for activation reads."""
        del ids
        return None


def test_decoder_layers_are_found_on_a_multimodal_wrapper() -> None:
    """Gemma 4 and Qwen3.5 nest their text decoder, and the search must reach it.

    Layer discovery is needed only when a roster requests activation capture.
    """
    layers = [StubLayer() for _ in range(DECODER_DEPTH)]
    found = backend_module._decoder_layers(_NestedTextModel(layers))  # noqa: SLF001
    assert list(found) == layers


def test_generation_needs_no_decoder_layers_when_no_activation_is_wanted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With `activation_layer` unset, a model with no locatable layers must still generate.

    Generation hooks a decoder layer only when one is requested, so a run that captures nothing does
    not depend on layer discovery.
    """

    class _NoLayers:
        device = "cpu"

        def eval(self) -> None:
            """Match the Transformers API."""

        def generate(self, ids: object, **kwargs: object) -> object:
            del ids, kwargs
            return StubTensor([[1, 2, 3, 4]])

    # StubTransformers is typed for StubModel; this stand-in deliberately has no locatable layers.
    transformers = StubTransformers(StubTokenizer(), cast("StubModel", _NoLayers()))
    modules: dict[str, object] = {"torch": StubTorch(), "transformers": transformers}
    monkeypatch.setattr(backend_module.importlib, "import_module", lambda name: modules[name])
    spec = ModelSpec(key="stub", repository="stub/model", activation_layer=None)
    result = backend_module.HuggingFaceCausalLMBackend(spec).generate(
        (ChatMessage(role="user", content="hello"),), seed=1
    )
    assert result.activation is None
    assert result.activation_layer is None
