"""Lazy Hugging Face causal-LM backend with residual-stream capture."""

from __future__ import annotations

import contextlib
import importlib
import os
from collections.abc import Sequence
from contextlib import contextmanager
from typing import Any

import numpy as np

from affective_contagion.generation import BackendGeneration, ChatMessage, FloatVector
from affective_contagion.roster import ModelSpec


def _decoder_layers(model: Any) -> Sequence[Any]:
    """Find decoder blocks on common Transformers causal-LM architectures."""
    # Tried in order. The nested paths cover multimodal wrappers such as Gemma 4 and Qwen3.5,
    # whose text decoder sits under `model.language_model`.
    candidates = (
        ("model", "layers"),
        ("transformer", "h"),
        ("gpt_neox", "layers"),
        ("model", "language_model", "layers"),
        ("language_model", "model", "layers"),
        ("language_model", "layers"),
    )
    for path in candidates:
        node: Any = model
        for attribute in path:
            node = getattr(node, attribute, None)
            if node is None:
                break
        if node is not None:
            return node
    msg = "unsupported model architecture: decoder layers could not be located"
    raise TypeError(msg)


class HuggingFaceCausalLMBackend:
    """Load one checkpoint lazily and reuse it across one or more assigned agents."""

    def __init__(self, spec: ModelSpec) -> None:
        """Store an immutable model spec without importing heavy optional dependencies."""
        self.spec = spec
        self._torch: Any = None
        self._tokenizer: Any = None
        self._model: Any = None
        self._outlines_model: Any = None
        self._processors: dict[str, Any] = {}

    def _constrained_logits_processors(self, output_regex: str) -> list[Any]:
        """Return a reusable grammar-constrained logits processor for one generation.

        The processor is cached per grammar and reset between generations. Outlines does not
        cache: `get_regex_logits_processor` calls `Index(regex, vocabulary)` on every call, and
        compiling that regex-to-vocabulary index takes seconds (about 3 s for this grammar against
        a 151k-token vocabulary, against about 0.8 s of sampling).

        This mirrors what `outlines.Generator` does internally: build the processor once, then
        `reset()` before each call. This backend keeps its own generate path because the Generator
        wrapper also owns tokenisation and decoding, and the backend needs the exact prompt ids for
        activation capture and for reproducible per-generation seeding.

        Reuse is safe for sequential generation because `reset()` sets `is_first_token`, which
        makes the next `process_logits` call re-enter `_setup` and rebuild the per-sequence `Guide`
        objects from the cached index. Only the immutable compiled index is shared, so no automaton
        state carries over. That includes after an exception mid-generation, or after
        `max_new_tokens` truncates before the grammar reaches an accepting state, since both leave
        a stale `Guide` that the next `reset()` discards.

        It is not safe concurrently. Agents sharing a `model_key` share one backend and so one
        processor, whose `_guides` and in-place bitmasks would interleave if agent turns were
        threaded or batched. The simulation loop is strictly sequential. Parallel agents would need
        a processor per concurrent stream, and a violation would corrupt masks silently.

        Constructing any Outlines logits processor sets `torch._dynamo.config.suppress_errors =
        True` process-wide, so the flag is set early and stays set for the whole run.
        """
        try:
            outlines = importlib.import_module("outlines")
            backends = importlib.import_module("outlines.backends")
        except ModuleNotFoundError as error:
            msg = (
                "constrained_decision_format requires the experiment extra: "
                "uv sync --extra experiment"
            )
            raise RuntimeError(msg) from error
        if self._outlines_model is None:
            self._outlines_model = outlines.from_transformers(self._model, self._tokenizer)
        processor = self._processors.get(output_regex)
        if processor is None:
            processor = backends.get_regex_logits_processor(
                None, self._outlines_model, output_regex
            )
            self._processors[output_regex] = processor
        else:
            processor.reset()
        return [processor]

    def _enable_determinism(self, torch: Any) -> None:
        """Force deterministic kernels, because the pipeline is otherwise not reproducible.

        Without them, an agent with a byte-identical prompt and generation seed can produce a
        different message. As allocation patterns shift, cuBLAS can select a different reduction
        order for the same bf16 matmul, and at temperature 0.7 a last-bit logit difference can
        change the sampled token. That agent's message enters every peer's next inbox, so one
        changed token desynchronises the whole run.

        Matched arms share their random streams, so a run pair should be byte-identical until the
        treatment lands. Every pre-treatment divergence adds nuisance variance to the paired
        difference.

        `warn_only=True` lets a run continue when an operation has no deterministic
        implementation. The pre-treatment identity check in `scripts/check_pairing.py` gates any
        paired claim.
        """
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)  # noqa: FBT003
        torch.backends.cudnn.benchmark = False
        if hasattr(torch.backends.cuda, "matmul"):
            # Autotuned TF32 paths trade the low bits for speed, which is exactly the precision a
            # sampled token can hinge on.
            torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = False

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            torch = importlib.import_module("torch")
            transformers = importlib.import_module("transformers")
        except ModuleNotFoundError as error:
            msg = "install experiment dependencies with: uv sync --extra experiment"
            raise RuntimeError(msg) from error
        if self.spec.deterministic:
            self._enable_determinism(torch)

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.spec.repository,
            revision=self.spec.revision,
            trust_remote_code=self.spec.trust_remote_code,
        )
        kwargs: dict[str, Any] = {
            "revision": self.spec.revision,
            "device_map": self.spec.device_map,
            "trust_remote_code": self.spec.trust_remote_code,
        }
        dtype_name = self.spec.dtype
        if dtype_name != "auto":
            kwargs["torch_dtype"] = getattr(torch, dtype_name)
        if self.spec.quantization == "4bit":
            kwargs["quantization_config"] = transformers.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )
        model = transformers.AutoModelForCausalLM.from_pretrained(
            self.spec.repository,
            **kwargs,
        )
        model.eval()
        self._torch, self._tokenizer, self._model = torch, tokenizer, model

    def _input_ids(self, messages: tuple[ChatMessage, ...]) -> Any:
        self._load()
        chat = [{"role": item.role, "content": item.content} for item in messages]
        ids = self._tokenizer.apply_chat_template(
            chat,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            # Transformers 5 returns a BatchEncoding by default; downstream code indexes a tensor.
            return_dict=False,
        )
        return ids.to(self._model.device)

    @contextmanager
    def _capture(self, layer: int) -> Any:
        layers = _decoder_layers(self._model)
        if not 0 <= layer < len(layers):
            msg = f"activation layer {layer} outside decoder with {len(layers)} layers"
            raise IndexError(msg)
        captured: list[FloatVector] = []

        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            hidden = output[0] if isinstance(output, tuple) else output
            if hidden.shape[1] > 1 and not captured:
                value = hidden[0, -1, :].detach().float().cpu().numpy()
                captured.append(np.asarray(value, dtype=np.float64))

        handle = layers[layer].register_forward_hook(hook)
        try:
            yield captured
        finally:
            handle.remove()

    def generate(
        self,
        messages: tuple[ChatMessage, ...],
        *,
        seed: int,
        output_regex: str | None = None,
    ) -> BackendGeneration:
        """Generate one completion and capture the configured prefill activation if requested."""
        self._load()
        ids = self._input_ids(messages)
        layer = self.spec.activation_layer
        capture_layer = 0 if layer is None else layer
        extra: dict[str, Any] = {}
        if output_regex is not None:
            extra["logits_processor"] = self._constrained_logits_processors(output_regex)
        # `generate` accepts no per-call generator, so reseed the global torch RNG from the
        # deterministic per-generation seed immediately before sampling.
        self._torch.manual_seed(seed)
        # Only hook a layer when an activation is wanted, so a model whose decoder layers sit at an
        # unexpected path can still generate with `activation_layer: null`.
        capture = self._capture(capture_layer) if layer is not None else contextlib.nullcontext([])
        with self._torch.inference_mode(), capture as captured:
            output = self._model.generate(
                ids,
                max_new_tokens=self.spec.max_new_tokens,
                do_sample=self.spec.temperature > 0,
                temperature=max(self.spec.temperature, 1e-6),
                pad_token_id=self._tokenizer.eos_token_id,
                **extra,
            )
        text = self._tokenizer.decode(output[0, ids.shape[1] :], skip_special_tokens=True)
        activation = captured[0] if layer is not None and captured else None
        return BackendGeneration(
            text=text,
            activation=activation,
            activation_layer=layer if activation is not None else None,
        )

    def read_activation(
        self,
        messages: tuple[ChatMessage, ...],
        *,
        layer: int,
    ) -> FloatVector:
        """Capture the last input-token residual stream from a forward-only prefill."""
        self._load()
        ids = self._input_ids(messages)
        with self._torch.inference_mode(), self._capture(layer) as captured:
            self._model(ids)
        if not captured:
            msg = "activation hook did not observe a multi-token prefill"
            raise RuntimeError(msg)
        return captured[0]
