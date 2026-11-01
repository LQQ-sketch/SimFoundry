"""Qwen3-VL clients for articulate-anything, served by Alibaba Bailian's OpenAI-compatible API.

The pipeline reaches a VLM two ways, so there are two shapes:

* ``QwenVLM`` is what ``query_vlm.make_vlm`` returns (part recognition, mesh merge, physics
  estimation). Same ``__call__`` / ``get_result_text`` as ``Claude``.
* ``QwenWrapper`` is what ``prompt_utils.setup_vlm_model`` returns for ``Agent`` (joint actor
  and critic). Same ``generate_content(prompt_parts, generation_config)`` as ``ClaudeWrapper``.

Like Claude, Qwen takes images and text but not video. ``supports_video_input`` only accepts
Gemini, so the joint critic falls back to sampled frames without any change there.

The key is read from the environment, then from ``api_keys.txt`` at the repo root (the same file
the main pipeline reads). It is never printed or written anywhere.
"""
import base64
import logging
import mimetypes
import os
import time
from io import BytesIO
from pathlib import Path

from openai import OpenAI
from PIL import Image

DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MAX_TOKENS = 8192
_KEY_ENVS = ("SIMFOUNDRY_OPENAI_COMPAT_API_KEY", "DASHSCOPE_API_KEY")


def _api_key():
    for name in _KEY_ENVS:
        if os.environ.get(name):
            return os.environ[name]
    # A subprocess started without the parent's environment still finds the key the way the
    # main pipeline does: api_keys.txt in the nearest enclosing directory that has one.
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "api_keys.txt"
        if not candidate.is_file():
            continue
        for line in candidate.read_text().splitlines():
            name, _, value = line.partition("=")
            if name.strip() in _KEY_ENVS and value.strip():
                return value.strip().strip("'\"")
        break
    raise ValueError(
        "Qwen needs an API key: set DASHSCOPE_API_KEY (or SIMFOUNDRY_OPENAI_COMPAT_API_KEY), "
        "or put it in api_keys.txt at the repo root."
    )


def _client():
    return OpenAI(
        api_key=_api_key(),
        base_url=os.environ.get("SIMFOUNDRY_OPENAI_COMPAT_BASE_URL") or DEFAULT_BASE_URL,
        timeout=300.0,
    )


def _max_tokens(generation_config=None):
    config = generation_config or {}
    value = config.get("max_tokens") or config.get("max_output_tokens")
    return int(value or os.environ.get("SIMFOUNDRY_OPENAI_COMPAT_MAX_TOKENS", DEFAULT_MAX_TOKENS))


def _image_block(data: bytes, mime: str):
    url = f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"
    return {"type": "image_url", "image_url": {"url": url}}


def _block_from_path(path):
    mime = mimetypes.guess_type(str(path))[0] or ""
    if not mime.startswith("image/"):
        raise ValueError(
            f"Qwen takes images, not '{path}'. Video must be sampled into frames first."
        )
    return _image_block(Path(path).read_bytes(), mime)


def _block_from_pil(image: Image.Image):
    # PNG, like the Claude path: lossless and it round-trips RGBA renders.
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return _image_block(buffered.getvalue(), "image/png")


def to_openai_blocks(prompt_parts):
    """Normalise the pipeline's prompt parts into OpenAI chat content blocks.

    Accepts what the agents produce: strings, PIL images, already-formed blocks (OpenAI or
    Anthropic style) and ``google.genai.types.Part`` objects (the SIMFOUNDRY video critic).
    """
    if not isinstance(prompt_parts, (list, tuple)):
        prompt_parts = [prompt_parts]

    blocks = []
    for part in prompt_parts:
        if isinstance(part, str):
            if part:  # the API rejects empty text blocks
                blocks.append({"type": "text", "text": part})
        elif isinstance(part, Image.Image):
            blocks.append(_block_from_pil(part))
        elif isinstance(part, dict):
            kind = part.get("type")
            source = part.get("source") or {}
            if kind in ("text", "image_url"):
                blocks.append(part)
            elif kind == "image" and source.get("type") == "base64":
                url = f"data:{source['media_type']};base64,{source['data']}"
                blocks.append({"type": "image_url", "image_url": {"url": url}})
            else:
                raise TypeError(f"Unsupported prompt block for Qwen: {kind!r}")
        elif getattr(part, "text", None) is not None:
            blocks.append({"type": "text", "text": part.text})
        elif getattr(part, "inline_data", None) is not None:
            inline = part.inline_data
            mime = getattr(inline, "mime_type", "") or ""
            data = getattr(inline, "data", None)
            if data is None:
                continue
            if not mime.startswith("image/"):
                raise ValueError(
                    f"Qwen cannot take '{mime}' as input. Sample frames with "
                    "`get_frames_from_video` and pass those images instead."
                )
            blocks.append(_image_block(bytes(data), mime))
        else:
            raise TypeError(f"Unsupported prompt part for Qwen: {type(part).__name__}")
    return blocks


class QwenResponse:
    """Mirrors the ``.text`` / ``.text_error`` attributes the agents read off ``ClaudeResponse``."""

    def __init__(self, text, message=None, text_error=None):
        self.text = text
        self.message = message
        self.text_error = text_error


def _chat(client, model, messages, max_tokens, temperature=None, top_p=None, seed=None,
          n_retries=3, verbose=False):
    request = dict(model=model, messages=messages, max_tokens=max_tokens)
    if temperature is not None:
        request["temperature"] = temperature
    if seed is not None:
        request["seed"] = seed
    if top_p and top_p > 0:  # callers pass top_p=0; OpenAI-style APIs require (0, 1]
        request["top_p"] = top_p

    last_error = None
    for attempt in range(n_retries):
        if verbose:
            print(f"Querying Qwen [{model}]: attempt {attempt + 1} of {n_retries}...", flush=True)
        try:
            response = client.chat.completions.create(**request)
            choice = response.choices[0]
            return choice.message.content or "", choice.finish_reason, response
        except Exception as error:
            last_error = error
            logging.warning(f"Qwen [{model}] attempt {attempt + 1} failed: {type(error).__name__}: {error}")
            time.sleep(min(2 ** attempt * 2, 30))
    raise RuntimeError(f"Qwen [{model}] failed after {n_retries} attempts") from last_error


def _response(text, finish_reason, message):
    # A reply cut off at the token limit parses as broken code or JSON; flag it so the
    # caller retries instead of acting on half an answer.
    if not text.strip():
        return QwenResponse("", message, "Qwen returned no text")
    if finish_reason == "length":
        return QwenResponse(text, message, "Qwen response truncated (finish_reason=length)")
    return QwenResponse(text, message)


class QwenVLM:
    """Drop-in for ``Claude`` in the text+image steps (``make_vlm``)."""

    def __init__(self, model="qwen3-vl-plus", verbose=False):
        self.model = model
        self.verbose = verbose
        self.client = _client()

    def __call__(self, prompt, system_prompt=None, image_paths=None, temperature=0, top_p=0,
                 seed=0, n_retries=3, print_results=False, **_ignored):
        if system_prompt is None:
            try:
                from articulate_anything.utils.vlm import BASIC_SYSTEM_PROMPT as system_prompt
            except ImportError:
                system_prompt = None
        if isinstance(image_paths, (str, os.PathLike)):
            image_paths = [image_paths]
        content = [_block_from_path(path) for path in (image_paths or [])]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})

        text, finish_reason, message = _chat(
            self.client, self.model, messages, _max_tokens(), temperature=temperature,
            top_p=top_p, seed=seed, n_retries=n_retries, verbose=self.verbose,
        )
        if print_results:
            print(text)
        return _response(text, finish_reason, message)

    def get_result_text(self, result):
        if result.text_error:
            raise ValueError(result.text_error)
        return result.text


class QwenWrapper:
    """Qwen behind the ``generate_content(prompt_parts, generation_config)`` interface ``Agent`` uses."""

    def __init__(self, model_name, system_instruction=None):
        self.model_name = model_name
        self.client = _client()
        self.system_text, self.lead_blocks = self._split_system(system_instruction)

    @staticmethod
    def _split_system(system_instruction):
        """A system instruction is a string or a list of parts; the chat API wants text there,
        so any image in it moves to the front of the user turn."""
        if not system_instruction:
            return None, []
        if isinstance(system_instruction, str):
            return system_instruction, []
        blocks = to_openai_blocks(system_instruction)
        text = "\n\n".join(block["text"] for block in blocks if block["type"] == "text")
        return text or None, [block for block in blocks if block["type"] != "text"]

    def generate_content(self, prompt_parts, generation_config=None):
        config = generation_config or {}
        content = self.lead_blocks + to_openai_blocks(prompt_parts)
        messages = [{"role": "user", "content": content}]
        if self.system_text:
            messages.insert(0, {"role": "system", "content": self.system_text})

        text, finish_reason, message = _chat(
            self.client, self.model_name, messages, _max_tokens(config),
            temperature=config.get("temperature"),
        )
        logging.info(f"Qwen [{self.model_name}] finish_reason: {finish_reason}")
        return _response(text, finish_reason, message)
