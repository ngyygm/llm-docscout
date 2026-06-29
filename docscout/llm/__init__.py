"""LLM API utilities for DocScout data generation & evaluation.

This package is the HOST-side bridge to an external LLM (GLM-5.2 via an
OpenAI-compatible endpoint). It is used for:
  - synthetic data generation / paraphrasing (v5 heterogeneous corpora),
  - LLM-as-judge answer scoring (alias / unit / phrasing tolerant),
  - closed-book audits (does the model answer without reading?).

It is NOT the policy/rollout client (that lives in docscout.agent.client and
talks to a local vLLM server on the GPU box). The GPU box has no internet, so
anything that calls this module must run on the host laptop.
"""

from docscout.llm.glm import LLMClient, LLMConfig, load_config

__all__ = ["LLMClient", "LLMConfig", "load_config"]
