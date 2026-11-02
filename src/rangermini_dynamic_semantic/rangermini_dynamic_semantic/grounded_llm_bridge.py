"""Optional text->bounded task JSON adapter for an OpenAI-compatible local API.

No LLM output is executable until it passes BoundedTaskCompiler validation.
In particular, the model never emits cmd_vel, Gazebo truth, or robot joints.
"""
from __future__ import annotations

import json
import os
from urllib.request import Request, urlopen


SYSTEM_PROMPT = """You compile Chinese indoor navigation instructions into one JSON object.
Allowed formats ONLY:
{"type":"visit_sequence","places":["place_id", "place_id"]}
{"type":"objectnav","category":"object_category"}
{"type":"human_assisted_service","source":"place_id","service":"place_id","destination":"place_id","object_category":"cup"}
Use only place IDs and object category strings provided by the user JSON context.
The robot has no arm. Manipulation always requires an explicit human step.
Do not invent labels, coordinates, velocities, nodes, or sensory evidence.
Return JSON only, no Markdown or explanations."""


class BoundedLLMClient:
    def __init__(self, endpoint: str | None = None, model: str | None = None,
                 api_key: str | None = None, timeout: float = 8.0):
        self.endpoint = endpoint or os.getenv("LLM_BASE_URL", "http://localhost:8000/v1")
        self.model = model or os.getenv("LLM_MODEL", "Qwen2.5-VL-7B-Instruct")
        self.api_key = api_key if api_key is not None else os.getenv("LLM_API_KEY", "local")
        self.timeout = max(0.2, min(20.0, float(timeout)))
        if not self.endpoint.startswith(("http://", "https://")):
            raise ValueError("LLM endpoint must be HTTP(S)")

    def compile(self, text: str, catalog, *, graph: dict | None = None) -> dict:
        if not text.strip():
            raise ValueError("empty instruction")
        # Graph summaries, if used, have no privileged ground-truth fields.
        visual_context = {"places": [{"id": i, "name": v["name"],
                                     "features": v["features"]}
                                     for i, v in catalog.places.items()],
                          "allowed_categories": sorted({c for place in catalog.places.values()
                                                        for c in place["object_priors"]})}
        if graph is not None:
            visual_context["graph_revision"] = graph.get("revision", 0)
            visual_context["non_free_edges"] = [
                {"from": e.get("from"), "to": e.get("to"), "state": e.get("state")}
                for e in graph.get("edges", []) if e.get("state") != "FREE"]
        request_body = {"model": self.model, "stream": False,
                        "response_format": {"type": "json_object"},
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": json.dumps(
                                {"instruction": text, "available": visual_context},
                                ensure_ascii=False)}]}
        url = self.endpoint.rstrip("/")
        if not url.endswith("/chat/completions"):
            url += "/chat/completions"
        request = Request(url, data=json.dumps(request_body).encode("utf-8"),
                          headers={"Content-Type": "application/json",
                                   "Authorization": "Bearer " + self.api_key}, method="POST")
        with urlopen(request, timeout=self.timeout) as result:
            raw = result.read(120000)
        obj = json.loads(raw)
        content = obj["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        if not isinstance(parsed, dict):
            raise ValueError("LLM must return a JSON object")
        # Further validated by BoundedTaskCompiler. Return no secret metadata.
        return parsed
