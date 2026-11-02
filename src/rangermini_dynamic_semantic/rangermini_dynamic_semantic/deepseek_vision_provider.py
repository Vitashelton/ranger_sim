"""Bounded DeepSeek visual semantics for laboratory assistance navigation.

The model can propose a semantic action and select only registered region/view
IDs. It never produces coordinates, chassis modes, trajectories, joints, or
velocity commands. Local validation and the geometry planner own execution.
"""
import base64
import json
import os
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


SYSTEM_PROMPT = """You are the high-level visual decision module for a laboratory assistant robot.
The request contains an assistance request, one current camera image, relevant
task memories, the current task context, and registered candidate regions and
viewpoints. Help the robot decide how to continue locating the requested person
and complete the assistance task. A past location is only a prior, not proof of
current presence or identity. A person detection alone does not confirm identity
or complete message delivery.

Return exactly one JSON object with exactly these fields:
{"action":"continue_area|inspect_view|visit_region|wait|unable_to_confirm",
 "region_id":null,
 "viewpoint_id":null,
 "target_match":"consistent|inconsistent|uncertain",
 "rationale_code":"CURRENT_CANDIDATE|STALE_MEMORY|VIEW_OCCLUDED|AREA_UNOBSERVED|NO_VALID_CANDIDATE"}

Rules:
- Use only region/viewpoint IDs provided in the request; never invent a pose.
- visit_region requires a registered region_id and no viewpoint_id.
- inspect_view requires a registered viewpoint_id and no region_id.
- Other actions must not nominate a region or viewpoint.
- Do not mark target_match consistent from memory alone or a generic person
  detection; use uncertain whenever the current image lacks identity evidence.
- If evidence is insufficient, choose unable_to_confirm or a registered action
  that can acquire better evidence.
- Do not output coordinates, paths, motion modes, velocities, ROS commands,
  joint targets, message-delivery claims, explanations, or extra fields.
- This is a semantic proposal only. The local task gate and geometry/safety
  planner decide whether any action can execute or the request is complete."""

ACTIONS = {"continue_area", "inspect_view", "visit_region", "wait", "unable_to_confirm"}
TARGET_MATCHES = {"consistent", "inconsistent", "uncertain"}
RATIONALES = {"CURRENT_CANDIDATE", "STALE_MEMORY", "VIEW_OCCLUDED",
              "AREA_UNOBSERVED", "NO_VALID_CANDIDATE"}
REQUIRED_FIELDS = {"action", "region_id", "viewpoint_id", "target_match", "rationale_code"}


class DeepSeekVisionProvider:
    """OpenAI-compatible DeepSeek vision client using environment credentials."""

    def __init__(self, api_key=None, base_url=None, model=None, timeout_s=None):
        self.api_key = api_key if api_key is not None else os.environ.get("DEEPSEEK_API_KEY", "")
        self.base_url = (base_url or os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")).rstrip("/")
        self.model = model or os.environ.get("DEEPSEEK_VISION_MODEL", "deepseek-flash")
        self.timeout_s = float(timeout_s or os.environ.get("DEEPSEEK_TIMEOUT_S", "30"))
        if not self.base_url.startswith("https://"):
            raise ValueError("DeepSeek endpoint must use HTTPS")
        if not self.model:
            raise ValueError("vision model is required")
        if not 0.1 <= self.timeout_s <= 180:
            raise ValueError("timeout must be between 0.1 and 180 seconds")
        self.last_metadata = {}

    @property
    def configured(self):
        return bool(self.api_key)

    @staticmethod
    def _validate_request(request):
        required = {"assistance_request", "current_context", "relevant_memories",
                    "candidate_regions", "candidate_viewpoints"}
        if not isinstance(request, dict) or set(request) != required:
            raise ValueError("request must contain exactly the bounded assistance fields")
        if not isinstance(request["assistance_request"], dict) or not isinstance(request["current_context"], dict):
            raise ValueError("assistance_request and current_context must be objects")
        memories = request["relevant_memories"]
        regions = request["candidate_regions"]
        views = request["candidate_viewpoints"]
        if not isinstance(memories, list) or not isinstance(regions, list) or not isinstance(views, list):
            raise ValueError("memories, regions, and viewpoints must be arrays")
        region_ids = [v.get("region_id") for v in regions if isinstance(v, dict)]
        view_ids = [v.get("viewpoint_id") for v in views if isinstance(v, dict)]
        if len(region_ids) != len(regions) or len(view_ids) != len(views):
            raise ValueError("candidate regions and viewpoints require bounded object records")
        if any(not isinstance(v, str) or not v for v in region_ids + view_ids):
            raise ValueError("candidate IDs must be nonempty strings")
        encoded = json.dumps(request, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        if len(encoded) > 12000:
            raise ValueError("semantic request exceeds 12 KB")
        return set(region_ids), set(view_ids)

    @staticmethod
    def validate_decision(raw, region_ids, view_ids):
        if not isinstance(raw, dict) or set(raw) != REQUIRED_FIELDS:
            raise ValueError("vision decision has missing or extra fields")
        if raw["action"] not in ACTIONS:
            raise ValueError("unknown assistance action")
        if raw["target_match"] not in TARGET_MATCHES:
            raise ValueError("unknown target-match state")
        if raw["rationale_code"] not in RATIONALES:
            raise ValueError("unknown rationale code")
        if raw["region_id"] is not None and raw["region_id"] not in region_ids:
            raise ValueError("decision references an unregistered region")
        if raw["viewpoint_id"] is not None and raw["viewpoint_id"] not in view_ids:
            raise ValueError("decision references an unregistered viewpoint")
        action = raw["action"]
        if action == "visit_region" and (raw["region_id"] is None or raw["viewpoint_id"] is not None):
            raise ValueError("visit_region requires only a registered region")
        if action == "inspect_view" and (raw["viewpoint_id"] is None or raw["region_id"] is not None):
            raise ValueError("inspect_view requires only a registered viewpoint")
        if action not in {"visit_region", "inspect_view"} and (
                raw["region_id"] is not None or raw["viewpoint_id"] is not None):
            raise ValueError("only region/view inspection actions may select a registered target")
        return dict(raw)

    def decide(self, request, image_bytes, mime_type="image/jpeg"):
        """Return a validated semantic proposal and non-sensitive call metadata."""
        if not self.configured:
            raise RuntimeError("DEEPSEEK_API_KEY is not configured")
        region_ids, view_ids = self._validate_request(request)
        if not isinstance(image_bytes, (bytes, bytearray)) or not image_bytes:
            raise ValueError("a current camera image is required")
        if mime_type not in {"image/jpeg", "image/png", "image/webp"}:
            raise ValueError("unsupported image MIME type")
        image_url = "data:" + mime_type + ";base64," + base64.b64encode(image_bytes).decode("ascii")
        body = {
            "model": self.model,
            "stream": False,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "text", "text": json.dumps(request, ensure_ascii=False,
                                                            separators=(",", ":"))},
                    {"type": "image_url", "image_url": {"url": image_url, "detail": "high"}},
                ]},
            ],
        }
        encoded = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        endpoint = self.base_url + "/chat/completions"
        http_request = Request(endpoint, data=encoded, method="POST", headers={
            "Authorization": "Bearer " + self.api_key,
            "Content-Type": "application/json",
        })
        started = time.monotonic()
        self.last_metadata = {"provider": "deepseek", "model": self.model,
                              "endpoint": endpoint, "request_bytes": len(encoded),
                              "image_bytes": len(image_bytes)}
        try:
            with urlopen(http_request, timeout=self.timeout_s) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            self.last_metadata.update({"latency_s": round(time.monotonic()-started, 3),
                                       "http_status": exc.code, "remote_success": False})
            raise RuntimeError(f"DeepSeek request failed (HTTP {exc.code})") from None
        except (URLError, TimeoutError, OSError):
            self.last_metadata.update({"latency_s": round(time.monotonic()-started, 3),
                                       "remote_success": False})
            raise RuntimeError("DeepSeek request failed (network/timeout)") from None
        self.last_metadata.update({"latency_s": round(time.monotonic()-started, 3),
                                   "remote_success": True})
        choices = payload.get("choices", [])
        if not choices:
            raise ValueError("DeepSeek returned no choices")
        content = choices[0].get("message", {}).get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("DeepSeek returned empty content")
        try:
            raw = json.loads(content)
        except json.JSONDecodeError:
            raise ValueError("DeepSeek returned invalid JSON") from None
        return self.validate_decision(raw, region_ids, view_ids)
