"""Pure deterministic primitives for the benchmark reset contract."""
import hashlib
import json


def canonical_json(payload):
    return json.dumps(payload, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"))


def canonical_hash(payload):
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()

