"""Verified registration facts for conversational service-task compilation.

This module supplies factual context. It does not infer motion modes, identify
people in images, or turn natural-language requests into hand-authored plans.
"""
import hashlib
import json
from pathlib import Path


def load_service_directory(path, graph):
    path = Path(path)
    raw = path.read_bytes()
    data = json.loads(raw)
    if not isinstance(data, dict) or data.get('version') != 1:
        raise ValueError('service directory requires version=1')
    if not isinstance(data.get('source'), str) or not data['source'].strip():
        raise ValueError('registration source must be explicit')
    entries = data.get('entities')
    if not isinstance(entries, list):
        raise ValueError('service directory requires an entity list')
    ids, names = set(), {}
    validated = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError('directory entity must be an object')
        identity, name, room = entry.get('id'), entry.get('name'), entry.get('room')
        if not isinstance(identity, str) or not identity or identity in ids:
            raise ValueError('entity ID missing or duplicated')
        if not isinstance(name, str) or not name.strip():
            raise ValueError('entity name is required')
        if not isinstance(room, str) or 'D' + room not in graph.doors:
            raise ValueError('unregistered room for ' + name)
        aliases = entry.get('aliases', [])
        if not isinstance(aliases, list) or any(not isinstance(x, str) or not x.strip() for x in aliases):
            raise ValueError('aliases must be nonempty strings')
        labels = [name] + aliases
        if len(set(labels)) != len(labels):
            raise ValueError('duplicated name within an entity')
        kind = entry.get('kind')
        if kind not in ('person', 'service_location'):
            raise ValueError('unsupported registration entity kind')
        ids.add(identity)
        for label in labels:
            names.setdefault(label, []).append(identity)
        validated.append(dict(id=identity, name=name, aliases=aliases, room=room,
                              kind=kind, knowledge_source='registration',
                              visual_identity_verified=False))
    return dict(version=1, source=data['source'], entities=validated,
                ambiguous_labels={k: v for k, v in names.items() if len(v) > 1},
                directory_sha256=hashlib.sha256(raw).hexdigest())


def task_compile_request(instruction, graph, service_directory=None):
    """Build model context without pre-solving the natural-language task."""
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError('a service request is required')
    context = dict(instruction=instruction, registered_rooms=sorted(d[1:] for d in graph.doors))
    if service_directory is not None:
        context['service_directory'] = load_service_directory(service_directory, graph)
    return context
