"""Structured JSON-Schema generation from Python dataclasses.

Every agent declares its input / output contract as dataclasses.  Schemas are
derived from type hints so the contract lives in exactly one place and can
never drift from the implementation.  Supports primitives, Optional, List,
Dict, Enum, Literal, nested dataclasses and dict/list defaults.

Deterministic code validates; LLM never decides what is valid JSON.
"""
from __future__ import annotations

import dataclasses
import enum
import json
import types
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

from ..errors import SchemaError

_UnionType = types.UnionType

_JSON_TYPES = {str: "string", int: "integer", float: "number", bool: "boolean", dict: "object"}


def _unpack(tp: Any):
    """Return (base_type, is_optional).

    Handles BOTH typing.Optional (typing.Union) and PEP 604 unions
    (`X | None`, types.UnionType) - the latter was previously missed, which
    silently skipped coercion/validation of Optional nested dataclasses.
    """
    origin = get_origin(tp)
    if origin in (Union, _UnionType):
        args = [a for a in get_args(tp) if a is not type(None)]
        if len(args) == 1:
            return args[0], True
        return tp, False
    return tp, False


def _type_of(tp: Any) -> str:
    base, _ = _unpack(tp)
    if base in _JSON_TYPES:
        return _JSON_TYPES[base]
    if isinstance(base, type) and issubclass(base, enum.Enum):
        return "string"
    if isinstance(base, type) and issubclass(base, bool):
        return "boolean"
    if get_origin(base) in (list, set, tuple):
        return "array"
    if get_origin(base) is dict:
        return "object"
    if dataclasses.is_dataclass(base) and isinstance(base, type):
        return "object"
    if get_origin(base) is Literal:
        return "string"
    return "string"  # fallback: opaque values are JSON-serialized


def json_schema_for(cls: type) -> dict:
    """Build a JSON Schema for a dataclass (object-rooted)."""
    base, _ = _unpack(cls)
    if not (isinstance(base, type) and dataclasses.is_dataclass(base)):
        raise SchemaError(f"json_schema_for expects a dataclass, got {cls!r}")
    hints = get_type_hints(base)
    props: dict[str, Any] = {}
    required: list[str] = []
    for f in dataclasses.fields(base):
        override = f.metadata.get("schema")
        if override is not None:
            props[f.name] = override
            if f.default is dataclasses.MISSING:
                required.append(f.name)
            continue
        tp = hints.get(f.name, Any)
        props[f.name] = _schema_for_type(tp)
        if f.default is dataclasses.MISSING:
            required.append(f.name)
    return {
        "type": "object",
        "title": base.__name__,
        "properties": props,
        "required": required,
        "additionalProperties": False,
    }


def _schema_for_type(tp: Any) -> dict:
    base, optional = _unpack(tp)
    if base is Any:
        return {}
    if base in _JSON_TYPES:
        sch = {"type": _JSON_TYPES[base]}
        return sch
    if isinstance(base, type) and issubclass(base, enum.Enum):
        return {"type": "string", "enum": [e.value for e in base]}
    if get_origin(base) is Literal:
        return {"type": "string", "enum": list(get_args(base))}
    if get_origin(base) in (list, set, tuple):
        item_tp = get_args(base)[0] if get_args(base) else Any
        return {"type": "array", "items": _schema_for_type(item_tp) if item_tp is not Any else {}}
    if get_origin(base) is dict:
        args = get_args(base)
        value_tp = args[1] if len(args) > 1 else Any
        return {"type": "object", "additionalProperties": _schema_for_type(value_tp) if value_tp is not Any else {}}
    if dataclasses.is_dataclass(base) and isinstance(base, type):
        return json_schema_for(base)
    return {}


def _coerce(base: Any, name: str, value: Any) -> Any:
    """Strictly coerce a JSON value to the declared Python type."""
    if base is Any or value is None:
        return value
    if base is str:
        if not isinstance(value, str):
            raise SchemaError(f"field '{name}' expects string, got {type(value).__name__}")
        return value
    if base is bool:
        if not isinstance(value, bool):
            raise SchemaError(f"field '{name}' expects bool, got {type(value).__name__}")
        return value
    if base is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise SchemaError(f"field '{name}' expects integer, got {type(value).__name__}")
        return value
    if base is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise SchemaError(f"field '{name}' expects number, got {type(value).__name__}")
        return float(value)
    if base is dict:
        if not isinstance(value, dict):
            raise SchemaError(f"field '{name}' expects object, got {type(value).__name__}")
        return value
    if base is list:
        if not isinstance(value, list):
            raise SchemaError(f"field '{name}' expects array, got {type(value).__name__}")
        return value
    if isinstance(base, type) and issubclass(base, enum.Enum):
        if isinstance(value, base):
            return value
        try:
            return base(value)
        except (ValueError, TypeError):
            raise SchemaError(f"field '{name}' expects one of {[e.value for e in base]}, got {value!r}") from None
    if dataclasses.is_dataclass(base) and isinstance(base, type):
        if not isinstance(value, dict):
            raise SchemaError(f"field '{name}' expects object, got {type(value).__name__}")
        return from_dict(base, value)
    origin = get_origin(base)
    if origin in (list, set, tuple):
        item_tp = get_args(base)[0] if get_args(base) else Any
        if not isinstance(value, list):
            raise SchemaError(f"field '{name}' expects array, got {type(value).__name__}")
        return [_coerce(item_tp, f"{name}[]", v) for v in value]
    if origin is dict:
        args = get_args(base)
        value_tp = args[1] if len(args) > 1 else Any
        if not isinstance(value, dict):
            raise SchemaError(f"field '{name}' expects object, got {type(value).__name__}")
        return {k: _coerce(value_tp, f"{name}.{k}", v) for k, v in value.items()}
    if origin is Literal:
        if value not in get_args(base):
            raise SchemaError(f"field '{name}' expects one of {list(get_args(base))}, got {value!r}")
        return value
    return value


def from_dict(cls: type, data: dict) -> Any:
    """Validate + build a dataclass instance from a JSON dict. Strict."""
    base, _ = _unpack(cls)
    if not (isinstance(base, type) and dataclasses.is_dataclass(base)):
        raise SchemaError(f"from_dict expects a dataclass, got {cls!r}")
    if not isinstance(data, dict):
        raise SchemaError(f"expected JSON object for {base.__name__}, got {type(data).__name__}")
    hints = get_type_hints(base)
    fields = {f.name: f for f in dataclasses.fields(base)}
    unknown = set(data) - set(fields)
    if unknown:
        raise SchemaError(f"unknown fields for {base.__name__}: {sorted(unknown)}")
    kwargs: dict[str, Any] = {}
    for name, f in fields.items():
        if name not in data:
            if f.default is not dataclasses.MISSING:
                continue
            if f.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
                continue
            raise SchemaError(f"missing required field '{name}' for {base.__name__}")
        tp = hints.get(name, Any)
        tp, _ = _unpack(tp)
        kwargs[name] = _coerce(tp, name, data[name])
    try:
        return base(**kwargs)
    except SchemaError:
        raise
    except TypeError as e:
        raise SchemaError(f"cannot build {base.__name__}: {e}") from e


def validate(cls: type, data: Any) -> None:
    """Raise SchemaError if `data` does not match the dataclass contract."""
    from_dict(cls, data)


def to_dict(obj: Any) -> dict:
    """Serialize a dataclass instance to a JSON dict (enums -> values)."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        out: dict[str, Any] = {}
        for f in dataclasses.fields(obj):
            out[f.name] = to_dict(getattr(obj, f.name))
        return out
    if isinstance(obj, enum.Enum):
        return obj.value
    if isinstance(obj, (list, tuple, set)):
        return [to_dict(v) for v in obj]
    if isinstance(obj, dict):
        return {str(k): to_dict(v) for k, v in obj.items()}
    return obj


def validate_instance(cls: type, obj: Any) -> dict:
    """Validate a dataclass instance against its schema and return its dict."""
    data = to_dict(obj)
    validate(cls, data)
    return data


def json_dumps(data: Any, indent: int = 2) -> str:
    return json.dumps(data, ensure_ascii=False, indent=indent)
