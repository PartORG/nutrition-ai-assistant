#!/usr/bin/env python3
"""
Generate Dart data classes from the FastAPI OpenAPI schema.

The Flutter client hand-writes its request/response handling: `ApiService.get`
returns `Future<dynamic>` and callers cast at the call site. Rename a field in
`src/adapters/rest/schemas.py` and Dart still compiles — the break surfaces at
runtime, on a user's phone.

This closes that gap from the Python side: the schema is the source of truth,
and the Dart models are derived from it. CI runs `--check`, which fails when the
committed models no longer match the API.

    uv run python scripts/gen_dart_models.py            # write the models
    uv run python scripts/gen_dart_models.py --check    # fail if out of date

Deliberately not the `openapi-generator` toolchain: that needs a JVM in CI and
emits a client far larger than this API warrants. This handles the subset
FastAPI actually produces here — flat Pydantic models of scalars, lists and
nested refs.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

OUTPUT = REPO_ROOT / "app" / "nutrition_ai_assistent" / "lib" / "models" / "api_models.dart"

_HEADER = """// GENERATED FILE — DO NOT EDIT BY HAND.
//
// Produced from the FastAPI OpenAPI schema by scripts/gen_dart_models.py.
// Regenerate with:  uv run python scripts/gen_dart_models.py
//
// These exist so a field rename in src/adapters/rest/schemas.py becomes a Dart
// compile error instead of a runtime null on someone's phone. CI fails if this
// file drifts from the API.

"""

_DART_SCALARS = {
    ("string", None): "String",
    ("string", "date-time"): "String",
    ("integer", None): "int",
    ("number", None): "double",
    ("boolean", None): "bool",
}


def _dart_type(schema: dict[str, Any]) -> str:
    """Map one OpenAPI property schema onto a Dart type."""
    if "$ref" in schema:
        return _dart_class_name(schema["$ref"].rsplit("/", 1)[-1])

    # anyOf[X, null] is how Pydantic v2 expresses Optional[X]
    if "anyOf" in schema:
        non_null = [s for s in schema["anyOf"] if s.get("type") != "null"]
        inner = _dart_type(non_null[0]) if non_null else "dynamic"
        return f"{inner}?"

    kind = schema.get("type")
    if kind == "array":
        return f"List<{_dart_type(schema.get('items', {}))}>"
    if kind == "object":
        return "Map<String, dynamic>"
    return _DART_SCALARS.get((kind, schema.get("format")), "dynamic")


def _dart_class_name(name: str) -> str:
    """Normalise an OpenAPI schema name into a valid public Dart class name.

    Pydantic private models arrive with a leading underscore, which in Dart
    means library-private — the class would be generated and then flagged as
    unused. Names are also forced to UpperCamelCase to satisfy the linter.
    """
    cleaned = name.lstrip("_")
    parts = [p for p in re.split(r"[_\s]+", cleaned) if p]
    return "".join(p[:1].upper() + p[1:] for p in parts)


def _camel(name: str) -> str:
    head, *rest = name.split("_")
    return head + "".join(w.capitalize() for w in rest)


def _from_json(dart_type: str, key: str) -> str:
    """Expression converting json['key'] into `dart_type`."""
    optional = dart_type.endswith("?")
    base = dart_type.rstrip("?")
    src = f"json['{key}']"

    if base.startswith("List<"):
        inner = base[5:-1]
        expr = (
            f"({src} as List<dynamic>? ?? const [])"
            f".map((e) => {inner}.fromJson(e as Map<String, dynamic>)).toList()"
            if inner not in ("String", "int", "double", "bool", "dynamic")
            else f"({src} as List<dynamic>? ?? const []).map((e) => e as {inner}).toList()"
        )
        return expr
    if base in ("String", "int", "bool"):
        return f"{src} as {base}{'?' if optional else ''}"
    if base == "double":
        return f"({src} as num?)?.toDouble(){'' if optional else ' ?? 0'}"
    if base == "Map<String, dynamic>":
        return f"{src} as Map<String, dynamic>{'?' if optional else ''}"
    if base == "dynamic":
        return src
    # nested model
    return (
        f"{src} == null ? null : {base}.fromJson({src} as Map<String, dynamic>)"
        if optional
        else f"{base}.fromJson({src} as Map<String, dynamic>)"
    )


def render_model(name: str, schema: dict[str, Any]) -> str:
    props: dict[str, Any] = schema.get("properties", {})
    required = set(schema.get("required", []))
    if not props:
        return ""

    fields, ctor, parse = [], [], []
    for key, prop in props.items():
        dart_type = _dart_type(prop)
        if key not in required and not dart_type.endswith("?"):
            dart_type += "?"
        field = _camel(key)
        fields.append(f"  final {dart_type} {field};")
        ctor.append(
            f"    {'required ' if dart_type.endswith('?') is False else ''}this.{field},"
        )
        parse.append(f"      {field}: {_from_json(dart_type, key)},")

    doc = schema.get("description", "").strip().splitlines()
    doc_comment = "".join(f"/// {line}\n" for line in doc[:3]) if doc else ""

    return (
        f"{doc_comment}class {name} {{\n"
        + "\n".join(fields)
        + "\n\n  const "
        + name
        + "({\n"
        + "\n".join(ctor)
        + "\n  });\n\n"
        + f"  factory {name}.fromJson(Map<String, dynamic> json) => {name}(\n"
        + "\n".join(parse)
        + "\n      );\n}\n"
    )


def build_dart() -> str:
    from adapters.rest.app import app

    schemas = app.openapi().get("components", {}).get("schemas", {})
    # HTTPValidationError/ValidationError are FastAPI plumbing. Body_* are
    # generated multipart form bodies for file uploads — the client posts those
    # as multipart, never as JSON, so a model for them is noise.
    skip = {"HTTPValidationError", "ValidationError"}

    body = [
        render_model(_dart_class_name(name), schema)
        for name, schema in sorted(schemas.items())
        if name not in skip and not name.startswith("Body_")
    ]
    return _HEADER + "\n".join(b for b in body if b)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true",
        help="exit non-zero if the committed file is out of date",
    )
    args = parser.parse_args(argv)

    try:
        generated = build_dart()
    except Exception as e:
        print(f"ERROR: could not build the OpenAPI schema: {e}", file=sys.stderr)
        return 1

    if args.check:
        current = OUTPUT.read_text() if OUTPUT.exists() else ""
        if _normalise(current) != _normalise(generated):
            print(
                "Dart models are out of date with the API schema.\n"
                "Run: uv run python scripts/gen_dart_models.py",
                file=sys.stderr,
            )
            return 1
        print(f"Dart models match the API schema ({OUTPUT.relative_to(REPO_ROOT)})")
        return 0

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(generated)
    count = len(re.findall(r"^class ", generated, re.M))
    print(f"Wrote {count} model(s) to {OUTPUT.relative_to(REPO_ROOT)}")
    return 0


def _normalise(text: str) -> str:
    """Collapse all whitespace before comparing.

    The committed file is run through `dart format`, which reflows the raw
    generator output. Comparing verbatim would fail on formatting alone, so the
    check looks at content only: a renamed field, a changed type or a new model
    still fails, which is the drift that matters.
    """
    collapsed = re.sub(r"\s+", " ", text)
    # dart format drops the trailing comma before a closing bracket...
    collapsed = re.sub(r",\s*([)}\]])", r"\1", collapsed)
    # ...and reflows padding around brackets and separators.
    collapsed = re.sub(r"\s*([(){}\[\],;])\s*", r"\1", collapsed)
    return collapsed.strip()


if __name__ == "__main__":
    raise SystemExit(main())
