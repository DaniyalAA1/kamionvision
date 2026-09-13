"""Anthropic structured-output schemas, without network calls.

The identity `model` field is an OpenAI-style nullable enum:

    {"type": ["string", "null"], "enum": ["F-MAX", ..., None]}

Anthropic cross-validates every enum value against a *single* declared type
and 400s with:

    Enum value 'F-MAX' does not match declared type '['string', 'null']'

That is why every identity sample failed through to this backend after cursor
and openai already had. The shared schema stays in the OpenAI shape; this
backend rewrites the copy it sends.
"""
from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from app.evidence import prompts
from app.vlm.anthropic_backend import AnthropicBackend, schema_for_anthropic


def _conflicts(node, path="$"):
    """Pointers where an enum cannot sit next to the declared type."""
    found = []
    if isinstance(node, dict):
        typ, enum = node.get("type"), node.get("enum")
        if isinstance(enum, list) and isinstance(typ, list):
            found.append(path)
        elif isinstance(enum, list) and isinstance(typ, str):
            kinds = {type(v).__name__ if v is not None else "NoneType" for v in enum}
            if typ == "string" and kinds - {"str"}:
                found.append(path)
            if typ in {"integer", "number", "boolean"} and "NoneType" in kinds:
                found.append(path)
        for key, value in node.items():
            found.extend(_conflicts(value, f"{path}.{key}"))
    elif isinstance(node, list):
        for i, value in enumerate(node):
            found.extend(_conflicts(value, f"{path}[{i}]"))
    return found


def _model_field(schema):
    return schema["properties"]["model"]


def _unsupported_constraints(node, path="$"):
    """Pointers Anthropic 400s on: maxItems, and minItems other than 0 or 1."""
    found = []
    if isinstance(node, dict):
        if "maxItems" in node:
            found.append(f"{path}.maxItems")
        if node.get("minItems") not in (None, 0, 1):
            found.append(f"{path}.minItems")
        for key, value in node.items():
            found.extend(_unsupported_constraints(value, f"{path}.{key}"))
    elif isinstance(node, list):
        for i, value in enumerate(node):
            found.extend(_unsupported_constraints(value, f"{path}[{i}]"))
    return found


def _allows(field, value):
    """Whether `value` is a legal answer for a rewritten or raw field."""
    if "enum" in field:
        return value in field["enum"]
    for variant in field.get("anyOf", ()):
        if value is None and variant.get("type") == "null":
            return True
        if value is not None and value in variant.get("enum", ()):
            return True
    return False


class SchemaForAnthropic(unittest.TestCase):
    def test_the_shared_identity_schema_is_the_shape_anthropic_rejects(self):
        # This is the bug, pinned: do not "fix" it by changing the shared
        # schema. OpenAI strict mode wants type-plus-enum including null.
        model = _model_field(prompts.identity_schema())
        self.assertEqual(model["type"], ["string", "null"])
        self.assertIn("F-MAX", model["enum"])
        self.assertIn(None, model["enum"])
        self.assertTrue(_conflicts(model))

    def test_the_copy_sent_to_anthropic_has_no_enum_on_a_union_type(self):
        raw = prompts.identity_schema()
        sent = schema_for_anthropic(raw)
        self.assertEqual(_conflicts(sent), [])
        self.assertTrue(_allows(_model_field(sent), "F-MAX"))
        self.assertTrue(_allows(_model_field(sent), None))

    def test_the_shared_schema_is_not_mutated(self):
        raw = prompts.identity_schema()
        before = raw["properties"]["model"].copy()
        schema_for_anthropic(raw)
        self.assertEqual(raw["properties"]["model"], before)

    def test_a_matching_string_enum_is_left_alone(self):
        # Close-up component / severity / impact: type is a scalar and every
        # enum value is a string. Rewriting those would be invention.
        raw = prompts.closeup_schema()
        sent = schema_for_anthropic(raw)
        item = sent["properties"]["observations"]["items"]["properties"]
        self.assertEqual(item["severity"],
                         raw["properties"]["observations"]["items"]["properties"]["severity"])
        self.assertEqual(item["severity"]["type"], "string")
        self.assertIn("enum", item["severity"])

    def test_nullable_fields_without_an_enum_are_left_alone(self):
        sent = schema_for_anthropic(prompts.identity_schema())
        self.assertEqual(sent["properties"]["make"], {"type": ["string", "null"]})

    def test_closeup_array_constraints_anthropic_rejects_are_stripped(self):
        # The close-up schema caps `component_regions` at 8 and a box at 4
        # numbers. Anthropic only accepts array minItems of 0 or 1, and it
        # does not accept maxItems at all — after identity succeeded, the
        # next pass 400'd on exactly that.
        raw = prompts.closeup_schema()
        self.assertTrue(_unsupported_constraints(raw))
        sent = schema_for_anthropic(raw)
        self.assertEqual(_unsupported_constraints(sent), [])
        self.assertIn("maxItems", raw["properties"]["component_regions"])

    def test_every_schema_this_backend_is_sent_is_acceptable(self):
        schemas = [
            prompts.identity_schema(),
            prompts.BADGE_SCHEMA,
            prompts.closeup_schema(),
            prompts.synthesis_schema(),
            prompts.CALIBRATION_SCHEMA,
        ]
        for schema in schemas:
            sent = schema_for_anthropic(schema)
            self.assertEqual(_conflicts(sent), [])
            self.assertEqual(_unsupported_constraints(sent), [])


class AnthropicSendsAdaptedSchema(unittest.TestCase):
    def test_complete_sends_the_rewritten_schema(self):
        captured = {}

        class Stream:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def get_final_message(self):
                return SimpleNamespace(
                    stop_reason="end_turn",
                    content=[SimpleNamespace(type="text", text="{}")],
                    usage=SimpleNamespace(input_tokens=1, output_tokens=1),
                )

        client = MagicMock()
        client.messages.stream.side_effect = lambda **kw: (
            captured.update(kw) or Stream())
        backend = AnthropicBackend()
        backend._client = client
        backend.complete("identify", [], json_schema=prompts.identity_schema())
        schema = captured["output_config"]["format"]["schema"]
        self.assertEqual(_conflicts(schema), [])
        self.assertTrue(_allows(_model_field(schema), "F-MAX"))
        self.assertTrue(_allows(_model_field(schema), None))
