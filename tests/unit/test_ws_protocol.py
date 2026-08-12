"""
Tests for the chat WebSocket frame protocol.

The defect being fixed: errors were sent with `send_text`, byte-identical to an
answer, so the Flutter client rendered "Sorry, an unexpected error occurred" as
if the assistant had said it. A client must be able to tell a failure from a
reply without parsing prose.
"""

from __future__ import annotations

import json

import pytest

from adapters.rest.ws_protocol import (
    STAGE_THINKING,
    answer_frame,
    error_frame,
    parse_frame,
    status_frame,
)


def test_answer_and_error_are_distinguishable():
    """The whole point: same text, different type."""
    text = "Sorry, an unexpected error occurred"
    assert parse_frame(answer_frame(text))["type"] == "answer"
    assert parse_frame(error_frame(text))["type"] == "error"


def test_answer_frame_carries_content_and_conversation():
    frame = parse_frame(answer_frame("Here are 3 recipes", conversation_id="abc123"))
    assert frame["content"] == "Here are 3 recipes"
    assert frame["conversation_id"] == "abc123"


def test_error_frame_marks_recoverability():
    """A recoverable error keeps the socket open; the client should not force a reconnect."""
    assert parse_frame(error_frame("try again", recoverable=True))["recoverable"] is True
    assert parse_frame(error_frame("fatal", recoverable=False))["recoverable"] is False


def test_status_frame_carries_stage_and_detail():
    frame = parse_frame(status_frame(STAGE_THINKING, "Working on it…"))
    assert frame["type"] == "status"
    assert frame["stage"] == "thinking"
    assert frame["detail"] == "Working on it…"


def test_frames_survive_newlines_and_quotes():
    """Recipe markdown is full of both; naive text framing would break on them."""
    payload = 'Line 1\n\n- "quoted" item\n{"looks": "like json"}'
    assert parse_frame(answer_frame(payload))["content"] == payload


def test_frames_are_valid_json():
    for raw in (answer_frame("a"), error_frame("b"), status_frame("thinking")):
        json.loads(raw)


def test_parse_rejects_a_frame_without_a_type():
    with pytest.raises(ValueError, match="Malformed frame"):
        parse_frame('{"content": "no type field"}')


def test_parse_rejects_non_object_json():
    with pytest.raises(ValueError, match="Malformed frame"):
        parse_frame('"just a string"')
