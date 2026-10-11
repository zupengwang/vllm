# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import json
from unittest.mock import patch

import pytest
from starlette.requests import Request
from starlette.responses import StreamingResponse

from vllm.entrypoints.serve.middleware.log_response import log_response


@pytest.mark.asyncio
async def test_response_logging_preserves_streaming():
    """The client must receive a chunk before the producer needs its next one."""
    received_first = asyncio.Event()
    chunks = [b": keep-alive\n\n", b"data: [DONE]\n\n", b": final comment\n\n"]

    async def body():
        yield chunks[0]
        assert received_first.is_set(), "Response logging consumed the stream eagerly"
        for chunk in chunks[1:]:
            yield chunk

    async def call_next(request):
        return StreamingResponse(body(), media_type="text/event-stream")

    with patch("vllm.entrypoints.serve.middleware.log_response.logger.info") as log:
        response = await log_response(Request({"type": "http"}), call_next)
        assert await anext(response.body_iterator) == chunks[0]
        received_first.set()
        assert [chunk async for chunk in response.body_iterator] == chunks[1:]
        log.assert_any_call(
            "response_body={streaming_complete: no_content, chunks=%d}", 2
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("response_type", ["chat.completion.chunk", "text_completion"])
@pytest.mark.parametrize(
    "content", ["Hello world", "x" * 2050], ids=["short", "truncated"]
)
async def test_response_logging_preserves_content_log(response_type, content):
    choice: dict = {"index": 0}
    if response_type == "chat.completion.chunk":
        choice["delta"] = {"content": content}
    else:
        choice["text"] = content
    event = {
        "id": "cmpl-test",
        "object": response_type,
        "created": 0,
        "model": "test",
        "choices": [choice],
    }
    data = ("data: " + json.dumps(event) + "\n\n").encode()
    chunks = [data[:20], data[20:], b"data: [DONE]\n\n"]

    async def body():
        for chunk in chunks:
            yield chunk

    async def call_next(request):
        return StreamingResponse(body(), media_type="text/event-stream")

    with patch("vllm.entrypoints.serve.middleware.log_response.logger.info") as log:
        response = await log_response(Request({"type": "http"}), call_next)
        assert [chunk async for chunk in response.body_iterator] == chunks
        expected = (
            content if len(content) <= 2048 else content[:2048] + "...[truncated]"
        )
        log.assert_any_call(
            "response_body={streaming_complete: content=%r, chunks=%d}", expected, 3
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("termination", ["error", "cancel", "close"])
async def test_response_logging_releases_stream(termination):
    entered = asyncio.Event()
    closed = asyncio.Event()

    async def body():
        try:
            yield b": keep-alive\n\n"
            entered.set()
            if termination == "error":
                raise ValueError("stream failed")
            await asyncio.Event().wait()
        finally:
            closed.set()

    async def call_next(request):
        return StreamingResponse(body(), media_type="text/event-stream")

    response = await log_response(Request({"type": "http"}), call_next)
    stream = response.body_iterator
    try:
        assert await anext(stream) == b": keep-alive\n\n"
        if termination == "close":
            await stream.aclose()
        elif termination == "error":
            with pytest.raises(ValueError, match="stream failed"):
                await anext(stream)
        else:
            next_chunk = asyncio.create_task(anext(stream))
            try:
                await asyncio.wait_for(entered.wait(), timeout=5)
            finally:
                next_chunk.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await next_chunk
        assert closed.is_set()
    finally:
        await stream.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("chunks", [[], [b'{"answer": 42}'], [b"\xff"]])
async def test_response_logging_preserves_non_streaming_body(chunks):
    async def body():
        for chunk in chunks:
            yield chunk

    async def call_next(request):
        return StreamingResponse(body(), media_type="application/json")

    response = await log_response(Request({"type": "http"}), call_next)
    assert [chunk async for chunk in response.body_iterator] == chunks
