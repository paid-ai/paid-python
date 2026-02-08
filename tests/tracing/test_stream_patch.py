from openinference.instrumentation.anthropic._stream import _MessagesStream, _Stream
from openinference.instrumentation.anthropic._with_span import _WithSpan
from opentelemetry.trace import INVALID_SPAN

from paid.tracing.anthropic_patches import _patch_stream_context_managers


class MockWithSpan(_WithSpan):
    def __init__(self):
        super().__init__(span=INVALID_SPAN)


class MockAsyncStream:
    def __init__(self, items=None):
        self.items = items or []
        self.entered = False
        self.exited = False

    async def __aenter__(self):
        self.entered = True
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.exited = True
        return False

    def __aiter__(self):
        return self._aiter_impl()

    async def _aiter_impl(self):
        for item in self.items:
            yield item


class MockSyncStream:
    def __init__(self, items=None):
        self.items = items or []
        self.entered = False
        self.exited = False

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.exited = True
        return False

    def __iter__(self):
        return iter(self.items)


_patch_stream_context_managers()


class TestPatchAddsMethodsToMessagesStream:
    def test_has_aenter(self):
        assert hasattr(_MessagesStream, "__aenter__")

    def test_has_aexit(self):
        assert hasattr(_MessagesStream, "__aexit__")

    def test_has_enter(self):
        assert hasattr(_MessagesStream, "__enter__")

    def test_has_exit(self):
        assert hasattr(_MessagesStream, "__exit__")


class TestPatchAddsMethodsToStream:
    def test_has_aenter(self):
        assert hasattr(_Stream, "__aenter__")

    def test_has_aexit(self):
        assert hasattr(_Stream, "__aexit__")

    def test_has_enter(self):
        assert hasattr(_Stream, "__enter__")

    def test_has_exit(self):
        assert hasattr(_Stream, "__exit__")


class TestPatchIdempotent:
    def test_calling_patch_twice_does_not_error(self):
        _patch_stream_context_managers()
        _patch_stream_context_managers()
        assert hasattr(_MessagesStream, "__aenter__")
        assert hasattr(_Stream, "__aenter__")


class TestMessagesStreamAsyncContextManager:

    async def test_aenter_returns_proxy_not_wrapped(self):
        mock_stream = MockAsyncStream()
        proxy = _MessagesStream(mock_stream, MockWithSpan())
        result = await proxy.__aenter__()
        assert result is proxy
        assert result is not mock_stream

    async def test_aenter_delegates_to_wrapped(self):
        mock_stream = MockAsyncStream()
        proxy = _MessagesStream(mock_stream, MockWithSpan())
        await proxy.__aenter__()
        assert mock_stream.entered is True

    async def test_aexit_delegates_to_wrapped(self):
        mock_stream = MockAsyncStream()
        proxy = _MessagesStream(mock_stream, MockWithSpan())
        await proxy.__aexit__(None, None, None)
        assert mock_stream.exited is True

    async def test_full_async_with_round_trip(self):
        mock_stream = MockAsyncStream()
        proxy = _MessagesStream(mock_stream, MockWithSpan())
        async with proxy as stream:
            assert stream is proxy
            assert mock_stream.entered is True
        assert mock_stream.exited is True


class TestMessagesStreamSyncContextManager:

    def test_enter_returns_proxy_not_wrapped(self):
        mock_stream = MockSyncStream()
        proxy = _MessagesStream(mock_stream, MockWithSpan())
        result = proxy.__enter__()
        assert result is proxy

    def test_exit_delegates_to_wrapped(self):
        mock_stream = MockSyncStream()
        proxy = _MessagesStream(mock_stream, MockWithSpan())
        proxy.__exit__(None, None, None)
        assert mock_stream.exited is True

    def test_full_sync_with_round_trip(self):
        mock_stream = MockSyncStream()
        proxy = _MessagesStream(mock_stream, MockWithSpan())
        with proxy as stream:
            assert stream is proxy
            assert mock_stream.entered is True
        assert mock_stream.exited is True


class TestStreamAsyncContextManager:

    async def test_aenter_returns_proxy(self):
        mock_stream = MockAsyncStream()
        proxy = _Stream(mock_stream, MockWithSpan())
        assert await proxy.__aenter__() is proxy

    async def test_full_async_with_round_trip(self):
        mock_stream = MockAsyncStream()
        proxy = _Stream(mock_stream, MockWithSpan())
        async with proxy as stream:
            assert stream is proxy
            assert mock_stream.entered is True
        assert mock_stream.exited is True


class TestStreamSyncContextManager:

    def test_full_sync_with_round_trip(self):
        mock_stream = MockSyncStream()
        proxy = _Stream(mock_stream, MockWithSpan())
        with proxy as stream:
            assert stream is proxy
            assert mock_stream.entered is True
        assert mock_stream.exited is True


class TestGracefulDegradationNoContextManager:

    async def test_aenter_works_without_wrapped_aenter(self):
        proxy = _MessagesStream(object(), MockWithSpan())
        assert await proxy.__aenter__() is proxy

    async def test_aexit_works_without_wrapped_aexit(self):
        proxy = _MessagesStream(object(), MockWithSpan())
        await proxy.__aexit__(None, None, None)
