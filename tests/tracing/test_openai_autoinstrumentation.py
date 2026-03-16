from openai.types import CreateEmbeddingResponse
from openai.types.create_embedding_response import Usage
from openai.types.embedding import Embedding
from openinference.instrumentation.openai._response_attributes_extractor import (
    _ResponseAttributesExtractor,
)

from paid.tracing.openai_patches import instrument_openai, uninstrument_openai


class TestOpenAIAutoinstrumentation:
    def test_embedding_spans_omit_raw_vectors(self):
        import openai

        extractor = _ResponseAttributesExtractor(openai=openai)
        response = CreateEmbeddingResponse.model_construct(
            model="text-embedding-3-small",
            usage=Usage.model_construct(prompt_tokens=3, total_tokens=3),
            data=[
                Embedding.model_construct(index=0, embedding=[0.1, 0.2, 0.3], object="embedding"),
            ],
            object="list",
        )

        try:
            instrument_openai()
            attrs = dict(extractor.get_attributes_from_response(response=response, request_parameters={}))
        finally:
            uninstrument_openai()

        assert attrs["embedding.model_name"] == "text-embedding-3-small"
        assert attrs["llm.token_count.total"] == 3
        assert attrs["llm.token_count.prompt"] == 3
        assert not any(key.endswith(".embedding.vector") for key in attrs)
