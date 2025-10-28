from typing import Any, Dict, List, Optional, Union

from opentelemetry.trace import Status, StatusCode

from paid.tracing.tracing import (
    get_paid_tracer,
)

try:
    from mistralai import Mistral, models
    from mistralai.types import UNSET, OptionalNullable
except ImportError:
    raise ImportError(
        "mistralai package is a peer-dependency. To use the Paid wrapper around mistralai "
        "you're assumed to already have mistralai package installed."
    )


class PaidMistral:
    def __init__(self, mistral_client: Mistral):
        self.mistral = mistral_client

    @property
    def ocr(self):
        return OCRWrapper(self.mistral)


class OCRWrapper:
    def __init__(self, mistral_client: Mistral):
        self.mistral = mistral_client

    def process(
        self,
        *,
        model: str,
        document: Union[models.Document, models.DocumentTypedDict],
        id: Optional[str] = None,
        pages: Optional[List[int]] = None,
        include_image_base64: Optional[bool] = None,
        image_limit: Optional[int] = None,
        image_min_size: Optional[int] = None,
        bbox_annotation_format: OptionalNullable[Union[models.ResponseFormat, models.ResponseFormatTypedDict]] = UNSET,
        document_annotation_format: OptionalNullable[
            Union[models.ResponseFormat, models.ResponseFormatTypedDict]
        ] = UNSET,
        retries: Optional[Any] = None,
        server_url: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        http_headers: Optional[Dict[str, str]] = None,
    ) -> Any:
        """
        Process document with OCR synchronously

        Args:
            model: OCR model name (e.g., "mistral-ocr-latest")
            document: Document to run OCR on
            id: Optional ID for the request
            pages: Specific pages user wants to process. List of page numbers starting from 0
            include_image_base64: Include image URLs in response
            image_limit: Max images to extract
            image_min_size: Minimum height and width of image to extract
            bbox_annotation_format: Structured output for extracted bounding boxes/images
            document_annotation_format: Structured output for entire document
            retries: Override default retry configuration
            server_url: Override default server URL
            timeout_ms: Override default request timeout in milliseconds
            http_headers: Additional headers to set or replace on requests
        """
        tracer = get_paid_tracer()

        with tracer.start_as_current_span("mistral.ocr.process") as span:
            attributes = {
                "gen_ai.system": "mistral",
                "gen_ai.operation.name": "ocr",
            }
            if bbox_annotation_format or document_annotation_format:
                attributes["gen_ai.ocr.annotated"] = "true"

            span.set_attributes(attributes)

            try:
                # Make the actual Mistral OCR API call
                response = self.mistral.ocr.process(
                    model=model,
                    document=document,
                    id=id,
                    pages=pages,
                    include_image_base64=include_image_base64,
                    image_limit=image_limit,
                    image_min_size=image_min_size,
                    bbox_annotation_format=bbox_annotation_format,
                    document_annotation_format=document_annotation_format,
                    retries=retries,
                    server_url=server_url,
                    timeout_ms=timeout_ms,
                    http_headers=http_headers,
                )

                if (
                    hasattr(response, "usage_info")
                    and response.usage_info
                    and hasattr(response.usage_info, "pages_processed")
                ):
                    span.set_attribute("gen_ai.ocr.pages_processed", response.usage_info.pages_processed)
                if hasattr(response, "model"):
                    span.set_attribute("gen_ai.response.model", response.model)

                # Mark span as successful
                span.set_status(Status(StatusCode.OK))
                return response

            except Exception as error:
                # Mark span as failed and record error
                span.set_status(Status(StatusCode.ERROR, str(error)))
                span.record_exception(error)
                raise error

    async def process_async(
        self,
        *,
        model: str,
        document: Union[models.Document, models.DocumentTypedDict],
        id: Optional[str] = None,
        pages: Optional[List[int]] = None,
        include_image_base64: Optional[bool] = None,
        image_limit: Optional[int] = None,
        image_min_size: Optional[int] = None,
        bbox_annotation_format: OptionalNullable[Union[models.ResponseFormat, models.ResponseFormatTypedDict]] = UNSET,
        document_annotation_format: OptionalNullable[
            Union[models.ResponseFormat, models.ResponseFormatTypedDict]
        ] = UNSET,
        retries: Optional[Any] = None,
        server_url: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        http_headers: Optional[Dict[str, str]] = None,
    ) -> Any:
        """
        Process document with OCR asynchronously

        Args:
            model: OCR model name (e.g., "mistral-ocr-latest")
            document: Document to run OCR on
            id: Optional ID for the request
            pages: Specific pages user wants to process. List of page numbers starting from 0
            include_image_base64: Include image URLs in response
            image_limit: Max images to extract
            image_min_size: Minimum height and width of image to extract
            bbox_annotation_format: Structured output for extracted bounding boxes/images
            document_annotation_format: Structured output for entire document
            retries: Override default retry configuration
            server_url: Override default server URL
            timeout_ms: Override default request timeout in milliseconds
            http_headers: Additional headers to set or replace on requests
        """
        tracer = get_paid_tracer()

        with tracer.start_as_current_span("mistral.ocr.process_async") as span:
            attributes = {
                "gen_ai.system": "mistral",
                "gen_ai.operation.name": "ocr",
            }
            if bbox_annotation_format or document_annotation_format:
                attributes["gen_ai.ocr.annotated"] = "true"

            span.set_attributes(attributes)

            try:
                # Make the actual Mistral OCR API call asynchronously
                response = await self.mistral.ocr.process_async(
                    model=model,
                    document=document,
                    id=id,
                    pages=pages,
                    include_image_base64=include_image_base64,
                    image_limit=image_limit,
                    image_min_size=image_min_size,
                    bbox_annotation_format=bbox_annotation_format,
                    document_annotation_format=document_annotation_format,
                    retries=retries,
                    server_url=server_url,
                    timeout_ms=timeout_ms,
                    http_headers=http_headers,
                )

                if (
                    hasattr(response, "usage_info")
                    and response.usage_info
                    and hasattr(response.usage_info, "pages_processed")
                ):
                    span.set_attribute("gen_ai.ocr.pages_processed", response.usage_info.pages_processed)
                if hasattr(response, "model"):
                    span.set_attribute("gen_ai.response.model", response.model)

                # Mark span as successful
                span.set_status(Status(StatusCode.OK))
                return response

            except Exception as error:
                # Mark span as failed and record error
                span.set_status(Status(StatusCode.ERROR, str(error)))
                span.record_exception(error)
                raise error
