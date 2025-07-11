# This file was auto-generated by Fern from our API Definition.

import typing
from json.decoder import JSONDecodeError

from ..core.api_error import ApiError
from ..core.client_wrapper import AsyncClientWrapper, SyncClientWrapper
from ..core.http_response import AsyncHttpResponse, HttpResponse
from ..core.jsonable_encoder import jsonable_encoder
from ..core.pydantic_utilities import parse_obj_as
from ..core.request_options import RequestOptions
from ..core.serialization import convert_and_respect_annotation_metadata
from ..types.order import Order
from ..types.order_line_create import OrderLineCreate

# this is used as the default value for optional parameters
OMIT = typing.cast(typing.Any, ...)


class RawOrdersClient:
    def __init__(self, *, client_wrapper: SyncClientWrapper):
        self._client_wrapper = client_wrapper

    def list(self, *, request_options: typing.Optional[RequestOptions] = None) -> HttpResponse[typing.List[Order]]:
        """
        Parameters
        ----------
        request_options : typing.Optional[RequestOptions]
            Request-specific configuration.

        Returns
        -------
        HttpResponse[typing.List[Order]]
            Success response
        """
        _response = self._client_wrapper.httpx_client.request(
            "orders",
            method="GET",
            request_options=request_options,
        )
        try:
            if 200 <= _response.status_code < 300:
                _data = typing.cast(
                    typing.List[Order],
                    parse_obj_as(
                        type_=typing.List[Order],  # type: ignore
                        object_=_response.json(),
                    ),
                )
                return HttpResponse(response=_response, data=_data)
            _response_json = _response.json()
        except JSONDecodeError:
            raise ApiError(status_code=_response.status_code, headers=dict(_response.headers), body=_response.text)
        raise ApiError(status_code=_response.status_code, headers=dict(_response.headers), body=_response_json)

    def create(
        self,
        *,
        customer_id: str,
        billing_contact_id: str,
        name: str,
        start_date: str,
        currency: str,
        customer_external_id: typing.Optional[str] = OMIT,
        description: typing.Optional[str] = OMIT,
        end_date: typing.Optional[str] = OMIT,
        order_lines: typing.Optional[typing.Sequence[OrderLineCreate]] = OMIT,
        request_options: typing.Optional[RequestOptions] = None,
    ) -> HttpResponse[Order]:
        """
        Parameters
        ----------
        customer_id : str

        billing_contact_id : str

        name : str

        start_date : str

        currency : str

        customer_external_id : typing.Optional[str]

        description : typing.Optional[str]

        end_date : typing.Optional[str]

        order_lines : typing.Optional[typing.Sequence[OrderLineCreate]]

        request_options : typing.Optional[RequestOptions]
            Request-specific configuration.

        Returns
        -------
        HttpResponse[Order]
            Success response
        """
        _response = self._client_wrapper.httpx_client.request(
            "orders",
            method="POST",
            json={
                "customerId": customer_id,
                "customerExternalId": customer_external_id,
                "billingContactId": billing_contact_id,
                "name": name,
                "description": description,
                "startDate": start_date,
                "endDate": end_date,
                "currency": currency,
                "orderLines": convert_and_respect_annotation_metadata(
                    object_=order_lines, annotation=typing.Sequence[OrderLineCreate], direction="write"
                ),
            },
            headers={
                "content-type": "application/json",
            },
            request_options=request_options,
            omit=OMIT,
        )
        try:
            if 200 <= _response.status_code < 300:
                _data = typing.cast(
                    Order,
                    parse_obj_as(
                        type_=Order,  # type: ignore
                        object_=_response.json(),
                    ),
                )
                return HttpResponse(response=_response, data=_data)
            _response_json = _response.json()
        except JSONDecodeError:
            raise ApiError(status_code=_response.status_code, headers=dict(_response.headers), body=_response.text)
        raise ApiError(status_code=_response.status_code, headers=dict(_response.headers), body=_response_json)

    def get(self, order_id: str, *, request_options: typing.Optional[RequestOptions] = None) -> HttpResponse[Order]:
        """
        Parameters
        ----------
        order_id : str

        request_options : typing.Optional[RequestOptions]
            Request-specific configuration.

        Returns
        -------
        HttpResponse[Order]
            Success response
        """
        _response = self._client_wrapper.httpx_client.request(
            f"orders/{jsonable_encoder(order_id)}",
            method="GET",
            request_options=request_options,
        )
        try:
            if 200 <= _response.status_code < 300:
                _data = typing.cast(
                    Order,
                    parse_obj_as(
                        type_=Order,  # type: ignore
                        object_=_response.json(),
                    ),
                )
                return HttpResponse(response=_response, data=_data)
            _response_json = _response.json()
        except JSONDecodeError:
            raise ApiError(status_code=_response.status_code, headers=dict(_response.headers), body=_response.text)
        raise ApiError(status_code=_response.status_code, headers=dict(_response.headers), body=_response_json)

    def delete(self, order_id: str, *, request_options: typing.Optional[RequestOptions] = None) -> HttpResponse[None]:
        """
        Parameters
        ----------
        order_id : str

        request_options : typing.Optional[RequestOptions]
            Request-specific configuration.

        Returns
        -------
        HttpResponse[None]
        """
        _response = self._client_wrapper.httpx_client.request(
            f"orders/{jsonable_encoder(order_id)}",
            method="DELETE",
            request_options=request_options,
        )
        try:
            if 200 <= _response.status_code < 300:
                return HttpResponse(response=_response, data=None)
            _response_json = _response.json()
        except JSONDecodeError:
            raise ApiError(status_code=_response.status_code, headers=dict(_response.headers), body=_response.text)
        raise ApiError(status_code=_response.status_code, headers=dict(_response.headers), body=_response_json)

    def activate(
        self, order_id: str, *, request_options: typing.Optional[RequestOptions] = None
    ) -> HttpResponse[Order]:
        """
        Parameters
        ----------
        order_id : str

        request_options : typing.Optional[RequestOptions]
            Request-specific configuration.

        Returns
        -------
        HttpResponse[Order]
            Success response
        """
        _response = self._client_wrapper.httpx_client.request(
            f"orders/{jsonable_encoder(order_id)}/activate",
            method="POST",
            request_options=request_options,
        )
        try:
            if 200 <= _response.status_code < 300:
                _data = typing.cast(
                    Order,
                    parse_obj_as(
                        type_=Order,  # type: ignore
                        object_=_response.json(),
                    ),
                )
                return HttpResponse(response=_response, data=_data)
            _response_json = _response.json()
        except JSONDecodeError:
            raise ApiError(status_code=_response.status_code, headers=dict(_response.headers), body=_response.text)
        raise ApiError(status_code=_response.status_code, headers=dict(_response.headers), body=_response_json)


class AsyncRawOrdersClient:
    def __init__(self, *, client_wrapper: AsyncClientWrapper):
        self._client_wrapper = client_wrapper

    async def list(
        self, *, request_options: typing.Optional[RequestOptions] = None
    ) -> AsyncHttpResponse[typing.List[Order]]:
        """
        Parameters
        ----------
        request_options : typing.Optional[RequestOptions]
            Request-specific configuration.

        Returns
        -------
        AsyncHttpResponse[typing.List[Order]]
            Success response
        """
        _response = await self._client_wrapper.httpx_client.request(
            "orders",
            method="GET",
            request_options=request_options,
        )
        try:
            if 200 <= _response.status_code < 300:
                _data = typing.cast(
                    typing.List[Order],
                    parse_obj_as(
                        type_=typing.List[Order],  # type: ignore
                        object_=_response.json(),
                    ),
                )
                return AsyncHttpResponse(response=_response, data=_data)
            _response_json = _response.json()
        except JSONDecodeError:
            raise ApiError(status_code=_response.status_code, headers=dict(_response.headers), body=_response.text)
        raise ApiError(status_code=_response.status_code, headers=dict(_response.headers), body=_response_json)

    async def create(
        self,
        *,
        customer_id: str,
        billing_contact_id: str,
        name: str,
        start_date: str,
        currency: str,
        customer_external_id: typing.Optional[str] = OMIT,
        description: typing.Optional[str] = OMIT,
        end_date: typing.Optional[str] = OMIT,
        order_lines: typing.Optional[typing.Sequence[OrderLineCreate]] = OMIT,
        request_options: typing.Optional[RequestOptions] = None,
    ) -> AsyncHttpResponse[Order]:
        """
        Parameters
        ----------
        customer_id : str

        billing_contact_id : str

        name : str

        start_date : str

        currency : str

        customer_external_id : typing.Optional[str]

        description : typing.Optional[str]

        end_date : typing.Optional[str]

        order_lines : typing.Optional[typing.Sequence[OrderLineCreate]]

        request_options : typing.Optional[RequestOptions]
            Request-specific configuration.

        Returns
        -------
        AsyncHttpResponse[Order]
            Success response
        """
        _response = await self._client_wrapper.httpx_client.request(
            "orders",
            method="POST",
            json={
                "customerId": customer_id,
                "customerExternalId": customer_external_id,
                "billingContactId": billing_contact_id,
                "name": name,
                "description": description,
                "startDate": start_date,
                "endDate": end_date,
                "currency": currency,
                "orderLines": convert_and_respect_annotation_metadata(
                    object_=order_lines, annotation=typing.Sequence[OrderLineCreate], direction="write"
                ),
            },
            headers={
                "content-type": "application/json",
            },
            request_options=request_options,
            omit=OMIT,
        )
        try:
            if 200 <= _response.status_code < 300:
                _data = typing.cast(
                    Order,
                    parse_obj_as(
                        type_=Order,  # type: ignore
                        object_=_response.json(),
                    ),
                )
                return AsyncHttpResponse(response=_response, data=_data)
            _response_json = _response.json()
        except JSONDecodeError:
            raise ApiError(status_code=_response.status_code, headers=dict(_response.headers), body=_response.text)
        raise ApiError(status_code=_response.status_code, headers=dict(_response.headers), body=_response_json)

    async def get(
        self, order_id: str, *, request_options: typing.Optional[RequestOptions] = None
    ) -> AsyncHttpResponse[Order]:
        """
        Parameters
        ----------
        order_id : str

        request_options : typing.Optional[RequestOptions]
            Request-specific configuration.

        Returns
        -------
        AsyncHttpResponse[Order]
            Success response
        """
        _response = await self._client_wrapper.httpx_client.request(
            f"orders/{jsonable_encoder(order_id)}",
            method="GET",
            request_options=request_options,
        )
        try:
            if 200 <= _response.status_code < 300:
                _data = typing.cast(
                    Order,
                    parse_obj_as(
                        type_=Order,  # type: ignore
                        object_=_response.json(),
                    ),
                )
                return AsyncHttpResponse(response=_response, data=_data)
            _response_json = _response.json()
        except JSONDecodeError:
            raise ApiError(status_code=_response.status_code, headers=dict(_response.headers), body=_response.text)
        raise ApiError(status_code=_response.status_code, headers=dict(_response.headers), body=_response_json)

    async def delete(
        self, order_id: str, *, request_options: typing.Optional[RequestOptions] = None
    ) -> AsyncHttpResponse[None]:
        """
        Parameters
        ----------
        order_id : str

        request_options : typing.Optional[RequestOptions]
            Request-specific configuration.

        Returns
        -------
        AsyncHttpResponse[None]
        """
        _response = await self._client_wrapper.httpx_client.request(
            f"orders/{jsonable_encoder(order_id)}",
            method="DELETE",
            request_options=request_options,
        )
        try:
            if 200 <= _response.status_code < 300:
                return AsyncHttpResponse(response=_response, data=None)
            _response_json = _response.json()
        except JSONDecodeError:
            raise ApiError(status_code=_response.status_code, headers=dict(_response.headers), body=_response.text)
        raise ApiError(status_code=_response.status_code, headers=dict(_response.headers), body=_response_json)

    async def activate(
        self, order_id: str, *, request_options: typing.Optional[RequestOptions] = None
    ) -> AsyncHttpResponse[Order]:
        """
        Parameters
        ----------
        order_id : str

        request_options : typing.Optional[RequestOptions]
            Request-specific configuration.

        Returns
        -------
        AsyncHttpResponse[Order]
            Success response
        """
        _response = await self._client_wrapper.httpx_client.request(
            f"orders/{jsonable_encoder(order_id)}/activate",
            method="POST",
            request_options=request_options,
        )
        try:
            if 200 <= _response.status_code < 300:
                _data = typing.cast(
                    Order,
                    parse_obj_as(
                        type_=Order,  # type: ignore
                        object_=_response.json(),
                    ),
                )
                return AsyncHttpResponse(response=_response, data=_data)
            _response_json = _response.json()
        except JSONDecodeError:
            raise ApiError(status_code=_response.status_code, headers=dict(_response.headers), body=_response.text)
        raise ApiError(status_code=_response.status_code, headers=dict(_response.headers), body=_response_json)
