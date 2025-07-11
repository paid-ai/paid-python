# This file was auto-generated by Fern from our API Definition.

import typing

from ..core.client_wrapper import AsyncClientWrapper, SyncClientWrapper
from ..core.request_options import RequestOptions
from ..types.order import Order
from ..types.order_line_create import OrderLineCreate
from .lines.client import AsyncLinesClient, LinesClient
from .raw_client import AsyncRawOrdersClient, RawOrdersClient

# this is used as the default value for optional parameters
OMIT = typing.cast(typing.Any, ...)


class OrdersClient:
    def __init__(self, *, client_wrapper: SyncClientWrapper):
        self._raw_client = RawOrdersClient(client_wrapper=client_wrapper)
        self.lines = LinesClient(client_wrapper=client_wrapper)

    @property
    def with_raw_response(self) -> RawOrdersClient:
        """
        Retrieves a raw implementation of this client that returns raw responses.

        Returns
        -------
        RawOrdersClient
        """
        return self._raw_client

    def list(self, *, request_options: typing.Optional[RequestOptions] = None) -> typing.List[Order]:
        """
        Parameters
        ----------
        request_options : typing.Optional[RequestOptions]
            Request-specific configuration.

        Returns
        -------
        typing.List[Order]
            Success response

        Examples
        --------
        from paid import Paid

        client = Paid(
            token="YOUR_TOKEN",
        )
        client.orders.list()
        """
        _response = self._raw_client.list(request_options=request_options)
        return _response.data

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
    ) -> Order:
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
        Order
            Success response

        Examples
        --------
        from paid import Paid

        client = Paid(
            token="YOUR_TOKEN",
        )
        client.orders.create(
            customer_id="customerId",
            billing_contact_id="billingContactId",
            name="name",
            start_date="startDate",
            currency="currency",
        )
        """
        _response = self._raw_client.create(
            customer_id=customer_id,
            billing_contact_id=billing_contact_id,
            name=name,
            start_date=start_date,
            currency=currency,
            customer_external_id=customer_external_id,
            description=description,
            end_date=end_date,
            order_lines=order_lines,
            request_options=request_options,
        )
        return _response.data

    def get(self, order_id: str, *, request_options: typing.Optional[RequestOptions] = None) -> Order:
        """
        Parameters
        ----------
        order_id : str

        request_options : typing.Optional[RequestOptions]
            Request-specific configuration.

        Returns
        -------
        Order
            Success response

        Examples
        --------
        from paid import Paid

        client = Paid(
            token="YOUR_TOKEN",
        )
        client.orders.get(
            order_id="orderId",
        )
        """
        _response = self._raw_client.get(order_id, request_options=request_options)
        return _response.data

    def delete(self, order_id: str, *, request_options: typing.Optional[RequestOptions] = None) -> None:
        """
        Parameters
        ----------
        order_id : str

        request_options : typing.Optional[RequestOptions]
            Request-specific configuration.

        Returns
        -------
        None

        Examples
        --------
        from paid import Paid

        client = Paid(
            token="YOUR_TOKEN",
        )
        client.orders.delete(
            order_id="orderId",
        )
        """
        _response = self._raw_client.delete(order_id, request_options=request_options)
        return _response.data

    def activate(self, order_id: str, *, request_options: typing.Optional[RequestOptions] = None) -> Order:
        """
        Parameters
        ----------
        order_id : str

        request_options : typing.Optional[RequestOptions]
            Request-specific configuration.

        Returns
        -------
        Order
            Success response

        Examples
        --------
        from paid import Paid

        client = Paid(
            token="YOUR_TOKEN",
        )
        client.orders.activate(
            order_id="orderId",
        )
        """
        _response = self._raw_client.activate(order_id, request_options=request_options)
        return _response.data


class AsyncOrdersClient:
    def __init__(self, *, client_wrapper: AsyncClientWrapper):
        self._raw_client = AsyncRawOrdersClient(client_wrapper=client_wrapper)
        self.lines = AsyncLinesClient(client_wrapper=client_wrapper)

    @property
    def with_raw_response(self) -> AsyncRawOrdersClient:
        """
        Retrieves a raw implementation of this client that returns raw responses.

        Returns
        -------
        AsyncRawOrdersClient
        """
        return self._raw_client

    async def list(self, *, request_options: typing.Optional[RequestOptions] = None) -> typing.List[Order]:
        """
        Parameters
        ----------
        request_options : typing.Optional[RequestOptions]
            Request-specific configuration.

        Returns
        -------
        typing.List[Order]
            Success response

        Examples
        --------
        import asyncio

        from paid import AsyncPaid

        client = AsyncPaid(
            token="YOUR_TOKEN",
        )


        async def main() -> None:
            await client.orders.list()


        asyncio.run(main())
        """
        _response = await self._raw_client.list(request_options=request_options)
        return _response.data

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
    ) -> Order:
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
        Order
            Success response

        Examples
        --------
        import asyncio

        from paid import AsyncPaid

        client = AsyncPaid(
            token="YOUR_TOKEN",
        )


        async def main() -> None:
            await client.orders.create(
                customer_id="customerId",
                billing_contact_id="billingContactId",
                name="name",
                start_date="startDate",
                currency="currency",
            )


        asyncio.run(main())
        """
        _response = await self._raw_client.create(
            customer_id=customer_id,
            billing_contact_id=billing_contact_id,
            name=name,
            start_date=start_date,
            currency=currency,
            customer_external_id=customer_external_id,
            description=description,
            end_date=end_date,
            order_lines=order_lines,
            request_options=request_options,
        )
        return _response.data

    async def get(self, order_id: str, *, request_options: typing.Optional[RequestOptions] = None) -> Order:
        """
        Parameters
        ----------
        order_id : str

        request_options : typing.Optional[RequestOptions]
            Request-specific configuration.

        Returns
        -------
        Order
            Success response

        Examples
        --------
        import asyncio

        from paid import AsyncPaid

        client = AsyncPaid(
            token="YOUR_TOKEN",
        )


        async def main() -> None:
            await client.orders.get(
                order_id="orderId",
            )


        asyncio.run(main())
        """
        _response = await self._raw_client.get(order_id, request_options=request_options)
        return _response.data

    async def delete(self, order_id: str, *, request_options: typing.Optional[RequestOptions] = None) -> None:
        """
        Parameters
        ----------
        order_id : str

        request_options : typing.Optional[RequestOptions]
            Request-specific configuration.

        Returns
        -------
        None

        Examples
        --------
        import asyncio

        from paid import AsyncPaid

        client = AsyncPaid(
            token="YOUR_TOKEN",
        )


        async def main() -> None:
            await client.orders.delete(
                order_id="orderId",
            )


        asyncio.run(main())
        """
        _response = await self._raw_client.delete(order_id, request_options=request_options)
        return _response.data

    async def activate(self, order_id: str, *, request_options: typing.Optional[RequestOptions] = None) -> Order:
        """
        Parameters
        ----------
        order_id : str

        request_options : typing.Optional[RequestOptions]
            Request-specific configuration.

        Returns
        -------
        Order
            Success response

        Examples
        --------
        import asyncio

        from paid import AsyncPaid

        client = AsyncPaid(
            token="YOUR_TOKEN",
        )


        async def main() -> None:
            await client.orders.activate(
                order_id="orderId",
            )


        asyncio.run(main())
        """
        _response = await self._raw_client.activate(order_id, request_options=request_options)
        return _response.data
