# This file was auto-generated by Fern from our API Definition.

import typing

from ..core.client_wrapper import AsyncClientWrapper, SyncClientWrapper
from ..core.request_options import RequestOptions
from ..types.contact import Contact
from ..types.salutation import Salutation
from .raw_client import AsyncRawContactsClient, RawContactsClient

# this is used as the default value for optional parameters
OMIT = typing.cast(typing.Any, ...)


class ContactsClient:
    def __init__(self, *, client_wrapper: SyncClientWrapper):
        self._raw_client = RawContactsClient(client_wrapper=client_wrapper)

    @property
    def with_raw_response(self) -> RawContactsClient:
        """
        Retrieves a raw implementation of this client that returns raw responses.

        Returns
        -------
        RawContactsClient
        """
        return self._raw_client

    def list(self, *, request_options: typing.Optional[RequestOptions] = None) -> typing.List[Contact]:
        """
        Parameters
        ----------
        request_options : typing.Optional[RequestOptions]
            Request-specific configuration.

        Returns
        -------
        typing.List[Contact]
            Success response

        Examples
        --------
        from paid import Paid

        client = Paid(
            token="YOUR_TOKEN",
        )
        client.contacts.list()
        """
        _response = self._raw_client.list(request_options=request_options)
        return _response.data

    def create(
        self,
        *,
        salutation: Salutation,
        first_name: str,
        last_name: str,
        email: str,
        billing_street: str,
        billing_city: str,
        billing_country: str,
        billing_postal_code: str,
        external_id: typing.Optional[str] = OMIT,
        customer_id: typing.Optional[str] = OMIT,
        customer_external_id: typing.Optional[str] = OMIT,
        phone: typing.Optional[str] = OMIT,
        billing_state_province: typing.Optional[str] = OMIT,
        request_options: typing.Optional[RequestOptions] = None,
    ) -> Contact:
        """
        Parameters
        ----------
        salutation : Salutation

        first_name : str

        last_name : str

        email : str

        billing_street : str

        billing_city : str

        billing_country : str

        billing_postal_code : str

        external_id : typing.Optional[str]

        customer_id : typing.Optional[str]

        customer_external_id : typing.Optional[str]

        phone : typing.Optional[str]

        billing_state_province : typing.Optional[str]

        request_options : typing.Optional[RequestOptions]
            Request-specific configuration.

        Returns
        -------
        Contact
            Success response

        Examples
        --------
        from paid import Paid

        client = Paid(
            token="YOUR_TOKEN",
        )
        client.contacts.create(
            salutation="Mr.",
            first_name="firstName",
            last_name="lastName",
            email="email",
            billing_street="billingStreet",
            billing_city="billingCity",
            billing_country="billingCountry",
            billing_postal_code="billingPostalCode",
        )
        """
        _response = self._raw_client.create(
            salutation=salutation,
            first_name=first_name,
            last_name=last_name,
            email=email,
            billing_street=billing_street,
            billing_city=billing_city,
            billing_country=billing_country,
            billing_postal_code=billing_postal_code,
            external_id=external_id,
            customer_id=customer_id,
            customer_external_id=customer_external_id,
            phone=phone,
            billing_state_province=billing_state_province,
            request_options=request_options,
        )
        return _response.data

    def get(self, contact_id: str, *, request_options: typing.Optional[RequestOptions] = None) -> Contact:
        """
        Parameters
        ----------
        contact_id : str

        request_options : typing.Optional[RequestOptions]
            Request-specific configuration.

        Returns
        -------
        Contact
            Success response

        Examples
        --------
        from paid import Paid

        client = Paid(
            token="YOUR_TOKEN",
        )
        client.contacts.get(
            contact_id="contactId",
        )
        """
        _response = self._raw_client.get(contact_id, request_options=request_options)
        return _response.data

    def delete(self, contact_id: str, *, request_options: typing.Optional[RequestOptions] = None) -> None:
        """
        Parameters
        ----------
        contact_id : str

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
        client.contacts.delete(
            contact_id="contactId",
        )
        """
        _response = self._raw_client.delete(contact_id, request_options=request_options)
        return _response.data

    def get_by_external_id(
        self, external_id: str, *, request_options: typing.Optional[RequestOptions] = None
    ) -> Contact:
        """
        Parameters
        ----------
        external_id : str

        request_options : typing.Optional[RequestOptions]
            Request-specific configuration.

        Returns
        -------
        Contact
            Success response

        Examples
        --------
        from paid import Paid

        client = Paid(
            token="YOUR_TOKEN",
        )
        client.contacts.get_by_external_id(
            external_id="externalId",
        )
        """
        _response = self._raw_client.get_by_external_id(external_id, request_options=request_options)
        return _response.data

    def delete_by_external_id(
        self, external_id: str, *, request_options: typing.Optional[RequestOptions] = None
    ) -> None:
        """
        Parameters
        ----------
        external_id : str

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
        client.contacts.delete_by_external_id(
            external_id="externalId",
        )
        """
        _response = self._raw_client.delete_by_external_id(external_id, request_options=request_options)
        return _response.data


class AsyncContactsClient:
    def __init__(self, *, client_wrapper: AsyncClientWrapper):
        self._raw_client = AsyncRawContactsClient(client_wrapper=client_wrapper)

    @property
    def with_raw_response(self) -> AsyncRawContactsClient:
        """
        Retrieves a raw implementation of this client that returns raw responses.

        Returns
        -------
        AsyncRawContactsClient
        """
        return self._raw_client

    async def list(self, *, request_options: typing.Optional[RequestOptions] = None) -> typing.List[Contact]:
        """
        Parameters
        ----------
        request_options : typing.Optional[RequestOptions]
            Request-specific configuration.

        Returns
        -------
        typing.List[Contact]
            Success response

        Examples
        --------
        import asyncio

        from paid import AsyncPaid

        client = AsyncPaid(
            token="YOUR_TOKEN",
        )


        async def main() -> None:
            await client.contacts.list()


        asyncio.run(main())
        """
        _response = await self._raw_client.list(request_options=request_options)
        return _response.data

    async def create(
        self,
        *,
        salutation: Salutation,
        first_name: str,
        last_name: str,
        email: str,
        billing_street: str,
        billing_city: str,
        billing_country: str,
        billing_postal_code: str,
        external_id: typing.Optional[str] = OMIT,
        customer_id: typing.Optional[str] = OMIT,
        customer_external_id: typing.Optional[str] = OMIT,
        phone: typing.Optional[str] = OMIT,
        billing_state_province: typing.Optional[str] = OMIT,
        request_options: typing.Optional[RequestOptions] = None,
    ) -> Contact:
        """
        Parameters
        ----------
        salutation : Salutation

        first_name : str

        last_name : str

        email : str

        billing_street : str

        billing_city : str

        billing_country : str

        billing_postal_code : str

        external_id : typing.Optional[str]

        customer_id : typing.Optional[str]

        customer_external_id : typing.Optional[str]

        phone : typing.Optional[str]

        billing_state_province : typing.Optional[str]

        request_options : typing.Optional[RequestOptions]
            Request-specific configuration.

        Returns
        -------
        Contact
            Success response

        Examples
        --------
        import asyncio

        from paid import AsyncPaid

        client = AsyncPaid(
            token="YOUR_TOKEN",
        )


        async def main() -> None:
            await client.contacts.create(
                salutation="Mr.",
                first_name="firstName",
                last_name="lastName",
                email="email",
                billing_street="billingStreet",
                billing_city="billingCity",
                billing_country="billingCountry",
                billing_postal_code="billingPostalCode",
            )


        asyncio.run(main())
        """
        _response = await self._raw_client.create(
            salutation=salutation,
            first_name=first_name,
            last_name=last_name,
            email=email,
            billing_street=billing_street,
            billing_city=billing_city,
            billing_country=billing_country,
            billing_postal_code=billing_postal_code,
            external_id=external_id,
            customer_id=customer_id,
            customer_external_id=customer_external_id,
            phone=phone,
            billing_state_province=billing_state_province,
            request_options=request_options,
        )
        return _response.data

    async def get(self, contact_id: str, *, request_options: typing.Optional[RequestOptions] = None) -> Contact:
        """
        Parameters
        ----------
        contact_id : str

        request_options : typing.Optional[RequestOptions]
            Request-specific configuration.

        Returns
        -------
        Contact
            Success response

        Examples
        --------
        import asyncio

        from paid import AsyncPaid

        client = AsyncPaid(
            token="YOUR_TOKEN",
        )


        async def main() -> None:
            await client.contacts.get(
                contact_id="contactId",
            )


        asyncio.run(main())
        """
        _response = await self._raw_client.get(contact_id, request_options=request_options)
        return _response.data

    async def delete(self, contact_id: str, *, request_options: typing.Optional[RequestOptions] = None) -> None:
        """
        Parameters
        ----------
        contact_id : str

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
            await client.contacts.delete(
                contact_id="contactId",
            )


        asyncio.run(main())
        """
        _response = await self._raw_client.delete(contact_id, request_options=request_options)
        return _response.data

    async def get_by_external_id(
        self, external_id: str, *, request_options: typing.Optional[RequestOptions] = None
    ) -> Contact:
        """
        Parameters
        ----------
        external_id : str

        request_options : typing.Optional[RequestOptions]
            Request-specific configuration.

        Returns
        -------
        Contact
            Success response

        Examples
        --------
        import asyncio

        from paid import AsyncPaid

        client = AsyncPaid(
            token="YOUR_TOKEN",
        )


        async def main() -> None:
            await client.contacts.get_by_external_id(
                external_id="externalId",
            )


        asyncio.run(main())
        """
        _response = await self._raw_client.get_by_external_id(external_id, request_options=request_options)
        return _response.data

    async def delete_by_external_id(
        self, external_id: str, *, request_options: typing.Optional[RequestOptions] = None
    ) -> None:
        """
        Parameters
        ----------
        external_id : str

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
            await client.contacts.delete_by_external_id(
                external_id="externalId",
            )


        asyncio.run(main())
        """
        _response = await self._raw_client.delete_by_external_id(external_id, request_options=request_options)
        return _response.data
