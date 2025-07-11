# This file was auto-generated by Fern from our API Definition.

import typing

import pydantic
import typing_extensions
from ..core.pydantic_utilities import IS_PYDANTIC_V2, UniversalBaseModel
from ..core.serialization import FieldMetadata
from .address import Address
from .creation_source import CreationSource
from .creation_state import CreationState
from .tax_exempt_status import TaxExemptStatus


class Customer(UniversalBaseModel):
    id: str
    organization_id: typing_extensions.Annotated[str, FieldMetadata(alias="organizationId")]
    name: str
    external_id: typing_extensions.Annotated[typing.Optional[str], FieldMetadata(alias="externalId")] = None
    phone: typing.Optional[str] = None
    employee_count: typing_extensions.Annotated[typing.Optional[float], FieldMetadata(alias="employeeCount")] = None
    annual_revenue: typing_extensions.Annotated[typing.Optional[float], FieldMetadata(alias="annualRevenue")] = None
    tax_exempt_status: typing_extensions.Annotated[
        typing.Optional[TaxExemptStatus], FieldMetadata(alias="taxExemptStatus")
    ] = None
    creation_source: typing_extensions.Annotated[
        typing.Optional[CreationSource], FieldMetadata(alias="creationSource")
    ] = None
    creation_state: typing_extensions.Annotated[
        typing.Optional[CreationState], FieldMetadata(alias="creationState")
    ] = None
    website: typing.Optional[str] = None
    billing_address: typing_extensions.Annotated[typing.Optional[Address], FieldMetadata(alias="billingAddress")] = None

    if IS_PYDANTIC_V2:
        model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(extra="allow", frozen=True)  # type: ignore # Pydantic v2
    else:

        class Config:
            frozen = True
            smart_union = True
            extra = pydantic.Extra.allow
