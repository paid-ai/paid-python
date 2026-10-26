# Reference
## Products
<details><summary><code>client.products.<a href="src/paid/products/client.py">list_products</a>(...) -&gt; AsyncHttpResponse[ProductListResponse]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Get a list of products for the organization
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.products.list_products()

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**limit:** `typing.Optional[int]` 
    
</dd>
</dl>

<dl>
<dd>

**offset:** `typing.Optional[int]` 
    
</dd>
</dl>

<dl>
<dd>

**name:** `typing.Optional[str]` — Search by product name (case-insensitive, matches anywhere in the name).
    
</dd>
</dl>

<dl>
<dd>

**active:** `typing.Optional[bool]` — Filter by the product's active flag: true or false.
    
</dd>
</dl>

<dl>
<dd>

**archived:** `typing.Optional[bool]` — Filter by archived state: true returns only archived products, false only non-archived. Omit to include both.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.products.<a href="src/paid/products/client.py">create_product</a>(...) -&gt; AsyncHttpResponse[Product]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Creates a new product for the organization. Products are created without pricing: to create product attributes and set their pricing, call the update product endpoint (updateProductById / updateProductByExternalId), which upserts productAttributes.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.products.create_product(
    name="name",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**name:** `str` 
    
</dd>
</dl>

<dl>
<dd>

**description:** `typing.Optional[str]` 
    
</dd>
</dl>

<dl>
<dd>

**active:** `typing.Optional[bool]` 
    
</dd>
</dl>

<dl>
<dd>

**product_code:** `typing.Optional[str]` 
    
</dd>
</dl>

<dl>
<dd>

**external_id:** `typing.Optional[str]` 
    
</dd>
</dl>

<dl>
<dd>

**metadata:** `typing.Optional[typing.Dict[str, typing.Any]]` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.products.<a href="src/paid/products/client.py">get_product_by_id</a>(...) -&gt; AsyncHttpResponse[ProductDetail]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Get a product by ID, including its product attributes with pricing details
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.products.get_product_by_id(
    id="id",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**id:** `str` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.products.<a href="src/paid/products/client.py">update_product_by_id</a>(...) -&gt; AsyncHttpResponse[ProductDetail]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Update a product by ID. Also creates and edits product attributes: productAttributes upserts attributes and sets their pricing (metering event, price points, credit brackets). This is the endpoint to use to add pricing to a product created without any.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.products.update_product_by_id(
    id="id",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**id:** `str` 
    
</dd>
</dl>

<dl>
<dd>

**name:** `typing.Optional[str]` 
    
</dd>
</dl>

<dl>
<dd>

**description:** `typing.Optional[str]` 
    
</dd>
</dl>

<dl>
<dd>

**active:** `typing.Optional[bool]` 
    
</dd>
</dl>

<dl>
<dd>

**product_code:** `typing.Optional[str]` 
    
</dd>
</dl>

<dl>
<dd>

**external_id:** `typing.Optional[str]` 
    
</dd>
</dl>

<dl>
<dd>

**metadata:** `typing.Optional[typing.Dict[str, typing.Any]]` 
    
</dd>
</dl>

<dl>
<dd>

**product_attributes:** `typing.Optional[typing.Sequence[ProductAttributeUpsert]]` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.products.<a href="src/paid/products/client.py">get_product_by_external_id</a>(...) -&gt; AsyncHttpResponse[ProductDetail]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Get a product by external ID, including its product attributes with pricing details
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.products.get_product_by_external_id(
    external_id="externalId",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**external_id:** `str` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.products.<a href="src/paid/products/client.py">update_product_by_external_id</a>(...) -&gt; AsyncHttpResponse[ProductDetail]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Update a product by external ID. Also creates and edits product attributes: productAttributes upserts attributes and sets their pricing (metering event, price points, credit brackets). This is the endpoint to use to add pricing to a product created without any.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.products.update_product_by_external_id(
    external_id_="externalId",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**external_id_:** `str` 
    
</dd>
</dl>

<dl>
<dd>

**name:** `typing.Optional[str]` 
    
</dd>
</dl>

<dl>
<dd>

**description:** `typing.Optional[str]` 
    
</dd>
</dl>

<dl>
<dd>

**active:** `typing.Optional[bool]` 
    
</dd>
</dl>

<dl>
<dd>

**product_code:** `typing.Optional[str]` 
    
</dd>
</dl>

<dl>
<dd>

**external_id:** `typing.Optional[str]` 
    
</dd>
</dl>

<dl>
<dd>

**metadata:** `typing.Optional[typing.Dict[str, typing.Any]]` 
    
</dd>
</dl>

<dl>
<dd>

**product_attributes:** `typing.Optional[typing.Sequence[ProductAttributeUpsert]]` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

## Plans
<details><summary><code>client.plans.<a href="src/paid/plans/client.py">list_plans</a>(...) -&gt; AsyncHttpResponse[PlanListResponse]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Returns plans for your organization, including archived plans by default, optionally filtered to a product.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.plans.list_plans()

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**limit:** `typing.Optional[int]` 
    
</dd>
</dl>

<dl>
<dd>

**offset:** `typing.Optional[int]` 
    
</dd>
</dl>

<dl>
<dd>

**product_id:** `typing.Optional[str]` 
    
</dd>
</dl>

<dl>
<dd>

**external_product_id:** `typing.Optional[str]` 
    
</dd>
</dl>

<dl>
<dd>

**include_archived:** `typing.Optional[bool]` — Whether to include archived plans in the response. Defaults to true.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.plans.<a href="src/paid/plans/client.py">create_plan</a>(...) -&gt; AsyncHttpResponse[Plan]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Creates a new plan for a product.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import (
    Paid,
    PlanAttributeInput,
    ProductPricingInput_RecurringPerUnit,
    ProductSimplePricePoint,
)

client = Paid(
    token="YOUR_TOKEN",
)
client.plans.create_plan(
    product_id="productId",
    attributes=[
        PlanAttributeInput(
            product_attribute_id="productAttributeId",
            pricing=ProductPricingInput_RecurringPerUnit(
                billing_frequency="Monthly",
                price_points=[
                    ProductSimplePricePoint(
                        currency="USD",
                        unit_price=99,
                    )
                ],
            ),
        )
    ],
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**product_id:** `str` — Paid product ID, for example `prod_abc123`
    
</dd>
</dl>

<dl>
<dd>

**attributes:** `typing.Sequence[PlanAttributeInput]` 
    
</dd>
</dl>

<dl>
<dd>

**name:** `typing.Optional[str]` 
    
</dd>
</dl>

<dl>
<dd>

**external_id:** `typing.Optional[str]` — Your stable identifier for this plan
    
</dd>
</dl>

<dl>
<dd>

**supported_currencies:** `typing.Optional[typing.Sequence[str]]` 
    
</dd>
</dl>

<dl>
<dd>

**is_default:** `typing.Optional[bool]` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.plans.<a href="src/paid/plans/client.py">update_plan_upgrade_path</a>(...) -&gt; AsyncHttpResponse[PlanUpgradePathResponse]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Updates upgrade path ordering for plans within a product, grouped by billing frequency.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid, PlanUpgradePathGroup

client = Paid(
    token="YOUR_TOKEN",
)
client.plans.update_plan_upgrade_path(
    product_id="productId",
    groups=[
        PlanUpgradePathGroup(
            plan_ids=["planIds"],
        )
    ],
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**product_id:** `str` — Paid product ID, for example `prod_abc123`
    
</dd>
</dl>

<dl>
<dd>

**groups:** `typing.Sequence[PlanUpgradePathGroup]` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.plans.<a href="src/paid/plans/client.py">get_plan_by_id</a>(...) -&gt; AsyncHttpResponse[Plan]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Get a plan by Paid plan ID.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.plans.get_plan_by_id(
    id="id",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**id:** `str` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.plans.<a href="src/paid/plans/client.py">update_plan_by_id</a>(...) -&gt; AsyncHttpResponse[Plan]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Update a plan by Paid plan ID. If attributes are provided, they replace the plan's existing attributes. Set status to archive or restore the plan.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.plans.update_plan_by_id(
    id="id",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**id:** `str` 
    
</dd>
</dl>

<dl>
<dd>

**name:** `typing.Optional[str]` 
    
</dd>
</dl>

<dl>
<dd>

**external_id:** `typing.Optional[str]` 
    
</dd>
</dl>

<dl>
<dd>

**supported_currencies:** `typing.Optional[typing.Sequence[str]]` 
    
</dd>
</dl>

<dl>
<dd>

**is_default:** `typing.Optional[bool]` 
    
</dd>
</dl>

<dl>
<dd>

**status:** `typing.Optional[UpdatePlanRequestStatus]` — Set to `archived` to archive this plan, or `active` to restore it.
    
</dd>
</dl>

<dl>
<dd>

**attributes:** `typing.Optional[typing.Sequence[PlanAttributeInput]]` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.plans.<a href="src/paid/plans/client.py">get_plan_by_external_id</a>(...) -&gt; AsyncHttpResponse[Plan]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Get a plan by your external plan ID.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.plans.get_plan_by_external_id(
    external_id="externalId",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**external_id:** `str` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.plans.<a href="src/paid/plans/client.py">update_plan_by_external_id</a>(...) -&gt; AsyncHttpResponse[Plan]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Update a plan by your external plan ID. If attributes are provided, they replace the plan's existing attributes. Set status to archive or restore the plan.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.plans.update_plan_by_external_id(
    external_id_="externalId",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**external_id_:** `str` 
    
</dd>
</dl>

<dl>
<dd>

**name:** `typing.Optional[str]` 
    
</dd>
</dl>

<dl>
<dd>

**external_id:** `typing.Optional[str]` 
    
</dd>
</dl>

<dl>
<dd>

**supported_currencies:** `typing.Optional[typing.Sequence[str]]` 
    
</dd>
</dl>

<dl>
<dd>

**is_default:** `typing.Optional[bool]` 
    
</dd>
</dl>

<dl>
<dd>

**status:** `typing.Optional[UpdatePlanRequestStatus]` — Set to `archived` to archive this plan, or `active` to restore it.
    
</dd>
</dl>

<dl>
<dd>

**attributes:** `typing.Optional[typing.Sequence[PlanAttributeInput]]` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

## Customers
<details><summary><code>client.customers.<a href="src/paid/customers/client.py">list_customers</a>(...) -&gt; AsyncHttpResponse[CustomerListResponse]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Get a list of customers for the organization
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.customers.list_customers()

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**limit:** `typing.Optional[int]` 
    
</dd>
</dl>

<dl>
<dd>

**offset:** `typing.Optional[int]` 
    
</dd>
</dl>

<dl>
<dd>

**name:** `typing.Optional[str]` — Search by customer name (case-insensitive, matches anywhere in the name).
    
</dd>
</dl>

<dl>
<dd>

**status:** `typing.Optional[ListCustomersRequestStatus]` — Filter by customer status. churned: customers marked as churned. active: everyone else.
    
</dd>
</dl>

<dl>
<dd>

**creation_state:** `typing.Optional[ListCustomersRequestCreationState]` — Filter by creation state: draft or active.
    
</dd>
</dl>

<dl>
<dd>

**created_at_from:** `typing.Optional[str]` — Only customers created on or after this date. Accepts an ISO 8601 date or date-time. Date-only values (e.g. 2026-06-30) are treated as UTC; date-times without an explicit timezone offset are ambiguous, so include one (e.g. 2026-06-30T00:00:00-05:00) when precision matters.
    
</dd>
</dl>

<dl>
<dd>

**created_at_to:** `typing.Optional[str]` — Only customers created on or before this date. Accepts an ISO 8601 date or date-time. Date-only values (e.g. 2026-06-30) are treated as UTC; date-times without an explicit timezone offset are ambiguous, so include one (e.g. 2026-06-30T00:00:00-05:00) when precision matters.
    
</dd>
</dl>

<dl>
<dd>

**external_id:** `typing.Optional[str]` — Filter by your external customer ID (exact match).
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.customers.<a href="src/paid/customers/client.py">create_customer</a>(...) -&gt; AsyncHttpResponse[Customer]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Creates a new customer for the organization. `externalId` shares one set of ids with customer unit ids and aliases, unique across the organization and ignoring letter case: an external id that another customer already holds as a unit id or alias, or as its external id in another case, is refused with `409 EXTERNAL_ID_IN_USE`. The same id in the same case is `400 DUPLICATE_EXTERNAL_ID`, as before.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.customers.create_customer(
    name="name",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**name:** `str` 
    
</dd>
</dl>

<dl>
<dd>

**legal_name:** `typing.Optional[str]` 
    
</dd>
</dl>

<dl>
<dd>

**email:** `typing.Optional[str]` 
    
</dd>
</dl>

<dl>
<dd>

**phone:** `typing.Optional[str]` 
    
</dd>
</dl>

<dl>
<dd>

**website:** `typing.Optional[str]` 
    
</dd>
</dl>

<dl>
<dd>

**external_id:** `typing.Optional[str]` 
    
</dd>
</dl>

<dl>
<dd>

**billing_address:** `typing.Optional[CustomerBillingAddressInput]` 
    
</dd>
</dl>

<dl>
<dd>

**creation_state:** `typing.Optional[CustomerCreationState]` 
    
</dd>
</dl>

<dl>
<dd>

**vat_number:** `typing.Optional[str]` 
    
</dd>
</dl>

<dl>
<dd>

**metadata:** `typing.Optional[typing.Dict[str, typing.Any]]` 
    
</dd>
</dl>

<dl>
<dd>

**default_currency:** `typing.Optional[str]` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.customers.<a href="src/paid/customers/client.py">list_customer_aliases</a>(...) -&gt; AsyncHttpResponse[CustomerAliasListResponse]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

List alternate external identifiers that resolve to a customer by Paid display ID.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.customers.list_customer_aliases(
    id="cus_abc123",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**id:** `str` — Paid customer display id
    
</dd>
</dl>

<dl>
<dd>

**limit:** `typing.Optional[int]` 
    
</dd>
</dl>

<dl>
<dd>

**offset:** `typing.Optional[int]` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.customers.<a href="src/paid/customers/client.py">create_customer_alias</a>(...) -&gt; AsyncHttpResponse[CustomerAlias]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Create an alternate external identifier for a customer by Paid display ID. The alias shares one set of ids with customer external ids and unit ids, unique within the organization and ignoring letter case: an alias that another customer already holds as its external id, a unit id or an alias is refused with `409 EXTERNAL_ID_IN_USE` (an exact match keeps its older code, `ALIAS_CONFLICTS_WITH_CUSTOMER_EXTERNAL_ID` or `CUSTOMER_ALIAS_ALREADY_EXISTS`).
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.customers.create_customer_alias(
    id="cus_abc123",
    alias="child-customer-1",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**id:** `str` — Paid customer display id
    
</dd>
</dl>

<dl>
<dd>

**alias:** `str` — Alternate external identifier that should resolve to this customer.
    
</dd>
</dl>

<dl>
<dd>

**name:** `typing.Optional[str]` — Optional display name for this alias.
    
</dd>
</dl>

<dl>
<dd>

**description:** `typing.Optional[str]` — Optional note describing where this alias comes from.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.customers.<a href="src/paid/customers/client.py">delete_customer_alias</a>(...) -&gt; AsyncHttpResponse[EmptyResponse]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Remove an alternate external identifier from a customer by Paid display ID.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.customers.delete_customer_alias(
    id="cus_abc123",
    alias="child-customer-1",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**id:** `str` — Paid customer display id
    
</dd>
</dl>

<dl>
<dd>

**alias:** `str` — Customer alias value.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.customers.<a href="src/paid/customers/client.py">get_customer_by_id</a>(...) -&gt; AsyncHttpResponse[Customer]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Get a customer by Paid display ID. Use the value returned as `customer.id`, for example `cus_abc123`. If you have your own customer ID, use `GET /api/v2/customers/external/{externalId}`.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.customers.get_customer_by_id(
    id="cus_abc123",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**id:** `str` — Paid customer display id
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.customers.<a href="src/paid/customers/client.py">update_customer_by_id</a>(...) -&gt; AsyncHttpResponse[Customer]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Update a customer by Paid display ID. Use the value returned as `customer.id`, for example `cus_abc123`. If you have your own customer ID, use `PUT /api/v2/customers/external/{externalId}`. Changing `externalId` follows the organization-wide id rule: a new id that another customer already holds as its external id, a unit id or an alias, ignoring letter case, is refused with `409 EXTERNAL_ID_IN_USE`; the same id in the same case stays `400 DUPLICATE_EXTERNAL_ID`. When the id changes, the previous id is kept as an alias of the customer, so signals that still use it keep resolving to it.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.customers.update_customer_by_id(
    id="cus_abc123",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**id:** `str` — Paid customer display id
    
</dd>
</dl>

<dl>
<dd>

**name:** `typing.Optional[str]` 
    
</dd>
</dl>

<dl>
<dd>

**legal_name:** `typing.Optional[str]` 
    
</dd>
</dl>

<dl>
<dd>

**email:** `typing.Optional[str]` 
    
</dd>
</dl>

<dl>
<dd>

**phone:** `typing.Optional[str]` 
    
</dd>
</dl>

<dl>
<dd>

**website:** `typing.Optional[str]` 
    
</dd>
</dl>

<dl>
<dd>

**external_id:** `typing.Optional[str]` 
    
</dd>
</dl>

<dl>
<dd>

**billing_address:** `typing.Optional[CustomerBillingAddressInput]` 
    
</dd>
</dl>

<dl>
<dd>

**creation_state:** `typing.Optional[CustomerCreationState]` 
    
</dd>
</dl>

<dl>
<dd>

**churn_date:** `typing.Optional[dt.datetime]` 
    
</dd>
</dl>

<dl>
<dd>

**vat_number:** `typing.Optional[str]` 
    
</dd>
</dl>

<dl>
<dd>

**metadata:** `typing.Optional[typing.Dict[str, typing.Any]]` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.customers.<a href="src/paid/customers/client.py">delete_customer_by_id</a>(...) -&gt; AsyncHttpResponse[EmptyResponse]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Delete a customer by Paid display ID. Use the value returned as `customer.id`, for example `cus_abc123`. If you have your own customer ID, use `DELETE /api/v2/customers/external/{externalId}`. A customer that still has orders, invoices, payments, contacts, child customers or other linked records is not deleted: the request returns 409 `CUSTOMER_HAS_ASSOCIATIONS` and `details` lists what is linked.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.customers.delete_customer_by_id(
    id="cus_abc123",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**id:** `str` — Paid customer display id
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.customers.<a href="src/paid/customers/client.py">get_customer_state_by_id</a>(...) -&gt; AsyncHttpResponse[CustomerState]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Get the current customer state by Paid display ID
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.customers.get_customer_state_by_id(
    id="cus_abc123",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**id:** `str` — Paid customer display id
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.customers.<a href="src/paid/customers/client.py">list_customer_aliases_by_external_id</a>(...) -&gt; AsyncHttpResponse[CustomerAliasListResponse]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

List alternate external identifiers that resolve to a customer by external ID.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.customers.list_customer_aliases_by_external_id(
    external_id="customer_123",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**external_id:** `str` — Customer ID from the integrator's system, stored on Paid as `externalId`.
    
</dd>
</dl>

<dl>
<dd>

**limit:** `typing.Optional[int]` 
    
</dd>
</dl>

<dl>
<dd>

**offset:** `typing.Optional[int]` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.customers.<a href="src/paid/customers/client.py">create_customer_alias_by_external_id</a>(...) -&gt; AsyncHttpResponse[CustomerAlias]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Create an alternate external identifier for a customer by external ID. The alias shares one set of ids with customer external ids and unit ids, unique within the organization and ignoring letter case: an alias that another customer already holds as its external id, a unit id or an alias is refused with `409 EXTERNAL_ID_IN_USE` (an exact match keeps its older code, `ALIAS_CONFLICTS_WITH_CUSTOMER_EXTERNAL_ID` or `CUSTOMER_ALIAS_ALREADY_EXISTS`).
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.customers.create_customer_alias_by_external_id(
    external_id="customer_123",
    alias="child-customer-1",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**external_id:** `str` — Customer ID from the integrator's system, stored on Paid as `externalId`.
    
</dd>
</dl>

<dl>
<dd>

**alias:** `str` — Alternate external identifier that should resolve to this customer.
    
</dd>
</dl>

<dl>
<dd>

**name:** `typing.Optional[str]` — Optional display name for this alias.
    
</dd>
</dl>

<dl>
<dd>

**description:** `typing.Optional[str]` — Optional note describing where this alias comes from.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.customers.<a href="src/paid/customers/client.py">delete_customer_alias_by_external_id</a>(...) -&gt; AsyncHttpResponse[EmptyResponse]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Remove an alternate external identifier from a customer by external ID.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.customers.delete_customer_alias_by_external_id(
    external_id="customer_123",
    alias="child-customer-1",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**external_id:** `str` — Customer ID from the integrator's system, stored on Paid as `externalId`.
    
</dd>
</dl>

<dl>
<dd>

**alias:** `str` — Customer alias value.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.customers.<a href="src/paid/customers/client.py">get_customer_by_external_id</a>(...) -&gt; AsyncHttpResponse[Customer]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Get a customer by external ID
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.customers.get_customer_by_external_id(
    external_id="customer_123",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**external_id:** `str` — Customer ID from the integrator's system, stored on Paid as `externalId`.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.customers.<a href="src/paid/customers/client.py">update_customer_by_external_id</a>(...) -&gt; AsyncHttpResponse[Customer]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Update a customer by external ID. Changing `externalId` follows the organization-wide id rule: a new id that another customer already holds as its external id, a unit id or an alias, ignoring letter case, is refused with `409 EXTERNAL_ID_IN_USE`; the same id in the same case stays `400 DUPLICATE_EXTERNAL_ID`. When the id changes, the previous id is kept as an alias of the customer, so signals that still use it keep resolving to it.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.customers.update_customer_by_external_id(
    external_id_="customer_123",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**external_id_:** `str` — Customer ID from the integrator's system, stored on Paid as `externalId`.
    
</dd>
</dl>

<dl>
<dd>

**name:** `typing.Optional[str]` 
    
</dd>
</dl>

<dl>
<dd>

**legal_name:** `typing.Optional[str]` 
    
</dd>
</dl>

<dl>
<dd>

**email:** `typing.Optional[str]` 
    
</dd>
</dl>

<dl>
<dd>

**phone:** `typing.Optional[str]` 
    
</dd>
</dl>

<dl>
<dd>

**website:** `typing.Optional[str]` 
    
</dd>
</dl>

<dl>
<dd>

**external_id:** `typing.Optional[str]` 
    
</dd>
</dl>

<dl>
<dd>

**billing_address:** `typing.Optional[CustomerBillingAddressInput]` 
    
</dd>
</dl>

<dl>
<dd>

**creation_state:** `typing.Optional[CustomerCreationState]` 
    
</dd>
</dl>

<dl>
<dd>

**churn_date:** `typing.Optional[dt.datetime]` 
    
</dd>
</dl>

<dl>
<dd>

**vat_number:** `typing.Optional[str]` 
    
</dd>
</dl>

<dl>
<dd>

**metadata:** `typing.Optional[typing.Dict[str, typing.Any]]` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.customers.<a href="src/paid/customers/client.py">delete_customer_by_external_id</a>(...) -&gt; AsyncHttpResponse[EmptyResponse]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Delete a customer by external ID. A customer that still has orders, invoices, payments, contacts, child customers or other linked records is not deleted: the request returns 409 `CUSTOMER_HAS_ASSOCIATIONS` and `details` lists what is linked.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.customers.delete_customer_by_external_id(
    external_id="customer_123",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**external_id:** `str` — Customer ID from the integrator's system, stored on Paid as `externalId`.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.customers.<a href="src/paid/customers/client.py">get_customer_state_by_external_id</a>(...) -&gt; AsyncHttpResponse[CustomerState]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Primary integration endpoint for agents and programmatic clients using their own customer IDs. Use the value you stored on `customer.externalId`, for example `customer_123`.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.customers.get_customer_state_by_external_id(
    external_id="customer_123",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**external_id:** `str` — Customer ID from the integrator's system, stored on Paid as `externalId`.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.customers.<a href="src/paid/customers/client.py">get_customer_credit_balances</a>(...) -&gt; AsyncHttpResponse[CreditBalanceListResponse]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Get current customer credit balances grouped by currency for a Paid customer display ID. Use the value returned as `customer.id`, for example `cus_abc123`. If you have your own customer ID, use `/api/v2/customers/external/{externalId}/credits/balances`.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.customers.get_customer_credit_balances(
    id="cus_abc123",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**id:** `str` — Paid customer display id
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.customers.<a href="src/paid/customers/client.py">grant_customer_credits</a>(...) -&gt; AsyncHttpResponse[GrantCustomerCreditsResponse]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Immediately grant credits to a customer using an active credit currency key.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
import datetime

from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.customers.grant_customer_credits(
    id="cus_abc123",
    credit_currency_key="api_credits",
    amount=10000.0,
    starts_at=datetime.datetime.fromisoformat(
        "2026-06-05 12:00:00+00:00",
    ),
    expires_at=datetime.datetime.fromisoformat(
        "2026-12-31 23:59:59+00:00",
    ),
    idempotency_key="grant_acme_2026-06-05",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**id:** `str` — Paid customer display id
    
</dd>
</dl>

<dl>
<dd>

**credit_currency_key:** `str` — Stable machine-readable key for the active credit currency to grant.
    
</dd>
</dl>

<dl>
<dd>

**amount:** `float` — Number of credits to grant, exact to at most 6 decimal places. This is not a monetary amount.
    
</dd>
</dl>

<dl>
<dd>

**starts_at:** `typing.Optional[dt.datetime]` — When these credits become spendable, as an RFC3339 datetime with timezone. Must be at or before the current server time. Defaults to the current server time when omitted.
    
</dd>
</dl>

<dl>
<dd>

**expires_at:** `typing.Optional[dt.datetime]` — When these credits expire, as an RFC3339 datetime with timezone. Omit or set null for credits that do not expire.
    
</dd>
</dl>

<dl>
<dd>

**idempotency_key:** `typing.Optional[str]` — Unique key for this grant within your organization. A retry with the same key and request body returns the original grant with `created: false` instead of granting again. Reusing a key for a different customer or request body returns 409.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.customers.<a href="src/paid/customers/client.py">get_customer_credit_balances_by_external_id</a>(...) -&gt; AsyncHttpResponse[CreditBalanceListResponse]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Get current customer credit balances grouped by currency, looked up by external ID
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.customers.get_customer_credit_balances_by_external_id(
    external_id="customer_123",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**external_id:** `str` — Customer ID from the integrator's system, stored on Paid as `externalId`.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.customers.<a href="src/paid/customers/client.py">list_customer_pending_credit_consumption</a>(...) -&gt; AsyncHttpResponse[PendingCreditConsumptionListResponse]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

List credit consumption that was recorded before a matching credit pool existed — for example usage that arrived before an invoice was paid or before a new period's credits were granted. Entries leave this list once they are applied to a pool or settled. Use the value returned as `customer.id`, for example `cus_abc123`. If you have your own customer ID, use `/api/v2/customers/external/{externalId}/credits/pending-consumption`.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.customers.list_customer_pending_credit_consumption(
    id="cus_abc123",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**id:** `str` — Paid customer display id
    
</dd>
</dl>

<dl>
<dd>

**limit:** `typing.Optional[int]` 
    
</dd>
</dl>

<dl>
<dd>

**offset:** `typing.Optional[int]` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.customers.<a href="src/paid/customers/client.py">list_customer_pending_credit_consumption_by_external_id</a>(...) -&gt; AsyncHttpResponse[PendingCreditConsumptionListResponse]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

List credit consumption recorded before a matching credit pool existed, for a customer looked up by external ID.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.customers.list_customer_pending_credit_consumption_by_external_id(
    external_id="customer_123",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**external_id:** `str` — Customer ID from the integrator's system, stored on Paid as `externalId`.
    
</dd>
</dl>

<dl>
<dd>

**limit:** `typing.Optional[int]` 
    
</dd>
</dl>

<dl>
<dd>

**offset:** `typing.Optional[int]` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.customers.<a href="src/paid/customers/client.py">grant_customer_credits_by_external_id</a>(...) -&gt; AsyncHttpResponse[GrantCustomerCreditsResponse]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Immediately grant credits to a customer looked up by external ID using an active credit currency key.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
import datetime

from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.customers.grant_customer_credits_by_external_id(
    external_id="customer_123",
    credit_currency_key="api_credits",
    amount=10000.0,
    starts_at=datetime.datetime.fromisoformat(
        "2026-06-05 12:00:00+00:00",
    ),
    expires_at=datetime.datetime.fromisoformat(
        "2026-12-31 23:59:59+00:00",
    ),
    idempotency_key="grant_acme_2026-06-05",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**external_id:** `str` — Customer ID from the integrator's system, stored on Paid as `externalId`.
    
</dd>
</dl>

<dl>
<dd>

**credit_currency_key:** `str` — Stable machine-readable key for the active credit currency to grant.
    
</dd>
</dl>

<dl>
<dd>

**amount:** `float` — Number of credits to grant, exact to at most 6 decimal places. This is not a monetary amount.
    
</dd>
</dl>

<dl>
<dd>

**starts_at:** `typing.Optional[dt.datetime]` — When these credits become spendable, as an RFC3339 datetime with timezone. Must be at or before the current server time. Defaults to the current server time when omitted.
    
</dd>
</dl>

<dl>
<dd>

**expires_at:** `typing.Optional[dt.datetime]` — When these credits expire, as an RFC3339 datetime with timezone. Omit or set null for credits that do not expire.
    
</dd>
</dl>

<dl>
<dd>

**idempotency_key:** `typing.Optional[str]` — Unique key for this grant within your organization. A retry with the same key and request body returns the original grant with `created: false` instead of granting again. Reusing a key for a different customer or request body returns 409.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.customers.<a href="src/paid/customers/client.py">upsert_customer_user_by_external_id</a>(...) -&gt; AsyncHttpResponse[CustomerUser]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Create or update a customer user using customer and user external IDs
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.customers.upsert_customer_user_by_external_id(
    customer_external_id="customerExternalId",
    user_external_id="userExternalId",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**customer_external_id:** `str` 
    
</dd>
</dl>

<dl>
<dd>

**user_external_id:** `str` 
    
</dd>
</dl>

<dl>
<dd>

**name:** `typing.Optional[str]` 
    
</dd>
</dl>

<dl>
<dd>

**email:** `typing.Optional[str]` 
    
</dd>
</dl>

<dl>
<dd>

**metadata:** `typing.Optional[typing.Dict[str, typing.Any]]` 
    
</dd>
</dl>

<dl>
<dd>

**status:** `typing.Optional[CustomerUserStatus]` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.customers.<a href="src/paid/customers/client.py">get_customer_funding_instructions</a>(...) -&gt; AsyncHttpResponse[CustomerFundingInstructions]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Returns the virtual bank account (Stripe funding instructions) a customer can pay by bank transfer into, by Paid display ID.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.customers.get_customer_funding_instructions(
    id="cus_abc123",
    currency="USD",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**id:** `str` — Paid customer display id
    
</dd>
</dl>

<dl>
<dd>

**currency:** `typing.Optional[str]` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.customers.<a href="src/paid/customers/client.py">create_customer_funding_instructions</a>(...) -&gt; AsyncHttpResponse[CustomerFundingInstructions]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Creates a dedicated virtual bank account for the customer to pay invoices by bank transfer, by Paid display ID. Returns the existing account if one already exists for the currency. Requires a live-mode organization with Stripe connected and Bank Transfers enabled on the Stripe account.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.customers.create_customer_funding_instructions(
    id="cus_abc123",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**id:** `str` — Paid customer display id
    
</dd>
</dl>

<dl>
<dd>

**currency:** `typing.Optional[str]` — Currency of the virtual bank account. Only USD is supported today; defaults to USD.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.customers.<a href="src/paid/customers/client.py">get_customer_funding_instructions_by_external_id</a>(...) -&gt; AsyncHttpResponse[CustomerFundingInstructions]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Returns the virtual bank account (Stripe funding instructions) a customer can pay by bank transfer into, by your external ID.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.customers.get_customer_funding_instructions_by_external_id(
    external_id="externalId",
    currency="USD",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**external_id:** `str` 
    
</dd>
</dl>

<dl>
<dd>

**currency:** `typing.Optional[str]` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.customers.<a href="src/paid/customers/client.py">create_customer_funding_instructions_by_external_id</a>(...) -&gt; AsyncHttpResponse[CustomerFundingInstructions]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Creates a dedicated virtual bank account for the customer to pay invoices by bank transfer, by your external ID. Returns the existing account if one already exists for the currency. Requires a live-mode organization with Stripe connected and Bank Transfers enabled on the Stripe account.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.customers.create_customer_funding_instructions_by_external_id(
    external_id="externalId",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**external_id:** `str` 
    
</dd>
</dl>

<dl>
<dd>

**currency:** `typing.Optional[str]` — Currency of the virtual bank account. Only USD is supported today; defaults to USD.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.customers.<a href="src/paid/customers/client.py">list_customer_units_by_external_id</a>(...) -&gt; AsyncHttpResponse[CustomerUnitListResponse]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Lists the customer's units as a flat list, newest last; assemble the tree from `parentExternalId` (`null` on the root unit, `isRoot: true`). Deleted units are hidden unless `status=DELETED` is given, drafts unless `status=DRAFT`. Filter by `externalType`, or by `parentExternalId` for one level of the tree. Addresses the customer by your external customer id.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.customers.list_customer_units_by_external_id(
    external_id="customer_123",
    parent_external_id="dept-rnd",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**external_id:** `str` — Customer ID from your system, stored on Paid as the customer's `externalId`.
    
</dd>
</dl>

<dl>
<dd>

**limit:** `typing.Optional[int]` 
    
</dd>
</dl>

<dl>
<dd>

**offset:** `typing.Optional[int]` 
    
</dd>
</dl>

<dl>
<dd>

**status:** `typing.Optional[ListCustomerUnitsByExternalIdRequestStatus]` — Filter by status (default ACTIVE; drafts are listed only with DRAFT).
    
</dd>
</dl>

<dl>
<dd>

**external_type:** `typing.Optional[str]` — Filter by external type.
    
</dd>
</dl>

<dl>
<dd>

**parent_external_id:** `typing.Optional[str]` — Your external ID of the parent unit; lists its direct children.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.customers.<a href="src/paid/customers/client.py">create_customer_unit_by_external_id</a>(...) -&gt; AsyncHttpResponse[CustomerUnit]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Creates a unit for this customer. `externalId` is your own key for it: required, immutable, and unique within the organization (no other customer's unit id, external id or alias may be the same, ignoring letter case); every unit route addresses the unit by it, and `name` defaults to it. Omit `parentExternalId` to create the customer's root unit (its first unit; `409 ROOT_EXISTS` if it already has one — a customer created with an external id usable as a unit key already has its root, keyed by that external id, so name it as the parent instead); otherwise the parent must exist (`409 PARENT_NOT_FOUND`) and be ACTIVE. A create whose `externalId` is one of this customer's `DRAFT` units promotes the draft in place: same unit, same id, now ACTIVE with the request's name, type, metadata and parent (`parentExternalId` is required; a draft sits under the root). A signal that names a unit before it exists is accepted; with draft units enabled for your organization it creates a `DRAFT` unit under the customer's root that holds the key, and creating the unit with that key later promotes the draft; otherwise its spend attaches to the unit once you create it with that key. `409` also when the externalId is taken on this customer (`CUSTOMER_UNIT_EXISTS`) or by another customer of the organization (`EXTERNAL_ID_IN_USE`), the tree would get too deep, the customer is on seat-based billing, or a root unit is sent with `billable: false` (`ROOT_UNIT_ALWAYS_BILLABLE`). `403 FEATURE_NOT_ENABLED` for `billable: false` when non-billable units are not enabled for your organization. Addresses the customer by your external customer id.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.customers.create_customer_unit_by_external_id(
    external_id_="customer_123",
    external_id="team-research",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**external_id_:** `str` — Customer ID from your system, stored on Paid as the customer's `externalId`.
    
</dd>
</dl>

<dl>
<dd>

**external_id:** `str` — Your own id for the unit: required, immutable, at most 255 characters, and unique within the organization: no other customer's unit id, external id or alias may be the same, ignoring letter case. Every unit route addresses the unit by it (percent-encode it in the path), and so do signals (`customerUnit.externalCustomerUnitId`). Cannot be `.` or `..`.
    
</dd>
</dl>

<dl>
<dd>

**name:** `typing.Optional[str]` — Display name (defaults to the external ID).
    
</dd>
</dl>

<dl>
<dd>

**external_type:** `typing.Optional[str]` — Your structural vocabulary for the unit (`department`, `tenant`, `team`, ...). Free text; filterable; Paid never branches on it.
    
</dd>
</dl>

<dl>
<dd>

**parent_external_id:** `typing.Optional[str]` — Your external ID of the parent unit; omit it to create the root unit.
    
</dd>
</dl>

<dl>
<dd>

**metadata:** `typing.Optional[typing.Dict[str, typing.Any]]` — Freeform JSON for your own use.
    
</dd>
</dl>

<dl>
<dd>

**billable:** `typing.Optional[bool]` — `false` makes the unit non-billable, together with every unit under it (a dev or QA environment, say). Defaults to `true`. The root unit is always billable (`409 ROOT_UNIT_ALWAYS_BILLABLE`): a customer that should never be billed simply has no order. Sending `false` needs non-billable units enabled for your organization (`403 FEATURE_NOT_ENABLED` otherwise). Usage on a non-billable unit is still recorded but not billed: it burns no credits, adds nothing to invoices or overage, and counts toward no cap, on a price whose usage comes from a meter (read from your events when the invoice closes) too. A meter's derived signal, and a correlation-window meter's window, follow the source event they take their unit from (the window's closing event, else its opening event): they are not billed when that event was not. Usage is still billed when it is sent with a customer API key, and on cost-plus pricing, which bills from costs that carry no unit. Only signals sent to `/api/v2/signals` that resolve to an active unit, by its key or through an alias on it, can be non-billable, and a change applies to signals received afterwards.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.customers.<a href="src/paid/customers/client.py">get_customer_unit_by_external_id</a>(...) -&gt; AsyncHttpResponse[CustomerUnit]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Returns one unit of this customer by its `externalId`, including a deleted one. `404` when the unit does not exist or belongs to another customer. Addresses the customer by your external customer id.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.customers.get_customer_unit_by_external_id(
    external_id="customer_123",
    external_customer_unit_id="team-research",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**external_id:** `str` — Customer ID from your system, stored on Paid as the customer's `externalId`.
    
</dd>
</dl>

<dl>
<dd>

**external_customer_unit_id:** `str` — Your own id for the unit (its `externalId`), unique within this customer.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.customers.<a href="src/paid/customers/client.py">delete_customer_unit_by_external_id</a>(...) -&gt; AsyncHttpResponse[CustomerUnit]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Soft-deletes a unit: it stays readable with `status: DELETED` and cannot be reactivated. Spend history that references it is kept, and signals that keep naming it are still attributed to it. A `DRAFT` unit is deleted the same way: it stays readable as DELETED, its key stays reserved, and spend already keyed to it stays its own. `409` while the unit has ACTIVE or DRAFT children or a cap in force or scheduled; the root follows the same rules, and once it is deleted a new root can be created. Addresses the customer by your external customer id.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.customers.delete_customer_unit_by_external_id(
    external_id="customer_123",
    external_customer_unit_id="team-research",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**external_id:** `str` — Customer ID from your system, stored on Paid as the customer's `externalId`.
    
</dd>
</dl>

<dl>
<dd>

**external_customer_unit_id:** `str` — Your own id for the unit (its `externalId`), unique within this customer.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.customers.<a href="src/paid/customers/client.py">update_customer_unit_by_external_id</a>(...) -&gt; AsyncHttpResponse[CustomerUnit]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Renames, re-types, re-parents or annotates a unit, the root included. `externalId` cannot change. Re-parenting (`parentExternalId`) moves the unit with everything under it. Spend already recorded keeps naming the unit it landed on; caps are evaluated on the current tree, so from the move on the unit's spend in the running cap period counts toward its new ancestors' caps and no longer toward the old ones. `billable` flips the unit's own setting; moving a unit under a non-billable one makes it non-billable in effect (`effectiveBillable`). `409` for a deleted unit (`CUSTOMER_UNIT_DELETED`), a draft (`CUSTOMER_UNIT_DRAFT`), a parent that does not exist or is not ACTIVE, a move of the root (`ROOT_UNIT_IMMOVABLE`), `billable: false` on the root (`ROOT_UNIT_ALWAYS_BILLABLE`), a move under the unit's own subtree, or a tree that would get too deep. `403 FEATURE_NOT_ENABLED` for `billable: false` when non-billable units are not enabled for your organization. Addresses the customer by your external customer id.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.customers.update_customer_unit_by_external_id(
    external_id="customer_123",
    external_customer_unit_id="team-research",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**external_id:** `str` — Customer ID from your system, stored on Paid as the customer's `externalId`.
    
</dd>
</dl>

<dl>
<dd>

**external_customer_unit_id:** `str` — Your own id for the unit (its `externalId`), unique within this customer.
    
</dd>
</dl>

<dl>
<dd>

**name:** `typing.Optional[str]` — Display name. Never used to address the unit.
    
</dd>
</dl>

<dl>
<dd>

**external_type:** `typing.Optional[str]` — Your structural vocabulary for the unit (`department`, `tenant`, `team`, ...). Free text; filterable; Paid never branches on it.
    
</dd>
</dl>

<dl>
<dd>

**parent_external_id:** `typing.Optional[str]` — Your external ID of the new parent unit; moves the unit and its subtree.
    
</dd>
</dl>

<dl>
<dd>

**metadata:** `typing.Optional[typing.Dict[str, typing.Any]]` — Freeform JSON for your own use.
    
</dd>
</dl>

<dl>
<dd>

**billable:** `typing.Optional[bool]` — Flips the unit's own setting either way. Under a non-billable ancestor the unit stays non-billable in effect (`effectiveBillable`) whatever its own value. Refused on the root unit, which is always billable (`409 ROOT_UNIT_ALWAYS_BILLABLE`). Sending `false` needs non-billable units enabled for your organization (`403 FEATURE_NOT_ENABLED` otherwise). Usage on a non-billable unit is still recorded but not billed: it burns no credits, adds nothing to invoices or overage, and counts toward no cap, on a price whose usage comes from a meter (read from your events when the invoice closes) too. A meter's derived signal, and a correlation-window meter's window, follow the source event they take their unit from (the window's closing event, else its opening event): they are not billed when that event was not. Usage is still billed when it is sent with a customer API key, and on cost-plus pricing, which bills from costs that carry no unit. Only signals sent to `/api/v2/signals` that resolve to an active unit, by its key or through an alias on it, can be non-billable, and a change applies to signals received afterwards.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.customers.<a href="src/paid/customers/client.py">list_customer_unit_aliases_by_external_id</a>(...) -&gt; AsyncHttpResponse[CustomerAliasListResponse]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Lists the aliases that land signals on this unit. A customer-level alias (created on the customer routes) points at the customer's root unit, so it is listed under the root. Addresses the customer by your external customer id.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.customers.list_customer_unit_aliases_by_external_id(
    external_id="customer_123",
    external_customer_unit_id="team-research",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**external_id:** `str` — Customer ID from your system, stored on Paid as the customer's `externalId`.
    
</dd>
</dl>

<dl>
<dd>

**external_customer_unit_id:** `str` — Your own id for the unit (its `externalId`), unique within this customer.
    
</dd>
</dl>

<dl>
<dd>

**limit:** `typing.Optional[int]` 
    
</dd>
</dl>

<dl>
<dd>

**offset:** `typing.Optional[int]` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.customers.<a href="src/paid/customers/client.py">create_customer_unit_alias_by_external_id</a>(...) -&gt; AsyncHttpResponse[CustomerAlias]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Creates an alias that lands signals on this unit: a signal whose `externalCustomerId` is the alias resolves to this customer and, when it names no `customerUnit`, to this unit. The alias shares one set of ids with customer external ids and unit ids, unique within the organization and ignoring letter case: an alias another customer holds is refused with `409 EXTERNAL_ID_IN_USE`. Addresses the customer by your external customer id.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.customers.create_customer_unit_alias_by_external_id(
    external_id="customer_123",
    external_customer_unit_id="team-research",
    alias="child-customer-1",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**external_id:** `str` — Customer ID from your system, stored on Paid as the customer's `externalId`.
    
</dd>
</dl>

<dl>
<dd>

**external_customer_unit_id:** `str` — Your own id for the unit (its `externalId`), unique within this customer.
    
</dd>
</dl>

<dl>
<dd>

**alias:** `str` — Alternate external identifier that should resolve to this customer.
    
</dd>
</dl>

<dl>
<dd>

**name:** `typing.Optional[str]` — Optional display name for this alias.
    
</dd>
</dl>

<dl>
<dd>

**description:** `typing.Optional[str]` — Optional note describing where this alias comes from.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.customers.<a href="src/paid/customers/client.py">delete_customer_unit_alias_by_external_id</a>(...) -&gt; AsyncHttpResponse[DeleteCustomerUnitAliasByExternalIdResponse]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Deletes an alias of this unit. Signals still sent with it then resolve to nothing again. Addresses the customer by your external customer id.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.customers.delete_customer_unit_alias_by_external_id(
    external_id="customer_123",
    external_customer_unit_id="team-research",
    alias="tenant-7-sandbox",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**external_id:** `str` — Customer ID from your system, stored on Paid as the customer's `externalId`.
    
</dd>
</dl>

<dl>
<dd>

**external_customer_unit_id:** `str` — Your own id for the unit (its `externalId`), unique within this customer.
    
</dd>
</dl>

<dl>
<dd>

**alias:** `str` — The alias, as created.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.customers.<a href="src/paid/customers/client.py">list_customer_units</a>(...) -&gt; AsyncHttpResponse[CustomerUnitListResponse]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Lists the customer's units as a flat list, newest last; assemble the tree from `parentExternalId` (`null` on the root unit, `isRoot: true`). Deleted units are hidden unless `status=DELETED` is given, drafts unless `status=DRAFT`. Filter by `externalType`, or by `parentExternalId` for one level of the tree. Use the value returned as `customer.id`, for example `cus_abc123`; if you have your own customer ID, use the `/api/v2/customers/external/{externalId}/…` twin.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.customers.list_customer_units(
    id="cus_abc123",
    parent_external_id="dept-rnd",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**id:** `str` — Paid customer display id
    
</dd>
</dl>

<dl>
<dd>

**limit:** `typing.Optional[int]` 
    
</dd>
</dl>

<dl>
<dd>

**offset:** `typing.Optional[int]` 
    
</dd>
</dl>

<dl>
<dd>

**status:** `typing.Optional[ListCustomerUnitsRequestStatus]` — Filter by status (default ACTIVE; drafts are listed only with DRAFT).
    
</dd>
</dl>

<dl>
<dd>

**external_type:** `typing.Optional[str]` — Filter by external type.
    
</dd>
</dl>

<dl>
<dd>

**parent_external_id:** `typing.Optional[str]` — Your external ID of the parent unit; lists its direct children.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.customers.<a href="src/paid/customers/client.py">create_customer_unit</a>(...) -&gt; AsyncHttpResponse[CustomerUnit]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Creates a unit for this customer. `externalId` is your own key for it: required, immutable, and unique within the organization (no other customer's unit id, external id or alias may be the same, ignoring letter case); every unit route addresses the unit by it, and `name` defaults to it. Omit `parentExternalId` to create the customer's root unit (its first unit; `409 ROOT_EXISTS` if it already has one — a customer created with an external id usable as a unit key already has its root, keyed by that external id, so name it as the parent instead); otherwise the parent must exist (`409 PARENT_NOT_FOUND`) and be ACTIVE. A create whose `externalId` is one of this customer's `DRAFT` units promotes the draft in place: same unit, same id, now ACTIVE with the request's name, type, metadata and parent (`parentExternalId` is required; a draft sits under the root). A signal that names a unit before it exists is accepted; with draft units enabled for your organization it creates a `DRAFT` unit under the customer's root that holds the key, and creating the unit with that key later promotes the draft; otherwise its spend attaches to the unit once you create it with that key. `409` also when the externalId is taken on this customer (`CUSTOMER_UNIT_EXISTS`) or by another customer of the organization (`EXTERNAL_ID_IN_USE`), the tree would get too deep, the customer is on seat-based billing, or a root unit is sent with `billable: false` (`ROOT_UNIT_ALWAYS_BILLABLE`). `403 FEATURE_NOT_ENABLED` for `billable: false` when non-billable units are not enabled for your organization. Use the value returned as `customer.id`, for example `cus_abc123`; if you have your own customer ID, use the `/api/v2/customers/external/{externalId}/…` twin.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.customers.create_customer_unit(
    id="cus_abc123",
    external_id="team-research",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**id:** `str` — Paid customer display id
    
</dd>
</dl>

<dl>
<dd>

**external_id:** `str` — Your own id for the unit: required, immutable, at most 255 characters, and unique within the organization: no other customer's unit id, external id or alias may be the same, ignoring letter case. Every unit route addresses the unit by it (percent-encode it in the path), and so do signals (`customerUnit.externalCustomerUnitId`). Cannot be `.` or `..`.
    
</dd>
</dl>

<dl>
<dd>

**name:** `typing.Optional[str]` — Display name (defaults to the external ID).
    
</dd>
</dl>

<dl>
<dd>

**external_type:** `typing.Optional[str]` — Your structural vocabulary for the unit (`department`, `tenant`, `team`, ...). Free text; filterable; Paid never branches on it.
    
</dd>
</dl>

<dl>
<dd>

**parent_external_id:** `typing.Optional[str]` — Your external ID of the parent unit; omit it to create the root unit.
    
</dd>
</dl>

<dl>
<dd>

**metadata:** `typing.Optional[typing.Dict[str, typing.Any]]` — Freeform JSON for your own use.
    
</dd>
</dl>

<dl>
<dd>

**billable:** `typing.Optional[bool]` — `false` makes the unit non-billable, together with every unit under it (a dev or QA environment, say). Defaults to `true`. The root unit is always billable (`409 ROOT_UNIT_ALWAYS_BILLABLE`): a customer that should never be billed simply has no order. Sending `false` needs non-billable units enabled for your organization (`403 FEATURE_NOT_ENABLED` otherwise). Usage on a non-billable unit is still recorded but not billed: it burns no credits, adds nothing to invoices or overage, and counts toward no cap, on a price whose usage comes from a meter (read from your events when the invoice closes) too. A meter's derived signal, and a correlation-window meter's window, follow the source event they take their unit from (the window's closing event, else its opening event): they are not billed when that event was not. Usage is still billed when it is sent with a customer API key, and on cost-plus pricing, which bills from costs that carry no unit. Only signals sent to `/api/v2/signals` that resolve to an active unit, by its key or through an alias on it, can be non-billable, and a change applies to signals received afterwards.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.customers.<a href="src/paid/customers/client.py">get_customer_unit</a>(...) -&gt; AsyncHttpResponse[CustomerUnit]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Returns one unit of this customer by its `externalId`, including a deleted one. `404` when the unit does not exist or belongs to another customer. Use the value returned as `customer.id`, for example `cus_abc123`; if you have your own customer ID, use the `/api/v2/customers/external/{externalId}/…` twin.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.customers.get_customer_unit(
    id="cus_abc123",
    external_customer_unit_id="team-research",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**id:** `str` — Paid customer display id
    
</dd>
</dl>

<dl>
<dd>

**external_customer_unit_id:** `str` — Your own id for the unit (its `externalId`), unique within this customer.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.customers.<a href="src/paid/customers/client.py">delete_customer_unit</a>(...) -&gt; AsyncHttpResponse[CustomerUnit]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Soft-deletes a unit: it stays readable with `status: DELETED` and cannot be reactivated. Spend history that references it is kept, and signals that keep naming it are still attributed to it. A `DRAFT` unit is deleted the same way: it stays readable as DELETED, its key stays reserved, and spend already keyed to it stays its own. `409` while the unit has ACTIVE or DRAFT children or a cap in force or scheduled; the root follows the same rules, and once it is deleted a new root can be created. Use the value returned as `customer.id`, for example `cus_abc123`; if you have your own customer ID, use the `/api/v2/customers/external/{externalId}/…` twin.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.customers.delete_customer_unit(
    id="cus_abc123",
    external_customer_unit_id="team-research",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**id:** `str` — Paid customer display id
    
</dd>
</dl>

<dl>
<dd>

**external_customer_unit_id:** `str` — Your own id for the unit (its `externalId`), unique within this customer.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.customers.<a href="src/paid/customers/client.py">update_customer_unit</a>(...) -&gt; AsyncHttpResponse[CustomerUnit]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Renames, re-types, re-parents or annotates a unit, the root included. `externalId` cannot change. Re-parenting (`parentExternalId`) moves the unit with everything under it. Spend already recorded keeps naming the unit it landed on; caps are evaluated on the current tree, so from the move on the unit's spend in the running cap period counts toward its new ancestors' caps and no longer toward the old ones. `billable` flips the unit's own setting; moving a unit under a non-billable one makes it non-billable in effect (`effectiveBillable`). `409` for a deleted unit (`CUSTOMER_UNIT_DELETED`), a draft (`CUSTOMER_UNIT_DRAFT`), a parent that does not exist or is not ACTIVE, a move of the root (`ROOT_UNIT_IMMOVABLE`), `billable: false` on the root (`ROOT_UNIT_ALWAYS_BILLABLE`), a move under the unit's own subtree, or a tree that would get too deep. `403 FEATURE_NOT_ENABLED` for `billable: false` when non-billable units are not enabled for your organization. Use the value returned as `customer.id`, for example `cus_abc123`; if you have your own customer ID, use the `/api/v2/customers/external/{externalId}/…` twin.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.customers.update_customer_unit(
    id="cus_abc123",
    external_customer_unit_id="team-research",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**id:** `str` — Paid customer display id
    
</dd>
</dl>

<dl>
<dd>

**external_customer_unit_id:** `str` — Your own id for the unit (its `externalId`), unique within this customer.
    
</dd>
</dl>

<dl>
<dd>

**name:** `typing.Optional[str]` — Display name. Never used to address the unit.
    
</dd>
</dl>

<dl>
<dd>

**external_type:** `typing.Optional[str]` — Your structural vocabulary for the unit (`department`, `tenant`, `team`, ...). Free text; filterable; Paid never branches on it.
    
</dd>
</dl>

<dl>
<dd>

**parent_external_id:** `typing.Optional[str]` — Your external ID of the new parent unit; moves the unit and its subtree.
    
</dd>
</dl>

<dl>
<dd>

**metadata:** `typing.Optional[typing.Dict[str, typing.Any]]` — Freeform JSON for your own use.
    
</dd>
</dl>

<dl>
<dd>

**billable:** `typing.Optional[bool]` — Flips the unit's own setting either way. Under a non-billable ancestor the unit stays non-billable in effect (`effectiveBillable`) whatever its own value. Refused on the root unit, which is always billable (`409 ROOT_UNIT_ALWAYS_BILLABLE`). Sending `false` needs non-billable units enabled for your organization (`403 FEATURE_NOT_ENABLED` otherwise). Usage on a non-billable unit is still recorded but not billed: it burns no credits, adds nothing to invoices or overage, and counts toward no cap, on a price whose usage comes from a meter (read from your events when the invoice closes) too. A meter's derived signal, and a correlation-window meter's window, follow the source event they take their unit from (the window's closing event, else its opening event): they are not billed when that event was not. Usage is still billed when it is sent with a customer API key, and on cost-plus pricing, which bills from costs that carry no unit. Only signals sent to `/api/v2/signals` that resolve to an active unit, by its key or through an alias on it, can be non-billable, and a change applies to signals received afterwards.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.customers.<a href="src/paid/customers/client.py">list_customer_unit_aliases</a>(...) -&gt; AsyncHttpResponse[CustomerAliasListResponse]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Lists the aliases that land signals on this unit. A customer-level alias (created on the customer routes) points at the customer's root unit, so it is listed under the root. Use the value returned as `customer.id`, for example `cus_abc123`; if you have your own customer ID, use the `/api/v2/customers/external/{externalId}/…` twin.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.customers.list_customer_unit_aliases(
    id="cus_abc123",
    external_customer_unit_id="team-research",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**id:** `str` — Paid customer display id
    
</dd>
</dl>

<dl>
<dd>

**external_customer_unit_id:** `str` — Your own id for the unit (its `externalId`), unique within this customer.
    
</dd>
</dl>

<dl>
<dd>

**limit:** `typing.Optional[int]` 
    
</dd>
</dl>

<dl>
<dd>

**offset:** `typing.Optional[int]` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.customers.<a href="src/paid/customers/client.py">create_customer_unit_alias</a>(...) -&gt; AsyncHttpResponse[CustomerAlias]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Creates an alias that lands signals on this unit: a signal whose `externalCustomerId` is the alias resolves to this customer and, when it names no `customerUnit`, to this unit. The alias shares one set of ids with customer external ids and unit ids, unique within the organization and ignoring letter case: an alias another customer holds is refused with `409 EXTERNAL_ID_IN_USE`. Use the value returned as `customer.id`, for example `cus_abc123`; if you have your own customer ID, use the `/api/v2/customers/external/{externalId}/…` twin.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.customers.create_customer_unit_alias(
    id="cus_abc123",
    external_customer_unit_id="team-research",
    alias="child-customer-1",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**id:** `str` — Paid customer display id
    
</dd>
</dl>

<dl>
<dd>

**external_customer_unit_id:** `str` — Your own id for the unit (its `externalId`), unique within this customer.
    
</dd>
</dl>

<dl>
<dd>

**alias:** `str` — Alternate external identifier that should resolve to this customer.
    
</dd>
</dl>

<dl>
<dd>

**name:** `typing.Optional[str]` — Optional display name for this alias.
    
</dd>
</dl>

<dl>
<dd>

**description:** `typing.Optional[str]` — Optional note describing where this alias comes from.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.customers.<a href="src/paid/customers/client.py">delete_customer_unit_alias</a>(...) -&gt; AsyncHttpResponse[DeleteCustomerUnitAliasResponse]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Deletes an alias of this unit. Signals still sent with it then resolve to nothing again. Use the value returned as `customer.id`, for example `cus_abc123`; if you have your own customer ID, use the `/api/v2/customers/external/{externalId}/…` twin.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.customers.delete_customer_unit_alias(
    id="cus_abc123",
    external_customer_unit_id="team-research",
    alias="tenant-7-sandbox",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**id:** `str` — Paid customer display id
    
</dd>
</dl>

<dl>
<dd>

**external_customer_unit_id:** `str` — Your own id for the unit (its `externalId`), unique within this customer.
    
</dd>
</dl>

<dl>
<dd>

**alias:** `str` — The alias, as created.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.customers.<a href="src/paid/customers/client.py">get_customer_unit_by_key</a>(...) -&gt; AsyncHttpResponse[CustomerUnitByKey]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Returns a unit by its key without naming its customer. The answer carries the customer's ids (`customerId`, `customerExternalId`) so the customer routes are one hop away, but, like the create on this path, not Paid's unit id: a caller of these routes names units by key. The key is looked up across the organization ignoring letter case. When more than one unit carries it in another letter case (legacy duplicates across customers, or case variants on one customer), the exact spelling decides; when that does not single one out either, the answer is `409 CUSTOMER_UNIT_AMBIGUOUS`: use the customer's unit routes, where the key is exact.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.customers.get_customer_unit_by_key(
    external_customer_unit_id="team-research",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**external_customer_unit_id:** `str` — Your id of the unit (its `externalId`), unique within the organization ignoring letter case.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.customers.<a href="src/paid/customers/client.py">list_customer_unit_aliases_by_key</a>(...) -&gt; AsyncHttpResponse[CustomerAliasListResponse]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Lists the aliases that land signals on this unit, without naming its customer. The key is looked up across the organization ignoring letter case. When more than one unit carries it in another letter case (legacy duplicates across customers, or case variants on one customer), the exact spelling decides; when that does not single one out either, the answer is `409 CUSTOMER_UNIT_AMBIGUOUS`: use the customer's unit routes, where the key is exact.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.customers.list_customer_unit_aliases_by_key(
    external_customer_unit_id="team-research",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**external_customer_unit_id:** `str` — Your id of the unit (its `externalId`), unique within the organization ignoring letter case.
    
</dd>
</dl>

<dl>
<dd>

**limit:** `typing.Optional[int]` 
    
</dd>
</dl>

<dl>
<dd>

**offset:** `typing.Optional[int]` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.customers.<a href="src/paid/customers/client.py">create_customer_unit_alias_by_key</a>(...) -&gt; AsyncHttpResponse[CustomerAlias]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Creates an alias that lands signals on this unit, without naming its customer: a signal whose `externalCustomerId` is the alias resolves to the unit's customer and is attributed to this unit. The alias must be unique within the organization ignoring letter case (`409 EXTERNAL_ID_IN_USE`). The key is looked up across the organization ignoring letter case. When more than one unit carries it in another letter case (legacy duplicates across customers, or case variants on one customer), the exact spelling decides; when that does not single one out either, the answer is `409 CUSTOMER_UNIT_AMBIGUOUS`: use the customer's unit routes, where the key is exact.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.customers.create_customer_unit_alias_by_key(
    external_customer_unit_id="team-research",
    alias="child-customer-1",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**external_customer_unit_id:** `str` — Your id of the unit (its `externalId`), unique within the organization ignoring letter case.
    
</dd>
</dl>

<dl>
<dd>

**alias:** `str` — Alternate external identifier that should resolve to this customer.
    
</dd>
</dl>

<dl>
<dd>

**name:** `typing.Optional[str]` — Optional display name for this alias.
    
</dd>
</dl>

<dl>
<dd>

**description:** `typing.Optional[str]` — Optional note describing where this alias comes from.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.customers.<a href="src/paid/customers/client.py">delete_customer_unit_alias_by_key</a>(...) -&gt; AsyncHttpResponse[DeleteCustomerUnitAliasByKeyResponse]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Deletes an alias of this unit, without naming its customer. Signals still sent with it then resolve to nothing again. The key is looked up across the organization ignoring letter case. When more than one unit carries it in another letter case (legacy duplicates across customers, or case variants on one customer), the exact spelling decides; when that does not single one out either, the answer is `409 CUSTOMER_UNIT_AMBIGUOUS`: use the customer's unit routes, where the key is exact.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.customers.delete_customer_unit_alias_by_key(
    external_customer_unit_id="team-research",
    alias="tenant-7",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**external_customer_unit_id:** `str` — Your id of the unit (its `externalId`), unique within the organization ignoring letter case.
    
</dd>
</dl>

<dl>
<dd>

**alias:** `str` — The alias, as created.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.customers.<a href="src/paid/customers/client.py">create_customer_unit_under_parent</a>(...) -&gt; AsyncHttpResponse[CustomerUnitCreatedUnderParent]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Creates a unit by naming its parent alone; the customer is whoever the parent belongs to, so a caller that only knows its own ids never needs the customer routes. `parentExternalId` is looked up across the organization, ignoring letter case, among unit ids, customer external ids and aliases: a unit becomes the parent; a customer (by external id or alias) stands for the top of its tree, so the new unit goes under the root unit, or becomes the root when the customer has none. The parent is only ever named by your id, never by a Paid id, so a customer with no external id takes its units through the customer's unit routes. The answer carries the unit's `externalId` and its customer's ids, not Paid's unit id. `externalId` follows the same rules as on the customer-scoped create: required, immutable, unique within the organization ignoring letter case. `409 PARENT_NOT_FOUND` when nothing matches, `PARENT_NOT_ACTIVE` for a deleted or draft parent, `PARENT_AMBIGUOUS` when legacy rows let the id name more than one customer (use that customer's unit routes), `PARENT_CHANGED` when the reference pointed elsewhere by the time the create ran (send it again), and otherwise the customer-scoped create's 409s (`CUSTOMER_UNIT_EXISTS`, `EXTERNAL_ID_IN_USE`, `DEPTH_EXCEEDED`, `SEAT_BILLING`, and `ROOT_UNIT_ALWAYS_BILLABLE` for `billable: false` on a unit that would become the root). `403 FEATURE_NOT_ENABLED` for `billable: false` when non-billable units are not enabled for your organization.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.customers.create_customer_unit_under_parent(
    external_id="team-research",
    parent_external_id="dept-rnd",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**external_id:** `str` — Your own id for the unit: required, immutable, at most 255 characters, and unique within the organization: no other customer's unit id, external id or alias may be the same, ignoring letter case. Every unit route addresses the unit by it (percent-encode it in the path), and so do signals (`customerUnit.externalCustomerUnitId`). Cannot be `.` or `..`.
    
</dd>
</dl>

<dl>
<dd>

**parent_external_id:** `str` — Your id of the parent, looked up across the organization ignoring letter case: a unit id makes the new unit its child; a customer external id or alias names the customer, so the new unit goes under that customer's root unit, or becomes the root when the customer has none. Required: a parent is only ever named by your id, never by a Paid id.
    
</dd>
</dl>

<dl>
<dd>

**name:** `typing.Optional[str]` — Display name (defaults to the external ID).
    
</dd>
</dl>

<dl>
<dd>

**external_type:** `typing.Optional[str]` — Your structural vocabulary for the unit (`department`, `tenant`, `team`, ...). Free text; filterable; Paid never branches on it.
    
</dd>
</dl>

<dl>
<dd>

**metadata:** `typing.Optional[typing.Dict[str, typing.Any]]` — Freeform JSON for your own use.
    
</dd>
</dl>

<dl>
<dd>

**billable:** `typing.Optional[bool]` — `false` makes the unit non-billable, together with every unit under it (a dev or QA environment, say). Defaults to `true`. The root unit is always billable (`409 ROOT_UNIT_ALWAYS_BILLABLE`): a customer that should never be billed simply has no order. Sending `false` needs non-billable units enabled for your organization (`403 FEATURE_NOT_ENABLED` otherwise). Usage on a non-billable unit is still recorded but not billed: it burns no credits, adds nothing to invoices or overage, and counts toward no cap, on a price whose usage comes from a meter (read from your events when the invoice closes) too. A meter's derived signal, and a correlation-window meter's window, follow the source event they take their unit from (the window's closing event, else its opening event): they are not billed when that event was not. Usage is still billed when it is sent with a customer API key, and on cost-plus pricing, which bills from costs that carry no unit. Only signals sent to `/api/v2/signals` that resolve to an active unit, by its key or through an alias on it, can be non-billable, and a change applies to signals received afterwards.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.customers.<a href="src/paid/customers/client.py">list_customer_api_keys_by_external_id</a>(...) -&gt; AsyncHttpResponse[CustomerApiKeyList]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Lists this customer's API keys, newest first and paginated, revoked and expired ones included. The secret is never returned; `keyPrefix` and `last4` identify each key. Addresses the customer by your external customer id.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.customers.list_customer_api_keys_by_external_id(
    external_id="customer_123",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**external_id:** `str` — Customer ID from your system, stored on Paid as the customer's `externalId`.
    
</dd>
</dl>

<dl>
<dd>

**limit:** `typing.Optional[int]` 
    
</dd>
</dl>

<dl>
<dd>

**offset:** `typing.Optional[int]` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.customers.<a href="src/paid/customers/client.py">create_customer_api_key_by_external_id</a>(...) -&gt; AsyncHttpResponse[CustomerApiKeyCreated]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Creates an API key bound to this customer. The response carries the secret in `key`, returned only this once: store it immediately, as Paid keeps only its hash. The key acts for this customer alone: it can send signals for this customer through `POST /api/v2/signals/bulk`, and every other route refuses it. Requires `write:api-keys`, which full-access keys hold; a restricted key needs the scope granted explicitly. Addresses the customer by your external customer id.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.customers.create_customer_api_key_by_external_id(
    external_id="customer_123",
    name="Tenant runtime (production)",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**external_id:** `str` — Customer ID from your system, stored on Paid as the customer's `externalId`.
    
</dd>
</dl>

<dl>
<dd>

**name:** `str` — A label for the key, shown in lists and in the audit log.
    
</dd>
</dl>

<dl>
<dd>

**description:** `typing.Optional[str]` — Optional free text about where the key is used.
    
</dd>
</dl>

<dl>
<dd>

**expires_at:** `typing.Optional[dt.datetime]` — Optional expiry (ISO 8601), which must be in the future. Without it the key works until it is revoked.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.customers.<a href="src/paid/customers/client.py">revoke_customer_api_key_by_external_id</a>(...) -&gt; AsyncHttpResponse[CustomerApiKey]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Revokes one of this customer's keys, which then stops authenticating. Keys are never deleted: a revoked key stays listed with `revokedAt` set, and deleting the customer revokes its keys too. Revoking an already revoked key returns it unchanged. `404` when the key does not belong to this customer. Addresses the customer by your external customer id.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.customers.revoke_customer_api_key_by_external_id(
    external_id="customer_123",
    api_key_id="apiKeyId",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**external_id:** `str` — Customer ID from your system, stored on Paid as the customer's `externalId`.
    
</dd>
</dl>

<dl>
<dd>

**api_key_id:** `str` — The key's id, as returned by create or list.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.customers.<a href="src/paid/customers/client.py">list_customer_api_keys</a>(...) -&gt; AsyncHttpResponse[CustomerApiKeyList]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Lists this customer's API keys, newest first and paginated, revoked and expired ones included. The secret is never returned; `keyPrefix` and `last4` identify each key. Use the value returned as `customer.id`, for example `cus_abc123`; if you have your own customer ID, use the `/api/v2/customers/external/{externalId}/…` twin.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.customers.list_customer_api_keys(
    id="cus_abc123",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**id:** `str` — Paid customer display id
    
</dd>
</dl>

<dl>
<dd>

**limit:** `typing.Optional[int]` 
    
</dd>
</dl>

<dl>
<dd>

**offset:** `typing.Optional[int]` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.customers.<a href="src/paid/customers/client.py">create_customer_api_key</a>(...) -&gt; AsyncHttpResponse[CustomerApiKeyCreated]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Creates an API key bound to this customer. The response carries the secret in `key`, returned only this once: store it immediately, as Paid keeps only its hash. The key acts for this customer alone: it can send signals for this customer through `POST /api/v2/signals/bulk`, and every other route refuses it. Requires `write:api-keys`, which full-access keys hold; a restricted key needs the scope granted explicitly. Use the value returned as `customer.id`, for example `cus_abc123`; if you have your own customer ID, use the `/api/v2/customers/external/{externalId}/…` twin.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.customers.create_customer_api_key(
    id="cus_abc123",
    name="Tenant runtime (production)",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**id:** `str` — Paid customer display id
    
</dd>
</dl>

<dl>
<dd>

**name:** `str` — A label for the key, shown in lists and in the audit log.
    
</dd>
</dl>

<dl>
<dd>

**description:** `typing.Optional[str]` — Optional free text about where the key is used.
    
</dd>
</dl>

<dl>
<dd>

**expires_at:** `typing.Optional[dt.datetime]` — Optional expiry (ISO 8601), which must be in the future. Without it the key works until it is revoked.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.customers.<a href="src/paid/customers/client.py">revoke_customer_api_key</a>(...) -&gt; AsyncHttpResponse[CustomerApiKey]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Revokes one of this customer's keys, which then stops authenticating. Keys are never deleted: a revoked key stays listed with `revokedAt` set, and deleting the customer revokes its keys too. Revoking an already revoked key returns it unchanged. `404` when the key does not belong to this customer. Use the value returned as `customer.id`, for example `cus_abc123`; if you have your own customer ID, use the `/api/v2/customers/external/{externalId}/…` twin.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.customers.revoke_customer_api_key(
    id="cus_abc123",
    api_key_id="apiKeyId",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**id:** `str` — Paid customer display id
    
</dd>
</dl>

<dl>
<dd>

**api_key_id:** `str` — The key's id, as returned by create or list.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.customers.<a href="src/paid/customers/client.py">get_customer_unit_cap_by_external_id</a>(...) -&gt; AsyncHttpResponse[CustomerUnitCapResponse]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Returns the cap in force on this customer unit for one credits currency, with usage in the current period when available. Select the currency with `creditsCurrencyId`; it may be omitted only when the organization has exactly one credits currency, which is then used and echoed back. `404` when the customer or the unit does not exist, or the unit has no cap in force for that currency. The usage figures are advisory: other spend may land between this read and the next burn. Addresses the customer by your external customer id.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.customers.get_customer_unit_cap_by_external_id(
    external_id="customer_123",
    external_customer_unit_id="tenant-a",
    credits_currency_id="7f4f5d4c-55e9-4d5b-a3e7-c9eb3d2d01bf",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**external_id:** `str` — Customer ID from your system, stored on Paid as the customer's `externalId`.
    
</dd>
</dl>

<dl>
<dd>

**external_customer_unit_id:** `str` — Your own id for the unit (its `externalId`), unique within this customer.
    
</dd>
</dl>

<dl>
<dd>

**credits_currency_id:** `typing.Optional[str]` — The credits currency to read. Omit it only when the organization has exactly one credits currency, which is then used; otherwise it is required.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.customers.<a href="src/paid/customers/client.py">set_customer_unit_cap_by_external_id</a>(...) -&gt; AsyncHttpResponse[CustomerUnitCapSetResponse]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Sets the cap on this customer unit for one credits currency. Returns `changed: false` without creating a version when the effective timeline is unchanged; otherwise records a new version, preserving earlier rows. The newest version wins where they overlap. `effectiveFrom` controls applicability; `schedule` defines monthly UTC boundaries independently. Omit schedule to inherit the version covering effectiveFrom, or use the creation-time UTC day if none applies. Select the currency with `creditsCurrencyId` in the body; it may be omitted only when the organization has exactly one credits currency. A cap covers the unit and every unit beneath it, so a cap on the customer's root unit covers every unit of that customer; spend from a signal that names no unit counts toward no cap. `404` when the customer or the unit does not exist. `409` for customers on seat-based billing, a deleted unit (`CUSTOMER_UNIT_DELETED`) or a draft (`CUSTOMER_UNIT_DRAFT`). Addresses the customer by your external customer id.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.customers.set_customer_unit_cap_by_external_id(
    external_id="customer_123",
    external_customer_unit_id="tenant-a",
    amount=10000.0,
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**external_id:** `str` — Customer ID from your system, stored on Paid as the customer's `externalId`.
    
</dd>
</dl>

<dl>
<dd>

**external_customer_unit_id:** `str` — Your own id for the unit (its `externalId`), unique within this customer.
    
</dd>
</dl>

<dl>
<dd>

**amount:** `float` — The cap, in credits of the currency, per period. Must be positive.
    
</dd>
</dl>

<dl>
<dd>

**frequency:** `typing.Optional[CustomerUnitCapSetFrequency]` — Monthly UTC periods, with boundaries defined by `schedule`.
    
</dd>
</dl>

<dl>
<dd>

**schedule:** `typing.Optional[CustomerUnitCapScheduleInput]` 
    
</dd>
</dl>

<dl>
<dd>

**credits_currency_id:** `typing.Optional[str]` — The credits currency to cap. Omit it only when the organization has exactly one credits currency, which is then used; otherwise it is required.
    
</dd>
</dl>

<dl>
<dd>

**effective_from:** `typing.Optional[dt.datetime]` — ISO 8601 timestamp. When the cap starts applying, independently of its schedule. Defaults to now. Spend earlier in the schedule’s current period still counts.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.customers.<a href="src/paid/customers/client.py">end_customer_unit_cap_by_external_id</a>(...) -&gt; AsyncHttpResponse[CustomerUnitCapEndResponse]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Ends the cap on this customer unit for one credits currency by setting `effectiveUntil` to now on every open version — the one in force, older overlapping versions still open, and versions scheduled to start later — so nothing can resurface or activate afterwards; nothing is deleted and history is kept. Select the currency with `creditsCurrencyId`; it may be omitted only when the organization has exactly one credits currency. `404` when the customer or the unit does not exist, or there is no open version for that currency. Addresses the customer by your external customer id.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.customers.end_customer_unit_cap_by_external_id(
    external_id="customer_123",
    external_customer_unit_id="tenant-a",
    credits_currency_id="7f4f5d4c-55e9-4d5b-a3e7-c9eb3d2d01bf",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**external_id:** `str` — Customer ID from your system, stored on Paid as the customer's `externalId`.
    
</dd>
</dl>

<dl>
<dd>

**external_customer_unit_id:** `str` — Your own id for the unit (its `externalId`), unique within this customer.
    
</dd>
</dl>

<dl>
<dd>

**credits_currency_id:** `typing.Optional[str]` — The credits currency whose cap to end. Omit it only when the organization has exactly one credits currency, which is then used; otherwise it is required.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.customers.<a href="src/paid/customers/client.py">get_customer_unit_spend_by_external_id</a>(...) -&gt; AsyncHttpResponse[CustomerUnitSpendResponse]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Returns the credits of one credits currency spent by this customer unit and every unit beneath it during a window, whether or not the unit has a cap. The window runs from `start` (inclusive) to `end` (exclusive), at most 366 days; omit both for the current calendar month (UTC). For a capped unit over its cap's current period, `spent` equals the cap's `used`. Spend from a signal that names no unit counts toward no unit. Select the currency with `creditsCurrencyId`; it may be omitted only when the organization has exactly one credits currency, which is then used and echoed back. `404` when the customer or the unit does not exist. Addresses the customer by your external customer id.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
import datetime

from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.customers.get_customer_unit_spend_by_external_id(
    external_id="customer_123",
    external_customer_unit_id="tenant-a",
    credits_currency_id="7f4f5d4c-55e9-4d5b-a3e7-c9eb3d2d01bf",
    start=datetime.datetime.fromisoformat(
        "2026-10-01 00:00:00+00:00",
    ),
    end=datetime.datetime.fromisoformat(
        "2026-11-01 00:00:00+00:00",
    ),
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**external_id:** `str` — Customer ID from your system, stored on Paid as the customer's `externalId`.
    
</dd>
</dl>

<dl>
<dd>

**external_customer_unit_id:** `str` — Your own id for the unit (its `externalId`), unique within this customer.
    
</dd>
</dl>

<dl>
<dd>

**credits_currency_id:** `typing.Optional[str]` — The credits currency to read. Omit it only when the organization has exactly one credits currency, which is then used; otherwise it is required.
    
</dd>
</dl>

<dl>
<dd>

**start:** `typing.Optional[dt.datetime]` — ISO 8601 timestamp, the window's inclusive start. Give `start` and `end` together, with `start` before `end`, or omit both for the current calendar month (UTC).
    
</dd>
</dl>

<dl>
<dd>

**end:** `typing.Optional[dt.datetime]` — ISO 8601 timestamp, the window's exclusive end. Give `start` and `end` together, with `start` before `end`, or omit both for the current calendar month (UTC).
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.customers.<a href="src/paid/customers/client.py">get_customer_unit_cap</a>(...) -&gt; AsyncHttpResponse[CustomerUnitCapResponse]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Returns the cap in force on this customer unit for one credits currency, with usage in the current period when available. Select the currency with `creditsCurrencyId`; it may be omitted only when the organization has exactly one credits currency, which is then used and echoed back. `404` when the customer or the unit does not exist, or the unit has no cap in force for that currency. The usage figures are advisory: other spend may land between this read and the next burn. Use the value returned as `customer.id`, for example `cus_abc123`; if you have your own customer ID, use the `/api/v2/customers/external/{externalId}/…` twin.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.customers.get_customer_unit_cap(
    id="cus_abc123",
    external_customer_unit_id="tenant-a",
    credits_currency_id="7f4f5d4c-55e9-4d5b-a3e7-c9eb3d2d01bf",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**id:** `str` — Paid customer display id
    
</dd>
</dl>

<dl>
<dd>

**external_customer_unit_id:** `str` — Your own id for the unit (its `externalId`), unique within this customer.
    
</dd>
</dl>

<dl>
<dd>

**credits_currency_id:** `typing.Optional[str]` — The credits currency to read. Omit it only when the organization has exactly one credits currency, which is then used; otherwise it is required.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.customers.<a href="src/paid/customers/client.py">set_customer_unit_cap</a>(...) -&gt; AsyncHttpResponse[CustomerUnitCapSetResponse]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Sets the cap on this customer unit for one credits currency. Returns `changed: false` without creating a version when the effective timeline is unchanged; otherwise records a new version, preserving earlier rows. The newest version wins where they overlap. `effectiveFrom` controls applicability; `schedule` defines monthly UTC boundaries independently. Omit schedule to inherit the version covering effectiveFrom, or use the creation-time UTC day if none applies. Select the currency with `creditsCurrencyId` in the body; it may be omitted only when the organization has exactly one credits currency. A cap covers the unit and every unit beneath it, so a cap on the customer's root unit covers every unit of that customer; spend from a signal that names no unit counts toward no cap. `404` when the customer or the unit does not exist. `409` for customers on seat-based billing, a deleted unit (`CUSTOMER_UNIT_DELETED`) or a draft (`CUSTOMER_UNIT_DRAFT`). Use the value returned as `customer.id`, for example `cus_abc123`; if you have your own customer ID, use the `/api/v2/customers/external/{externalId}/…` twin.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.customers.set_customer_unit_cap(
    id="cus_abc123",
    external_customer_unit_id="tenant-a",
    amount=10000.0,
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**id:** `str` — Paid customer display id
    
</dd>
</dl>

<dl>
<dd>

**external_customer_unit_id:** `str` — Your own id for the unit (its `externalId`), unique within this customer.
    
</dd>
</dl>

<dl>
<dd>

**amount:** `float` — The cap, in credits of the currency, per period. Must be positive.
    
</dd>
</dl>

<dl>
<dd>

**frequency:** `typing.Optional[CustomerUnitCapSetFrequency]` — Monthly UTC periods, with boundaries defined by `schedule`.
    
</dd>
</dl>

<dl>
<dd>

**schedule:** `typing.Optional[CustomerUnitCapScheduleInput]` 
    
</dd>
</dl>

<dl>
<dd>

**credits_currency_id:** `typing.Optional[str]` — The credits currency to cap. Omit it only when the organization has exactly one credits currency, which is then used; otherwise it is required.
    
</dd>
</dl>

<dl>
<dd>

**effective_from:** `typing.Optional[dt.datetime]` — ISO 8601 timestamp. When the cap starts applying, independently of its schedule. Defaults to now. Spend earlier in the schedule’s current period still counts.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.customers.<a href="src/paid/customers/client.py">end_customer_unit_cap</a>(...) -&gt; AsyncHttpResponse[CustomerUnitCapEndResponse]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Ends the cap on this customer unit for one credits currency by setting `effectiveUntil` to now on every open version — the one in force, older overlapping versions still open, and versions scheduled to start later — so nothing can resurface or activate afterwards; nothing is deleted and history is kept. Select the currency with `creditsCurrencyId`; it may be omitted only when the organization has exactly one credits currency. `404` when the customer or the unit does not exist, or there is no open version for that currency. Use the value returned as `customer.id`, for example `cus_abc123`; if you have your own customer ID, use the `/api/v2/customers/external/{externalId}/…` twin.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.customers.end_customer_unit_cap(
    id="cus_abc123",
    external_customer_unit_id="tenant-a",
    credits_currency_id="7f4f5d4c-55e9-4d5b-a3e7-c9eb3d2d01bf",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**id:** `str` — Paid customer display id
    
</dd>
</dl>

<dl>
<dd>

**external_customer_unit_id:** `str` — Your own id for the unit (its `externalId`), unique within this customer.
    
</dd>
</dl>

<dl>
<dd>

**credits_currency_id:** `typing.Optional[str]` — The credits currency whose cap to end. Omit it only when the organization has exactly one credits currency, which is then used; otherwise it is required.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.customers.<a href="src/paid/customers/client.py">get_customer_unit_spend</a>(...) -&gt; AsyncHttpResponse[CustomerUnitSpendResponse]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Returns the credits of one credits currency spent by this customer unit and every unit beneath it during a window, whether or not the unit has a cap. The window runs from `start` (inclusive) to `end` (exclusive), at most 366 days; omit both for the current calendar month (UTC). For a capped unit over its cap's current period, `spent` equals the cap's `used`. Spend from a signal that names no unit counts toward no unit. Select the currency with `creditsCurrencyId`; it may be omitted only when the organization has exactly one credits currency, which is then used and echoed back. `404` when the customer or the unit does not exist. Use the value returned as `customer.id`, for example `cus_abc123`; if you have your own customer ID, use the `/api/v2/customers/external/{externalId}/…` twin.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
import datetime

from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.customers.get_customer_unit_spend(
    id="cus_abc123",
    external_customer_unit_id="tenant-a",
    credits_currency_id="7f4f5d4c-55e9-4d5b-a3e7-c9eb3d2d01bf",
    start=datetime.datetime.fromisoformat(
        "2026-10-01 00:00:00+00:00",
    ),
    end=datetime.datetime.fromisoformat(
        "2026-11-01 00:00:00+00:00",
    ),
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**id:** `str` — Paid customer display id
    
</dd>
</dl>

<dl>
<dd>

**external_customer_unit_id:** `str` — Your own id for the unit (its `externalId`), unique within this customer.
    
</dd>
</dl>

<dl>
<dd>

**credits_currency_id:** `typing.Optional[str]` — The credits currency to read. Omit it only when the organization has exactly one credits currency, which is then used; otherwise it is required.
    
</dd>
</dl>

<dl>
<dd>

**start:** `typing.Optional[dt.datetime]` — ISO 8601 timestamp, the window's inclusive start. Give `start` and `end` together, with `start` before `end`, or omit both for the current calendar month (UTC).
    
</dd>
</dl>

<dl>
<dd>

**end:** `typing.Optional[dt.datetime]` — ISO 8601 timestamp, the window's exclusive end. Give `start` and `end` together, with `start` before `end`, or omit both for the current calendar month (UTC).
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

## Contacts
<details><summary><code>client.contacts.<a href="src/paid/contacts/client.py">list_contacts</a>(...) -&gt; AsyncHttpResponse[ContactListResponse]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Get a list of contacts for the organization
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.contacts.list_contacts()

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**limit:** `typing.Optional[int]` 
    
</dd>
</dl>

<dl>
<dd>

**offset:** `typing.Optional[int]` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.contacts.<a href="src/paid/contacts/client.py">create_contact</a>(...) -&gt; AsyncHttpResponse[Contact]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Creates a new contact for the organization
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.contacts.create_contact(
    customer_id="customerId",
    email="email",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**customer_id:** `str` 
    
</dd>
</dl>

<dl>
<dd>

**email:** `str` 
    
</dd>
</dl>

<dl>
<dd>

**first_name:** `typing.Optional[str]` 
    
</dd>
</dl>

<dl>
<dd>

**last_name:** `typing.Optional[str]` 
    
</dd>
</dl>

<dl>
<dd>

**phone:** `typing.Optional[str]` 
    
</dd>
</dl>

<dl>
<dd>

**billing_address:** `typing.Optional[ContactBillingAddress]` 
    
</dd>
</dl>

<dl>
<dd>

**external_id:** `typing.Optional[str]` 
    
</dd>
</dl>

<dl>
<dd>

**roles:** `typing.Optional[typing.Sequence[CreateContactRequestRolesItem]]` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.contacts.<a href="src/paid/contacts/client.py">get_contact_by_id</a>(...) -&gt; AsyncHttpResponse[Contact]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Get a contact by its ID
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.contacts.get_contact_by_id(
    id="id",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**id:** `str` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.contacts.<a href="src/paid/contacts/client.py">update_contact_by_id</a>(...) -&gt; AsyncHttpResponse[Contact]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Update a contact by its ID. A contact can move to another customer only if no orders or invoices reference it; otherwise a different `customerId` returns 400 `CONTACT_CUSTOMER_IMMUTABLE`.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.contacts.update_contact_by_id(
    id="id",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**id:** `str` 
    
</dd>
</dl>

<dl>
<dd>

**customer_id:** `typing.Optional[str]` — Destination customer ID. A contact can move only if no orders or invoices reference it.
    
</dd>
</dl>

<dl>
<dd>

**first_name:** `typing.Optional[str]` 
    
</dd>
</dl>

<dl>
<dd>

**last_name:** `typing.Optional[str]` 
    
</dd>
</dl>

<dl>
<dd>

**email:** `typing.Optional[str]` 
    
</dd>
</dl>

<dl>
<dd>

**phone:** `typing.Optional[str]` 
    
</dd>
</dl>

<dl>
<dd>

**billing_address:** `typing.Optional[ContactBillingAddress]` 
    
</dd>
</dl>

<dl>
<dd>

**external_id:** `typing.Optional[str]` 
    
</dd>
</dl>

<dl>
<dd>

**roles:** `typing.Optional[typing.Sequence[UpdateContactRequestRolesItem]]` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.contacts.<a href="src/paid/contacts/client.py">delete_contact_by_id</a>(...) -&gt; AsyncHttpResponse[EmptyResponse]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Delete a contact by its ID. A contact referenced by an order or an unsettled invoice cannot be deleted (409 `CONTACT_HAS_ASSOCIATIONS`). Posted paid invoices, voided invoices and canceled invoices do not block deletion.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.contacts.delete_contact_by_id(
    id="id",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**id:** `str` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.contacts.<a href="src/paid/contacts/client.py">get_contact_by_external_id</a>(...) -&gt; AsyncHttpResponse[Contact]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Get a contact by its external ID
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.contacts.get_contact_by_external_id(
    external_id="externalId",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**external_id:** `str` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.contacts.<a href="src/paid/contacts/client.py">update_contact_by_external_id</a>(...) -&gt; AsyncHttpResponse[Contact]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Update a contact by its external ID. A contact can move to another customer only if no orders or invoices reference it; otherwise a different `customerId` returns 400 `CONTACT_CUSTOMER_IMMUTABLE`.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.contacts.update_contact_by_external_id(
    external_id_="externalId",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**external_id_:** `str` 
    
</dd>
</dl>

<dl>
<dd>

**customer_id:** `typing.Optional[str]` — Destination customer ID. A contact can move only if no orders or invoices reference it.
    
</dd>
</dl>

<dl>
<dd>

**first_name:** `typing.Optional[str]` 
    
</dd>
</dl>

<dl>
<dd>

**last_name:** `typing.Optional[str]` 
    
</dd>
</dl>

<dl>
<dd>

**email:** `typing.Optional[str]` 
    
</dd>
</dl>

<dl>
<dd>

**phone:** `typing.Optional[str]` 
    
</dd>
</dl>

<dl>
<dd>

**billing_address:** `typing.Optional[ContactBillingAddress]` 
    
</dd>
</dl>

<dl>
<dd>

**external_id:** `typing.Optional[str]` 
    
</dd>
</dl>

<dl>
<dd>

**roles:** `typing.Optional[typing.Sequence[UpdateContactRequestRolesItem]]` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.contacts.<a href="src/paid/contacts/client.py">delete_contact_by_external_id</a>(...) -&gt; AsyncHttpResponse[EmptyResponse]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Delete a contact by its external ID. A contact referenced by an order or an unsettled invoice cannot be deleted (409 `CONTACT_HAS_ASSOCIATIONS`). Posted paid invoices, voided invoices and canceled invoices do not block deletion.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.contacts.delete_contact_by_external_id(
    external_id="externalId",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**external_id:** `str` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

## Orders
<details><summary><code>client.orders.<a href="src/paid/orders/client.py">list_orders</a>(...) -&gt; AsyncHttpResponse[OrderListResponse]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Get a list of orders for the organization
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.orders.list_orders()

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**limit:** `typing.Optional[int]` 
    
</dd>
</dl>

<dl>
<dd>

**offset:** `typing.Optional[int]` 
    
</dd>
</dl>

<dl>
<dd>

**customer_id:** `typing.Optional[str]` — Filter by customer ID.
    
</dd>
</dl>

<dl>
<dd>

**external_customer_id:** `typing.Optional[str]` — Filter by customer external ID.
    
</dd>
</dl>

<dl>
<dd>

**external_id:** `typing.Optional[str]` — Filter by the order's external ID (exact match).
    
</dd>
</dl>

<dl>
<dd>

**creation_state:** `typing.Optional[ListOrdersRequestCreationState]` — Filter by creation state: draft or active.
    
</dd>
</dl>

<dl>
<dd>

**status:** `typing.Optional[OrderStatusFilter]` — Filter by derived order status. draft: not yet activated. paused: billing is paused. ended: end date is in the past. active: activated, not paused, and not ended.
    
</dd>
</dl>

<dl>
<dd>

**start_date_from:** `typing.Optional[str]` — Only orders whose start date is on or after this date. Accepts an ISO 8601 date or date-time. Date-only values (e.g. 2026-06-30) are treated as UTC; date-times without an explicit timezone offset are ambiguous, so include one (e.g. 2026-06-30T00:00:00-05:00) when precision matters.
    
</dd>
</dl>

<dl>
<dd>

**start_date_to:** `typing.Optional[str]` — Only orders whose start date is on or before this date. Accepts an ISO 8601 date or date-time. Date-only values (e.g. 2026-06-30) are treated as UTC; date-times without an explicit timezone offset are ambiguous, so include one (e.g. 2026-06-30T00:00:00-05:00) when precision matters.
    
</dd>
</dl>

<dl>
<dd>

**end_date_from:** `typing.Optional[str]` — Only orders whose end date is on or after this date. Orders without an end date are not matched. Accepts an ISO 8601 date or date-time. Date-only values (e.g. 2026-06-30) are treated as UTC; date-times without an explicit timezone offset are ambiguous, so include one (e.g. 2026-06-30T00:00:00-05:00) when precision matters.
    
</dd>
</dl>

<dl>
<dd>

**end_date_to:** `typing.Optional[str]` — Only orders whose end date is on or before this date. Orders without an end date are not matched. Accepts an ISO 8601 date or date-time. Date-only values (e.g. 2026-06-30) are treated as UTC; date-times without an explicit timezone offset are ambiguous, so include one (e.g. 2026-06-30T00:00:00-05:00) when precision matters.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.orders.<a href="src/paid/orders/client.py">create_order</a>(...) -&gt; AsyncHttpResponse[Order]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Creates a new order for the organization
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.orders.create_order(
    customer_id="customerId",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**customer_id:** `str` 
    
</dd>
</dl>

<dl>
<dd>

**billing_customer_id:** `typing.Optional[str]` 
    
</dd>
</dl>

<dl>
<dd>

**billing_contact_ids:** `typing.Optional[typing.Sequence[str]]` 
    
</dd>
</dl>

<dl>
<dd>

**name:** `typing.Optional[str]` 
    
</dd>
</dl>

<dl>
<dd>

**start_date:** `typing.Optional[dt.datetime]` 
    
</dd>
</dl>

<dl>
<dd>

**end_date:** `typing.Optional[dt.datetime]` 
    
</dd>
</dl>

<dl>
<dd>

**subscription_terms:** `typing.Optional[int]` 
    
</dd>
</dl>

<dl>
<dd>

**creation_state:** `typing.Optional[OrderCreationState]` 
    
</dd>
</dl>

<dl>
<dd>

**billing_anchor:** `typing.Optional[int]` — Day of month for billing anchor (1-31). Defaults to start date day if not provided.
    
</dd>
</dl>

<dl>
<dd>

**payment_terms:** `typing.Optional[CreateOrderRequestPaymentTerms]` 
    
</dd>
</dl>

<dl>
<dd>

**external_id:** `typing.Optional[str]` 
    
</dd>
</dl>

<dl>
<dd>

**metadata:** `typing.Optional[typing.Dict[str, typing.Any]]` 
    
</dd>
</dl>

<dl>
<dd>

**currency:** `typing.Optional[str]` 
    
</dd>
</dl>

<dl>
<dd>

**auto_post_invoices:** `typing.Optional[bool]` 
    
</dd>
</dl>

<dl>
<dd>

**auto_send_billing_emails:** `typing.Optional[bool]` 
    
</dd>
</dl>

<dl>
<dd>

**auto_send_payment_emails:** `typing.Optional[bool]` 
    
</dd>
</dl>

<dl>
<dd>

**lines:** `typing.Optional[typing.Sequence[CreateOrderLineRequest]]` 
    
</dd>
</dl>

<dl>
<dd>

**billing_frequency_override:** `typing.Optional[OrderBillingFrequencyOverride]` 
    
</dd>
</dl>

<dl>
<dd>

**purchase_order_reference:** `typing.Optional[str]` — Purchase order number printed on invoices generated from this order.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.orders.<a href="src/paid/orders/client.py">get_order_by_id</a>(...) -&gt; AsyncHttpResponse[Order]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Get an order by ID
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.orders.get_order_by_id(
    id="id",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**id:** `str` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.orders.<a href="src/paid/orders/client.py">update_order_by_id</a>(...) -&gt; AsyncHttpResponse[Order]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Update an order by ID. creationState must equal the order's current state; activate a draft order with POST /orders/{id}/activate.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.orders.update_order_by_id(
    id="id",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**id:** `str` 
    
</dd>
</dl>

<dl>
<dd>

**name:** `typing.Optional[str]` 
    
</dd>
</dl>

<dl>
<dd>

**start_date:** `typing.Optional[dt.datetime]` 
    
</dd>
</dl>

<dl>
<dd>

**end_date:** `typing.Optional[dt.datetime]` 
    
</dd>
</dl>

<dl>
<dd>

**subscription_terms:** `typing.Optional[int]` 
    
</dd>
</dl>

<dl>
<dd>

**creation_state:** `typing.Optional[OrderCreationState]` 
    
</dd>
</dl>

<dl>
<dd>

**billing_anchor:** `typing.Optional[int]` — Day of month for billing anchor (1-31). Defaults to start date day if not provided.
    
</dd>
</dl>

<dl>
<dd>

**payment_terms:** `typing.Optional[UpdateOrderRequestPaymentTerms]` 
    
</dd>
</dl>

<dl>
<dd>

**external_id:** `typing.Optional[str]` 
    
</dd>
</dl>

<dl>
<dd>

**metadata:** `typing.Optional[typing.Dict[str, typing.Any]]` 
    
</dd>
</dl>

<dl>
<dd>

**billing_customer_id:** `typing.Optional[str]` 
    
</dd>
</dl>

<dl>
<dd>

**billing_contact_ids:** `typing.Optional[typing.Sequence[str]]` 
    
</dd>
</dl>

<dl>
<dd>

**auto_post_invoices:** `typing.Optional[bool]` 
    
</dd>
</dl>

<dl>
<dd>

**auto_send_billing_emails:** `typing.Optional[bool]` 
    
</dd>
</dl>

<dl>
<dd>

**auto_send_payment_emails:** `typing.Optional[bool]` 
    
</dd>
</dl>

<dl>
<dd>

**purchase_order_reference:** `typing.Optional[str]` — Purchase order number printed on invoices generated from this order.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.orders.<a href="src/paid/orders/client.py">delete_order_by_id</a>(...) -&gt; AsyncHttpResponse[EmptyResponse]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Delete an order by ID
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.orders.delete_order_by_id(
    id="id",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**id:** `str` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.orders.<a href="src/paid/orders/client.py">activate_order_by_id</a>(...) -&gt; AsyncHttpResponse[Order]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Activate a draft order by ID. Activation starts billing for the order using the same validation and side effects as the dashboard activation flow.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.orders.activate_order_by_id(
    id="id",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**id:** `str` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.orders.<a href="src/paid/orders/client.py">get_order_lines</a>(...) -&gt; AsyncHttpResponse[OrderLinesResponse]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Get the order lines for an order by ID
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.orders.get_order_lines(
    id="id",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**id:** `str` 
    
</dd>
</dl>

<dl>
<dd>

**limit:** `typing.Optional[int]` 
    
</dd>
</dl>

<dl>
<dd>

**offset:** `typing.Optional[int]` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.orders.<a href="src/paid/orders/client.py">list_order_seats</a>(...) -&gt; AsyncHttpResponse[OrderSeatListResponse]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

List seats for an order
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.orders.list_order_seats(
    id="id",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**id:** `str` 
    
</dd>
</dl>

<dl>
<dd>

**limit:** `typing.Optional[int]` 
    
</dd>
</dl>

<dl>
<dd>

**offset:** `typing.Optional[int]` 
    
</dd>
</dl>

<dl>
<dd>

**product_external_id:** `typing.Optional[str]` 
    
</dd>
</dl>

<dl>
<dd>

**status:** `typing.Optional[ListOrderSeatsRequestStatus]` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.orders.<a href="src/paid/orders/client.py">update_order_seat_assignment</a>(...) -&gt; AsyncHttpResponse[OrderSeat]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Assign or unassign a single seat on an order
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.orders.update_order_seat_assignment(
    id="id",
    seat_id="seatId",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**id:** `str` 
    
</dd>
</dl>

<dl>
<dd>

**seat_id:** `str` 
    
</dd>
</dl>

<dl>
<dd>

**user_external_id:** `typing.Optional[str]` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.orders.<a href="src/paid/orders/client.py">batch_order_seat_assignments</a>(...) -&gt; AsyncHttpResponse[BatchSeatAssignmentsResponse]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Assign or unassign seats in batch for an order
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid
from paid.orders import BatchSeatAssignmentsRequestAssignmentsItem

client = Paid(
    token="YOUR_TOKEN",
)
client.orders.batch_order_seat_assignments(
    id="id",
    assignments=[
        BatchSeatAssignmentsRequestAssignmentsItem(
            seat_id="seatId",
        )
    ],
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**id:** `str` 
    
</dd>
</dl>

<dl>
<dd>

**assignments:** `typing.Sequence[BatchSeatAssignmentsRequestAssignmentsItem]` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.orders.<a href="src/paid/orders/client.py">upsert_order_by_external_id</a>(...) -&gt; AsyncHttpResponse[ProvisionResponse]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Creates or updates a customer, an optional contact, the products on the order, and the order with its lines, in one call. Each record is matched by its `externalId`, so you can send the same deal again after it changes in your CRM. Returns 201 Created when the order is new, and 200 OK when an existing draft order with the same `externalId` was updated. Treat both as success. Re-provisioning an order that is already active returns 409 `ORDER_ALREADY_ACTIVE`; change an active order through the amendments API instead. Amounts are whole numbers in minor currency units (cents).
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
import datetime

from paid import (
    CustomerBillingAddressInput,
    Paid,
    ProvisionCustomer,
    ProvisionExternalReference,
    ProvisionOrder,
    ProvisionOrderLine,
    ProvisionProduct,
    ProvisionProductAttributesItem,
    ProvisionProductAttributesItemPricing,
    ProvisionProductAttributesItemPricingPricePoints,
)

client = Paid(
    token="YOUR_TOKEN",
)
client.orders.upsert_order_by_external_id(
    idempotency_key="deal_88213-2026-09-30T14:05:00Z",
    external_reference=ProvisionExternalReference(
        source="standard",
        metadata={"crmDealId": "deal_88213", "crmAccountId": "acct_4410"},
    ),
    customer=ProvisionCustomer(
        external_id="acct_4410",
        name="Northwind Logistics",
        email="billing@northwind.example",
        billing_address=CustomerBillingAddressInput(
            line1="200 Harbor Way",
            city="Seattle",
            state="WA",
            zip_code="98101",
            country="US",
        ),
    ),
    order=ProvisionOrder(
        external_id="deal_88213",
        name="Northwind Logistics - Growth plan",
        currency="USD",
        start_date=datetime.datetime.fromisoformat(
            "2026-10-01 00:00:00+00:00",
        ),
        lines=[
            ProvisionOrderLine(
                external_id="deal_88213_platform",
                start_date=datetime.datetime.fromisoformat(
                    "2026-10-01 00:00:00+00:00",
                ),
                product=ProvisionProduct(
                    external_id="growth_plan",
                    name="Growth plan",
                    attributes=[
                        ProvisionProductAttributesItem(
                            name="Platform fee",
                            pricing=ProvisionProductAttributesItemPricing(
                                charge_type="recurring",
                                pricing_model="perUnit",
                                billing_frequency="monthly",
                                billing_type="advance",
                                price_points=ProvisionProductAttributesItemPricingPricePoints(
                                    currency="USD",
                                    unit_price=49900,
                                ),
                            ),
                        )
                    ],
                ),
            ),
            ProvisionOrderLine(
                external_id="deal_88213_launch_discount",
                line_type="CREDIT",
                start_date=datetime.datetime.fromisoformat(
                    "2026-10-01 00:00:00+00:00",
                ),
                end_date=datetime.datetime.fromisoformat(
                    "2027-01-01 00:00:00+00:00",
                ),
                product=ProvisionProduct(
                    external_id="launch_discount",
                    name="Launch discount",
                    attributes=[
                        ProvisionProductAttributesItem(
                            name="Launch discount",
                            pricing=ProvisionProductAttributesItemPricing(
                                charge_type="recurring",
                                pricing_model="perUnit",
                                billing_frequency="monthly",
                                billing_type="advance",
                                price_points=ProvisionProductAttributesItemPricingPricePoints(
                                    currency="USD",
                                    unit_price=10000,
                                ),
                            ),
                        )
                    ],
                ),
            ),
        ],
    ),
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**idempotency_key:** `str` 
    
</dd>
</dl>

<dl>
<dd>

**external_reference:** `ProvisionExternalReference` 
    
</dd>
</dl>

<dl>
<dd>

**customer:** `ProvisionCustomer` 
    
</dd>
</dl>

<dl>
<dd>

**order:** `ProvisionOrder` 
    
</dd>
</dl>

<dl>
<dd>

**contact:** `typing.Optional[ProvisionContact]` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

## Invoices
<details><summary><code>client.invoices.<a href="src/paid/invoices/client.py">list_invoices</a>(...) -&gt; AsyncHttpResponse[InvoiceListResponse]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Get a list of invoices for the organization
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.invoices.list_invoices()

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**limit:** `typing.Optional[int]` 
    
</dd>
</dl>

<dl>
<dd>

**offset:** `typing.Optional[int]` 
    
</dd>
</dl>

<dl>
<dd>

**customer_id:** `typing.Optional[str]` — Filter by customer ID.
    
</dd>
</dl>

<dl>
<dd>

**external_customer_id:** `typing.Optional[str]` — Filter by customer external ID.
    
</dd>
</dl>

<dl>
<dd>

**order_id:** `typing.Optional[str]` — Filter by the order this invoice was generated from.
    
</dd>
</dl>

<dl>
<dd>

**status:** `typing.Optional[ListInvoicesRequestStatus]` — Filter by invoice status.
    
</dd>
</dl>

<dl>
<dd>

**payment_status:** `typing.Optional[ListInvoicesRequestPaymentStatus]` — Filter by payment status.
    
</dd>
</dl>

<dl>
<dd>

**issue_date_from:** `typing.Optional[str]` — Only invoices whose issue date is on or after this date. Accepts an ISO 8601 date or date-time. Date-only values (e.g. 2026-06-30) are treated as UTC; date-times without an explicit timezone offset are ambiguous, so include one (e.g. 2026-06-30T00:00:00-05:00) when precision matters.
    
</dd>
</dl>

<dl>
<dd>

**issue_date_to:** `typing.Optional[str]` — Only invoices whose issue date is on or before this date. Accepts an ISO 8601 date or date-time. Date-only values (e.g. 2026-06-30) are treated as UTC; date-times without an explicit timezone offset are ambiguous, so include one (e.g. 2026-06-30T00:00:00-05:00) when precision matters.
    
</dd>
</dl>

<dl>
<dd>

**due_date_from:** `typing.Optional[str]` — Only invoices whose due date is on or after this date. Invoices without a due date are not matched. Accepts an ISO 8601 date or date-time. Date-only values (e.g. 2026-06-30) are treated as UTC; date-times without an explicit timezone offset are ambiguous, so include one (e.g. 2026-06-30T00:00:00-05:00) when precision matters.
    
</dd>
</dl>

<dl>
<dd>

**due_date_to:** `typing.Optional[str]` — Only invoices whose due date is on or before this date. Invoices without a due date are not matched. Accepts an ISO 8601 date or date-time. Date-only values (e.g. 2026-06-30) are treated as UTC; date-times without an explicit timezone offset are ambiguous, so include one (e.g. 2026-06-30T00:00:00-05:00) when precision matters.
    
</dd>
</dl>

<dl>
<dd>

**display_number:** `typing.Optional[str]` — Filter by the invoice number shown on the invoice, whether draft or posted (exact match).
    
</dd>
</dl>

<dl>
<dd>

**purchase_order_reference:** `typing.Optional[str]` — Filter by purchase order reference (exact match, whitespace-sensitive).
    
</dd>
</dl>

<dl>
<dd>

**currency:** `typing.Optional[str]` — Filter by invoice currency code (case-insensitive, e.g. USD).
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.invoices.<a href="src/paid/invoices/client.py">get_invoice_by_id</a>(...) -&gt; AsyncHttpResponse[Invoice]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Get an invoice by ID
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.invoices.get_invoice_by_id(
    id="id",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**id:** `str` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.invoices.<a href="src/paid/invoices/client.py">update_invoice_by_id</a>(...) -&gt; AsyncHttpResponse[Invoice]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Update an invoice by ID
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.invoices.update_invoice_by_id(
    id="id",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**id:** `str` 
    
</dd>
</dl>

<dl>
<dd>

**metadata:** `typing.Optional[typing.Dict[str, typing.Any]]` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.invoices.<a href="src/paid/invoices/client.py">get_invoice_lines</a>(...) -&gt; AsyncHttpResponse[InvoiceLinesResponse]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Get the invoice lines for an invoice by ID
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.invoices.get_invoice_lines(
    id="id",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**id:** `str` 
    
</dd>
</dl>

<dl>
<dd>

**limit:** `typing.Optional[int]` 
    
</dd>
</dl>

<dl>
<dd>

**offset:** `typing.Optional[int]` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

## Signals
<details><summary><code>client.signals.<a href="src/paid/signals/client.py">list_signals</a>(...) -&gt; AsyncHttpResponse[SignalListResponse]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Returns ingested signals (usage events) for your organization, newest first. Filter by signal name, customer, product, and creation date range.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.signals.list_signals()

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**limit:** `typing.Optional[int]` 
    
</dd>
</dl>

<dl>
<dd>

**offset:** `typing.Optional[int]` 
    
</dd>
</dl>

<dl>
<dd>

**signal_name:** `typing.Optional[str]` — Filter by signal event name (exact match).
    
</dd>
</dl>

<dl>
<dd>

**customer_id:** `typing.Optional[str]` — Filter by the Paid customer ID the signal is attributed to.
    
</dd>
</dl>

<dl>
<dd>

**external_customer_id:** `typing.Optional[str]` — Filter by your external customer ID. Aliases resolve to the attributed customer, and unresolved IDs match raw ingest data.
    
</dd>
</dl>

<dl>
<dd>

**product_id:** `typing.Optional[str]` — Filter by the Paid product ID the signal is attributed to.
    
</dd>
</dl>

<dl>
<dd>

**external_product_id:** `typing.Optional[str]` — Filter by your external product ID.
    
</dd>
</dl>

<dl>
<dd>

**created_at_from:** `typing.Optional[str]` — Only signals created on or after this date. Accepts an ISO 8601 date or date-time. Date-only values (e.g. 2026-06-30) are treated as UTC; date-times without an explicit timezone offset are ambiguous, so include one (e.g. 2026-06-30T00:00:00-05:00) when precision matters.
    
</dd>
</dl>

<dl>
<dd>

**created_at_to:** `typing.Optional[str]` — Only signals created on or before this date. Accepts an ISO 8601 date or date-time. Date-only values (e.g. 2026-06-30) are treated as UTC; date-times without an explicit timezone offset are ambiguous, so include one (e.g. 2026-06-30T00:00:00-05:00) when precision matters.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.signals.<a href="src/paid/signals/client.py">get_signal_by_id</a>(...) -&gt; AsyncHttpResponse[SignalListItem]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Get a single ingested signal (usage event) by its ID, including the data payload submitted at ingest.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.signals.get_signal_by_id(
    id="6890b0e2a6f2c30012f0a1b3",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**id:** `str` — Signal ID.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.signals.<a href="src/paid/signals/client.py">get_signals_by_ids</a>(...) -&gt; AsyncHttpResponse[SignalLookupResponse]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Fetch up to 500 ingested signals (usage events) by ID in one request, each with the data payload submitted at ingest. Use it to resolve the `signalId` values on a page of credit transactions, or the signals behind one run or ticket, instead of calling `GET /api/v2/signals/{id}` once per signal. Signals are returned in request order; IDs that match nothing in your organization are listed under `notFound` rather than failing the request. This is a read-only operation: it requires the `read:usage` scope and is not available to customer API keys.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.signals.get_signals_by_ids(
    ids=["6890b0e2a6f2c30012f0a1b3", "6890b0e2a6f2c30012f0a1b4"],
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**ids:** `typing.Sequence[str]` — Signal IDs to fetch, up to 500 per request. Duplicates are collapsed. For example the `signalId` values from a page of `GET /api/v2/credits/transactions`.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.signals.<a href="src/paid/signals/client.py">create_signals</a>(...) -&gt; AsyncHttpResponse[BulkSignalsResponse]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Create multiple signals (usage events) in a single request. Each signal names its customer (`customerId` or `externalCustomerId`; an alias of the customer resolves too), its customer unit (`externalCustomerUnitId`: the unit's key, or an alias that points at a unit), or both, plus an optional product attribution (`productId` or `externalProductId`). A unit alone derives the customer from the unit. A unit sent with a customer must belong to it, else the batch is refused with `400 CUSTOMER_UNIT_MISMATCH`. A unit is never named by Paid's unit id. A key or alias that matches nothing is stored without a customer, keeping the id, so it can be pointed at a unit later. An identifier is matched as a unit key first, then a customer external id, then an alias. Unit attribution is enabled per organization: where it is not, a `customerUnit` sent next to a `customer` is accepted and ignored rather than checked, as before, while a unit named alone still derives its customer. Authenticate with your organization API key, or with a customer API key (prefixed `paid_ck_live_` or `paid_ck_test_`, created with `POST /api/v2/customers/{id}/api-keys`) so a customer's own runtime can report its usage without holding your organization key. A customer API key can call this endpoint only, and only for its own customer: if any signal names another customer, or one Paid does not know, the whole batch is refused with `403` `CUSTOMER_MISMATCH` and nothing is stored. Its idempotency keys are scoped to that customer. See [Send signals with a customer API key](/documentation/signals/send-signals-with-a-customer-api-key).
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
import datetime

from paid import (
    CustomerByExternalId,
    CustomerUnitAttribution,
    Paid,
    ProductByExternalId,
    Signal,
)

client = Paid(
    token="YOUR_TOKEN",
)
client.signals.create_signals(
    signals=[
        Signal(
            event_name="email_sent",
            customer=CustomerByExternalId(
                external_customer_id="acme-corp",
            ),
            attribution=ProductByExternalId(
                external_product_id="email-agent",
            ),
            timestamp=datetime.datetime.fromisoformat(
                "2026-01-31 23:00:00+00:00",
            ),
            data={"recipients": 3},
            idempotency_key="email_sent_msg_8f3a2c",
        ),
        Signal(
            event_name="email_sent",
            customer_unit=CustomerUnitAttribution(
                external_customer_unit_id="acme-marketing",
            ),
            idempotency_key="email_sent_msg_9b41d7",
        ),
    ],
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**signals:** `typing.Sequence[Signal]` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.signals.<a href="src/paid/signals/client.py">create_signal_summary_export</a>(...) -&gt; AsyncHttpResponse[SignalSummaryExport]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Produces a usage summary file: one row per customer per month, with columns broken down by the signal payload fields you name. Choose `format` to get JSON or a wide CSV; JSON is the default. The export runs in the background, so this returns immediately with a pending export and the file appears on it once it succeeds. Give `spec` to describe the columns, or `copyFrom` with an earlier export's id to reuse that export's columns over a new window. The spec is stored on the export, so an old file can always be explained by the export that produced it. Signals are bucketed by month in UTC on their effective timestamp, the same basis billing uses, so the totals reconcile with invoices. One export runs per organization at a time.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import (
    Paid,
    SignalSummaryBandsDimensionBracketsItem,
    SignalSummaryMeasure,
    SignalSummarySignalSelection,
    SignalSummarySpecInput,
    SignalSummarySpecInputDimensionsItem_Bands,
)

client = Paid(
    token="YOUR_TOKEN",
)
client.signals.create_signal_summary_export(
    spec=SignalSummarySpecInput(
        name="Monthly workflow usage",
        signal_names=[
            SignalSummarySignalSelection(
                name="workflow_run",
                group="Workflow runs",
                slug="runs",
            )
        ],
        dimensions=[
            SignalSummarySpecInputDimensionsItem_Bands(
                path="steps",
                applies_to=["workflow_run"],
                brackets=[
                    SignalSummaryBandsDimensionBracketsItem(
                        from_=1,
                        up_to=10,
                        slug="1_10_steps",
                        group="1-10 steps",
                    ),
                    SignalSummaryBandsDimensionBracketsItem(
                        slug="11_plus_steps",
                        group="11+ steps",
                    ),
                ],
            )
        ],
        measures=[
            SignalSummaryMeasure(
                fn="count",
            )
        ],
    ),
    from_="2026-01-01",
    to="2026-04-01",
    format="csv",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**from_:** `str` — Start of the window, inclusive, snapped down to a period start. Cannot be earlier than 2025-09-01.
    
</dd>
</dl>

<dl>
<dd>

**to:** `str` — End of the window, exclusive, snapped down to a period start.
    
</dd>
</dl>

<dl>
<dd>

**spec:** `typing.Optional[SignalSummarySpecInput]` 
    
</dd>
</dl>

<dl>
<dd>

**copy_from:** `typing.Optional[str]` — Reuse the shape of an earlier export, so a monthly run produces the same columns without restating them. Give this or spec, not both.
    
</dd>
</dl>

<dl>
<dd>

**customer_ids:** `typing.Optional[typing.Sequence[str]]` — Restrict the file to these customers. Accepts Paid IDs, UUIDs, or your external IDs. Omit for every customer.
    
</dd>
</dl>

<dl>
<dd>

**format:** `typing.Optional[CreateSignalSummaryExportRequestFormat]` — Format of the produced file. `json` gives the columns and one object per customer per month; `csv` gives the wide spreadsheet with three header rows.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.signals.<a href="src/paid/signals/client.py">get_signal_summary_export</a>(...) -&gt; AsyncHttpResponse[SignalSummaryExport]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Returns one export with its status and the shape it ran. While it is succeeded and unexpired, each read issues a fresh short-lived download link.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.signals.get_signal_summary_export(
    id="sse_5pZa1Kd8Nqx",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**id:** `str` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.signals.<a href="src/paid/signals/client.py">get_signal_summary</a>(...) -&gt; AsyncHttpResponse[SignalSummaryResponse]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Counts signals into period buckets. Use it to chart signal volume over time, split by signal name or by a signal-payload key such as team_id.

Signals count against their effective timestamp in UTC. This is the same basis billing uses, so these figures reconcile with an invoice.

The endpoint reads a pre-aggregated hourly rollup, not the raw signals. An hour is therefore the finest interval. One request may span at most 250 periods.

A grouped request keeps the 100 groups with the most signals. It sums the rest under the key "other". Each period total stays complete, but a group inside "other" loses its own figure. Read groups.truncated to learn how many groups lost it. Zero means every group kept its own figure.

Results are paged with limit and cursor. A page is never truncated in silence.

To download the series as a file, call this endpoint over HTTP with format=csv. It returns the same points as a text/csv table with a single header row. A CSV cannot carry a cursor, so a result that does not fit one page returns 400 rather than a file that is missing rows.

For a wide per-customer spreadsheet over arbitrary payload fields, use the batch export at /signals/summary/batch instead.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.signals.get_signal_summary(
    start_date="2026-08-01",
    end_date="2026-08-31",
    interval="day",
    group_by="signalName",
    limit=1000,
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**start_date:** `str` — Window start, inclusive. Accepts an ISO 8601 date or date-time. Date-only values (e.g. 2026-06-30) are treated as UTC; date-times without an explicit timezone offset are ambiguous, so include one (e.g. 2026-06-30T00:00:00-05:00) when precision matters.
    
</dd>
</dl>

<dl>
<dd>

**end_date:** `str` — Window end, inclusive — a date-only value means through the end of that UTC day. Accepts an ISO 8601 date or date-time. Date-only values (e.g. 2026-06-30) are treated as UTC; date-times without an explicit timezone offset are ambiguous, so include one (e.g. 2026-06-30T00:00:00-05:00) when precision matters.
    
</dd>
</dl>

<dl>
<dd>

**interval:** `GetSignalSummaryRequestInterval` 

Period size. Buckets align to UTC hours, calendar days, ISO weeks that start on Monday, or calendar months. An hour is the finest period available.

The first and last bucket report bounds clipped to your window.

A request may span at most 250 periods. That is about 10 days at hour, 8 months at day, and 20 years at month. A longer window returns 400 WINDOW_TOO_LARGE.
    
</dd>
</dl>

<dl>
<dd>

**customer_id:** `typing.Optional[str]` — Narrow the summary to one customer. Accepts a Paid customer ID or, as a fallback, your external customer ID. Mutually exclusive with externalCustomerId. Omit both to summarize every customer.
    
</dd>
</dl>

<dl>
<dd>

**external_customer_id:** `typing.Optional[str]` — Your external customer ID. Mutually exclusive with customerId.
    
</dd>
</dl>

<dl>
<dd>

**signal_names:** `typing.Optional[typing.Union[str, typing.Sequence[str]]]` — Narrow the summary to these signal names. Repeat the parameter for more than one. Omit to count every signal.
    
</dd>
</dl>

<dl>
<dd>

**group_by:** `typing.Optional[GetSignalSummaryRequestGroupBy]` 

Split each period into one series per group. signalName groups by the signal's name. dimension groups by a signal-payload key named in dimensionKey — signals per team, for example. Omit for one total per period.

A grouped request keeps the 100 groups with the most signals. It sums the rest under the key "other".

Each period total stays complete, but a group inside "other" loses its own figure. Read groups.truncated to learn how many groups lost it. To name every group, narrow the request with signalNames, customerId, or a shorter window.
    
</dd>
</dl>

<dl>
<dd>

**dimension_key:** `typing.Optional[str]` — The signal-payload key to group by. Required when groupBy is dimension; rejected otherwise. The key must be one your organization has declared for grouping — an undeclared key is refused rather than answered by a full scan. A signal whose payload lacks the key, or holds it empty, groups under an empty groupKey.
    
</dd>
</dl>

<dl>
<dd>

**limit:** `typing.Optional[int]` — Maximum points to return in this page. The response is never silently truncated: when more points exist, hasMore is true and nextCursor is set.
    
</dd>
</dl>

<dl>
<dd>

**cursor:** `typing.Optional[str]` — Continue from a previous response's nextCursor. Every other parameter must be unchanged.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

## Credits
<details><summary><code>client.credits.<a href="src/paid/credits/client.py">list_credit_currencies</a>(...) -&gt; AsyncHttpResponse[CreditCurrencyListResponse]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

List credit currencies for the organization. Includes active and archived currencies by default; use the status query parameter to filter.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.credits.list_credit_currencies()

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**status:** `typing.Optional[ListCreditCurrenciesRequestStatus]` — Filter credit currencies by status. Defaults to `all` so archived currencies remain readable after they are archived.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.credits.<a href="src/paid/credits/client.py">create_credit_currency</a>(...) -&gt; AsyncHttpResponse[CreditCurrency]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Creates a credit currency for the organization.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.credits.create_credit_currency(
    name="API Credits",
    key="api_credits",
    description="Credits consumed by API calls.",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**name:** `str` — Human-readable name shown for this credit currency.
    
</dd>
</dl>

<dl>
<dd>

**key:** `str` — Stable machine-readable key for this credit currency. Use lowercase letters, numbers, underscores, and hyphens. Keys are unique within an organization.
    
</dd>
</dl>

<dl>
<dd>

**description:** `typing.Optional[str]` — Optional description for this credit currency.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.credits.<a href="src/paid/credits/client.py">list_credit_transactions</a>(...) -&gt; AsyncHttpResponse[CreditTransactionListResponse]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

List credit ledger transactions (grants, spends, and pending grants) for the organization, newest first. Filter by customer, credit currency, type, order, or date range.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.credits.list_credit_transactions()

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**limit:** `typing.Optional[int]` 
    
</dd>
</dl>

<dl>
<dd>

**offset:** `typing.Optional[int]` 
    
</dd>
</dl>

<dl>
<dd>

**customer_id:** `typing.Optional[str]` — Filter by customer ID.
    
</dd>
</dl>

<dl>
<dd>

**external_customer_id:** `typing.Optional[str]` — Filter by customer external ID.
    
</dd>
</dl>

<dl>
<dd>

**credits_currency_id:** `typing.Optional[str]` — Filter by credit currency ID.
    
</dd>
</dl>

<dl>
<dd>

**credit_currency_key:** `typing.Optional[str]` — Filter by the stable machine-readable key of the credit currency.
    
</dd>
</dl>

<dl>
<dd>

**type:** `typing.Optional[ListCreditTransactionsRequestType]` — Filter by transaction type.
    
</dd>
</dl>

<dl>
<dd>

**order_id:** `typing.Optional[str]` — Filter by the order this transaction is linked to.
    
</dd>
</dl>

<dl>
<dd>

**event_name:** `typing.Optional[str]` — Filter by the usage event that produced the transaction (exact match, untrimmed). Transactions with no event (grants, invoice-void clawbacks, manual adjustments) never match.
    
</dd>
</dl>

<dl>
<dd>

**created_at_from:** `typing.Optional[str]` — Only transactions recorded on or after this date. Accepts an ISO 8601 date or date-time. Date-only values (e.g. 2026-06-30) are treated as UTC; date-times without an explicit timezone offset are ambiguous, so include one (e.g. 2026-06-30T00:00:00-05:00) when precision matters.
    
</dd>
</dl>

<dl>
<dd>

**created_at_to:** `typing.Optional[str]` — Only transactions recorded on or before this date. Accepts an ISO 8601 date or date-time. Date-only values (e.g. 2026-06-30) are treated as UTC; date-times without an explicit timezone offset are ambiguous, so include one (e.g. 2026-06-30T00:00:00-05:00) when precision matters.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.credits.<a href="src/paid/credits/client.py">get_credit_transaction_summary</a>(...) -&gt; AsyncHttpResponse[CreditTransactionSummaryResponse]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Aggregate credit ledger transactions of one type into period buckets grouped by a dimension, for charting credit consumption over time. Accepts the same filters as the transaction list and reads the same ledger, so bucket totals always match summing that list over the same filters and the same `type` — pass this endpoint's effective `type` (it defaults to `spend`) to the list, which has no default and otherwise returns every type.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.credits.get_credit_transaction_summary(
    created_at_from="createdAtFrom",
    group_by="eventName",
    interval="day",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**created_at_from:** `str` — Start of the window (inclusive). Required: the window may span at most 92 days. Accepts an ISO 8601 date or date-time. Date-only values (e.g. 2026-06-30) are treated as UTC; date-times without an explicit timezone offset are ambiguous, so include one (e.g. 2026-06-30T00:00:00-05:00) when precision matters.
    
</dd>
</dl>

<dl>
<dd>

**group_by:** `GetCreditTransactionSummaryRequestGroupBy` — Dimension to group buckets by. `eventName` groups signal-driven spends by the usage event that produced them; transactions without one (grants, invoice-void clawbacks, manual adjustments) share a bucket whose `groupKey` is null.
    
</dd>
</dl>

<dl>
<dd>

**interval:** `GetCreditTransactionSummaryRequestInterval` — Period bucket size. Buckets align to UTC calendar days, ISO weeks (Monday start), or calendar months; the first and last bucket's published bounds are clipped to the requested window.
    
</dd>
</dl>

<dl>
<dd>

**customer_id:** `typing.Optional[str]` — Filter by customer ID.
    
</dd>
</dl>

<dl>
<dd>

**external_customer_id:** `typing.Optional[str]` — Filter by customer external ID.
    
</dd>
</dl>

<dl>
<dd>

**credits_currency_id:** `typing.Optional[str]` — Filter by credit currency ID.
    
</dd>
</dl>

<dl>
<dd>

**credit_currency_key:** `typing.Optional[str]` — Filter by the stable machine-readable key of the credit currency.
    
</dd>
</dl>

<dl>
<dd>

**type:** `typing.Optional[GetCreditTransactionSummaryRequestType]` — Ledger type to summarize. Each call summarizes exactly one type, so a spend chart never nets grants against spends. Defaults to `spend`.
    
</dd>
</dl>

<dl>
<dd>

**order_id:** `typing.Optional[str]` — Filter by the order this transaction is linked to.
    
</dd>
</dl>

<dl>
<dd>

**event_name:** `typing.Optional[str]` — Filter by the usage event that produced the transaction (exact match, untrimmed). Transactions with no event (grants, invoice-void clawbacks, manual adjustments) never match.
    
</dd>
</dl>

<dl>
<dd>

**created_at_to:** `typing.Optional[str]` — End of the window (inclusive). Defaults to now. Accepts an ISO 8601 date or date-time. Date-only values (e.g. 2026-06-30) are treated as UTC; date-times without an explicit timezone offset are ambiguous, so include one (e.g. 2026-06-30T00:00:00-05:00) when precision matters.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.credits.<a href="src/paid/credits/client.py">update_credit_currency_by_id</a>(...) -&gt; AsyncHttpResponse[CreditCurrency]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Update a credit currency description or set its active/archive status.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.credits.update_credit_currency_by_id(
    id="7f4f5d4c-55e9-4d5b-a3e7-c9eb3d2d01bf",
    description="Credits consumed by developer API calls.",
    status="archived",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**id:** `str` — Credit currency ID.
    
</dd>
</dl>

<dl>
<dd>

**description:** `typing.Optional[str]` — Updated description for this credit currency. Use null to clear the description.
    
</dd>
</dl>

<dl>
<dd>

**status:** `typing.Optional[UpdateCreditCurrencyRequestStatus]` — Set to `archived` to archive this credit currency, or `active` to restore it.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

## Checkouts
<details><summary><code>client.checkouts.<a href="src/paid/checkouts/client.py">list_checkouts</a>(...) -&gt; AsyncHttpResponse[CheckoutListResponse]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Get a list of checkouts for the organization
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.checkouts.list_checkouts()

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**limit:** `typing.Optional[int]` 
    
</dd>
</dl>

<dl>
<dd>

**offset:** `typing.Optional[int]` 
    
</dd>
</dl>

<dl>
<dd>

**status:** `typing.Optional[ListCheckoutsRequestStatus]` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.checkouts.<a href="src/paid/checkouts/client.py">create_checkout</a>(...) -&gt; AsyncHttpResponse[Checkout]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Creates a checkout link that generates a URL for a customer to complete a purchase
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import CheckoutProductInput, Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.checkouts.create_checkout(
    products=[
        CheckoutProductInput(
            id="id",
        )
    ],
    success_url="successUrl",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**products:** `typing.Sequence[CheckoutProductInput]` 
    
</dd>
</dl>

<dl>
<dd>

**success_url:** `str` 
    
</dd>
</dl>

<dl>
<dd>

**customer_id:** `typing.Optional[str]` 
    
</dd>
</dl>

<dl>
<dd>

**external_customer_id:** `typing.Optional[str]` — External customer identifier. Creates the customer on first use, resolves to the existing customer on subsequent uses.
    
</dd>
</dl>

<dl>
<dd>

**cancel_url:** `typing.Optional[str]` 
    
</dd>
</dl>

<dl>
<dd>

**expires_at:** `typing.Optional[dt.datetime]` — When the checkout link expires. Must be in the future. Defaults to 24 hours after creation.
    
</dd>
</dl>

<dl>
<dd>

**metadata:** `typing.Optional[typing.Dict[str, typing.Any]]` 
    
</dd>
</dl>

<dl>
<dd>

**collect_address:** `typing.Optional[bool]` 
    
</dd>
</dl>

<dl>
<dd>

**collect_phone:** `typing.Optional[bool]` 
    
</dd>
</dl>

<dl>
<dd>

**success_title:** `typing.Optional[str]` — Heading on the confirmation screen the customer sees after completing checkout, shown instead of "Payment successful". Plain text, up to 80 characters. Omit it to keep the default.
    
</dd>
</dl>

<dl>
<dd>

**success_message:** `typing.Optional[str]` — Text under the heading on the confirmation screen, shown instead of "Thank you for your purchase!". Plain text, up to 500 characters. Omit it to keep the default. Right after checkout, the page still adds "You will be redirected shortly." before sending the customer to `successUrl`.
    
</dd>
</dl>

<dl>
<dd>

**single_use:** `typing.Optional[bool]` 
    
</dd>
</dl>

<dl>
<dd>

**currency:** `typing.Optional[str]` — Lock checkout to a specific currency. Omit to allow all currencies supported by the selected plans. If the checkout is for a customer with an active subscription, the currency must match that subscription's currency — subscriptions cannot change currency.
    
</dd>
</dl>

<dl>
<dd>

**custom_cards:** `typing.Optional[typing.Sequence[CheckoutCustomCardInput]]` — Additional informational pricing cards rendered alongside the plans.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.checkouts.<a href="src/paid/checkouts/client.py">get_checkout</a>(...) -&gt; AsyncHttpResponse[CheckoutDetails]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Get a checkout by ID
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.checkouts.get_checkout(
    id="id",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**id:** `str` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.checkouts.<a href="src/paid/checkouts/client.py">archive_checkout</a>(...) -&gt; AsyncHttpResponse[EmptyResponse]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Archive a checkout by ID so it can no longer be paid. A completed (paid) checkout can't be archived and returns 409.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.checkouts.archive_checkout(
    id="id",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**id:** `str` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

## CustomerPortals
<details><summary><code>client.customer_portals.<a href="src/paid/customer_portals/client.py">create_customer_portal</a>(...) -&gt; AsyncHttpResponse[CustomerPortal]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Deprecated — the customer portal is being retired, and this endpoint will be removed in a future release. It keeps working for existing integrations meanwhile: it creates a portal session for the customer and returns a short-lived URL to the customer portal. Contact Paid before building anything new on it.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.customer_portals.create_customer_portal()

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**customer_id:** `typing.Optional[str]` — The Paid customer ID (display ID or UUID). Either this or externalCustomerId must be provided.
    
</dd>
</dl>

<dl>
<dd>

**external_customer_id:** `typing.Optional[str]` — Your external customer ID. Either this or customerId must be provided.
    
</dd>
</dl>

<dl>
<dd>

**return_url:** `typing.Optional[str]` 
    
</dd>
</dl>

<dl>
<dd>

**expires_at:** `typing.Optional[dt.datetime]` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

## ValueReceipts
<details><summary><code>client.value_receipts.<a href="src/paid/value_receipts/client.py">list_value_receipts</a>(...) -&gt; AsyncHttpResponse[ValueReceiptListResponse]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

List value receipts for the organization
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.value_receipts.list_value_receipts()

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**limit:** `typing.Optional[int]` 
    
</dd>
</dl>

<dl>
<dd>

**offset:** `typing.Optional[int]` 
    
</dd>
</dl>

<dl>
<dd>

**customer_id:** `typing.Optional[str]` — Filter by customer display ID.
    
</dd>
</dl>

<dl>
<dd>

**external_customer_id:** `typing.Optional[str]` — Filter by customer external ID.
    
</dd>
</dl>

<dl>
<dd>

**order_id:** `typing.Optional[str]` 
    
</dd>
</dl>

<dl>
<dd>

**product_id:** `typing.Optional[str]` 
    
</dd>
</dl>

<dl>
<dd>

**archived:** `typing.Optional[ListValueReceiptsRequestArchived]` — Include archived value receipts. Defaults to false.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.value_receipts.<a href="src/paid/value_receipts/client.py">create_value_receipt</a>(...) -&gt; AsyncHttpResponse[ValueReceiptSyncResponse]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Creates a value receipt for a customer and date range, optionally scoped to a product or an order. Every call creates a receipt, so calling twice for the same period gives the customer two. The date range must have ended; a range with nothing delivered in it reports zero. Returns the receipt's ID and public URL.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
import datetime

from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.value_receipts.create_value_receipt(
    start_date=datetime.datetime.fromisoformat(
        "2024-01-15 09:30:00+00:00",
    ),
    end_date=datetime.datetime.fromisoformat(
        "2024-01-15 09:30:00+00:00",
    ),
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**start_date:** `dt.datetime` 
    
</dd>
</dl>

<dl>
<dd>

**end_date:** `dt.datetime` 
    
</dd>
</dl>

<dl>
<dd>

**customer_id:** `typing.Optional[str]` — Mutually exclusive with externalCustomerId. Exactly one is required.
    
</dd>
</dl>

<dl>
<dd>

**external_customer_id:** `typing.Optional[str]` — Mutually exclusive with customerId. Exactly one is required.
    
</dd>
</dl>

<dl>
<dd>

**product:** `typing.Optional[SyncValueReceiptRequestProduct]` — Mutually exclusive with orderId. Provide at most one.
    
</dd>
</dl>

<dl>
<dd>

**order_id:** `typing.Optional[str]` — Mutually exclusive with product. Provide at most one.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.value_receipts.<a href="src/paid/value_receipts/client.py">sync_value_receipt</a>(...) -&gt; AsyncHttpResponse[ValueReceiptSyncResponse]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Deprecated — use POST /value-receipts. Returns the receipt this customer already has for the date range (200), and creates one only if there is none (201), so calling twice does not give the customer two receipts. A receipt does not change after it is created, so a repeated call reports the same figures. Create a new receipt to report the period again with newer data.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
import datetime

from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.value_receipts.sync_value_receipt(
    start_date=datetime.datetime.fromisoformat(
        "2024-01-15 09:30:00+00:00",
    ),
    end_date=datetime.datetime.fromisoformat(
        "2024-01-15 09:30:00+00:00",
    ),
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**start_date:** `dt.datetime` 
    
</dd>
</dl>

<dl>
<dd>

**end_date:** `dt.datetime` 
    
</dd>
</dl>

<dl>
<dd>

**customer_id:** `typing.Optional[str]` — Mutually exclusive with externalCustomerId. Exactly one is required.
    
</dd>
</dl>

<dl>
<dd>

**external_customer_id:** `typing.Optional[str]` — Mutually exclusive with customerId. Exactly one is required.
    
</dd>
</dl>

<dl>
<dd>

**product:** `typing.Optional[SyncValueReceiptRequestProduct]` — Mutually exclusive with orderId. Provide at most one.
    
</dd>
</dl>

<dl>
<dd>

**order_id:** `typing.Optional[str]` — Mutually exclusive with product. Provide at most one.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.value_receipts.<a href="src/paid/value_receipts/client.py">get_value_receipt_by_id</a>(...) -&gt; AsyncHttpResponse[ValueReceiptDetail]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Get a value receipt by ID, including its publish/share state.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.value_receipts.get_value_receipt_by_id(
    id="id",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**id:** `str` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.value_receipts.<a href="src/paid/value_receipts/client.py">refresh_value_receipt</a>(...) -&gt; AsyncHttpResponse[ValueReceiptSyncResponse]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Deprecated. A value receipt is immutable from creation, so it cannot be re-populated. A receipt you own answers 409, and one that does not exist or belongs to another organization answers 404. To get current data, create a new receipt for the period with `POST /value-receipts`.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.value_receipts.refresh_value_receipt(
    id="id",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**id:** `str` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.value_receipts.<a href="src/paid/value_receipts/client.py">seal_value_receipt</a>(...) -&gt; AsyncHttpResponse[SuccessResponse]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Deprecated. A value receipt is immutable from creation, so it is already sealed and this call has nothing to do. It answers 200 for any receipt the organization owns.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.value_receipts.seal_value_receipt(
    id="id",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**id:** `str` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.value_receipts.<a href="src/paid/value_receipts/client.py">archive_value_receipt</a>(...) -&gt; AsyncHttpResponse[SuccessResponse]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Soft-archive a value receipt. Archived VRs are hidden from list by default.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.value_receipts.archive_value_receipt(
    id="id",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**id:** `str` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.value_receipts.<a href="src/paid/value_receipts/client.py">unarchive_value_receipt</a>(...) -&gt; AsyncHttpResponse[SuccessResponse]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Restore an archived value receipt. A receipt with no content, or one that failed to build, stays archived and answers 409. So does one whose period another value receipt now covers.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.value_receipts.unarchive_value_receipt(
    id="id",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**id:** `str` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.value_receipts.<a href="src/paid/value_receipts/client.py">publish_value_receipt</a>(...) -&gt; AsyncHttpResponse[ValueReceiptDetail]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Make a value receipt publicly accessible via URL. An archived receipt is rejected with 409 — unarchive it first. So is one with no content, which has nothing to show.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.value_receipts.publish_value_receipt(
    id="id",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**id:** `str` 
    
</dd>
</dl>

<dl>
<dd>

**publish_expires_at:** `typing.Optional[dt.datetime]` — When the public link stops working. Must be in the future, or the request fails with 409. Omit or set null for a link that does not expire.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.value_receipts.<a href="src/paid/value_receipts/client.py">unpublish_value_receipt</a>(...) -&gt; AsyncHttpResponse[ValueReceiptDetail]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Revoke public access to a value receipt. Available for archived receipts too, so a live link can always be revoked.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.value_receipts.unpublish_value_receipt(
    id="id",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**id:** `str` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

## DeliveredValue
<details><summary><code>client.delivered_value.<a href="src/paid/delivered_value/client.py">get_delivered_value</a>(...) -&gt; AsyncHttpResponse[DeliveredValueResponse]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Returns the value delivered to a customer over a period, computed live from your active value model on every call — per value metric: its definition, a display-ready calculation, the measured unit total, monetary total, and signal counts, plus a per-event signal breakdown. The same data a value receipt presents, without creating one — built for value dashboards and reporting. Figures always reflect the currently active value model. Shortly after you publish a model change, totals can still be priced under the previous model's rates.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.delivered_value.get_delivered_value(
    external_customer_id="acme-corp",
    start_date="2026-08-01",
    end_date="2026-08-31",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**start_date:** `str` — Period start, inclusive. Accepts an ISO 8601 date or date-time. Date-only values (e.g. 2026-06-30) are treated as UTC; date-times without an explicit timezone offset are ambiguous, so include one (e.g. 2026-06-30T00:00:00-05:00) when precision matters.
    
</dd>
</dl>

<dl>
<dd>

**end_date:** `str` — Period end, inclusive — a date-only value means through the end of that UTC day. Accepts an ISO 8601 date or date-time. Date-only values (e.g. 2026-06-30) are treated as UTC; date-times without an explicit timezone offset are ambiguous, so include one (e.g. 2026-06-30T00:00:00-05:00) when precision matters.
    
</dd>
</dl>

<dl>
<dd>

**customer_id:** `typing.Optional[str]` — The customer to compute delivered value for. Accepts a Paid customer ID or, as a fallback, your external customer ID. Mutually exclusive with externalCustomerId; exactly one is required.
    
</dd>
</dl>

<dl>
<dd>

**external_customer_id:** `typing.Optional[str]` — Your external customer ID. Mutually exclusive with customerId; exactly one is required.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.delivered_value.<a href="src/paid/delivered_value/client.py">get_delivered_value_summary</a>(...) -&gt; AsyncHttpResponse[DeliveredValueSummaryResponse]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Aggregates delivered value into period buckets for charting value over time, grouped by value metric, by a signal-payload key (for example, value delivered per team_id), or by the formula condition band that priced each row (for example, value delivered per workflow complexity). Any grouping can be narrowed to a single value metric, which is also what lets a dimension grouping state a unit total. Reads the same rows as GET /delivered-value, so summing every bucket's monetaryTotal matches that endpoint's total for the same customer and window. Figures always reflect the currently active value model. Shortly after you publish a model change, totals can still be priced under the previous model's rates. Grouping by band reads a figure recorded when the row was priced, so value delivered before this grouping existed reports under the "none" band until a value-model backfill re-derives it.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.delivered_value.get_delivered_value_summary(
    external_customer_id="acme-corp",
    start_date="2026-08-01",
    end_date="2026-08-31",
    group_by="valueMetric",
    interval="month",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**start_date:** `str` — Window start, inclusive. Accepts an ISO 8601 date or date-time. Date-only values (e.g. 2026-06-30) are treated as UTC; date-times without an explicit timezone offset are ambiguous, so include one (e.g. 2026-06-30T00:00:00-05:00) when precision matters.
    
</dd>
</dl>

<dl>
<dd>

**end_date:** `str` — Window end, inclusive — a date-only value means through the end of that UTC day. Accepts an ISO 8601 date or date-time. Date-only values (e.g. 2026-06-30) are treated as UTC; date-times without an explicit timezone offset are ambiguous, so include one (e.g. 2026-06-30T00:00:00-05:00) when precision matters.
    
</dd>
</dl>

<dl>
<dd>

**group_by:** `GetDeliveredValueSummaryRequestGroupBy` — How to group the buckets. valueMetric groups by the value metric that delivered the money. dimension groups by a signal-payload key named in dimensionKey, summed across metrics unless valueMetric narrows it to one — for example, value delivered per team when your signals carry a team_id field. formulaCondition cuts each metric by the condition band that priced it, for a value model that prices in graduated bands — for example, value delivered per workflow complexity when the metric bands runs by step count. A band the window never hit still reports a zero in every period, so the series keeps its shape. signal groups by the source event, one row per period per signal the value model binds to a metric, summed across the metrics the signal feeds unless valueMetric narrows it to one — the per-event breakdown GET /delivered-value states for a whole window, cut into periods. A signal the window never fired still reports a zero in every period.
    
</dd>
</dl>

<dl>
<dd>

**interval:** `GetDeliveredValueSummaryRequestInterval` — Period bucket size. Buckets align to UTC hours, calendar days, ISO weeks (Monday start), calendar months or calendar quarters; the first and last bucket's published bounds are clipped to the requested window. The window cannot hold more than 400 buckets.
    
</dd>
</dl>

<dl>
<dd>

**customer_id:** `typing.Optional[str]` — The customer to summarize delivered value for. Accepts a Paid customer ID or, as a fallback, your external customer ID. Mutually exclusive with externalCustomerId; exactly one is required.
    
</dd>
</dl>

<dl>
<dd>

**external_customer_id:** `typing.Optional[str]` — Your external customer ID. Mutually exclusive with customerId; exactly one is required.
    
</dd>
</dl>

<dl>
<dd>

**dimension_key:** `typing.Optional[str]` — The signal-payload key to group by (up to 64 letters, digits, or underscores). Required when groupBy is dimension; rejected otherwise. A signal whose payload lacks the key, or holds it empty, groups under an empty groupKey.
    
</dd>
</dl>

<dl>
<dd>

**value_metric:** `typing.Optional[str]` — Narrow the summary to a single value metric, by slug. Every grouping accepts it: it restricts which rows are read, so the totals are that metric's alone. Under groupBy=dimension and groupBy=signal it is also what makes unitTotal stateable — one metric is one unit, where a sum across metrics is in none. Under groupBy=signal it also keeps only the signals bound to that metric as groups. An archived metric may be named; a slug the value model has never defined is rejected.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

## Webhooks
<details><summary><code>client.webhooks.<a href="src/paid/webhooks/client.py">list_webhooks</a>() -&gt; AsyncHttpResponse[WebhookListResponse]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

List customer-facing billing webhooks for the authenticated organization, along with whether the organization has generated a signing secret.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.webhooks.list_webhooks()

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.webhooks.<a href="src/paid/webhooks/client.py">update_webhook</a>(...) -&gt; AsyncHttpResponse[WebhookUpdateResponse]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Enable or disable a webhook and configure the destination URL for the authenticated organization.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.webhooks.update_webhook(
    webhook_name="billing-invoice-created",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**webhook_name:** `UpdateWebhookRequestWebhookName` 
    
</dd>
</dl>

<dl>
<dd>

**enabled:** `typing.Optional[bool]` — Whether the webhook is enabled for delivery.
    
</dd>
</dl>

<dl>
<dd>

**url:** `typing.Optional[str]` — The HTTPS endpoint Paid should deliver this webhook to. It must use HTTPS and a public host: localhost, private networks (10.x, 172.16-31.x, 192.168.x), link-local addresses and .local names are rejected. Set to null to clear it while disabled.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.webhooks.<a href="src/paid/webhooks/client.py">test_webhook</a>(...) -&gt; AsyncHttpResponse[WebhookTestResponse]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Send a synthetic webhook delivery to the configured destination for this webhook.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.webhooks.test_webhook(
    webhook_name="billing-invoice-created",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**webhook_name:** `TestWebhookRequestWebhookName` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.webhooks.<a href="src/paid/webhooks/client.py">rotate_webhook_secret</a>() -&gt; AsyncHttpResponse[RotateWebhookSecretResponse]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Generate a new HMAC signing secret used by every webhook in this organization and return it exactly once. The previous secret is invalidated immediately on next delivery.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.webhooks.rotate_webhook_secret()

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

## Pricing
<details><summary><code>client.pricing.<a href="src/paid/pricing/client.py">list_pricing</a>(...) -&gt; AsyncHttpResponse[PricingListResponse]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Returns pricing for all product attributes of a product. Each entry includes the attribute's pricing configuration and credit benefits.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.pricing.list_pricing(
    product_id="productId",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**product_id:** `str` — Product display ID or UUID
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.pricing.<a href="src/paid/pricing/client.py">get_pricing</a>(...) -&gt; AsyncHttpResponse[PricingResponse]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Returns pricing and credit benefits for a single product attribute.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.pricing.get_pricing(
    product_attribute_id="productAttributeId",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**product_attribute_id:** `str` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.pricing.<a href="src/paid/pricing/client.py">update_pricing</a>(...) -&gt; AsyncHttpResponse[PricingResponse]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Updates pricing on an existing product attribute. To create a new attribute, use the update product endpoint (updateProductById), which upserts productAttributes. If creditBenefits is provided, it fully replaces existing benefits. If omitted, existing benefits are preserved.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid, PricingInput_RecurringPerUnit, SimplePricePoint

client = Paid(
    token="YOUR_TOKEN",
)
client.pricing.update_pricing(
    product_attribute_id="productAttributeId",
    pricing=PricingInput_RecurringPerUnit(
        billing_frequency="Monthly",
        price_points=[
            SimplePricePoint(
                currency="currency",
                unit_price=1,
            )
        ],
    ),
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**product_attribute_id:** `str` 
    
</dd>
</dl>

<dl>
<dd>

**pricing:** `PricingInput` 
    
</dd>
</dl>

<dl>
<dd>

**credit_benefits:** `typing.Optional[typing.Sequence[CreditBenefitInput]]` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

## Costs
<details><summary><code>client.costs.<a href="src/paid/costs/client.py">create_costs</a>(...) -&gt; AsyncHttpResponse[CostIngestResponse]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Ingests a batch of cost records. Each record is either a pre-computed `cost` (caller supplies amount + currency) or a `usage` record (caller supplies vendor/model/token counts and Paid prices it server-side). The batch is all-or-nothing: if any record fails validation, the entire request is rejected with a 400 and nothing is persisted.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Cost_Cost, CustomerById, Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.costs.create_costs(
    costs=[
        Cost_Cost(
            customer=CustomerById(
                customer_id="customerId",
            ),
            amount=1.1,
            currency="currency",
        )
    ],
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**costs:** `typing.Sequence[Cost]` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

## Analytics
<details><summary><code>client.analytics.<a href="src/paid/analytics/client.py">execute_analytics_query</a>(...) -&gt; AsyncHttpResponse[AnalyticsQueryResponse]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Runs a single ClickHouse SELECT (or WITH … SELECT) against your organization's analytics views. Before writing a query, call `getAnalyticsSchema` (GET /schema) for the available views and columns, and `getSignalsMetadata` (GET /signals/metadata) for the JSON paths inside `fact_signal.data`. Results are automatically scoped to your organization — no org filter is needed or possible. Only SELECT/WITH statements are accepted.

Conventions: monetary amounts are minor units (cents — divide by 100 for the major unit); most are integers, but `fact_cost.cost_amount` is fractional cents (Decimal) since a single AI call usually costs less than a cent; 64-bit integers (counts, ids, amounts) are returned as JSON strings to preserve precision, so parse them client-side; Decimal columns (fractional cents, and credit amounts, which are counts of credits rather than cents and are never divided by 100) come back as JSON numbers instead, so a value beyond 2^53 is already rounded — select toString(col) when you need its exact digits. Query signal payloads via JSON paths, e.g. `SELECT data.country::String AS country, count() FROM fact_signal GROUP BY country`.

Limits: 30 seconds of execution time and 10,000 result rows (truncation is flagged via `meta.truncated`). Queries exceeding the 4 GB memory allowance return HTTP 422 with code `QUERY_MEMORY_LIMIT_EXCEEDED`. Prefer aggregates and a `created_at` date filter on large tables — this endpoint is for interactive analytics, not bulk export.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.analytics.execute_analytics_query(
    query="SELECT signal_name, count() AS signals FROM fact_signal WHERE created_at > now() - INTERVAL 30 DAY GROUP BY signal_name ORDER BY signals DESC",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**query:** `str` — A single ClickHouse SELECT (or WITH ... SELECT) statement against the analytics views. Results are automatically scoped to your organization. See GET /schema for the available views and columns.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.analytics.<a href="src/paid/analytics/client.py">get_analytics_schema</a>() -&gt; AsyncHttpResponse[AnalyticsSchemaResponse]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Returns the analytics views available to POST /query, with column names, ClickHouse types, and descriptions. Dimensions (`dim_*`) describe entities; facts (`fact_*`) are event/transaction tables that join to dimensions via the `*_id` columns described in each comment.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.analytics.get_analytics_schema()

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.analytics.<a href="src/paid/analytics/client.py">get_signals_metadata</a>(...) -&gt; AsyncHttpResponse[SignalsMetadataResponse]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Lists the JSON paths (and their observed types) present in the `data` payload of your signals within a time window (default: last 30 days), grouped by signal name. Use the returned paths in queries against `fact_signal`, e.g. `WHERE data.<path>::String = '...'`. Keys declared as grouping dimensions are marked `declared` with their `slot` and listed first. Where available, each path also carries `firstSeen`, `lastSeen` and a `signals` count; these span all time, and the window only selects which paths and types are returned.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.analytics.get_signals_metadata()

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**signal_name:** `typing.Optional[str]` — Restrict discovery to a single signal name.
    
</dd>
</dl>

<dl>
<dd>

**from_date:** `typing.Optional[dt.datetime]` — Start of the discovery window. Defaults to 30 days ago.
    
</dd>
</dl>

<dl>
<dd>

**to_date:** `typing.Optional[dt.datetime]` — End of the discovery window. Defaults to now.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

## CustomViewsExperimental
<details><summary><code>client.custom_views_experimental.<a href="src/paid/custom_views_experimental/client.py">list_custom_views</a>() -&gt; AsyncHttpResponse[CustomViewListResponse]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

⚠️ **Experimental** — this endpoint may change or be removed without notice and is not subject to v2 backwards-compatibility guarantees. Do not build production-critical integrations against it yet.

Lists the organization's custom views (newest first) with lightweight summary info — name, status, query count, default date range, created date. Does not return the SQL or render bundle; fetch a single view via getCustomView for those.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.custom_views_experimental.list_custom_views()

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.custom_views_experimental.<a href="src/paid/custom_views_experimental/client.py">create_custom_view</a>(...) -&gt; AsyncHttpResponse[CustomView]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

⚠️ **Experimental** — this endpoint may change or be removed without notice and is not subject to v2 backwards-compatibility guarantees. Do not build production-critical integrations against it yet.

⚠️ Only call this when the user has EXPLICITLY asked to save or create the view. After generating or previewing a dashboard, do NOT automatically save it — show it to the user and wait for them to ask you to save it. A customer-scoped view is created as a DRAFT — creating it is NOT permission to publish; never chain a publish onto a create. An organization-scoped view is created already PUBLISHED instead: it has no draft state and no publish step at all (never call publishCustomView on one — it's a no-op, and unpublishView refuses it outright). After saving, hand the user the previewUrl and wait for their feedback before doing anything else. Saves named analytics queries + a self-contained HTML render bundle. **Call getCustomViewAuthoringGuide (GET /experimental/views/authoring-guide) first** — it returns the full guide and a copy-paste interactive template. Key rules: (1) Do NOT add a customer filter to the SQL — the database scopes every query to the viewing customer at embed time. (2) Each query's SQL must be SELECT-only; return clearly-named columns. Compute metric VALUES in SQL (e.g. (count()*2)/5 AS custom_metric) — derive a number in the render bundle only when it depends on user interaction (toggle/filter/hover) or is pure formatting of a value a query already returns. (3) The render bundle must be SELF-CONTAINED — inline all CSS/JS/charting, NO external loads or fetch (the sandbox has connect-src 'none'); it must listen for the `paid:data` message (data keyed by query id) and re-render on each one. (4) Make it INTERACTIVE — mousemove hover tooltips and at least one addEventListener-wired control that re-renders (a static chart feels broken). (5) The render bundle is the single source of truth — BEFORE saving, preview the EXACT bundle in the user's current client (call getCustomViewPreviewHarness with your bundle + sample data and render the HTML it returns) and show it to the user; that preview in the current client is how the user first sees the dashboard. Do NOT save a draft just to preview it in Paid — creating writes to the user's real account and is never a preview step. Do NOT build a separate chart, and only show numbers that come from a declared query. (6) A view is a FULL dashboard — include as many charts/KPIs as the analysis has. Keep every element derived from the single viewing customer (KPIs, trends, type mix); drop only cross-customer comparisons (rankings, share-of-total, 'N customers'). Don't simplify to one chart. (7) To make the date range adjustable (e.g. the user says 'last month'), write the date boundary as `{period_start:DateTime}` / `{period_end:DateTime}` placeholders in the SQL and pass a default `period` (relative like {kind:'relative',unit:'month',amount:1}, or absolute start/end). The org user can then change it in Paid without re-authoring. A query using the placeholders REQUIRES a period. Do NOT add your own date-range picker to the render bundle — Paid owns the timeframe and the bundle receives already-filtered data; a second in-bundle picker cannot re-run the SQL. (8) Check your draft with validateCustomView (POST /experimental/views/validate) BEFORE asking the user to save — it runs these same gates without persisting and reports every problem at once. The response returns a `previewUrl` — give it to the user so they can open the new view in Paid.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import CustomViewQuery, Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.custom_views_experimental.create_custom_view(
    name="Usage over time",
    queries=[
        CustomViewQuery(
            id="usage",
            sql="SELECT toDate(created_at) AS day, count() AS signals FROM fact_signal GROUP BY day ORDER BY day",
        )
    ],
    render_bundle="<!doctype html><body>…</body>",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**name:** `str` — Human-readable view name (shown in preview + audit log).
    
</dd>
</dl>

<dl>
<dd>

**queries:** `typing.Sequence[CustomViewQuery]` — One or more named queries. Each becomes a separately-keyed result set in the embed.
    
</dd>
</dl>

<dl>
<dd>

**render_bundle:** `str` — Self-contained HTML document that renders the result sets. Inline ALL CSS/JS/charting libraries — the render sandbox has no network access. It receives the keyed result sets via a `message` event (`data[<queryId>]`) and must not fetch anything itself.
    
</dd>
</dl>

<dl>
<dd>

**hierarchy_enabled:** `typing.Optional[bool]` — Enable customer unit navigation for a customer-scoped view. Setting true requires customer hierarchies to be enabled for your organization, and every query must support unit selection. Existing views default to false.
    
</dd>
</dl>

<dl>
<dd>

**caps_enabled:** `typing.Optional[bool]` — Send the customer's unit caps to the render bundle as `context.caps`. Needs `hierarchyEnabled` on a customer-scoped view. Turn it on only for views that draw cap usage: each load then reads cap usage from billing. Caps are spend policy, so an API key also needs `read:billing` to receive them in the data (they are left out otherwise) or to mint this view's embed token. Defaults to false.
    
</dd>
</dl>

<dl>
<dd>

**description:** `typing.Optional[str]` — Optional longer description of what the view shows.
    
</dd>
</dl>

<dl>
<dd>

**period:** `typing.Optional[CreateCustomViewRequestPeriod]` — Optional default date range. Required if any query uses the `{period_start:DateTime}` / `{period_end:DateTime}` placeholders. Can be changed later in Paid without re-authoring.
    
</dd>
</dl>

<dl>
<dd>

**filters:** `typing.Optional[typing.Sequence[CustomViewFilter]]` — Optional per-request filter parameters. Required for every {filter_<name>:String} placeholder the queries reference.
    
</dd>
</dl>

<dl>
<dd>

**scope:** `typing.Optional[CreateCustomViewRequestScope]` — 'customer' (default): data is scoped to one viewing customer and the view is embeddable per-customer. 'organization': data is org-wide; the view is internal-only (visible to org members in Paid, never embeddable).
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.custom_views_experimental.<a href="src/paid/custom_views_experimental/client.py">publish_custom_view</a>(...) -&gt; AsyncHttpResponse[CustomView]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

⚠️ **Experimental** — this endpoint may change or be removed without notice and is not subject to v2 backwards-compatibility guarantees. Do not build production-critical integrations against it yet.

⚠️ Never publish as an automatic follow-up to creating or generating a view. Only call this after you have shown the user the built/previewed view and they have EXPLICITLY approved publishing — building and publishing are separate user decisions, and answering an earlier question (e.g. the view's scope) is NOT publish approval. Flips the view from DRAFT to PUBLISHED. Only PUBLISHED views are served on the embed data path — this is the gate that stops an unreviewed view reaching end-customers. Idempotent: publishing an already-published view is a no-op success. The response returns a `previewUrl` — give it to the user so they can open the view in Paid.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.custom_views_experimental.publish_custom_view(
    display_id="displayId",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**display_id:** `str` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.custom_views_experimental.<a href="src/paid/custom_views_experimental/client.py">update_custom_view_period</a>(...) -&gt; AsyncHttpResponse[CustomView]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

⚠️ **Experimental** — this endpoint may change or be removed without notice and is not subject to v2 backwards-compatibility guarantees. Do not build production-critical integrations against it yet.

Updates the view's default date range (the period applied to queries that use the `{period_start:DateTime}` / `{period_end:DateTime}` placeholders). Accepts a relative rolling window (e.g. last 30 days) or a fixed start/end range. Lets the period be changed after deployment without re-authoring the SQL. Applies to DRAFT or PUBLISHED views.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid
from paid.custom_views_experimental import UpdateCustomViewPeriodRequestPeriod

client = Paid(
    token="YOUR_TOKEN",
)
client.custom_views_experimental.update_custom_view_period(
    display_id="displayId",
    period=UpdateCustomViewPeriodRequestPeriod(
        kind="relative",
    ),
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**display_id:** `str` 
    
</dd>
</dl>

<dl>
<dd>

**period:** `UpdateCustomViewPeriodRequestPeriod` — Default date range for the view's queries. Use `{period_start:DateTime}` / `{period_end:DateTime}` placeholders in your SQL to make the range adjustable. Either relative (unit+amount) or absolute (start+end).
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.custom_views_experimental.<a href="src/paid/custom_views_experimental/client.py">get_custom_view</a>(...) -&gt; AsyncHttpResponse[CustomViewDetail]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

⚠️ **Experimental** — this endpoint may change or be removed without notice and is not subject to v2 backwards-compatibility guarantees. Do not build production-critical integrations against it yet.

Returns the view's name, status, query ids, and the author render bundle for the owning organization. Only PUBLISHED views are returned; a DRAFT returns 404 (use getCustomViewAuthoring to read a DRAFT). Used by the trusted preview/embed frame to render the sandbox; the per-customer data is fetched separately via /:displayId/data.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.custom_views_experimental.get_custom_view(
    display_id="displayId",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**display_id:** `str` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.custom_views_experimental.<a href="src/paid/custom_views_experimental/client.py">update_custom_view</a>(...) -&gt; AsyncHttpResponse[CustomView]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

⚠️ **Experimental** — this endpoint may change or be removed without notice and is not subject to v2 backwards-compatibility guarantees. Do not build production-critical integrations against it yet.

Partially updates a view. Omitted fields are left unchanged. For REVISIONS, prefer the incremental fields — `bundleEdits` (exact search-and-replace on the stored render bundle) and `queryUpserts`/`queryRemovals` (per-query changes) — so you transmit only what changed instead of re-sending the whole payload. The full-replacement fields remain for rewrites: `renderBundle`, and `queries` (a FULL replacement of the query list — never drop queries the user didn't ask to remove). Replacement and incremental forms of the same aspect cannot be combined. The resulting SQL and bundle pass the same validation as createCustomView (SELECT-only, size cap, self-contained, paid:data listener). Works on DRAFT or PUBLISHED views — published embeds pick the change up on their next load. If the queries or hierarchy setting change during validation, returns 409 CUSTOM_VIEW_UPDATE_CONFLICT; fetch the current authoring definition and reapply your changes before retrying.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.custom_views_experimental.update_custom_view(
    display_id="displayId",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**display_id:** `str` 
    
</dd>
</dl>

<dl>
<dd>

**hierarchy_enabled:** `typing.Optional[bool]` — Enable customer unit navigation for a customer-scoped view. Setting true requires customer hierarchies to be enabled for your organization, and every query must support unit selection. Existing views default to false.
    
</dd>
</dl>

<dl>
<dd>

**caps_enabled:** `typing.Optional[bool]` — Send the customer's unit caps to the render bundle as `context.caps`. Needs `hierarchyEnabled` on a customer-scoped view. Turn it on only for views that draw cap usage: each load then reads cap usage from billing. Caps are spend policy, so an API key also needs `read:billing` to receive them in the data (they are left out otherwise) or to mint this view's embed token. Defaults to false.
    
</dd>
</dl>

<dl>
<dd>

**name:** `typing.Optional[str]` — New human-readable view name.
    
</dd>
</dl>

<dl>
<dd>

**description:** `typing.Optional[str]` — New description; pass null to clear it.
    
</dd>
</dl>

<dl>
<dd>

**queries:** `typing.Optional[typing.Sequence[CustomViewQuery]]` — Full replacement of the view's query list. Each SQL is re-validated (SELECT-only) exactly like createCustomView. For changing one or two queries, prefer `queryUpserts`/`queryRemovals` instead. Cannot be combined with them.
    
</dd>
</dl>

<dl>
<dd>

**render_bundle:** `typing.Optional[str]` — Replacement render bundle. Re-validated (size cap, self-contained, paid:data listener) exactly like createCustomView. For small changes, prefer `bundleEdits` instead. Cannot be combined with `bundleEdits`.
    
</dd>
</dl>

<dl>
<dd>

**bundle_edits:** `typing.Optional[typing.Sequence[RenderBundleEdit]]` — PREFERRED for revisions: exact search-and-replace edits applied in order to the stored render bundle, so you send only the changed text instead of re-transmitting the whole bundle. The edited result passes the same validation as a full replacement. Cannot be combined with `renderBundle`.
    
</dd>
</dl>

<dl>
<dd>

**query_upserts:** `typing.Optional[typing.Sequence[CustomViewQuery]]` — PREFERRED for revisions: per-query changes — each entry replaces the stored query with the same id, or is appended as a new query. Queries not mentioned are left unchanged. Cannot be combined with `queries`.
    
</dd>
</dl>

<dl>
<dd>

**query_removals:** `typing.Optional[typing.Sequence[str]]` — Ids of stored queries to remove (applied before `queryUpserts`). Rejected if an id does not exist. Cannot be combined with `queries`.
    
</dd>
</dl>

<dl>
<dd>

**filters:** `typing.Optional[typing.Sequence[CustomViewFilter]]` — Full replacement of the view's declared filter parameters; pass null to remove them all. Every {filter_<name>:String} placeholder the (resulting) queries reference must stay declared.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.custom_views_experimental.<a href="src/paid/custom_views_experimental/client.py">get_custom_view_customer_units</a>(...) -&gt; AsyncHttpResponse[ViewCustomerUnitsResponse]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

⚠️ **Experimental** — this endpoint may change or be removed without notice and is not subject to v2 backwards-compatibility guarantees. Do not build production-critical integrations against it yet.

Lists active customer units for a published customer-scoped view with hierarchyEnabled set to true. Hierarchy eligibility is independent of the requested page or usage; requires credit hierarchies to be enabled for the organization.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.custom_views_experimental.get_custom_view_customer_units(
    display_id="displayId",
    customer_id="cus_acme",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**display_id:** `str` 
    
</dd>
</dl>

<dl>
<dd>

**customer_id:** `str` — The customer's external ID, Paid customer ID, or internal ID.
    
</dd>
</dl>

<dl>
<dd>

**limit:** `typing.Optional[int]` 
    
</dd>
</dl>

<dl>
<dd>

**offset:** `typing.Optional[int]` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.custom_views_experimental.<a href="src/paid/custom_views_experimental/client.py">get_custom_view_data</a>(...) -&gt; AsyncHttpResponse[ViewDataResponse]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

⚠️ **Experimental** — this endpoint may change or be removed without notice and is not subject to v2 backwards-compatibility guarantees. Do not build production-critical integrations against it yet.

Runs every stored query of the view on the read-only analytics database and returns the result sets keyed by query id. For a customer-scoped view (the default), the query is scoped to the caller's organization AND the given `customerId` (both enforced as ClickHouse row filters) — `customerId` is required. For an organization-scoped view, the data is org-wide (scoped only to the caller's organization) and `customerId` is ignored. The scope is enforced by the database — it cannot be widened by the stored SQL. If the view declares filters, pass per-request values as `filter_<name>` query parameters (e.g. `filter_region=eu`); undeclared names or disallowed values are rejected with 400. Filters narrow data within the scope — never widen it. Only PUBLISHED views are served by default; a DRAFT returns 404. To check a saved DRAFT's real data before publishing, pass `includeDraft=true`. Embeds and share links never serve a DRAFT.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.custom_views_experimental.get_custom_view_data(
    display_id="displayId",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**display_id:** `str` 
    
</dd>
</dl>

<dl>
<dd>

**customer_id:** `typing.Optional[str]` — Customer to scope the data to (dev/preview only; the embed derives this from the verified token). Required for customer-scoped views; ignored for organization-scoped views (their data is org-wide).
    
</dd>
</dl>

<dl>
<dd>

**period_kind:** `typing.Optional[GetCustomViewDataRequestPeriodKind]` — Override the view's default date range for this request only. 'relative' = rolling window (set periodUnit + periodAmount); 'absolute' = fixed range (set periodStart + periodEnd).
    
</dd>
</dl>

<dl>
<dd>

**period_unit:** `typing.Optional[GetCustomViewDataRequestPeriodUnit]` — relative override only: unit of the rolling window.
    
</dd>
</dl>

<dl>
<dd>

**period_amount:** `typing.Optional[int]` — relative override only: how many units back from today.
    
</dd>
</dl>

<dl>
<dd>

**period_start:** `typing.Optional[str]` — absolute override only: inclusive start date (YYYY-MM-DD).
    
</dd>
</dl>

<dl>
<dd>

**period_end:** `typing.Optional[str]` — absolute override only: inclusive end date (YYYY-MM-DD).
    
</dd>
</dl>

<dl>
<dd>

**include_draft:** `typing.Optional[GetCustomViewDataRequestIncludeDraft]` — Pass 'true' to also serve a DRAFT view, so you can check its real data before publishing. Without it a DRAFT returns 404. End-customer embeds and share links never serve a DRAFT.
    
</dd>
</dl>

<dl>
<dd>

**unit_mode:** `typing.Optional[GetCustomViewDataRequestUnitMode]` — Unit selection for this request. Omit both unit parameters for all customer activity. Shared links instead use their saved starting selection when omitted; pass all to reset a shared link.
    
</dd>
</dl>

<dl>
<dd>

**unit_external_id:** `typing.Optional[str]` — The unit's external ID. Required with direct or subtree, and omitted with all.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.custom_views_experimental.<a href="src/paid/custom_views_experimental/client.py">get_custom_view_embed_token</a>(...) -&gt; AsyncHttpResponse[CustomViewEmbedTokenResponse]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

⚠️ **Experimental** — this endpoint may change or be removed without notice and is not subject to v2 backwards-compatibility guarantees. Do not build production-critical integrations against it yet.

Mints a short-lived, customer-scoped token for embedding a published custom view. Call this from your server with your API key, then pass the returned token to the embed SDK. Organization-scoped views cannot be embedded per-customer — this returns a 400 (`ORG_SCOPED_VIEW_NOT_EMBEDDABLE`) for one.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.custom_views_experimental.get_custom_view_embed_token(
    display_id="displayId",
    customer_id="customerId",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**display_id:** `str` 
    
</dd>
</dl>

<dl>
<dd>

**customer_id:** `str` — The customer to scope the view to: external id, Paid display id (cus_…), or internal id.
    
</dd>
</dl>

<dl>
<dd>

**ttl_seconds:** `typing.Optional[int]` — Token lifetime in seconds (capped at 3600).
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

## ValueModels
<details><summary><code>client.value_models.<a href="src/paid/value_models/client.py">get_current_value_model</a>() -&gt; AsyncHttpResponse[ValueModelDetail]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Returns the current (latest active) value model for the organization.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.value_models.get_current_value_model()

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.value_models.<a href="src/paid/value_models/client.py">update_value_model</a>(...) -&gt; AsyncHttpResponse[ValueModelDetail]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Uploads a new value model version. Validates the content, creates a new version, archives the previous active version, and syncs to ClickHouse. This updates how value is calculated going forward — new signals are valued under the new model. It does NOT re-rate historical signals; run POST /api/v2/value-models/backfill to re-rate history as a separate, explicit action. Any unpublished draft saved in the Paid app is discarded. Send expectedActiveVersion to get a 409 instead of overwriting a version published since you read it.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
import datetime

from paid import (
    Paid,
    ValueModelContent,
    ValueModelContentFormulasItem,
    ValueModelContentFormulasItemVariablesItem,
    ValueModelContentSignalsItem,
    ValueModelContentValueTypesItem,
    ValueModelContentValueTypesItemCalculationTimelineItem,
    ValueModelContentValueTypesItemCalculationTimelineItemCalculation,
    ValueModelContentValueTypesItemCalculationTimelineItemCalculationUnitZero,
)

client = Paid(
    token="YOUR_TOKEN",
)
client.value_models.update_value_model(
    content=ValueModelContent(
        currency="currency",
        value_types=[
            ValueModelContentValueTypesItem(
                slug="slug",
                name="name",
                calculation_timeline=[
                    ValueModelContentValueTypesItemCalculationTimelineItem(
                        effective_from=datetime.datetime.fromisoformat(
                            "2024-01-15 09:30:00+00:00",
                        ),
                        calculation=ValueModelContentValueTypesItemCalculationTimelineItemCalculation(
                            unit=ValueModelContentValueTypesItemCalculationTimelineItemCalculationUnitZero(
                                type="monetary",
                            ),
                            formula_ids=["formulaIds"],
                            signal_event_names=["signalEventNames"],
                            segment_table_ids=["segmentTableIds"],
                            override_ids=["overrideIds"],
                        ),
                    )
                ],
            )
        ],
        formulas=[
            ValueModelContentFormulasItem(
                id="id",
                value_type_slug="valueTypeSlug",
                label="label",
                variables=[
                    ValueModelContentFormulasItemVariablesItem(
                        id="id",
                        label="label",
                    )
                ],
                expression="expression",
            )
        ],
        signals=[
            ValueModelContentSignalsItem(
                event_name="eventName",
                label="label",
            )
        ],
    ),
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**content:** `ValueModelContent` 
    
</dd>
</dl>

<dl>
<dd>

**effective_from:** `typing.Optional[dt.datetime]` 
    
</dd>
</dl>

<dl>
<dd>

**effective_to:** `typing.Optional[dt.datetime]` 
    
</dd>
</dl>

<dl>
<dd>

**expected_active_version:** `typing.Optional[int]` — Optimistic concurrency guard: the active version this upload is based on. When supplied and it does not match the live active version, the upload fails with 409 instead of overwriting a concurrent change.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.value_models.<a href="src/paid/value_models/client.py">list_value_model_versions</a>(...) -&gt; AsyncHttpResponse[ValueModelVersionListResponse]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Returns all value model versions sorted by version descending. Does not include the full content.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.value_models.list_value_model_versions()

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**limit:** `typing.Optional[int]` 
    
</dd>
</dl>

<dl>
<dd>

**offset:** `typing.Optional[int]` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.value_models.<a href="src/paid/value_models/client.py">get_value_model_version</a>(...) -&gt; AsyncHttpResponse[ValueModelDetail]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Returns a specific historical value model version by version number, including full content.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.value_models.get_value_model_version(
    version=1,
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**version:** `int` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.value_models.<a href="src/paid/value_models/client.py">refresh_value_model_backfill</a>() -&gt; AsyncHttpResponse[BackfillResponse]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Manually triggers a recalculation of all ClickHouse rows against the current value model. Returns 422 when the backfill would scan more signals than the ceiling allows — that is deterministic, so retrying without narrowing the scope will not help.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.value_models.refresh_value_model_backfill()

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

## ValueMetrics
<details><summary><code>client.value_metrics.<a href="src/paid/value_metrics/client.py">list_value_metrics</a>(...) -&gt; AsyncHttpResponse[ValueMetricListResponse]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Returns the value metrics in the current value model, without their formulas. Archived metrics are hidden unless includeArchived is true.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.value_metrics.list_value_metrics()

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**limit:** `typing.Optional[int]` 
    
</dd>
</dl>

<dl>
<dd>

**offset:** `typing.Optional[int]` 
    
</dd>
</dl>

<dl>
<dd>

**include_archived:** `typing.Optional[bool]` — Whether to include archived metrics in the response. Defaults to false.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.value_metrics.<a href="src/paid/value_metrics/client.py">create_value_metric</a>(...) -&gt; AsyncHttpResponse[ValueMetricWriteAck]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Adds one value metric — its unit, formula, signal binding and optional monetary conversion — to the value model. The signal must already exist: an event name your organization has sent, or one referenced by usage pricing on an active product. Publishes a new value model version that values NEW signals going forward; it does NOT recalculate historical delivered value. Re-rating history is a separate, explicit action (POST /api/v2/value-models/backfill).
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import (
    Paid,
    ValueMetricFormula,
    ValueMetricFormulaVariable,
    ValueMetricSignalBinding,
    ValueMetricUnit,
)

client = Paid(
    token="YOUR_TOKEN",
)
client.value_metrics.create_value_metric(
    name="Time saved",
    unit=ValueMetricUnit(
        type="monetary",
    ),
    formula=ValueMetricFormula(
        expression="minutes_saved / 60",
        variables=[
            ValueMetricFormulaVariable(
                id="id",
                label="label",
            )
        ],
    ),
    signal=ValueMetricSignalBinding(
        event_name="ticket_resolved",
    ),
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**name:** `str` — Customer-facing metric name.
    
</dd>
</dl>

<dl>
<dd>

**unit:** `ValueMetricUnit` 
    
</dd>
</dl>

<dl>
<dd>

**formula:** `ValueMetricFormula` 
    
</dd>
</dl>

<dl>
<dd>

**signal:** `ValueMetricSignalBinding` 
    
</dd>
</dl>

<dl>
<dd>

**slug:** `typing.Optional[str]` — Stable identifier. Derived from the name when omitted.
    
</dd>
</dl>

<dl>
<dd>

**monetary_conversion:** `typing.Optional[ValueMetricMonetaryConversion]` 
    
</dd>
</dl>

<dl>
<dd>

**category:** `typing.Optional[CreateValueMetricRequestCategory]` — Classification: hve = human value equivalent, time = time saved, cost = cost savings, revenue = revenue generated, risk = risk avoided.
    
</dd>
</dl>

<dl>
<dd>

**description:** `typing.Optional[str]` — Short customer-facing copy shown on value receipts.
    
</dd>
</dl>

<dl>
<dd>

**long_description:** `typing.Optional[str]` 
    
</dd>
</dl>

<dl>
<dd>

**sources:** `typing.Optional[typing.Sequence[CreateValueMetricRequestSourcesItem]]` — Up to 3 customer-facing citations backing this metric.
    
</dd>
</dl>

<dl>
<dd>

**expected_active_version:** `typing.Optional[int]` — Optimistic concurrency guard. When supplied and it does not match the live active version, the request fails with 409 instead of overwriting a concurrent change.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.value_metrics.<a href="src/paid/value_metrics/client.py">get_value_metric</a>(...) -&gt; AsyncHttpResponse[ValueMetricDetail]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Returns one value metric with its formula, signal binding and monetary conversion joined together. Call getCurrentValueModel if you need the active version number to guard a follow-up write.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.value_metrics.get_value_metric(
    slug="slug",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**slug:** `str` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.value_metrics.<a href="src/paid/value_metrics/client.py">archive_value_metric</a>(...) -&gt; AsyncHttpResponse[ValueMetricWriteAck]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Marks a value metric archived so it stops appearing in listValueMetrics. Its formula and signal bindings are deliberately kept, so historical delivered value and sealed value receipts still resolve — which also means an archived metric's signals continue to be ingested and can still surface on value receipts. Removing it from receipts entirely requires deleting its signal bindings. Restore it with updateValueMetric and archivedAt null.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.value_metrics.archive_value_metric(
    slug="slug",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**slug:** `str` 
    
</dd>
</dl>

<dl>
<dd>

**expected_active_version:** `typing.Optional[int]` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.value_metrics.<a href="src/paid/value_metrics/client.py">update_value_metric</a>(...) -&gt; AsyncHttpResponse[ValueMetricWriteAck]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Changes one value metric. Omitted fields are left alone. You can change its name, category, unit, customer-facing copy, sources, monetary rate, archive state, and the value, label or display format of any variable its formula declares. The formula expression and the signal it is bound to cannot be changed — recreate the metric, or use the whole value-model upload. Publishes a new value model version.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.value_metrics.update_value_metric(
    slug="slug",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**slug:** `str` 
    
</dd>
</dl>

<dl>
<dd>

**name:** `typing.Optional[str]` 
    
</dd>
</dl>

<dl>
<dd>

**unit:** `typing.Optional[ValueMetricUnit]` 
    
</dd>
</dl>

<dl>
<dd>

**monetary_conversion:** `typing.Optional[ValueMetricMonetaryConversion]` 
    
</dd>
</dl>

<dl>
<dd>

**variables:** `typing.Optional[typing.Dict[str, ValueMetricVariableEdit]]` — Per-variable edits, keyed by the variable id the formula declares. Merged: a variable you do not name is untouched. Naming one the formula does not declare is an error rather than a no-op.
    
</dd>
</dl>

<dl>
<dd>

**archived_at:** `typing.Optional[dt.datetime]` — Set null to restore an archived metric.
    
</dd>
</dl>

<dl>
<dd>

**category:** `typing.Optional[UpdateValueMetricRequestCategory]` — Classification: hve = human value equivalent, time = time saved, cost = cost savings, revenue = revenue generated, risk = risk avoided.
    
</dd>
</dl>

<dl>
<dd>

**description:** `typing.Optional[str]` — Short customer-facing copy shown on value receipts.
    
</dd>
</dl>

<dl>
<dd>

**long_description:** `typing.Optional[str]` 
    
</dd>
</dl>

<dl>
<dd>

**sources:** `typing.Optional[typing.Sequence[UpdateValueMetricRequestSourcesItem]]` — Up to 3 customer-facing citations backing this metric.
    
</dd>
</dl>

<dl>
<dd>

**expected_active_version:** `typing.Optional[int]` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

## PaymentMethods
<details><summary><code>client.payment_methods.<a href="src/paid/payment_methods/client.py">list_payment_methods</a>(...) -&gt; AsyncHttpResponse[PaymentMethodListResponse]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Lists the payment methods saved for a customer
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.payment_methods.list_payment_methods(
    customer_id="cus_1234abcd",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**limit:** `typing.Optional[int]` 
    
</dd>
</dl>

<dl>
<dd>

**offset:** `typing.Optional[int]` 
    
</dd>
</dl>

<dl>
<dd>

**customer_id:** `typing.Optional[str]` — Filter by Paid customer ID.
    
</dd>
</dl>

<dl>
<dd>

**external_customer_id:** `typing.Optional[str]` — Filter by your external customer ID.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.payment_methods.<a href="src/paid/payment_methods/client.py">create_payment_method</a>(...) -&gt; AsyncHttpResponse[PaymentMethodSetup]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Starts attaching a payment method to a customer by exchanging a client-side confirmation token for a setup intent. Complete any additional authentication client-side using the returned client secret.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.payment_methods.create_payment_method(
    confirmation_token="ctoken_1NXWPnLkdIwHu7ix",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**confirmation_token:** `str` — Confirmation token generated client-side by the payment processor's elements (e.g. a Stripe ConfirmationToken ID).
    
</dd>
</dl>

<dl>
<dd>

**customer_id:** `typing.Optional[str]` — Paid customer ID to attach the payment method to.
    
</dd>
</dl>

<dl>
<dd>

**external_customer_id:** `typing.Optional[str]` — Your external customer ID to attach the payment method to.
    
</dd>
</dl>

<dl>
<dd>

**return_url:** `typing.Optional[str]` — URL the customer is redirected to after completing any additional authentication step (e.g. 3-D Secure).
    
</dd>
</dl>

<dl>
<dd>

**metadata:** `typing.Optional[typing.Dict[str, str]]` — Key-value metadata stored on the setup intent.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.payment_methods.<a href="src/paid/payment_methods/client.py">get_payment_method</a>(...) -&gt; AsyncHttpResponse[PaymentMethod]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Get a payment method by its ID
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.payment_methods.get_payment_method(
    id="id",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**id:** `str` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.payment_methods.<a href="src/paid/payment_methods/client.py">delete_payment_method</a>(...) -&gt; AsyncHttpResponse[EmptyResponse]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Detaches a payment method from the customer and removes it from the payment processor
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.payment_methods.delete_payment_method(
    id="id",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**id:** `str` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.payment_methods.<a href="src/paid/payment_methods/client.py">update_default_payment_method</a>(...) -&gt; AsyncHttpResponse[PaymentMethod]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Makes this payment method the customer's default for future charges
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.payment_methods.update_default_payment_method(
    id="id",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**id:** `str` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

## Payments
<details><summary><code>client.payments.<a href="src/paid/payments/client.py">list_payments</a>(...) -&gt; AsyncHttpResponse[PaymentListResponse]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Lists payments for your organization
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.payments.list_payments()

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**limit:** `typing.Optional[int]` 
    
</dd>
</dl>

<dl>
<dd>

**offset:** `typing.Optional[int]` 
    
</dd>
</dl>

<dl>
<dd>

**customer_id:** `typing.Optional[str]` — Filter by Paid customer ID.
    
</dd>
</dl>

<dl>
<dd>

**external_customer_id:** `typing.Optional[str]` — Filter by your external customer ID.
    
</dd>
</dl>

<dl>
<dd>

**status:** `typing.Optional[ListPaymentsRequestStatus]` — Filter by payment status.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.payments.<a href="src/paid/payments/client.py">create_payment</a>(...) -&gt; AsyncHttpResponse[Payment]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Records a payment received from a customer, e.g. a bank transfer or check collected outside Paid. Allocate it to invoice lines with the payment allocations endpoints.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.payments.create_payment(
    amount=15000,
    currency="USD",
    payment_type="creditCard",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**amount:** `int` — Payment amount in cents (minor currency units).
    
</dd>
</dl>

<dl>
<dd>

**currency:** `str` — Three-letter ISO 4217 currency code. Lowercase is accepted and stored upper-cased.
    
</dd>
</dl>

<dl>
<dd>

**payment_type:** `PaymentType` 
    
</dd>
</dl>

<dl>
<dd>

**customer_id:** `typing.Optional[str]` — Paid customer ID the payment belongs to.
    
</dd>
</dl>

<dl>
<dd>

**external_customer_id:** `typing.Optional[str]` — Your external customer ID the payment belongs to.
    
</dd>
</dl>

<dl>
<dd>

**payment_date:** `typing.Optional[dt.datetime]` — When the payment was made (ISO 8601). Defaults to the current time.
    
</dd>
</dl>

<dl>
<dd>

**status:** `typing.Optional[PaymentCreateStatus]` 
    
</dd>
</dl>

<dl>
<dd>

**metadata:** `typing.Optional[typing.Dict[str, typing.Any]]` — Key-value metadata stored on the payment.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.payments.<a href="src/paid/payments/client.py">get_payment</a>(...) -&gt; AsyncHttpResponse[Payment]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Get a payment by its ID
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.payments.get_payment(
    id="id",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**id:** `str` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

## PaymentAllocations
<details><summary><code>client.payment_allocations.<a href="src/paid/payment_allocations/client.py">list_payment_allocations</a>(...) -&gt; AsyncHttpResponse[PaymentAllocationListResponse]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Lists payment allocations for a payment or an invoice
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.payment_allocations.list_payment_allocations(
    payment_id="pay_1234abcd",
    invoice_id="inv_1234abcd",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**limit:** `typing.Optional[int]` 
    
</dd>
</dl>

<dl>
<dd>

**offset:** `typing.Optional[int]` 
    
</dd>
</dl>

<dl>
<dd>

**payment_id:** `typing.Optional[str]` — Filter by the payment the amounts were allocated from.
    
</dd>
</dl>

<dl>
<dd>

**invoice_id:** `typing.Optional[str]` — Filter by the invoice the allocated lines belong to.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.payment_allocations.<a href="src/paid/payment_allocations/client.py">create_payment_allocation</a>(...) -&gt; AsyncHttpResponse[PaymentAllocationCreateResponse]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Allocates a payment across one or more invoice lines of posted invoices, in the payment's currency. When an invoice becomes fully paid, its credit entitlements are processed. Returns 409 when a concurrent allocation has already spent the balance.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid, PaymentAllocationInput

client = Paid(
    token="YOUR_TOKEN",
)
client.payment_allocations.create_payment_allocation(
    payment_id="pay_1234abcd",
    allocations=[
        PaymentAllocationInput(
            invoice_line_id="invoiceLineId",
            amount=15000,
        )
    ],
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**payment_id:** `str` — The payment to allocate from.
    
</dd>
</dl>

<dl>
<dd>

**allocations:** `typing.Sequence[PaymentAllocationInput]` — Invoice lines to allocate the payment to.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

## Amendments
<details><summary><code>client.amendments.<a href="src/paid/amendments/client.py">get_order_amendment_options</a>(...) -&gt; AsyncHttpResponse[AmendmentOptions]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Returns which amendments the order admits right now: per-attribute intents and treatment axes with choosable options, defaults, and unavailability reasons, plus the order version, currency, and effective date an amendment request needs.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.amendments.get_order_amendment_options(
    order_id="orderId",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**order_id:** `str` — Display id of the order (for example `ord_5rLZXDFSHNw`). Line and attribute ids in amendment bodies are UUIDs from the options response.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.amendments.<a href="src/paid/amendments/client.py">preview_order_amendment</a>(...) -&gt; AsyncHttpResponse[AmendmentPlan]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Compiles amendment intents into a plan (operations, money effects, credit effects, state diff) without executing. The returned planHash can be passed to the execute endpoint for two-phase, drift-guarded execution.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid, UnifiedAmendmentIntent_UpdateQuantity

client = Paid(
    token="YOUR_TOKEN",
)
client.amendments.preview_order_amendment(
    order_id="orderId",
    order_version=1,
    intents=[
        UnifiedAmendmentIntent_UpdateQuantity(
            order_line_attribute_id="orderLineAttributeId",
            new_quantity=1,
        )
    ],
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**order_id:** `str` — Display id of the order (for example `ord_5rLZXDFSHNw`). Line and attribute ids in amendment bodies are UUIDs from the options response.
    
</dd>
</dl>

<dl>
<dd>

**order_version:** `int` — Must match `orderVersion` from the options response. Returns 409 if the order has been amended since.
    
</dd>
</dl>

<dl>
<dd>

**intents:** `typing.Sequence[UnifiedAmendmentIntent]` — At least one intent, discriminated by type.
    
</dd>
</dl>

<dl>
<dd>

**default_treatment:** `typing.Optional[UnifiedAmendmentPreviewRequestDefaultTreatment]` — Recurring charges pick next-cycle vs settle-now. Usage price changes pick new usage only vs the whole current period.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.amendments.<a href="src/paid/amendments/client.py">execute_order_amendment</a>(...) -&gt; AsyncHttpResponse[UnifiedAmendmentResponse]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Executes amendment intents against an order. One-shot by default; pass the previewed planHash to require the recomputed plan to match (409 PLAN_CONFLICT on drift).
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid, UnifiedAmendmentIntent_UpdateQuantity

client = Paid(
    token="YOUR_TOKEN",
)
client.amendments.execute_order_amendment(
    order_id="orderId",
    order_version=1,
    intents=[
        UnifiedAmendmentIntent_UpdateQuantity(
            order_line_attribute_id="orderLineAttributeId",
            new_quantity=1,
        )
    ],
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**order_id:** `str` — Display id of the order (for example `ord_5rLZXDFSHNw`). Line and attribute ids in amendment bodies are UUIDs from the options response.
    
</dd>
</dl>

<dl>
<dd>

**order_version:** `int` — Must match `orderVersion` from the options response. Returns 409 if the order has been amended since.
    
</dd>
</dl>

<dl>
<dd>

**intents:** `typing.Sequence[UnifiedAmendmentIntent]` — At least one intent, discriminated by type.
    
</dd>
</dl>

<dl>
<dd>

**default_treatment:** `typing.Optional[UnifiedAmendmentExecuteRequestDefaultTreatment]` — Recurring charges pick next-cycle vs settle-now. Usage price changes pick new usage only vs the whole current period.
    
</dd>
</dl>

<dl>
<dd>

**plan_hash:** `typing.Optional[str]` — From preview. When set, execute recomputes the plan and returns 409 `PLAN_CONFLICT` if billing state has drifted. Omit only for one-shot execute.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

## AnalyticsExperimental
<details><summary><code>client.analytics_experimental.<a href="src/paid/analytics_experimental/client.py">execute_experimental_analytics_query</a>(...) -&gt; AsyncHttpResponse[AnalyticsQueryResponse]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

This experimental path is deprecated and is not supported for new integrations. Use `POST /api/v2/analytics/query` instead; the old path remains available for existing integrations.

Runs a single ClickHouse SELECT (or WITH … SELECT) against your organization's analytics views. Before writing a query, call `getAnalyticsSchema` (GET /schema) for the available views and columns, and `getSignalsMetadata` (GET /signals/metadata) for the JSON paths inside `fact_signal.data`. Results are automatically scoped to your organization — no org filter is needed or possible. Only SELECT/WITH statements are accepted.

Conventions: monetary amounts are minor units (cents — divide by 100 for the major unit); most are integers, but `fact_cost.cost_amount` is fractional cents (Decimal) since a single AI call usually costs less than a cent; 64-bit integers (counts, ids, amounts) are returned as JSON strings to preserve precision, so parse them client-side; Decimal columns (fractional cents, and credit amounts, which are counts of credits rather than cents and are never divided by 100) come back as JSON numbers instead, so a value beyond 2^53 is already rounded — select toString(col) when you need its exact digits. Query signal payloads via JSON paths, e.g. `SELECT data.country::String AS country, count() FROM fact_signal GROUP BY country`.

Limits: 30 seconds of execution time and 10,000 result rows (truncation is flagged via `meta.truncated`). Queries exceeding the 4 GB memory allowance return HTTP 422 with code `QUERY_MEMORY_LIMIT_EXCEEDED`. Prefer aggregates and a `created_at` date filter on large tables — this endpoint is for interactive analytics, not bulk export.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.analytics_experimental.execute_experimental_analytics_query(
    query="SELECT signal_name, count() AS signals FROM fact_signal WHERE created_at > now() - INTERVAL 30 DAY GROUP BY signal_name ORDER BY signals DESC",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**query:** `str` — A single ClickHouse SELECT (or WITH ... SELECT) statement against the analytics views. Results are automatically scoped to your organization. See GET /schema for the available views and columns.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.analytics_experimental.<a href="src/paid/analytics_experimental/client.py">get_experimental_analytics_schema</a>() -&gt; AsyncHttpResponse[AnalyticsSchemaResponse]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

This experimental path is deprecated and is not supported for new integrations. Use `GET /api/v2/analytics/schema` instead; the old path remains available for existing integrations.

Returns the analytics views available to POST /query, with column names, ClickHouse types, and descriptions. Dimensions (`dim_*`) describe entities; facts (`fact_*`) are event/transaction tables that join to dimensions via the `*_id` columns described in each comment.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.analytics_experimental.get_experimental_analytics_schema()

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.analytics_experimental.<a href="src/paid/analytics_experimental/client.py">get_experimental_signals_metadata</a>(...) -&gt; AsyncHttpResponse[SignalsMetadataResponse]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

This experimental path is deprecated and is not supported for new integrations. Use `GET /api/v2/analytics/signals/metadata` instead; the old path remains available for existing integrations.

Lists the JSON paths (and their observed types) present in the `data` payload of your signals within a time window (default: last 30 days), grouped by signal name. Use the returned paths in queries against `fact_signal`, e.g. `WHERE data.<path>::String = '...'`. Keys declared as grouping dimensions are marked `declared` with their `slot` and listed first. Where available, each path also carries `firstSeen`, `lastSeen` and a `signals` count; these span all time, and the window only selects which paths and types are returned.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from paid import Paid

client = Paid(
    token="YOUR_TOKEN",
)
client.analytics_experimental.get_experimental_signals_metadata()

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**signal_name:** `typing.Optional[str]` — Restrict discovery to a single signal name.
    
</dd>
</dl>

<dl>
<dd>

**from_date:** `typing.Optional[dt.datetime]` — Start of the discovery window. Defaults to 30 days ago.
    
</dd>
</dl>

<dl>
<dd>

**to_date:** `typing.Optional[dt.datetime]` — End of the discovery window. Defaults to now.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

