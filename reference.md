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

Creates a new product for the organization
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

Update a product by ID. Optionally upsert product attributes with pricing.
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

Update a product by external ID. Optionally upsert product attributes with pricing.
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
                        unit_price=99.0,
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

Creates a new customer for the organization
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

Create an alternate external identifier for a customer by Paid display ID.
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

Update a customer by Paid display ID. Use the value returned as `customer.id`, for example `cus_abc123`. If you have your own customer ID, use `PUT /api/v2/customers/external/{externalId}`.
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

Delete a customer by Paid display ID. Use the value returned as `customer.id`, for example `cus_abc123`. If you have your own customer ID, use `DELETE /api/v2/customers/external/{externalId}`.
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

Create an alternate external identifier for a customer by external ID.
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

Update a customer by external ID
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

Delete a customer by external ID
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
    amount=10000,
    starts_at=datetime.datetime.fromisoformat(
        "2026-06-05 12:00:00+00:00",
    ),
    expires_at=datetime.datetime.fromisoformat(
        "2026-12-31 23:59:59+00:00",
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

**credit_currency_key:** `str` — Stable machine-readable key for the active credit currency to grant.
    
</dd>
</dl>

<dl>
<dd>

**amount:** `int` — Number of credits to grant. This is not a monetary amount.
    
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
    amount=10000,
    starts_at=datetime.datetime.fromisoformat(
        "2026-06-05 12:00:00+00:00",
    ),
    expires_at=datetime.datetime.fromisoformat(
        "2026-12-31 23:59:59+00:00",
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

**amount:** `int` — Number of credits to grant. This is not a monetary amount.
    
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

Update a contact by its ID
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

**customer_id:** `typing.Optional[str]` 
    
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

Delete a contact by its ID
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

Update a contact by its external ID
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

**customer_id:** `typing.Optional[str]` 
    
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

Delete a contact by its external ID
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

**payment_terms:** `typing.Optional[str]` 
    
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

Update an order by ID
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

**payment_terms:** `typing.Optional[str]` 
    
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
<details><summary><code>client.signals.<a href="src/paid/signals/client.py">create_signals</a>(...) -&gt; AsyncHttpResponse[BulkSignalsResponse]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Create multiple signals (usage events) in a single request. Each signal must include a customer attribution (either customerId or externalCustomerId) and a product attribution (either productId or externalProductId).
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
from paid import CustomerById, Paid, Signal

client = Paid(
    token="YOUR_TOKEN",
)
client.signals.create_signals(
    signals=[
        Signal(
            event_name="eventName",
            customer=CustomerById(
                customer_id="customerId",
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

**expires_at:** `typing.Optional[dt.datetime]` 
    
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

Archive a checkout by ID
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

Creates a portal session for the customer. Returns a short-lived URL to the customer portal.
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
<details><summary><code>client.value_receipts.<a href="src/paid/value_receipts/client.py">sync_value_receipt</a>(...) -&gt; AsyncHttpResponse[ValueReceiptSyncResponse]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Find or create a value receipt by natural key (customer + product/order + dates), then populate it with current data inline. Returns the ID, status, and public URL. Posted (sealed) VRs are returned as-is without re-populating.
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

Re-populate an existing draft value receipt with current data inline. Returns the slim sync response. Sealed VRs cannot be refreshed.
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

Transition a draft value receipt to sealed (posted) status. Sealed VRs are immutable — they cannot be updated or re-populated.
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

Restore an archived value receipt.
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

Make a value receipt publicly accessible via URL.
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

**publish_expires_at:** `typing.Optional[dt.datetime]` 
    
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

Revoke public access to a value receipt.
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

**url:** `typing.Optional[str]` — The HTTPS endpoint Paid should deliver this webhook to. Set to null to clear it while disabled.
    
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

Updates pricing on an existing product attribute. If creditBenefits is provided, it fully replaces existing benefits. If omitted, existing benefits are preserved.
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
                unit_price=1.1,
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

## AnalyticsExperimental
<details><summary><code>client.analytics_experimental.<a href="src/paid/analytics_experimental/client.py">execute_analytics_query</a>(...) -&gt; AsyncHttpResponse[AnalyticsQueryResponse]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

⚠️ **Experimental** — this endpoint may change or be removed without notice and is not subject to v2 backwards-compatibility guarantees. Do not build production-critical integrations against it yet.

Runs a single ClickHouse SELECT (or WITH … SELECT) against your organization's analytics views. Before writing a query, call `getAnalyticsSchema` (GET /schema) for the available views and columns, and `getSignalsMetadata` (GET /signals/metadata) for the JSON paths inside `fact_signal.data`. Results are automatically scoped to your organization — no org filter is needed or possible. Only SELECT/WITH statements are accepted.

Conventions: monetary amounts are minor units (cents — divide by 100 for the major unit); most are integers, but `fact_cost.cost_amount` is fractional cents (Decimal) since a single AI call usually costs less than a cent; 64-bit integers (counts, ids, amounts) are returned as JSON strings to preserve precision, so parse them client-side. Query signal payloads via JSON paths, e.g. `SELECT data.country::String AS country, count() FROM fact_signal GROUP BY country`.

Limits: 30 seconds of execution time and 10,000 result rows (truncation is flagged via `meta.truncated`). Prefer aggregates and a `created_at` date filter on large tables — this endpoint is for interactive analytics, not bulk export.
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
client.analytics_experimental.execute_analytics_query(
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

<details><summary><code>client.analytics_experimental.<a href="src/paid/analytics_experimental/client.py">get_analytics_schema</a>() -&gt; AsyncHttpResponse[AnalyticsSchemaResponse]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

⚠️ **Experimental** — this endpoint may change or be removed without notice and is not subject to v2 backwards-compatibility guarantees. Do not build production-critical integrations against it yet.

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
client.analytics_experimental.get_analytics_schema()

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

<details><summary><code>client.analytics_experimental.<a href="src/paid/analytics_experimental/client.py">get_signals_metadata</a>(...) -&gt; AsyncHttpResponse[SignalsMetadataResponse]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

⚠️ **Experimental** — this endpoint may change or be removed without notice and is not subject to v2 backwards-compatibility guarantees. Do not build production-critical integrations against it yet.

Lists the JSON paths (and their observed types) present in the `data` payload of your signals within a time window (default: last 30 days), grouped by signal name. Use the returned paths in queries against `fact_signal`, e.g. `WHERE data.<path>::String = '...'`.
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
client.analytics_experimental.get_signals_metadata()

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

⚠️ Only call this when the user has EXPLICITLY asked to save, create, or publish the view. After generating or previewing a dashboard, do NOT automatically save a draft — show it to the user and wait for them to ask you to save it. Saves named analytics queries + a self-contained HTML render bundle as a DRAFT custom view. **Call getCustomViewAuthoringGuide (GET /experimental/views/authoring-guide) first** — it returns the full guide and a copy-paste interactive template. Key rules: (1) Do NOT add a customer filter to the SQL — the database scopes every query to the viewing customer at embed time. (2) Each query's SQL must be SELECT-only; return clearly-named columns. Compute metric VALUES in SQL (e.g. (count()*2)/5 AS custom_metric) — derive a number in the render bundle only when it depends on user interaction (toggle/filter/hover) or is pure formatting of a value a query already returns. (3) The render bundle must be SELF-CONTAINED — inline all CSS/JS/charting, NO external loads or fetch (the sandbox has connect-src 'none'); it must listen for the `paid:data` message (data keyed by query id) and re-render on each one. (4) Make it INTERACTIVE — mousemove hover tooltips and at least one addEventListener-wired control that re-renders (a static chart feels broken). (5) The render bundle is the single source of truth — preview the EXACT bundle you save (call getCustomViewPreviewHarness with your bundle + sample data and render the HTML it returns) or review it in the Paid preview; do NOT build a separate chart, and only show numbers that come from a declared query. (6) A view is a FULL dashboard — include as many charts/KPIs as the analysis has. Keep every element derived from the single viewing customer (KPIs, trends, type mix); drop only cross-customer comparisons (rankings, share-of-total, 'N customers'). Don't simplify to one chart. (7) To make the date range adjustable (e.g. the user says 'last month'), write the date boundary as `{period_start:DateTime}` / `{period_end:DateTime}` placeholders in the SQL and pass a default `period` (relative like {kind:'relative',unit:'month',amount:1}, or absolute start/end). The org user can then change it in Paid without re-authoring. A query using the placeholders REQUIRES a period. Do NOT add your own date-range picker to the render bundle — Paid owns the timeframe and the bundle receives already-filtered data; a second in-bundle picker cannot re-run the SQL. The response returns a `previewUrl` — give it to the user so they can open the new view in Paid.
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

Flips the view from DRAFT to PUBLISHED. Only PUBLISHED views are served on the embed data path — this is the gate that stops an unreviewed view reaching end-customers. Idempotent: publishing an already-published view is a no-op success. The response returns a `previewUrl` — give it to the user so they can open the view in Paid.
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

Returns the view's name, status, query ids, and the author render bundle for the owning organization (DRAFT or PUBLISHED). Used by the trusted preview/embed frame to render the sandbox; the per-customer data is fetched separately via /:displayId/data.
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

Partially updates a view's name, description, queries, or render bundle. Omitted fields are left unchanged; `queries` is a FULL replacement of the query list. Updated SQL and bundles pass the same validation as createCustomView (SELECT-only, size cap, self-contained, paid:data listener). Works on DRAFT or PUBLISHED views — published embeds pick the change up on their next load.
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

**queries:** `typing.Optional[typing.Sequence[CustomViewQuery]]` — Full replacement of the view's query list. Each SQL is re-validated (SELECT-only) exactly like createCustomView.
    
</dd>
</dl>

<dl>
<dd>

**render_bundle:** `typing.Optional[str]` — Replacement render bundle. Re-validated (size cap, self-contained, paid:data listener) exactly like createCustomView.
    
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

Runs every stored query of the view on the read-only analytics database, scoped to the caller's organization AND the given customer (both enforced as ClickHouse row filters), and returns the result sets keyed by query id. The customer scope is enforced by the database — it cannot be widened by the stored SQL.
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

**customer_id:** `str` — Customer to scope the data to (dev/preview only; the embed derives this from the verified token).
    
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

Mints a short-lived, customer-scoped token for embedding a published custom view. Call this from your server with your API key, then pass the returned token to the embed SDK.
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

Uploads a new value model version. Validates the content, creates a new version, archives the previous active version, syncs to ClickHouse, and triggers a backfill of all historical signals.
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
    ValueModelContent,
    ValueModelContentFormulasItem,
    ValueModelContentFormulasItemVariablesItem,
    ValueModelContentSignalsItem,
    ValueModelContentValueTypesItem,
    ValueModelContentValueTypesItemUnitZero,
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
                unit=ValueModelContentValueTypesItemUnitZero(
                    type="monetary",
                ),
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

Manually triggers a recalculation of all ClickHouse rows against the current value model.
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

## CustomerGroups
<details><summary><code>client.customer_groups.<a href="src/paid/customer_groups/client.py">list_customer_groups</a>(...) -&gt; AsyncHttpResponse[CustomerGroupListResponse]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

List all customer groups for the organization.
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
client.customer_groups.list_customer_groups()

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

<details><summary><code>client.customer_groups.<a href="src/paid/customer_groups/client.py">create_customer_group</a>(...) -&gt; AsyncHttpResponse[CustomerGroupDetail]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Create a new customer group. Names must be unique per org.
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
client.customer_groups.create_customer_group(
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

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.customer_groups.<a href="src/paid/customer_groups/client.py">get_customer_group</a>(...) -&gt; AsyncHttpResponse[CustomerGroupDetail]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Returns group details including member list (capped at 500).
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
client.customer_groups.get_customer_group(
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

<details><summary><code>client.customer_groups.<a href="src/paid/customer_groups/client.py">delete_customer_group</a>(...) -&gt; AsyncHttpResponse[CustomerGroupDeleteResponse]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Deletes the group and unbinds all members. Unbound customers fall back to the base value model config.
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
client.customer_groups.delete_customer_group(
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

<details><summary><code>client.customer_groups.<a href="src/paid/customer_groups/client.py">update_customer_group</a>(...) -&gt; AsyncHttpResponse[CustomerGroupDetail]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Update a customer group's name or description.
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
client.customer_groups.update_customer_group(
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

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.customer_groups.<a href="src/paid/customer_groups/client.py">create_customer_group_members</a>(...) -&gt; AsyncHttpResponse[AddMembersResponse]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Additive. Adds customers to the group without removing existing members.
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
client.customer_groups.create_customer_group_members(
    id="id",
    customer_ids=["customerIds"],
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

**customer_ids:** `typing.Sequence[str]` — External customer IDs.
    
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

<details><summary><code>client.customer_groups.<a href="src/paid/customer_groups/client.py">update_customer_group_members</a>(...) -&gt; AsyncHttpResponse[SetMembersResponse]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Reconcile membership. The provided list is the complete desired membership. Customers not in the list are removed. Customers in the list but not currently in the group are added.
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
client.customer_groups.update_customer_group_members(
    id="id",
    customer_ids=["customerIds"],
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

**customer_ids:** `typing.Sequence[str]` — External customer IDs.
    
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

<details><summary><code>client.customer_groups.<a href="src/paid/customer_groups/client.py">delete_customer_group_members</a>(...) -&gt; AsyncHttpResponse[RemoveMembersResponse]</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Removes specific customers from the group.
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
client.customer_groups.delete_customer_group_members(
    id="id",
    customer_ids=["customerIds"],
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

**customer_ids:** `typing.Sequence[str]` — External customer IDs.
    
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

