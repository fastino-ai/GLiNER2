# GLiNER2 Regex Validators

Regex validators filter extracted spans to ensure they match expected patterns, improving extraction quality and reducing false positives.

## Quick Start

```python
from gliner2 import GLiNER2, RegexValidator

extractor = GLiNER2.from_pretrained("your-model")

# Create validator and apply to field
email_validator = RegexValidator(r"^[\w\.-]+@[\w\.-]+\.\w+$")
schema = (extractor.create_schema()
    .structure("contact")
        .field("email", dtype="str", validators=[email_validator])
)
```

## RegexValidator Parameters

- **pattern**: Regex pattern (string or compiled Pattern)
- **mode**: `"full"` (exact match) or `"partial"` (substring match)
- **exclude**: `False` (keep matches) or `True` (exclude matches)
- **flags**: Regex flags like `re.IGNORECASE` (for string patterns only)

## Examples

### Email Validation
```python
email_validator = RegexValidator(r"^[\w\.-]+@[\w\.-]+\.\w+$")

text = "Contact: john@company.com, not-an-email, jane@domain.org"
# Output: ['john@company.com', 'jane@domain.org']
```

### Phone Numbers (US Format)
```python
phone_validator = RegexValidator(r"\(\d{3}\)\s\d{3}-\d{4}", mode="partial")

text = "Call (555) 123-4567 or 5551234567"
# Output: ['(555) 123-4567']  # Second number filtered out
```

### URLs Only
```python
url_validator = RegexValidator(r"^https?://", mode="partial")

text = "Visit https://example.com or www.site.com"
# Output: ['https://example.com']  # www.site.com filtered out
```

### Exclude Test Data
```python
no_test_validator = RegexValidator(r"^(test|demo|sample)", exclude=True, flags=re.IGNORECASE)

text = "Products: iPhone, Test Phone, Samsung Galaxy"
# Output: ['iPhone', 'Samsung Galaxy']  # Test Phone excluded
```

### Length Constraints
```python
length_validator = RegexValidator(r"^.{5,50}$")  # 5-50 characters

text = "Names: Jo, Alexander, A Very Long Name That Exceeds Fifty Characters"
# Output: ['Alexander']  # Others filtered by length
```

### Multiple Validators
```python
# All validators must pass
username_validators = [
    RegexValidator(r"^[a-zA-Z0-9_]+$"),  # Alphanumeric + underscore
    RegexValidator(r"^.{3,20}$"),        # 3-20 characters
    RegexValidator(r"^(?!admin)", exclude=True, flags=re.IGNORECASE)  # No "admin"
]

schema = (extractor.create_schema()
    .structure("user")
        .field("username", dtype="str", validators=username_validators)
)

text = "Users: ab, john_doe, user@domain, admin, valid_user123"
# Output: ['john_doe', 'valid_user123']
```

## Common Patterns

| Use Case | Pattern | Mode |
|----------|---------|------|
| Email | `r"^[\w\.-]+@[\w\.-]+\.\w+$"` | full |
| Phone (US) | `r"\(\d{3}\)\s\d{3}-\d{4}"` | partial |
| URL | `r"^https?://"` | partial |
| Numbers only | `r"^\d+$"` | full |
| No spaces | `r"^\S+$"` | full |
| Min length | `r"^.{5,}$"` | full |
| Alphanumeric | `r"^[a-zA-Z0-9]+$"` | full |

## Best Practices

1. **Use specific patterns** - More specific = fewer false positives
2. **Test your regex** - Validate patterns before deployment
3. **Combine validators** - Chain multiple simple validators
4. **Consider case sensitivity** - Use `re.IGNORECASE` when needed
5. **Start simple** - Begin with basic patterns, refine as needed

## Performance Notes

- Validators run after span extraction but before formatting
- Failed validation simply excludes the span (no errors)
- Multiple validators use short-circuit evaluation (stops at first failure)
- Compiled patterns are cached automatically
