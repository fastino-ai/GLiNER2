# GLiNER2 Classification Tutorial

This tutorial covers all the ways to perform text classification with GLiNER2, from simple single-label classification to complex multi-label tasks with custom configurations.

## Table of Contents
- [Setup](#setup)
- [Single-Label Classification](#single-label-classification)
- [Multi-Label Classification](#multi-label-classification)
- [Classification with Descriptions](#classification-with-descriptions)
- [Using the Quick API](#using-the-quick-api)
- [Multiple Classification Tasks](#multiple-classification-tasks)
- [Advanced Configurations](#advanced-configurations)
- [Best Practices](#best-practices)

## Setup

```python
from gliner2 import GLiNER2

# Load the pre-trained model
extractor = GLiNER2.from_pretrained("your-model-name")
```

## Single-Label Classification

The simplest form - classify text into one of several categories.

### Basic Example

```python
# Define the schema
schema = extractor.create_schema().classification(
    "sentiment",
    ["positive", "negative", "neutral"]
)

# Extract
text = "This product exceeded my expectations! Absolutely love it."
results = extractor.extract(text, schema)
print(results)
# Expected output: {'sentiment': 'positive'}
```

### With Confidence Scores

```python
# Same schema as above
schema = extractor.create_schema().classification(
    "sentiment",
    ["positive", "negative", "neutral"]
)

text = "The service was okay, nothing special but not bad either."
results = extractor.extract(text, schema, include_confidence=True)
print(results)
# Expected output: {'sentiment': {'label': 'neutral', 'confidence': 0.82}}
```

## Multi-Label Classification

When text can belong to multiple categories simultaneously.

```python
# Multi-label classification
schema = extractor.create_schema().classification(
    "topics",
    ["technology", "business", "health", "politics", "sports"],
    multi_label=True,
    cls_threshold=0.3  # Lower threshold for multi-label
)

text = "Apple announced new health monitoring features in their latest smartwatch, boosting their stock price."
results = extractor.extract(text, schema)
print(results)
# Expected output: {'topics': ['technology', 'business', 'health']}

# With confidence scores
results = extractor.extract(text, schema, include_confidence=True)
print(results)
# Expected output: {'topics': [
#     {'label': 'technology', 'confidence': 0.92},
#     {'label': 'business', 'confidence': 0.78},
#     {'label': 'health', 'confidence': 0.65}
# ]}
```

## Classification with Descriptions

Adding descriptions significantly improves accuracy by providing context.

```python
# With label descriptions
schema = extractor.create_schema().classification(
    "document_type",
    {
        "invoice": "A bill for goods or services with payment details",
        "receipt": "Proof of payment for a completed transaction",
        "contract": "Legal agreement between parties with terms and conditions",
        "proposal": "Document outlining suggested plans or services with pricing"
    }
)

text = "Please find attached the itemized bill for consulting services rendered in Q3 2024. Payment is due within 30 days."
results = extractor.extract(text, schema)
print(results)
# Expected output: {'document_type': 'invoice'}

# Another example
text2 = "Thank you for your payment of $500. This confirms your transaction was completed on March 1st, 2024."
results2 = extractor.extract(text2, schema)
print(results2)
# Expected output: {'document_type': 'receipt'}
```

## Using the Quick API

For simple classification tasks without building a schema.

### Single Task

```python
text = "The new AI model shows remarkable performance improvements."
results = extractor.classify_text(
    text,
    {"sentiment": ["positive", "negative", "neutral"]}
)
print(results)
# Expected output: {'sentiment': 'positive'}

# Another example
text2 = "The software keeps crashing and customer support is unresponsive."
results2 = extractor.classify_text(
    text2,
    {"sentiment": ["positive", "negative", "neutral"]}
)
print(results2)
# Expected output: {'sentiment': 'negative'}
```

### Multiple Tasks

```python
text = "Breaking: Tech giant announces major layoffs amid market downturn"
results = extractor.classify_text(
    text,
    {
        "sentiment": ["positive", "negative", "neutral"],
        "urgency": ["high", "medium", "low"],
        "category": {
            "labels": ["tech", "finance", "politics", "sports"],
            "multi_label": False
        }
    }
)
print(results)
# Expected output: {
#     'sentiment': 'negative',
#     'urgency': 'high',
#     'category': 'tech'
# }
```

### Multi-Label with Config

```python
text = "The smartphone features an amazing camera but disappointing battery life and overheats frequently."
results = extractor.classify_text(
    text,
    {
        "product_aspects": {
            "labels": ["camera", "battery", "display", "performance", "design", "heating"],
            "multi_label": True,
            "cls_threshold": 0.4
        }
    }
)
print(results)
# Expected output: {'product_aspects': ['camera', 'battery', 'heating']}

# Another example
text2 = "Beautiful design with vibrant display, though the camera could be better."
results2 = extractor.classify_text(
    text2,
    {
        "product_aspects": {
            "labels": ["camera", "battery", "display", "performance", "design", "heating"],
            "multi_label": True,
            "cls_threshold": 0.4
        }
    }
)
print(results2)
# Expected output: {'product_aspects': ['design', 'display', 'camera']}
```

## Multiple Classification Tasks

You can include multiple classification tasks in a single schema for comprehensive text analysis.

### Basic Multiple Classifications

```python
# Multiple independent classifications
schema = (extractor.create_schema()
    .classification("sentiment", ["positive", "negative", "neutral"])
    .classification("language", ["english", "spanish", "french", "german", "other"])
    .classification("formality", ["formal", "informal", "semi-formal"])
    .classification("intent", ["question", "statement", "request", "complaint"])
)

text = "Could you please help me with my order? The service has been disappointing."
results = extractor.extract(text, schema)
print(results)
# Expected output: {
#     'sentiment': 'negative',
#     'language': 'english',
#     'formality': 'formal',
#     'intent': 'question'
# }

# Another example
text2 = "Hey! Just wanted to say your product rocks! 🎉"
results2 = extractor.extract(text2, schema)
print(results2)
# Expected output: {
#     'sentiment': 'positive',
#     'language': 'english',
#     'formality': 'informal',
#     'intent': 'statement'
# }
```

### Mixed Single and Multi-Label Classifications

```python
# Combine different classification types
schema = (extractor.create_schema()
    # Single-label classifications
    .classification("primary_topic", ["tech", "business", "health", "sports", "politics"])
    .classification("urgency", ["immediate", "soon", "later", "not_urgent"])

    # Multi-label classifications
    .classification("emotions",
        ["happy", "sad", "angry", "surprised", "fearful", "disgusted"],
        multi_label=True,
        cls_threshold=0.4
    )
    .classification("content_flags",
        ["inappropriate", "spam", "promotional", "personal_info", "financial_info"],
        multi_label=True,
        cls_threshold=0.3
    )
)

text = "URGENT: I'm thrilled to announce our new product! But concerned about competitor reactions. Please keep confidential."
results = extractor.extract(text, schema)
print(results)
# Expected output: {
#     'primary_topic': 'business',
#     'urgency': 'immediate',
#     'emotions': ['happy', 'fearful'],
#     'content_flags': ['promotional', 'personal_info']
# }

# Another example
text2 = "Just saw the game - absolutely devastated by the loss. Can't believe the referee's terrible decision!"
results2 = extractor.extract(text2, schema)
print(results2)
# Expected output: {
#     'primary_topic': 'sports',
#     'urgency': 'not_urgent',
#     'emotions': ['sad', 'angry'],
#     'content_flags': []
# }
```

### Domain-Specific Multiple Classifications

```python
# Customer support ticket classification
support_schema = (extractor.create_schema()
    .classification("ticket_type",
        ["technical_issue", "billing", "feature_request", "bug_report", "other"])
    .classification("priority",
        ["critical", "high", "medium", "low"],
        cls_threshold=0.7
    )
    .classification("product_area",
        {
            "authentication": "Login, passwords, security",
            "payment": "Payment processing, subscriptions",
            "ui": "User interface, design issues",
            "performance": "Speed, loading, responsiveness",
            "data": "Data loss, corruption, sync issues"
        },
        multi_label=True,
        cls_threshold=0.5
    )
    .classification("customer_sentiment",
        ["very_satisfied", "satisfied", "neutral", "frustrated", "very_frustrated"],
        cls_threshold=0.6
    )
    .classification("requires_action",
        ["immediate_response", "investigation_needed", "waiting_customer", "resolved"],
        multi_label=True
    )
)

ticket_text = """
Subject: Cannot login - Urgent!

I've been trying to login for the past hour but keep getting error messages.
This is critical as I need to process payments for my customers today.
The page just keeps spinning and then times out. I'm extremely frustrated
as this is costing me business. Please fix this immediately!
"""

results = extractor.extract(ticket_text, support_schema)
print(results)
# Expected output: {
#     'ticket_type': 'technical_issue',
#     'priority': 'critical',
#     'product_area': ['authentication', 'payment', 'performance'],
#     'customer_sentiment': 'very_frustrated',
#     'requires_action': ['immediate_response', 'investigation_needed']
# }

# Another support ticket example
ticket_text2 = """
Hi team,

Thanks for the great product! I was wondering if you could add a dark mode feature?
It would really help with eye strain during late night work sessions.

Best regards,
Happy Customer
"""

results2 = extractor.extract(ticket_text2, support_schema)
print(results2)
# Expected output: {
#     'ticket_type': 'feature_request',
#     'priority': 'low',
#     'product_area': ['ui'],
#     'customer_sentiment': 'satisfied',
#     'requires_action': ['waiting_customer']
# }
```

### Sequential Classification with Dependencies

```python
# Email routing and handling classification
email_schema = (extractor.create_schema()
    # Primary classification
    .classification("email_category",
        ["sales", "support", "hr", "legal", "general"],
        cls_threshold=0.6
    )

    # Secondary classifications based on context
    .classification("sales_stage",
        ["lead", "qualified", "proposal", "negotiation", "closed"],
        cls_threshold=0.5
    )
    .classification("support_type",
        ["pre_sales", "technical", "account", "billing"],
        cls_threshold=0.5
    )

    # Action classifications
    .classification("required_action",
        ["reply_needed", "forward_to_team", "schedule_meeting", "no_action"],
        multi_label=True,
        cls_threshold=0.4
    )
    .classification("response_timeframe",
        ["within_1_hour", "within_24_hours", "within_week", "non_urgent"],
        cls_threshold=0.6
    )
)

email = """
Hi Sales Team,

I'm interested in your enterprise solution. We're currently evaluating vendors
for our upcoming project. Could we schedule a demo next week? We need to make
a decision by month end.

Best regards,
John from TechCorp
"""

results = extractor.extract(email, email_schema)
print(results)
# Expected output: {
#     'email_category': 'sales',
#     'sales_stage': 'qualified',
#     'support_type': 'pre_sales',
#     'required_action': ['reply_needed', 'schedule_meeting'],
#     'response_timeframe': 'within_24_hours'
# }

# HR email example
email2 = """
Dear HR Department,

I need to update my tax withholding information. Could someone please send me
the necessary forms? This is somewhat urgent as I need this changed before the
next payroll cycle.

Thank you,
Sarah
"""

results2 = extractor.extract(email2, email_schema)
print(results2)
# Expected output: {
#     'email_category': 'hr',
#     'sales_stage': 'lead',  # May have noise in non-sales emails
#     'support_type': 'account',
#     'required_action': ['reply_needed'],
#     'response_timeframe': 'within_24_hours'
# }
```

### Complex Analysis with Multiple Classifications

```python
# Content moderation and analysis
content_schema = (extractor.create_schema()
    # Content classifications
    .classification("content_type",
        ["article", "comment", "review", "social_post", "message"])
    .classification("primary_language",
        ["english", "spanish", "french", "other"])

    # Quality assessments
    .classification("quality_score",
        ["excellent", "good", "average", "poor", "spam"],
        cls_threshold=0.7
    )
    .classification("originality",
        ["original", "derivative", "duplicate", "plagiarized"],
        cls_threshold=0.8
    )

    # Safety and compliance
    .classification("safety_flags",
        {
            "hate_speech": "Contains discriminatory or hateful content",
            "violence": "Contains violent or threatening content",
            "adult": "Contains adult or explicit content",
            "misinformation": "Contains potentially false information",
            "personal_info": "Contains personal identifying information"
        },
        multi_label=True,
        cls_threshold=0.3
    )

    # Engagement predictions
    .classification("engagement_potential",
        ["viral", "high", "medium", "low"],
        cls_threshold=0.6
    )
    .classification("audience_fit",
        ["general", "professional", "academic", "youth", "senior"],
        multi_label=True,
        cls_threshold=0.5
    )
)

content_text = """
Just discovered this amazing productivity hack that doubled my output!
Here's what I do: I wake up at 5 AM, meditate for 20 minutes, then work
in 90-minute focused blocks. The results have been incredible. My email
is john.doe@example.com if you want more tips!
"""

results = extractor.extract(content_text, content_schema)
print(results)
# Expected output: {
#     'content_type': 'social_post',
#     'primary_language': 'english',
#     'quality_score': 'good',
#     'originality': 'original',
#     'safety_flags': ['personal_info'],
#     'engagement_potential': 'high',
#     'audience_fit': ['general', 'professional']
# }

# Review example
review_text = """
Worst product ever!!! Total scam! Don't buy this garbage. The company should
be shut down for selling this junk. I'm going to report them to authorities.
"""

results2 = extractor.extract(review_text, content_schema)
print(results2)
# Expected output: {
#     'content_type': 'review',
#     'primary_language': 'english',
#     'quality_score': 'poor',
#     'originality': 'original',
#     'safety_flags': ['violence'],  # Due to aggressive language
#     'engagement_potential': 'low',
#     'audience_fit': ['general']
# }
```

## Advanced Configurations

### Custom Thresholds

```python
# High-precision classification
schema = extractor.create_schema().classification(
    "is_spam",
    ["spam", "not_spam"],
    cls_threshold=0.9  # Very high confidence required
)

text = "Congratulations! You've won $1,000,000! Click here to claim your prize now!"
results = extractor.extract(text, schema)
print(results)
# Expected output: {'is_spam': 'spam'}

# Different thresholds for different tasks
schema = (extractor.create_schema()
    .classification("priority", ["urgent", "high", "normal", "low"], cls_threshold=0.8)
    .classification("department", ["sales", "support", "billing", "other"], cls_threshold=0.5)
)

text = "URGENT: Customer threatening to cancel $50k contract due to billing error"
results = extractor.extract(text, schema)
print(results)
# Expected output: {
#     'priority': 'urgent',
#     'department': 'billing'
# }
```

### Custom Activation Functions

```python
# Force specific activation
schema = extractor.create_schema().classification(
    "category",
    ["A", "B", "C", "D"],
    class_act="softmax"  # Options: "sigmoid", "softmax", "auto"
)

text = "This clearly belongs to category B based on the criteria."
results = extractor.extract(text, schema)
print(results)
# Expected output: {'category': 'B'}
```

### Complex Multi-Label Example

```python
# Email classification system
schema = extractor.create_schema().classification(
    "email_tags",
    {
        "action_required": "Email requires recipient to take action",
        "meeting_request": "Email contains meeting invitation or scheduling",
        "project_update": "Email contains project status or updates",
        "urgent": "Email marked as urgent or time-sensitive",
        "question": "Email contains questions requiring answers",
        "fyi": "Informational email requiring no action"
    },
    multi_label=True,
    cls_threshold=0.35
)

email_text = """
Hi team,

Quick update on Project Alpha: We're ahead of schedule!

However, I need your input on the design mockups by EOD tomorrow.
Can we schedule a 30-min call this week to discuss?

This is quite urgent as the client is waiting.

Best,
Sarah
"""

results = extractor.extract(email_text, schema)
print(results)
# Expected output: {
#     'email_tags': ['action_required', 'meeting_request', 'project_update', 'urgent', 'question']
# }

# FYI email example
email_text2 = """
Team,

Just wanted to let everyone know that I'll be out of office next Monday for a
doctor's appointment. I'll be back Tuesday morning.

Thanks,
Mark
"""

results2 = extractor.extract(email_text2, schema)
print(results2)
# Expected output: {
#     'email_tags': ['fyi']
# }
```

## Best Practices

1. **Use Descriptions**: Always provide label descriptions when possible
   ```python
   # Good - with descriptions
   schema = extractor.create_schema().classification(
       "intent",
       {
           "purchase": "User wants to buy a product",
           "return": "User wants to return a product",
           "inquiry": "User asking for information"
       }
   )

   # Less effective - no context
   schema = extractor.create_schema().classification(
       "intent",
       ["purchase", "return", "inquiry"]
   )
   ```

2. **Adjust Thresholds**: Lower thresholds for multi-label (0.3-0.5), higher for single-label (0.5-0.7)

3. **Multi-Label Strategy**: Use multi-label when categories aren't mutually exclusive
   ```python
   # Good use of multi-label
   schema = extractor.create_schema().classification(
       "product_features",
       ["waterproof", "wireless", "rechargeable", "portable"],
       multi_label=True
   )

   # Should be single-label
   schema = extractor.create_schema().classification(
       "size",
       ["small", "medium", "large"],
       multi_label=False  # Sizes are mutually exclusive
   )
   ```

4. **Test with Real Examples**: Always test with actual text samples from your domain

## Common Use Cases

- **Sentiment Analysis**: Customer feedback, reviews, social media
- **Intent Classification**: Chatbots, customer service routing
- **Document Classification**: Email filtering, document management
- **Content Moderation**: Toxic content, spam detection
- **Topic Classification**: News categorization, content tagging
