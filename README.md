# HackRx Bill Extractor API

## API Endpoint
```
POST /extract-bill-data
Content-Type: application/json

{
    "document": "https://example.com/bill.pdf"
}
```

## Response Format
```json
{
    "is_success": true,
    "token_usage": {
        "total_tokens": 1500,
        "input_tokens": 1200,
        "output_tokens": 300
    },
    "data": {
        "pagewise_line_items": [
            {
                "page_no": "1",
                "page_type": "Bill Detail",
                "bill_items": [
                    {
                        "item_name": "Consultation Fee",
                        "item_amount": 500.0,
                        "item_rate": 500.0,
                        "item_quantity": 1.0
                    }
                ]
            }
        ],
        "total_item_count": 1
    }
}
```

## Technology Stack
- **Framework:** FastAPI
- **LLM:** Google Gemini 2.0 Flash / 2.5 Flash (failover)
- **PDF Processing:** PyMuPDF
- **Deployment:** Docker on Render

## Features
- Instant model failover (zero delay)
- Memory-optimized for 512MB RAM
- Unbuffered real-time logging
- Handles PDFs and images
