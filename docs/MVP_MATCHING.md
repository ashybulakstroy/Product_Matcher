# MVP matching project

This project follows the same high-level folder logic as the parent OCR project:

- App/ for runnable application code
- data/ for inputs, references, and test fixtures
- docs/ for project notes
- tests/ for validation

Current goal:

- take a list of products from the Torgsoft receipt/check
- compare them with stock items from 1C
- return TOP-N candidates for each checked item
- sort candidates by similarity descending
