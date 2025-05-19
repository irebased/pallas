# Pallas

A Python package for generating and executing tool chains for base conversion-based ciphers.

## Installation

```bash
pip install -e .
```

## Development

1. Create a virtual environment:
```bash
python -m venv venv
source .venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install development dependencies:
```bash
pip install -r requirements.txt
```

3. Install the package in development mode:
```bash
pip install -e .
```

## Testing

Run tests using pytest:
```bash
invoke test
```

Run tests with coverage:
```
invoke test-coverage
```

Run a specific test file. Example for testing the octal encoder:
```
pytest tests/tools/encoders/test_octal.py
```

## Other Commands

Clean build artifacts:
```bash
invoke clean

## License

MIT License