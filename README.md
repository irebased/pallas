# Pallas

A Python package for generating and executing tool chains for base conversion-based ciphers.

## Installation

```bash
pip install -e .
```

## Running

1. Install command:

```
pip install -e .
```

Then start by running the following command:
```
python -m pallas.main -h
```

It will give you options and explain how to use the tool!


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

### Adding tools

#### 1. Tool location

Add all tools to the /tools directory. Please organize into subdirectories as it makes sense. Note the encoder/decoder folders.

#### 2. Tool Abstraction

All tools should extend the `Tool` class. This abstracts a lot of complexity so you can focus on implementing your tool!

#### 3. Metadata convention

Follow these conventions when creating your tool metadata:

```
  name: snake_case. if encoder/decoder, do base_encoder or base_decoder.
  description: simple enough
  domain_chars: the valid input. For decoders this is particularly important. Define a constant in common/charsets.py and import.
  range_chars: the possible output. For decoders, this is typically going to be EXTENDED_ASCII_CHARSET.
  separator: if your scheme requires a separator, please add one to your constructor and default to space.
```

#### 4. Logic implementation

All tools are required to place their core logic in the `_process` function with the following header:

```
def _process(self, input_str: str) -> str
```

This aligns with the schema of all other tools and is required to ensure the toolchain continues to work.

#### 5. Testing

All tools are expected to have unit tests in the /tests directory. This directory mirrors the `/src/pallas` directory in structure. So if your tool is in `src/pallas/tools/decoders` directory, your test file should be in the `test/tools/decoders` directory for consistency.

You can run `invoke test` to check that your tests are passing, and `invoke test-coverage` to check the line coverage of your new tool. Ensure that your tool is above 75% coverage or it will fail the CI automation and not get merged in.

#### 6. Contributing

Please open a PR to submit your contribution for review. You may receive comments to improve unit test coverage or otherwise.

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