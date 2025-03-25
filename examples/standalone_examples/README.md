# Standalone Examples

This directory contains examples demonstrating how to use LangMem independently of LangGraph's context. These examples show how to integrate LangMem into your own applications and frameworks.

## Examples

### Custom Store Example

`custom_store_example.py`: Shows how to use a custom store with the memory manager outside of LangGraph's context.

### Future Examples (Coming Soon)

- Persistent Store Example: Using persistent storage for memories
- Multi-User Memory Example: Managing memories for multiple users
- Memory Migration Example: Moving memories between different stores

## Running the Examples

1. Make sure you have the required dependencies installed:

```bash
uv venv
source .venv/bin/activate
uv sync
```

2. Set up your OpenAI API key:

```bash
export OPENAI_API_KEY=your_api_key_here
```

3. Run any example:

```bash
cd examples/standalone_examples
uv run custom_store_example.py
```

## What You'll Learn

- How to use LangMem independently of LangGraph
- How to integrate LangMem with your own infrastructure
- How to customize memory storage and retrieval
- Best practices for memory management in production

## Notes

- The examples use OpenAI's embedding model. Make sure you have appropriate API access.
- Some examples use InMemoryStore for demonstration. In production, you might want to use a persistent store.
- Each example is self-contained and includes detailed comments explaining the implementation.
