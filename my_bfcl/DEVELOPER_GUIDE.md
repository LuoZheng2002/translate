# Developer Guide - Model Interface System

## Quick Start

### Using the New Model Interface

```python
from models.model_factory import create_model_interface
from config import ApiModel, LocalModelStruct, LocalModel

# Get functions from test case
functions = test_case['function']
user_query = test_case['question'][0][0]['content']

# Create interface for any model
interface = create_model_interface(ApiModel.GPT_4O_MINI)

# Run inference
raw_output = interface.infer(
    functions=functions,
    user_query=user_query,
    prompt_passing_in_english=True
)

# Parse output
parsed = interface.parse_output(raw_output)
```

## Model-Specific Usage

### API Models (GPT-4o, Claude, DeepSeek)

```python
from models.model_factory import create_model_interface
from config import ApiModel

# Create interface
interface = create_model_interface(ApiModel.GPT_4O_MINI)
# or ApiModel.CLAUDE_SONNET
# or ApiModel.CLAUDE_HAIKU
# or ApiModel.DEEPSEEK_CHAT

# Single inference
result = interface.infer(
    functions=functions,
    user_query=query,
    prompt_passing_in_english=True
)

# Parse (returns dict list)
parsed = interface.parse_output(result)
# Format: [{func_name: {arg1: val1, ...}}, ...]
```

### Local Model (Granite)

```python
from models.model_factory import create_model_interface
from config import LocalModelStruct, LocalModel
from call_llm import make_chat_pipeline

# Setup
generator = make_chat_pipeline(LocalModel.GRANITE_3_1_8B_INSTRUCT)
config = LocalModelStruct(model=LocalModel.GRANITE_3_1_8B_INSTRUCT, generator=generator)
interface = create_model_interface(config, generator=generator)

# Single inference
result = interface.infer(
    functions=functions,
    user_query=query,
    prompt_passing_in_english=True
)

# Batch inference (recommended for efficiency)
batch_results = interface.infer_batch(
    functions_list=[functions] * 3,
    user_queries=[query1, query2, query3],
    prompt_passing_in_english=True
)

# Parse (same format as API models)
for result in batch_results:
    parsed = interface.parse_output(result)
    print(parsed)
```

## Understanding the Interface

### Method: `infer()`

```python
def infer(self,
         functions: List[Dict[str, Any]],
         user_query: str,
         prompt_passing_in_english: bool = True,
         model=None) -> str:
    """
    Run inference with the model.

    Args:
        functions: List of function definitions in JSON format
        user_query: User query/question as a string
        prompt_passing_in_english: Add English parameter instruction (default: True)
        model: Optional model type (kept for compatibility)

    Returns:
        Raw model output as a string
    """
```

### Method: `parse_output()`

```python
def parse_output(self, raw_output: str) -> Union[List[Dict], str]:
    """
    Parse raw model output.

    Args:
        raw_output: Raw string from infer()

    Returns:
        List of [{func_name: {arguments}}] on success
        Error string on failure
    """
```

### Error Handling

```python
# Parse output
parsed = interface.parse_output(raw_output)

# Check for error
if isinstance(parsed, str):
    print(f"Parse error: {parsed}")
else:
    for func_call in parsed:
        func_name = list(func_call.keys())[0]
        arguments = func_call[func_name]
        print(f"Call: {func_name}({arguments})")
```

## Understanding System Prompts

### How Prompts Are Generated

Each model has a `_generate_system_prompt()` method that creates appropriate prompts:

**API Models** (GPT-4o, Claude, DeepSeek):
- Generates prompt requesting Python function call syntax
- Example: `[func_name1(param1=value1, param2=value2)]`
- Uses AST parsing strategy

**Granite Model**:
- Generates prompt requesting JSON function call syntax
- Example: `[{"name": "func_name1", "arguments": {...}}]`
- Uses JSON parsing strategy

### Customizing Prompts (Advanced)

If you need explicit system prompt control, Granite model supports:

```python
# Use custom system prompt
result = interface.infer_with_functions(
    system_prompt="Your custom prompt here",
    user_query=query,
    functions=functions
)

# Batch with custom prompts
results = interface.infer_batch_with_functions(
    system_prompts=[prompt1, prompt2, ...],
    user_queries=[query1, query2, ...],
    batch_functions=[funcs1, funcs2, ...]
)
```

## Understanding the Factory

The `model_factory.py` handles model instantiation:

```python
from models.model_factory import create_model_interface

# For API models
interface = create_model_interface(ApiModel.GPT_4O_MINI)

# For local models (requires generator)
interface = create_model_interface(local_config, generator=generator)
```

## Parsing Strategies

### API Models - AST Parsing

API models output Python function call syntax and parse using Python's AST module:

```
Input:  "Your function call here"
Output: [func_name1(param1=value1, param2=value2)]
Parsed: [{func_name1: {param1: value1, param2: value2}}]
```

Key features:
- Handles nested function calls
- Supports various Python types (int, str, list, dict, bool, etc.)
- References parse_ast.py lines 170-189

### Granite Model - JSON Parsing

Granite model outputs JSON function call format:

```
Input:  "<tool_call>[{"name": "func_name1", "arguments": {...}}]"
Output: [{"name": "func_name1", "arguments": {...}}]
Parsed: [{func_name1: {...}}]
```

Key features:
- Handles `<tool_call>` wrapper
- Strips backticks and whitespace
- Converts JSON format to unified format
- References parse_ast.py lines 131-167

## File Organization

```
models/
├── base.py                        # Abstract interface
├── model_factory.py              # Factory function
├── gpt_4o_mini_interface.py      # GPT-4o-mini implementation
├── claude_sonnet_interface.py    # Claude Sonnet implementation
├── claude_haiku_interface.py     # Claude Haiku implementation
├── deepseek_chat_interface.py    # DeepSeek implementation
├── granite_3_1_8b_instruct_interface.py  # Granite implementation
├── PARSING_STRATEGY.md           # Detailed parsing docs
└── PARSE_STRATEGY_QUICK_REF.md   # Quick parsing reference
```

## Adding a New Model

To add a new model to the system:

1. Create `models/new_model_interface.py`
2. Implement the base interface:
   ```python
   from models.base import ModelInterface

   class NewModelInterface(ModelInterface):
       def __init__(self):
           # Initialize your model/client
           pass

       def infer(self, functions, user_query, prompt_passing_in_english=True, model=None):
           system_prompt = self._generate_system_prompt(functions, prompt_passing_in_english)
           # Call your model
           return raw_output

       def parse_output(self, raw_output):
           # Parse model output
           return parsed_output

       def _generate_system_prompt(self, functions, prompt_passing_in_english=True):
           # Generate appropriate prompt
           return system_prompt
   ```

3. Update `models/model_factory.py` to handle your new model:
   ```python
   def create_model_interface(model, generator=None):
       if isinstance(model, NewModel):
           return NewModelInterface()
       # ... other models
   ```

## Common Patterns

### Single Query Inference

```python
functions = test_case['function']
query = test_case['question'][0][0]['content']
interface = create_model_interface(model)
raw = interface.infer(functions=functions, user_query=query)
parsed = interface.parse_output(raw)
```

### Batch Processing (API Models)

```python
interface = create_model_interface(api_model)
results = []
for case in cases:
    raw = interface.infer(
        functions=case['function'],
        user_query=case['question'][0][0]['content']
    )
    results.append(interface.parse_output(raw))
```

### Batch Processing (Granite)

```python
interface = create_model_interface(granite_config, generator=gen)
functions_list = [case['function'] for case in cases]
queries = [case['question'][0][0]['content'] for case in cases]
batch_results = interface.infer_batch(
    functions_list=functions_list,
    user_queries=queries
)
parsed_results = [interface.parse_output(r) for r in batch_results]
```

## Troubleshooting

### Parse Error

If `parse_output()` returns an error string:
1. Check the raw model output for expected format
2. Verify the model produced valid Python syntax (API) or JSON (Granite)
3. Check for special characters that need escaping

### API Errors

If API model inference fails:
1. Verify API key is set in `.env` file
2. Check network connectivity
3. Verify model name is correct for the API

### Local Model Issues

If Granite inference fails:
1. Verify generator is initialized with `make_chat_pipeline()`
2. Check GPU memory availability
3. Verify functions list is properly formatted JSON

## Performance Tips

1. **Granite Batch Processing**: Use `infer_batch()` for better efficiency
2. **Prompt Passing**: Set `prompt_passing_in_english=True` only if needed
3. **Error Handling**: Cache parsed results to avoid re-parsing
4. **Memory**: Clean up generators between configs for local models

## References

- `models_QUICK_START.md` - Quick reference for all models
- `INFER_INTERFACE_UPDATE.md` - Interface signature details
- `PARSING_STRATEGY.md` - Detailed parsing documentation
- `MAIN_PY_MIGRATION.md` - Examples of migration
