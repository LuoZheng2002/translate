# Model Interfaces - Quick Start Guide

## Load Functions

```python
import json

# From file
with open("function_call.txt") as f:
    functions = json.load(f)

# Or directly
functions = [
    {
        "name": "triangle_properties.get",
        "description": "...",
        "parameters": {...}
    }
]
```

## API Models (GPT-4o, Claude, DeepSeek)

### Single Inference
```python
from models.model_factory import create_model_interface
from config import ApiModel

# Create interface
interface = create_model_interface(ApiModel.GPT_4O_MINI)

# Inference
raw_output = interface.infer(
    functions=functions,
    user_query="Can I find triangle properties for sides 3, 4, 5?"
)

# Parse
parsed = interface.parse_output(raw_output)
```

### Models Available
- `ApiModel.GPT_4O_MINI` → GPT4oMiniInterface
- `ApiModel.CLAUDE_SONNET` → ClaudeSonnetInterface
- `ApiModel.CLAUDE_HAIKU` → ClaudeHaikuInterface
- `ApiModel.DEEPSEEK_CHAT` → DeepseekChatInterface

## Granite Model (Local)

### Single Inference
```python
from models.model_factory import create_model_interface
from config import LocalModelStruct, LocalModel
from call_llm import make_chat_pipeline

# Setup
generator = make_chat_pipeline(LocalModel.GRANITE_3_1_8B_INSTRUCT)
config = LocalModelStruct(model=LocalModel.GRANITE_3_1_8B_INSTRUCT, generator=generator)
interface = create_model_interface(config, generator=generator)

# Inference
raw_output = interface.infer(
    functions=functions,
    user_query="Can I find triangle properties for sides 3, 4, 5?"
)

# Parse
parsed = interface.parse_output(raw_output)
```

### Batch Inference (Most Efficient)
```python
# Prepare batch data
functions_list = [functions] * 3
user_queries = [
    "Triangle properties for 3,4,5",
    "Circle area for radius 5",
    "Circle circumference for radius 10"
]

# Batch inference
batch_results = interface.infer_batch(
    functions_list=functions_list,
    user_queries=user_queries,
    prompt_passing_in_english=True
)

# Parse each
for raw_output in batch_results:
    parsed = interface.parse_output(raw_output)
    print(parsed)
```

## Method Signatures

### All Models
```python
def infer(self,
         functions: List[Dict[str, Any]],
         user_query: str,
         prompt_passing_in_english: bool = True,
         model=None) -> str
```

### All Models
```python
def parse_output(self, raw_output: str) -> Union[List[Dict], str]
# Returns: [{func_name: {arguments}}, ...] or error string
```

### Granite Only - Batch
```python
def infer_batch(self,
               functions_list: List[List[Dict[str, Any]]],
               user_queries: List[str],
               prompt_passing_in_english: bool = True) -> List[str]
```

### Granite Only - Explicit System Prompt
```python
def infer_with_functions(self,
                        system_prompt: str,
                        user_query: str,
                        functions: List[Dict[str, Any]]) -> str
```

## System Prompt Generation

Models automatically generate appropriate system prompts from functions:

**API Models:**
```
You are an expert in composing functions. You are given a question and a set of
possible functions. Based on the question, you will need to make one or more
function/tool calls to achieve the purpose...

You should only return the function calls in your response.

If you decide to invoke any of the function(s), you MUST put it in the format of
[func_name1(param1=value1, param2=value2), func_name2(...)].
You SHOULD NOT include any other text in the response.

Here is a list of functions in json format that you can invoke.
[functions in JSON]
```

**Granite Model:**
```
You are an expert in composing functions. You are given a question and a set of
possible functions. Based on the question, you will need to make one or more
function/tool calls to achieve the purpose...

You should only return the function calls in your response, in JSON format as a
list where each element has the format {"name": "function_name", "arguments":
{param1: value1, param2: value2, ...}}.

Here is a list of functions in json format that you can invoke.
[functions in JSON]
```

## Output Format (Same for All Models)

```python
[
    {"function_name": {"param1": value1, "param2": value2}},
    {"another_function": {"param3": value3}}
]
```

## Error Handling

```python
result = interface.infer(functions=functions, user_query=query)
parsed = interface.parse_output(result)

if isinstance(parsed, str):
    # It's an error message
    print(f"Parse error: {parsed}")
else:
    # It's valid parsed output
    for func_call in parsed:
        func_name = next(iter(func_call))
        arguments = func_call[func_name]
        print(f"Call: {func_name}({arguments})")
```

## Common Parameters

### `prompt_passing_in_english`
- **Type:** bool (default: True)
- **Effect:** Adds "Pass in all parameters in function calls in English" to system prompt
- **Use when:** Working with multilingual prompts

### `model`
- **Type:** LocalModel enum (optional)
- **Effect:** For API models, kept for interface compatibility; for Granite, used in system prompt generation
- **Note:** Usually not needed - system prompts are generated automatically

## Comparing Models

| Aspect | API Models | Granite |
|--------|---|---|
| **Setup** | Just create interface | Need generator + config |
| **Batch** | Sequential (concurrent inside) | True batch processing |
| **Speed** | Depends on API latency | Local, fastest option |
| **Output Format** | Python syntax | JSON syntax |
| **System Prompt** | Python function calls | JSON function calls |

## Main.py Integration

```python
# Setup
model_interface = create_model_interface(model)
if isinstance(model, LocalModelStruct):
    # Get/create generator
    generator = get_or_create_pipeline(model.model)
    model_interface = create_model_interface(model, generator=generator)

# Inference
for case in batch_cases:
    raw_output = model_interface.infer(
        functions=case['function'],
        user_query=case["question"][0][0]['content'],
        prompt_passing_in_english=True
    )

    # Parse
    parsed = model_interface.parse_output(raw_output)

    # Store result
    result = {"id": case["id"], "result": raw_output}
    inference_results.append(result)
```

## Files Reference

- **base.py** - Abstract ModelInterface class
- **gpt_4o_mini_interface.py** - GPT-4o-mini handler
- **claude_sonnet_interface.py** - Claude Sonnet handler
- **claude_haiku_interface.py** - Claude Haiku handler
- **deepseek_chat_interface.py** - DeepSeek handler
- **granite_3_1_8b_instruct_interface.py** - Granite handler
- **model_factory.py** - Factory function create_model_interface()

## Documentation Files

- **INFER_INTERFACE_UPDATE.md** - Detailed interface changes
- **PARSING_STRATEGY.md** - Output parsing logic details
- **PARSE_STRATEGY_QUICK_REF.md** - Parsing reference guide
- **USAGE_GUIDE.md** - Comprehensive usage guide
