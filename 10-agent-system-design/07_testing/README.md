# Module 7: Testing Agent Systems

Beyond unit tests: testing non-deterministic systems.

## Overview

Agent systems are hard to test. They're non-deterministic, multi-step, and interact with external services. Traditional testing approaches need adaptation.

## Key Topics

### The Testing Pyramid for Agents

```
        /\
       /  \  End-to-end (few, expensive, flaky)
      /----\
     /      \ Behavioral evals (core layer)
    /--------\
   /          \ Tool & integration tests
  /------------\
 /              \ Unit tests (many, fast, deterministic)
```

### Unit Testing Tool Implementations
```python
def test_search_tool():
    # Tools should be deterministic and testable
    result = search_tool.execute({"query": "test"})
    assert "results" in result
    assert isinstance(result["results"], list)
```

### Mocking LLM Responses
```python
@patch('openai.ChatCompletion.create')
def test_agent_flow(mock_llm):
    mock_llm.return_value = {
        "choices": [{"message": {"content": "search_database(query='test')"}}]
    }
    result = agent.run("Find test data")
    assert agent.tools_called == ["search_database"]
```

### Behavioral Testing
Test what the agent *does*, not what it *says*:
```python
def test_agent_uses_correct_tool():
    result = agent.run("What's the weather in NYC?")
    assert "get_weather" in result.tools_called
    assert result.tools_called["get_weather"]["location"] == "NYC"
```

### Trajectory Testing
Was the path reasonable?
```python
def test_efficient_trajectory():
    result = agent.run("Simple math: 2+2")
    # Should NOT call search tools for basic math
    assert "web_search" not in result.tools_called
    assert len(result.steps) < 3  # Should be quick
```

### Regression Testing
```python
# Save known-good trajectories
def test_regression_suite():
    for case in load_regression_cases():
        result = agent.run(case.input)
        assert result.final_answer == case.expected_answer
        assert set(result.tools_called) == set(case.expected_tools)
```

### Property-Based Testing
```python
from hypothesis import given, strategies as st

@given(st.text(min_size=1, max_size=100))
def test_agent_never_crashes(query):
    # Agent should handle any input gracefully
    result = agent.run(query)
    assert result.status in ["success", "error", "timeout"]
    assert result.error_message if result.status == "error" else True
```

### Integration Testing
Test tool chains work together:
```python
def test_search_then_summarize():
    result = agent.run("Summarize recent AI news")
    # Should search, then summarize
    assert result.tools_called[0] == "web_search"
    assert result.tools_called[-1] == "summarize"
```

## Test Environment Setup

- Sandboxed tool implementations
- Deterministic random seeds
- Recorded/replayed API responses
- Cost-free test mode

## Exercises

1. Write unit tests for a set of agent tools
2. Build a mock LLM for testing
3. Implement behavioral assertions
4. Create a regression test suite

## Key Insight

Test behavior, not text. "Did it call the right tool with right args?" is a better assertion than "Did it say the right words?"
