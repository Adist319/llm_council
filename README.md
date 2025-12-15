# LM-Council MCP Server

An MCP (Model Context Protocol) server that exposes the [lm-council](https://github.com/machine-theory/lm-council) library for democratic LLM evaluation. Multiple language models evaluate each other's responses and vote on the best performers.

## Features

- **Democratic Evaluation**: LLMs judge each other's responses using configurable rubrics
- **Two Evaluation Modes**: Direct assessment (rubric scoring) or pairwise comparison (ELO rankings)
- **Built-in Rubric Presets**: Default, LinkedIn content, and code review rubrics
- **Session Management**: Save and restore evaluation sessions
- **Analysis Tools**: Judge agreement, affinity analysis, and win rate heatmaps
- **HuggingFace Integration**: Publish results to HuggingFace Hub

## Installation

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/Adist319/llm_council.git
cd llm_council
uv sync
```

## Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENROUTER_API_KEY` | Yes | Your OpenRouter API key |
| `HF_TOKEN` | No | HuggingFace token (for uploads) |
| `COUNCIL_STORAGE_PATH` | No | Session storage directory (default: `.council_sessions`) |

### Claude Desktop Setup

Add to your Claude Desktop configuration (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "lm-council": {
      "command": "uv",
      "args": ["--directory", "/path/to/llm_council", "run", "lm-council-mcp"],
      "env": {
        "OPENROUTER_API_KEY": "your-api-key-here"
      }
    }
  }
}
```

## Available Tools

| Tool | Description |
|------|-------------|
| `council_configure` | Set models, judges, evaluation mode, and rubric preset |
| `council_execute` | Run evaluation on one or more prompts |
| `council_get_leaderboard` | Get model rankings (JSON format) |
| `council_save_session` | Save current session to disk |
| `council_load_session` | Restore a saved session |
| `council_get_judge_agreement` | Analyze inter-rater agreement |
| `council_get_affinity_analysis` | Detect judge biases |
| `council_get_win_rate_heatmap` | Get pairwise win rates (pairwise mode only) |
| `council_upload_to_huggingface` | Publish results to HuggingFace Hub |

## Usage Examples

### Basic Evaluation

```
Configure the council with GPT-4, Claude, and Gemini as both respondents and judges.
Then evaluate the prompt: "Write a haiku about artificial intelligence"
```

### LinkedIn Content Evaluation

```
Configure the council with the linkedin_content rubric preset.
Use these models: openai/gpt-4, anthropic/claude-3-opus, google/gemini-pro
Evaluate: "Write a LinkedIn post about the importance of continuous learning"
```

### Pairwise Comparison Mode

```
Configure the council in pairwise_comparison mode with all_pairs algorithm.
Execute on multiple prompts, then get the leaderboard to see ELO rankings.
```

## Rubric Presets

### `default_rubric`
General-purpose evaluation: relevance, accuracy, clarity, completeness

### `linkedin_content`
LinkedIn post optimization: hook strength, value delivery, authenticity, engagement potential, formatting, professional tone

### `code_review`
Code/technical content: technical accuracy, clarity, completeness, practical applicability

## Development

### Run Tests

```bash
uv run pytest tests/ -v
```

### Run Server Directly

```bash
uv run lm-council-mcp
```
