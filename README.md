# Council MCP

An MCP (Model Context Protocol) server that provides access to a "council" of LLM models, enabling AI-assisted editors like Claude to consult multiple models for diverse opinions, peer-ranked evaluations, and synthesized answers.

## Why This Exists

When working with an AI coding assistant, you're getting one model's perspective. Sometimes that's exactly what you need. But for important decisions—architecture choices, code review, debugging complex issues, or creative writing—a plurality of opinions can surface blind spots and alternative approaches.

**Council MCP brings democratic AI consensus directly into your editing workflow.**

Instead of manually querying multiple AI services, you can ask Claude (or any MCP-compatible client) to consult the council, get ranked opinions from multiple frontier models, and receive a synthesized answer that represents collective AI wisdom.

> Inspired by [Andrej Karpathy's llm-council](https://github.com/karpathy/llm-council) concept—a Saturday hack for exploring LLM comparisons. This project reimplements the core ideas as an MCP server for seamless integration with AI-assisted development tools.

## How It Works

The council operates in up to 3 stages:

```
┌─────────────────────────────────────────────────────────────────┐
│  Stage 1: OPINIONS                                              │
│  Query multiple LLMs in parallel for independent responses      │
│  (GPT, Claude, Gemini, Kimi, etc.)                             │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  Stage 2: PEER RANKING                                          │
│  Each model anonymously evaluates and ranks all responses       │
│  Aggregate "street cred" scores reveal best performers          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  Stage 3: SYNTHESIS                                             │
│  Chairman model synthesizes final answer from collective wisdom │
│  Consensus level reported (strong/moderate/weak/split)          │
│  Tiebreaker vote cast if council is split                       │
└─────────────────────────────────────────────────────────────────┘
```

## Features

- **Tiered queries**: Choose cost/depth tradeoff (quick | ranked | full)
- **Consensus protocol**: Detects agreement level, triggers tiebreaker on splits
- **Odd council size**: Ensures tiebreaker votes can break deadlocks
- **Rotating chairmanship**: Weekly rotation prevents single-model bias
- **Chairman presets**: Context-aware chairman selection (code, creative, reasoning)
- **Cost estimation**: Know what you'll spend before querying

## Installation

### Prerequisites

1. Get an OpenRouter API key from https://openrouter.ai/keys
2. Add credits to your OpenRouter account (pay-as-you-go)

### Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/council-mcp.git
cd council-mcp

# Create .env file with your API key
echo "OPENROUTER_API_KEY=sk-or-v1-your-key-here" > .env

# Install dependencies
uv sync
```

### Configure Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "council": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/council-mcp", "python", "server.py"],
      "env": {
        "OPENROUTER_API_KEY": "sk-or-v1-your-key-here"
      }
    }
  }
}
```

### Configure Claude Code

Add to your Claude Code MCP settings:

```json
{
  "council": {
    "command": "uv",
    "args": ["run", "--directory", "/path/to/council-mcp", "python", "server.py"],
    "env": {
      "OPENROUTER_API_KEY": "sk-or-v1-your-key-here"
    }
  }
}
```

## Available Tools

### `council_quick`

Fast parallel opinions (Stage 1 only). Queries all council models and returns individual responses.

**Cost**: ~$0.01-0.03 per query

**Use for**: Quick brainstorming, getting diverse perspectives fast

### `council_ranked`

Opinions with peer rankings (Stage 1 + 2). Shows which model performed best on this specific question.

**Cost**: ~$0.05-0.10 per query

**Use for**: Code review, comparing approaches, seeing which model "won"

### `council_full`

Complete council with synthesis (all 3 stages). Includes consensus detection and chairman tiebreaker.

**Cost**: ~$0.10-0.20 per query

**Options**:
- `chairman`: Override chairman model (e.g., `"anthropic/claude-sonnet-4"`)
- `chairman_preset`: Use a preset (`"code"`, `"creative"`, `"reasoning"`, `"concise"`, `"balanced"`)

**Use for**: Important decisions, architecture choices, complex debugging

### `council_config`

View current configuration: council members, chairman rotation status, consensus thresholds.

### `council_estimate`

Estimate costs before running a query.

## Configuration

Edit `config.py` to customize:

### Council Members

```python
# Default council (4 members + 1 chairman = 5, odd)
COUNCIL_MODELS = [
    "openai/gpt-4.1",
    "anthropic/claude-sonnet-4",
    "google/gemini-2.5-pro",
    "moonshotai/kimi-k2",
]
```

### Chairman Rotation

```python
CHAIRMAN_ROTATION_ENABLED = True
CHAIRMAN_ROTATION_DAYS = 7  # Rotate weekly

CHAIRMAN_POOL = [
    "google/gemini-2.5-pro",
    "anthropic/claude-sonnet-4",
    "openai/gpt-4.1",
    "moonshotai/kimi-k2",
]
```

### Consensus Thresholds

```python
CONSENSUS_STRONG_THRESHOLD = 0.75   # 75%+ agreement
CONSENSUS_MODERATE_THRESHOLD = 0.50  # 50-75% agreement
CHAIRMAN_TIEBREAKER_ENABLED = True   # Chairman breaks ties
```

### Budget-Friendly Council

```python
COUNCIL_MODELS = [
    "deepseek/deepseek-chat",      # Very cheap
    "google/gemini-2.5-flash",     # Fast and cheap
    "moonshotai/kimi-k2",          # Good value
    "qwen/qwen3-32b",              # Open weights
]
```

## Use Cases

| Scenario | Recommended Tool | Why |
|----------|------------------|-----|
| "Review this function" | `council_ranked` | See which model catches the most issues |
| "Redis vs PostgreSQL for sessions?" | `council_full` | Important decision, need synthesis |
| "Ideas for this feature" | `council_quick` | Fast diverse brainstorming |
| "Debug this error" | `council_quick` | Quick parallel diagnosis |
| "Rewrite this paragraph" | `council_full` + `chairman_preset="creative"` | Creative synthesis |
| "Is this architecture sound?" | `council_full` + `chairman_preset="code"` | Technical synthesis |

## Example Output

```
## Council Full Result

**Consensus: ✅ STRONG** (75% agreement)

---

### Chairman's Synthesis

_Chairman: google/gemini-2.5-pro_

[Synthesized answer incorporating best points from all models...]

---

### Model Rankings (lower is better)

1. **claude-sonnet-4**: 1.50
2. **gpt-4.1**: 2.00
3. **gemini-2.5-pro**: 2.75
4. **kimi-k2**: 3.75

_First-place votes:_ claude-sonnet-4=3, gpt-4.1=1
```

## Adding Models

OpenRouter supports 200+ models. Find model IDs at https://openrouter.ai/models

```python
# Add to COUNCIL_MODELS in config.py
"x-ai/grok-3"                    # xAI Grok
"meta-llama/llama-4-maverick"    # Meta Llama
"mistralai/mistral-large-2"      # Mistral
"deepseek/deepseek-r1"           # DeepSeek reasoning
```

## How OpenRouter Works

OpenRouter is a unified API gateway—you don't need separate accounts with OpenAI, Google, Anthropic, etc. One API key, one credit balance, access to all models.

- Sign up: https://openrouter.ai
- Add credits (prepaid, or enable auto-top-up)
- Use your single API key for all models

## License

MIT

## Attribution

Inspired by [Andrej Karpathy's llm-council](https://github.com/karpathy/llm-council). The original is a web application for interactively exploring LLM comparisons. This project reimplements the council concept as an MCP server for integration with AI-assisted editors, adding consensus protocol and tiebreaker mechanics.
