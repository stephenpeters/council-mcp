#!/usr/bin/env python3
"""
Conclave MCP Server

Exposes the Conclave (LLM Council) as MCP tools for use with Claude Desktop/Code.

Tools:
- conclave_quick: Fast parallel opinions (Stage 1 only)
- conclave_ranked: Opinions + peer rankings (Stage 1 + 2)
- conclave_full: Complete conclave with synthesis (all 3 stages)
- conclave_config: View/check current configuration
- conclave_estimate: Estimate cost before running
- conclave_models: List all available models with selection numbers
- conclave_select: Create custom conclave from model numbers
- conclave_reset: Clear custom selection and return to tier-based config
"""

import json
from typing import Optional

from mcp.server.fastmcp import FastMCP

from config import (
    OPENROUTER_API_KEY,
    COUNCIL_MODELS,
    COUNCIL_PREMIUM,
    COUNCIL_STANDARD,
    COUNCIL_BUDGET,
    CHAIRMAN_POOL,
    CHAIRMAN_PRESETS,
    CONSENSUS_STRONG_THRESHOLD,
    CONSENSUS_MODERATE_THRESHOLD,
    CHAIRMAN_TIEBREAKER_ENABLED,
    get_current_chairman,
    get_rotation_info,
    estimate_cost,
    validate_council_size,
    get_council_by_tier,
    get_tier_info,
)
from conclave import (
    run_council_quick,
    run_council_ranked,
    run_council_full,
)

# Initialize FastMCP server
mcp = FastMCP("conclave")


# =============================================================================
# CUSTOM CONCLAVE SELECTION (in-memory, persists until restart)
# =============================================================================

# Custom conclave state - None means use tier-based config
_custom_conclave: dict | None = None


def get_all_models_numbered() -> list[tuple[int, str, str]]:
    """
    Get all available models with unique numbers.

    Returns list of (number, model_id, tier) tuples.
    Numbers are stable: Premium 1-10, Standard 11-20, Budget 21-30, Chairman 31-40
    """
    models = []

    # Premium tier: 1-10
    for i, model in enumerate(COUNCIL_PREMIUM, start=1):
        models.append((i, model, "premium"))

    # Standard tier: 11-20
    for i, model in enumerate(COUNCIL_STANDARD, start=11):
        models.append((i, model, "standard"))

    # Budget tier: 21-30
    for i, model in enumerate(COUNCIL_BUDGET, start=21):
        models.append((i, model, "budget"))

    # Chairman pool: 31-40
    for i, model in enumerate(CHAIRMAN_POOL, start=31):
        models.append((i, model, "chairman"))

    return models


def get_model_by_number(num: int) -> str | None:
    """Get model ID by its number."""
    for n, model, _ in get_all_models_numbered():
        if n == num:
            return model
    return None


def get_active_models() -> tuple[list[str], str | None, str]:
    """
    Get the currently active models and chairman.

    Returns:
        (models, chairman, source) where source is "custom" or "tier"
    """
    global _custom_conclave

    if _custom_conclave:
        return (
            _custom_conclave["models"],
            _custom_conclave["chairman"],
            "custom"
        )

    # Default to standard tier
    return (COUNCIL_STANDARD, None, "tier")


# =============================================================================
# TOOLS
# =============================================================================

@mcp.tool()
async def conclave_quick(question: str, tier: str = "standard") -> str:
    """Query the conclave for quick parallel opinions (Stage 1 only).

    Fast and cheap - queries all conclave models in parallel and returns their
    individual responses. No peer ranking or synthesis. Good for getting
    diverse perspectives quickly.

    If a custom conclave is active (via conclave_select), it will be used
    instead of the tier-based config.

    Args:
        question: The question to ask the conclave
        tier: Model tier - "premium" (frontier), "standard" (default), "budget" (cheap/fast)
              Ignored if custom conclave is active.

    Returns:
        Individual responses from each conclave model
    """
    if not question:
        return "Error: 'question' is required"

    if not OPENROUTER_API_KEY:
        return "Error: OPENROUTER_API_KEY not configured"

    # Check for custom conclave
    custom_models, custom_chairman, source = get_active_models()

    if source == "custom":
        models = custom_models
        tier_label = "custom"
    else:
        if tier not in ("premium", "standard", "budget"):
            tier = "standard"
        models = get_council_by_tier(tier)
        tier_label = tier

    try:
        result = await run_council_quick(question, models=models)
        result["tier"] = tier_label
        return format_quick_result(result)
    except Exception as e:
        return f"Error: {str(e)}"


@mcp.tool()
async def conclave_ranked(question: str, tier: str = "standard") -> str:
    """Query the conclave with peer rankings (Stage 1 + 2).

    Medium cost - gets individual opinions, then has each model anonymously
    evaluate and rank all responses. Returns aggregate "street cred" scores
    showing which models performed best on this specific question.

    If a custom conclave is active (via conclave_select), it will be used
    instead of the tier-based config.

    Args:
        question: The question to ask the conclave
        tier: Model tier - "premium" (frontier), "standard" (default), "budget" (cheap/fast)
              Ignored if custom conclave is active.

    Returns:
        Individual responses plus aggregate rankings
    """
    if not question:
        return "Error: 'question' is required"

    if not OPENROUTER_API_KEY:
        return "Error: OPENROUTER_API_KEY not configured"

    # Check for custom conclave
    custom_models, custom_chairman, source = get_active_models()

    if source == "custom":
        models = custom_models
        tier_label = "custom"
    else:
        if tier not in ("premium", "standard", "budget"):
            tier = "standard"
        models = get_council_by_tier(tier)
        tier_label = tier

    try:
        result = await run_council_ranked(question, models=models)
        result["tier"] = tier_label
        return format_ranked_result(result)
    except Exception as e:
        return f"Error: {str(e)}"


@mcp.tool()
async def conclave_full(
    question: str,
    tier: str = "standard",
    chairman: Optional[str] = None,
    chairman_preset: Optional[str] = None,
) -> str:
    """Run the full conclave with synthesis (all 3 stages).

    Most comprehensive - collects opinions, peer rankings, then has a Chairman
    model synthesize the best possible answer from the collective wisdom.

    If a custom conclave is active (via conclave_select), it will be used
    instead of the tier-based config. The custom chairman overrides the
    chairman and chairman_preset parameters.

    Args:
        question: The question to ask the conclave
        tier: Model tier - "premium" (complex), "standard" (default), "budget" (simple)
              Ignored if custom conclave is active.
        chairman: Override chairman model (e.g., 'anthropic/claude-sonnet-4')
                  Ignored if custom conclave is active.
        chairman_preset: Use a context-based preset - "code", "creative", "reasoning", "concise", "balanced"
                         Ignored if custom conclave is active.

    Returns:
        Chairman's synthesis, consensus level, rankings, and individual responses
    """
    if not question:
        return "Error: 'question' is required"

    if not OPENROUTER_API_KEY:
        return "Error: OPENROUTER_API_KEY not configured"

    # Check for custom conclave
    custom_models, custom_chairman, source = get_active_models()

    if source == "custom":
        models = custom_models
        tier_label = "custom"
        # Custom chairman overrides parameters
        chairman = custom_chairman
        chairman_preset = None
    else:
        if tier not in ("premium", "standard", "budget"):
            tier = "standard"

        if chairman_preset and chairman_preset not in CHAIRMAN_PRESETS:
            return f"Error: Invalid chairman_preset. Valid options: {list(CHAIRMAN_PRESETS.keys())}"

        models = get_council_by_tier(tier)
        tier_label = tier

    try:
        result = await run_council_full(
            question,
            models=models,
            chairman=chairman,
            chairman_preset=chairman_preset,
        )
        result["tier"] = tier_label
        return format_full_result(result)
    except Exception as e:
        return f"Error: {str(e)}"


@mcp.tool()
async def conclave_config() -> str:
    """View current conclave configuration.

    Shows conclave member models, current chairman with rotation info,
    available chairman presets, consensus thresholds, and API key status.

    Also shows custom conclave selection if active.

    Returns:
        Current configuration as formatted JSON
    """
    rotation_info = get_rotation_info()

    # Check for custom conclave
    custom_models, custom_chairman, source = get_active_models()

    if source == "custom":
        size_validation = validate_council_size(custom_models, custom_chairman)
        active_config = {
            "mode": "custom",
            "chairman": custom_chairman,
            "members": custom_models,
            "size": {
                "total_members": size_validation["total_size"],
                "is_odd": size_validation["valid"],
                "status": size_validation["message"],
            },
        }
    else:
        size_validation = validate_council_size()
        active_config = {
            "mode": "tier-based",
            "default_tier": "standard",
            "active_models": COUNCIL_MODELS,
            "size": {
                "total_members": size_validation["total_size"],
                "is_odd": size_validation["valid"],
                "chairman_in_conclave": size_validation["chairman_included"],
                "status": size_validation["message"],
            },
        }

    config = {
        "api_key_configured": bool(OPENROUTER_API_KEY),
        "active_conclave": active_config,
        "tiers": get_tier_info(),
        "chairman": {
            "current": custom_chairman if source == "custom" else rotation_info["current_chairman"],
            "rotation_enabled": rotation_info["rotation_enabled"] if source != "custom" else False,
            "rotation_period_days": rotation_info["rotation_period_days"],
            "days_until_rotation": rotation_info["days_until_rotation"] if source != "custom" else "N/A (custom)",
            "chairman_pool": rotation_info["chairman_pool"],
        },
        "consensus": {
            "strong_threshold": f"{CONSENSUS_STRONG_THRESHOLD:.0%}",
            "moderate_threshold": f"{CONSENSUS_MODERATE_THRESHOLD:.0%}",
            "tiebreaker_enabled": CHAIRMAN_TIEBREAKER_ENABLED,
        },
        "presets": CHAIRMAN_PRESETS,
    }

    return f"## Conclave Configuration\n\n```json\n{json.dumps(config, indent=2)}\n```"


@mcp.tool()
async def conclave_estimate(question: str, tier: Optional[str] = None) -> str:
    """Estimate cost for a conclave query before running it.

    Provides approximate cost breakdown for quick/ranked/full query types.

    Args:
        question: The question (used to estimate token count)
        tier: Which tier to estimate - "quick", "ranked", "full" (default: all)

    Returns:
        Cost estimates for each query type
    """
    # Rough token estimate (4 chars per token)
    query_tokens = len(question) // 4 + 50

    tiers_to_estimate = [tier] if tier in ("quick", "ranked", "full") else ["quick", "ranked", "full"]

    estimates = {}
    for t in tiers_to_estimate:
        estimates[t] = estimate_cost(query_tokens, tier=t)

    output = "## Cost Estimates\n\n"
    for t, est in estimates.items():
        output += f"### {t.title()}\n"
        output += f"- Stage 1: ${est['stage1']:.4f}\n"
        output += f"- Stage 2: ${est['stage2']:.4f}\n"
        output += f"- Stage 3: ${est['stage3']:.4f}\n"
        output += f"- **Total: ${est['total']:.4f}**\n\n"

    output += f"_Estimates based on ~{query_tokens} input tokens. Actual costs may vary._"

    return output


@mcp.tool()
async def conclave_models() -> str:
    """List all available models with selection numbers.

    Shows all models from all tiers with unique numbers that can be used
    with conclave_select to create a custom conclave.

    Numbers are stable:
    - Premium tier: 1-10
    - Standard tier: 11-20
    - Budget tier: 21-30
    - Chairman pool: 31-40

    Returns:
        Numbered list of all available models grouped by tier
    """
    models = get_all_models_numbered()
    models_active, chairman_active, source = get_active_models()

    output = "## Available Models\n\n"

    # Show current selection status
    if source == "custom":
        output += f"**Current selection**: Custom conclave active\n"
        output += f"**Chairman**: {chairman_active}\n"
        output += f"**Members**: {', '.join(m.split('/')[-1] for m in models_active)}\n\n"
        output += "_Use `conclave_reset` to clear custom selection_\n\n"
    else:
        output += "_No custom selection active - using tier-based config_\n\n"

    output += "---\n\n"

    # Group by tier
    current_tier = None
    tier_labels = {
        "premium": "### Premium Tier (1-10)\n_Frontier models for complex questions_\n\n",
        "standard": "### Standard Tier (11-20)\n_Balanced models for typical questions_\n\n",
        "budget": "### Budget Tier (21-30)\n_Fast/cheap models for simple questions_\n\n",
        "chairman": "### Chairman Pool (31-40)\n_Reasoning models for synthesis_\n\n",
    }

    for num, model, tier in models:
        if tier != current_tier:
            current_tier = tier
            output += tier_labels[tier]

        model_short = model.split("/")[-1]
        # Mark if currently selected
        marker = ""
        if source == "custom":
            if model == chairman_active:
                marker = " ⭐ (chairman)"
            elif model in models_active:
                marker = " ✓ (selected)"

        output += f"  {num:2d}. `{model}`{marker}\n"

    output += "\n---\n\n"
    output += "**Usage**: `conclave_select(models=\"1,5,11,14\")` - first model becomes chairman\n"

    return output


@mcp.tool()
async def conclave_select(models: str) -> str:
    """Create a custom conclave from model numbers.

    Select specific models by their numbers (from conclave_models).
    The first model in the list becomes the chairman.

    This custom selection persists until server restart or conclave_reset.

    Args:
        models: Comma-separated model numbers, e.g. "1,5,11,14"
                First number = chairman, rest = conclave members

    Returns:
        Confirmation of the new conclave configuration

    Example:
        conclave_select(models="31,1,11,21") creates:
        - Chairman: model #31 (deepseek-r1)
        - Members: models #1, #11, #21
    """
    global _custom_conclave

    # Parse model numbers
    try:
        numbers = [int(n.strip()) for n in models.split(",") if n.strip()]
    except ValueError:
        return "Error: Invalid format. Use comma-separated numbers, e.g. '1,5,11,14'"

    if len(numbers) < 2:
        return "Error: Need at least 2 models (1 chairman + 1 member)"

    # Resolve model IDs
    resolved = []
    invalid = []
    for num in numbers:
        model = get_model_by_number(num)
        if model:
            resolved.append((num, model))
        else:
            invalid.append(num)

    if invalid:
        return f"Error: Invalid model numbers: {invalid}. Use `conclave_models` to see valid numbers."

    # First model is chairman
    chairman_num, chairman = resolved[0]
    members = [model for _, model in resolved[1:]]

    # Validate: check for duplicates
    all_models = [chairman] + members
    if len(all_models) != len(set(all_models)):
        return "Error: Duplicate models selected. Each model can only appear once."

    # Validate: warn if even number (no proper tiebreaker)
    total_size = len(members) + 1  # members + chairman
    is_odd = total_size % 2 == 1

    # Store custom config
    _custom_conclave = {
        "chairman": chairman,
        "models": members,
        "numbers": numbers,
    }

    # Build response
    output = "## Custom Conclave Created\n\n"

    if not is_odd:
        output += f"⚠️ **Warning**: Total size is {total_size} (even). Add/remove 1 model for proper tiebreaker support.\n\n"

    output += f"**Chairman** (#{chairman_num}): `{chairman}`\n\n"
    output += "**Members**:\n"
    for num, model in resolved[1:]:
        output += f"  - #{num}: `{model}`\n"

    output += f"\n**Total size**: {total_size} {'(odd ✓)' if is_odd else '(even ⚠️)'}\n\n"
    output += "_This selection persists until restart or `conclave_reset`_\n"
    output += "_Use `conclave_quick`, `conclave_ranked`, or `conclave_full` to query_"

    return output


@mcp.tool()
async def conclave_reset() -> str:
    """Clear custom conclave selection and return to tier-based config.

    After reset, queries will use the tier parameter (premium/standard/budget)
    instead of the custom model selection.

    Returns:
        Confirmation that custom selection was cleared
    """
    global _custom_conclave

    if _custom_conclave is None:
        return "No custom conclave was active. Already using tier-based config."

    old_chairman = _custom_conclave["chairman"]
    old_members = _custom_conclave["models"]
    _custom_conclave = None

    output = "## Custom Conclave Cleared\n\n"
    output += "**Previous selection removed**:\n"
    output += f"  - Chairman: `{old_chairman}`\n"
    output += f"  - Members: {', '.join(f'`{m}`' for m in old_members)}\n\n"
    output += "_Now using tier-based config (premium/standard/budget)_"

    return output


# =============================================================================
# RESULT FORMATTERS
# =============================================================================

def format_quick_result(result: dict) -> str:
    """Format quick conclave result for display."""
    output = "## Conclave Quick Opinions\n\n"

    for resp in result["stage1"]:
        model_name = resp["model"].split("/")[-1]  # Just the model name
        output += f"### {model_name}\n\n{resp['content']}\n\n---\n\n"

    return output


def format_ranked_result(result: dict) -> str:
    """Format ranked conclave result for display."""
    output = "## Conclave Opinions with Rankings\n\n"

    # Stage 1 responses
    output += "### Individual Responses\n\n"
    for resp in result["stage1"]:
        model_name = resp["model"].split("/")[-1]
        output += f"#### {model_name}\n\n{resp['content']}\n\n---\n\n"

    # Aggregate rankings
    output += "### Aggregate Rankings (lower is better)\n\n"
    sorted_rankings = sorted(
        result["stage2"]["aggregate"].items(),
        key=lambda x: x[1]
    )
    for i, (model, score) in enumerate(sorted_rankings, 1):
        model_name = model.split("/")[-1]
        output += f"{i}. **{model_name}**: {score:.2f} avg rank\n"

    return output


def format_full_result(result: dict) -> str:
    """Format full conclave result for display."""
    output = "## Conclave Full Result\n\n"

    # Consensus status badge
    consensus = result.get("consensus", {})
    consensus_level = consensus.get("level", "unknown")
    consensus_emoji = {
        "strong": "✅",
        "moderate": "🟡",
        "weak": "🟠",
        "split": "⚖️",
    }.get(consensus_level, "❓")

    output += f"**Consensus: {consensus_emoji} {consensus_level.upper()}**"
    if consensus.get("ranking_agreement"):
        output += f" ({consensus['ranking_agreement']:.0%} agreement)"
    output += "\n\n"

    # Tiebreaker info if used
    tiebreaker = result.get("tiebreaker")
    if tiebreaker and tiebreaker.get("valid_vote"):
        output += f"⚖️ **Tiebreaker Vote**: Chairman selected **{tiebreaker['vote'].split('/')[-1]}** (Response {tiebreaker['vote_label']})\n\n"

    # Council size warning if even
    size_info = result.get("council_size", {})
    if size_info and not size_info.get("valid", True):
        output += f"⚠️ {size_info.get('message', 'Council size is even')}\n\n"

    output += "---\n\n"

    # Final synthesis (most important)
    output += "### Chairman's Synthesis\n\n"
    output += f"_Chairman: {result['stage3']['chairman']}_\n"
    if result['stage3'].get('tiebreaker_used'):
        output += "_Tiebreaker vote was cast_\n"
    output += f"\n{result['stage3']['synthesis']}\n\n"
    output += "---\n\n"

    # Aggregate rankings
    output += "### Model Rankings (lower is better)\n\n"
    sorted_rankings = sorted(
        result["stage2"]["aggregate"].items(),
        key=lambda x: x[1]
    )
    for i, (model, score) in enumerate(sorted_rankings, 1):
        model_name = model.split("/")[-1]
        # Mark tied models
        is_tied = consensus_level == "split" and model in consensus.get("split_details", {}).get("tied_models", [])
        tie_marker = " ⚖️" if is_tied else ""
        output += f"{i}. **{model_name}**: {score:.2f}{tie_marker}\n"

    # First place vote distribution
    if consensus.get("first_place_votes"):
        output += "\n_First-place votes:_ "
        votes = [f"{m.split('/')[-1]}={v}" for m, v in consensus["first_place_votes"].items()]
        output += ", ".join(votes)
        output += "\n"

    output += "\n---\n\n"

    # Individual responses (collapsed by default in most renderers)
    output += "<details>\n<summary>Individual Responses</summary>\n\n"
    for resp in result["stage1"]:
        model_name = resp["model"].split("/")[-1]
        output += f"#### {model_name}\n\n{resp['content']}\n\n---\n\n"
    output += "</details>\n"

    # Tiebreaker reasoning if available
    if tiebreaker and tiebreaker.get("reasoning"):
        output += "\n<details>\n<summary>Tiebreaker Reasoning</summary>\n\n"
        output += f"{tiebreaker['reasoning']}\n"
        output += "</details>\n"

    return output


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    mcp.run(transport='stdio')
