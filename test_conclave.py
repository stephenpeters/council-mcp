#!/usr/bin/env python3
"""
Test script for conclave-mcp.

Run with: python test_conclave.py

Requires OPENROUTER_API_KEY in .env or environment.
"""

import asyncio
import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Check API key
if not os.getenv("OPENROUTER_API_KEY"):
    print("❌ OPENROUTER_API_KEY not set")
    print("   Create a .env file with: OPENROUTER_API_KEY=sk-or-v1-...")
    print("   Get your key at: https://openrouter.ai/keys")
    exit(1)

print("✓ API key configured")

from config import (
    COUNCIL_MODELS,
    COUNCIL_PREMIUM,
    COUNCIL_STANDARD,
    COUNCIL_BUDGET,
    CHAIRMAN_POOL,
    get_current_chairman,
    get_rotation_info,
    validate_council_size,
    estimate_cost,
)

# Test config
print(f"✓ Council models: {len(COUNCIL_MODELS)}")
for m in COUNCIL_MODELS:
    print(f"    - {m}")

rotation = get_rotation_info()
print(f"✓ Current chairman: {rotation['current_chairman']}")
print(f"    Rotation: {'enabled' if rotation['rotation_enabled'] else 'disabled'}")
print(f"    Days until next: {rotation['days_until_rotation']}")

size = validate_council_size()
print(f"✓ Council size: {size['total_size']} ({'odd ✓' if size['valid'] else 'even ⚠️'})")

# Cost estimate
est = estimate_cost(100, tier="full")
print(f"✓ Estimated cost (full council): ${est['total']:.4f}")


# =============================================================================
# CUSTOM CONCLAVE SELECTION TESTS
# =============================================================================

def test_custom_conclave_selection():
    """Test the custom conclave selection feature."""
    from server import (
        get_all_models_numbered,
        get_model_by_number,
        get_active_models,
        conclave_select,
        conclave_reset,
        conclave_models,
        _custom_conclave,
    )
    import server  # For modifying global state

    print("\n" + "="*50)
    print("TESTING CUSTOM CONCLAVE SELECTION")
    print("="*50)

    # Test 1: get_all_models_numbered returns correct structure
    print("\n--- Test 1: Model numbering ---")
    models = get_all_models_numbered()
    assert len(models) > 0, "Should have models"
    assert all(len(m) == 3 for m in models), "Each model should be (num, id, tier)"

    # Check number ranges
    premium_nums = [n for n, _, t in models if t == "premium"]
    standard_nums = [n for n, _, t in models if t == "standard"]
    budget_nums = [n for n, _, t in models if t == "budget"]
    chairman_nums = [n for n, _, t in models if t == "chairman"]

    assert all(1 <= n <= 10 for n in premium_nums), "Premium should be 1-10"
    assert all(11 <= n <= 20 for n in standard_nums), "Standard should be 11-20"
    assert all(21 <= n <= 30 for n in budget_nums), "Budget should be 21-30"
    assert all(31 <= n <= 40 for n in chairman_nums), "Chairman should be 31-40"
    print(f"✓ Model numbering correct: {len(models)} models in 4 tiers")

    # Test 2: get_model_by_number
    print("\n--- Test 2: Model lookup by number ---")
    model_1 = get_model_by_number(1)
    assert model_1 == COUNCIL_PREMIUM[0], f"Model #1 should be first premium model"
    model_11 = get_model_by_number(11)
    assert model_11 == COUNCIL_STANDARD[0], f"Model #11 should be first standard model"
    model_31 = get_model_by_number(31)
    assert model_31 == CHAIRMAN_POOL[0], f"Model #31 should be first chairman"
    model_99 = get_model_by_number(99)
    assert model_99 is None, "Invalid number should return None"
    print("✓ Model lookup working correctly")

    # Test 3: Default state (no custom selection)
    print("\n--- Test 3: Default state ---")
    server._custom_conclave = None  # Reset state
    models_active, chairman, source = get_active_models()
    assert source == "tier", "Default source should be 'tier'"
    assert chairman is None, "Default chairman should be None (uses rotation)"
    assert models_active == COUNCIL_STANDARD, "Default should be standard tier"
    print("✓ Default state correct")

    # Test 4: conclave_select with valid input
    print("\n--- Test 4: Custom selection ---")
    async def test_select():
        result = await conclave_select("31,1,11,21")
        assert "Custom Conclave Created" in result, "Should confirm creation"
        assert "deepseek/deepseek-r1" in result, "Should show chairman"

        models_active, chairman, source = get_active_models()
        assert source == "custom", "Source should be 'custom'"
        assert chairman == CHAIRMAN_POOL[0], "Chairman should be #31"
        assert len(models_active) == 3, "Should have 3 members"
        print("✓ Custom selection working")

        # Test 5: conclave_models shows selection
        models_output = await conclave_models()
        assert "Custom conclave active" in models_output, "Should show custom status"
        assert "⭐ (chairman)" in models_output, "Should mark chairman"
        assert "✓ (selected)" in models_output, "Should mark selected models"
        print("✓ conclave_models shows selection status")

        # Test 6: conclave_reset
        reset_result = await conclave_reset()
        assert "Custom Conclave Cleared" in reset_result, "Should confirm reset"

        models_active, chairman, source = get_active_models()
        assert source == "tier", "After reset, source should be 'tier'"
        print("✓ conclave_reset working")

        # Test 7: Reset when already reset
        reset_again = await conclave_reset()
        assert "No custom conclave was active" in reset_again, "Should indicate no-op"
        print("✓ Double reset handled correctly")

    asyncio.run(test_select())

    # Test 8: Invalid inputs
    print("\n--- Test 5: Error handling ---")
    async def test_errors():
        # Invalid format
        result = await conclave_select("abc,def")
        assert "Error" in result, "Should error on invalid format"

        # Too few models
        result = await conclave_select("1")
        assert "Need at least 2 models" in result, "Should require 2+ models"

        # Invalid model numbers
        result = await conclave_select("99,100")
        assert "Invalid model numbers" in result, "Should error on invalid numbers"

        # Duplicate models
        result = await conclave_select("1,1,2")
        assert "Duplicate" in result, "Should error on duplicates"

        print("✓ Error handling correct")

    asyncio.run(test_errors())

    # Test 9: Even/odd size warnings
    print("\n--- Test 6: Size validation ---")
    async def test_size():
        # Even size (4 models = warning)
        result = await conclave_select("31,1,11,21")
        assert "even" in result.lower(), "Should warn about even size"
        await conclave_reset()

        # Odd size (5 models = ok)
        result = await conclave_select("31,1,2,11,21")
        assert "odd ✓" in result, "Should confirm odd size"
        await conclave_reset()

        print("✓ Size validation correct")

    asyncio.run(test_size())

    print("\n" + "="*50)
    print("✓ ALL CUSTOM CONCLAVE TESTS PASSED")
    print("="*50)


# Test actual API call (optional - costs money)
async def test_quick_query():
    from conclave import run_council_quick

    print("\n--- Testing council_quick (will call OpenRouter API) ---")
    print("Question: What is 2+2?")

    try:
        result = await run_council_quick("What is 2+2? Answer briefly.")
        print(f"✓ Got {len(result['stage1'])} responses:")
        for resp in result['stage1']:
            model = resp['model'].split('/')[-1]
            content = resp['content'][:100].replace('\n', ' ')
            print(f"    {model}: {content}...")
        print("\n✓ API integration working!")
    except Exception as e:
        print(f"❌ API error: {e}")


async def test_custom_conclave_query():
    """Test that custom conclave selection works with actual queries."""
    from server import conclave_select, conclave_reset, conclave_quick

    print("\n--- Testing custom conclave with API query ---")

    # Select a small custom conclave (2 cheap models)
    await conclave_select("31,21,22")  # Chairman + 2 budget models

    result = await conclave_quick("What is 1+1? Answer in one word.")

    if "Error" not in result:
        print("✓ Custom conclave query successful")
        print(result[:500] + "..." if len(result) > 500 else result)
    else:
        print(f"⚠️ Query had errors (may be rate limiting): {result[:200]}")

    await conclave_reset()


if __name__ == "__main__":
    # Always run unit tests (no API calls)
    test_custom_conclave_selection()

    print("\n" + "="*50)
    response = input("Run live API test? (costs ~$0.01) [y/N]: ")
    if response.lower() == 'y':
        asyncio.run(test_quick_query())

        response2 = input("Run custom conclave API test? (costs ~$0.01) [y/N]: ")
        if response2.lower() == 'y':
            asyncio.run(test_custom_conclave_query())
    else:
        print("Skipped live tests. All unit tests passed!")
