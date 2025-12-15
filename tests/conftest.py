"""Pytest configuration and fixtures for LM Council MCP tests."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def mock_api_key():
    """Ensure OPENROUTER_API_KEY is set for all tests."""
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-api-key"}):
        yield


@pytest.fixture
def mock_council():
    """Create a mock LanguageModelCouncil instance."""
    council = MagicMock()

    # Mock execute to return DataFrames
    completions_df = pd.DataFrame(
        {
            "model": ["model-a", "model-b"],
            "prompt": ["test prompt", "test prompt"],
            "completion": ["Response A", "Response B"],
        }
    )
    judgments_df = pd.DataFrame(
        {
            "judge": ["model-a", "model-b"],
            "respondent": ["model-b", "model-a"],
            "score": [0.8, 0.7],
        }
    )
    council.execute = AsyncMock(return_value=(completions_df, judgments_df))

    # Mock leaderboard
    leaderboard_df = pd.DataFrame(
        {"model": ["model-a", "model-b"], "mean_score": [0.75, 0.72]}
    )
    council.leaderboard = MagicMock(return_value=leaderboard_df)

    # Mock analysis methods
    council.judge_agreement = MagicMock(return_value=pd.DataFrame())
    council.affinity = MagicMock(return_value=pd.DataFrame())
    council.win_rate_heatmap = MagicMock(return_value=pd.DataFrame())

    # Mock save/load
    council.save = MagicMock()

    return council


@pytest.fixture
def sample_models():
    """Sample model list for testing."""
    return ["openai/gpt-4", "anthropic/claude-3-opus", "google/gemini-pro"]
