"""Unit tests for CouncilManager."""

import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from lm_council_mcp.council_manager import CouncilManager
from lm_council_mcp.models import EvalConfig, EvalType, PairwiseConfig, RubricPreset
from lm_council_mcp.utils import ConfigurationError, SessionError


class TestCouncilManagerConfiguration:
    """Tests for council configuration."""

    def test_configure_with_models(self, mock_council, sample_models):
        """Test basic configuration with model list."""
        with patch("lm_council_mcp.council_manager.LanguageModelCouncil") as MockCouncil:
            MockCouncil.return_value = mock_council
            manager = CouncilManager()

            result = manager.configure(models=sample_models)

            assert result["configured"] is True
            assert result["models"] == sample_models
            MockCouncil.assert_called_once()

    def test_configure_with_separate_judges(self, mock_council, sample_models):
        """Test configuration with separate judge models."""
        judge_models = ["openai/gpt-4"]

        with patch("lm_council_mcp.council_manager.LanguageModelCouncil") as MockCouncil:
            MockCouncil.return_value = mock_council
            manager = CouncilManager()

            result = manager.configure(models=sample_models, judge_models=judge_models)

            assert result["judge_models"] == judge_models

    def test_configure_with_eval_config(self, mock_council, sample_models):
        """Test configuration with evaluation settings."""
        eval_config = EvalConfig(
            type=EvalType.DIRECT_ASSESSMENT,
            preset=RubricPreset.LINKEDIN_CONTENT,
            temperature=0.5,
        )

        with patch("lm_council_mcp.council_manager.LanguageModelCouncil") as MockCouncil:
            MockCouncil.return_value = mock_council
            manager = CouncilManager()

            result = manager.configure(models=sample_models, eval_config=eval_config)

            assert result["eval_config"]["preset"] == "linkedin_content"


class TestCouncilManagerExecution:
    """Tests for council execution."""

    @pytest.mark.asyncio
    async def test_execute_single_prompt(self, mock_council, sample_models):
        """Test execution with a single prompt."""
        with patch("lm_council_mcp.council_manager.LanguageModelCouncil") as MockCouncil:
            MockCouncil.return_value = mock_council
            manager = CouncilManager()
            manager.configure(models=sample_models)

            result = await manager.execute_council(prompts="Write a haiku about AI")

            assert result["num_prompts"] == 1
            assert "completions" in result
            assert "judgments" in result

    @pytest.mark.asyncio
    async def test_execute_multiple_prompts(self, mock_council, sample_models):
        """Test execution with multiple prompts."""
        prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]

        with patch("lm_council_mcp.council_manager.LanguageModelCouncil") as MockCouncil:
            MockCouncil.return_value = mock_council
            manager = CouncilManager()
            manager.configure(models=sample_models)

            result = await manager.execute_council(prompts=prompts)

            assert result["num_prompts"] == 3
            assert result["prompts"] == prompts

    @pytest.mark.asyncio
    async def test_execute_auto_configures_with_models(self, mock_council):
        """Test that execute auto-configures when models provided."""
        models = ["openai/gpt-4"]

        with patch("lm_council_mcp.council_manager.LanguageModelCouncil") as MockCouncil:
            MockCouncil.return_value = mock_council
            manager = CouncilManager()

            result = await manager.execute_council(prompts="Test", models=models)

            assert "completions" in result


class TestCouncilManagerLeaderboard:
    """Tests for leaderboard retrieval."""

    def test_get_leaderboard(self, mock_council, sample_models):
        """Test getting leaderboard data."""
        with patch("lm_council_mcp.council_manager.LanguageModelCouncil") as MockCouncil:
            MockCouncil.return_value = mock_council
            manager = CouncilManager()
            manager.configure(models=sample_models)
            manager._has_results = True

            result = manager.get_leaderboard()

            assert "leaderboard" in result


class TestCouncilManagerSessions:
    """Tests for session management."""

    def test_save_session(self, mock_council, sample_models, tmp_path):
        """Test saving a session."""
        with patch("lm_council_mcp.council_manager.LanguageModelCouncil") as MockCouncil:
            MockCouncil.return_value = mock_council
            manager = CouncilManager(storage_path=str(tmp_path))
            manager.configure(models=sample_models)

            result = manager.save_session("test_session")

            assert result["saved"] is True
            assert (tmp_path / "test_session").exists()

    def test_save_session_no_overwrite(self, mock_council, sample_models, tmp_path):
        """Test that save fails when session exists and overwrite=False."""
        with patch("lm_council_mcp.council_manager.LanguageModelCouncil") as MockCouncil:
            MockCouncil.return_value = mock_council
            manager = CouncilManager(storage_path=str(tmp_path))
            manager.configure(models=sample_models)

            manager.save_session("test_session")

            with pytest.raises(SessionError):
                manager.save_session("test_session", overwrite=False)

    def test_load_nonexistent_session(self, tmp_path):
        """Test that loading a nonexistent session fails."""
        with patch("lm_council_mcp.council_manager.LanguageModelCouncil"):
            manager = CouncilManager(storage_path=str(tmp_path))

            with pytest.raises(SessionError):
                manager.load_session("nonexistent")

    def test_list_sessions(self, mock_council, sample_models, tmp_path):
        """Test listing saved sessions."""
        with patch("lm_council_mcp.council_manager.LanguageModelCouncil") as MockCouncil:
            MockCouncil.return_value = mock_council
            manager = CouncilManager(storage_path=str(tmp_path))
            manager.configure(models=sample_models)

            manager.save_session("session1")
            manager.save_session("session2")

            sessions = manager.list_sessions()

            assert "session1" in sessions
            assert "session2" in sessions
