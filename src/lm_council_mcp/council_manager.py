"""Business logic layer for LM Council MCP server."""

import json
import os
import sys
from pathlib import Path
from typing import Any

# Patch huggingface_hub before importing lm_council (HfFolder deprecated in newer versions)
import huggingface_hub
if not hasattr(huggingface_hub, 'HfFolder'):
    class HfFolder:
        @staticmethod
        def get_token():
            return os.environ.get("HF_TOKEN")
    huggingface_hub.HfFolder = HfFolder

import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend for headless operation

from lm_council import LanguageModelCouncil

# Try to import PRESET_EVAL_CONFIGS, fall back to empty dict if not available
try:
    from lm_council.constants import PRESET_EVAL_CONFIGS
except ImportError:
    PRESET_EVAL_CONFIGS = {}

from .models import EvalConfig, EvalType, PairwiseConfig, RubricPreset
from .rubrics import RUBRIC_PRESETS, RubricCriteria
from .utils import (
    AnalysisError,
    ConfigurationError,
    ExecutionError,
    SessionError,
    df_to_json,
    get_storage_path,
    validate_api_key,
    validate_hf_token,
)


class CouncilManager:
    """Manager class for LM Council operations.

    Handles council lifecycle, configuration, execution, and analysis.
    Maintains state between MCP tool calls.
    """

    def __init__(self, storage_path: str | None = None):
        """Initialize the CouncilManager.

        Args:
            storage_path: Directory for session storage. Defaults to COUNCIL_STORAGE_PATH env var.
        """
        self._storage_path = Path(storage_path or get_storage_path())
        self._storage_path.mkdir(parents=True, exist_ok=True)

        self._council: LanguageModelCouncil | None = None
        self._current_config: dict[str, Any] | None = None
        self._has_results: bool = False

        # Validate API key on initialization
        validate_api_key()

    def _build_eval_config(
        self, eval_config: EvalConfig | None, pairwise_config: PairwiseConfig | None
    ) -> dict[str, Any]:
        """Build evaluation config dict for lm-council library.

        Args:
            eval_config: Evaluation configuration from user
            pairwise_config: Pairwise comparison configuration

        Returns:
            Config dict compatible with lm-council
        """
        if eval_config is None:
            # Use default rubric preset
            return PRESET_EVAL_CONFIGS.get("default_rubric", {})

        # Get base config from preset or build custom
        if eval_config.preset:
            preset_name = eval_config.preset.value
            if preset_name in PRESET_EVAL_CONFIGS:
                config = dict(PRESET_EVAL_CONFIGS[preset_name])
            else:
                # Use our custom rubric presets
                rubric = RUBRIC_PRESETS.get(preset_name, RUBRIC_PRESETS["default_rubric"])
                config = self._rubric_to_config(rubric)
        else:
            config = {}

        # Apply evaluation type
        if eval_config.type == EvalType.PAIRWISE_COMPARISON:
            config["eval_type"] = "pairwise_comparison"
            if pairwise_config:
                config["pairing_algorithm"] = pairwise_config.algorithm.value
                if pairwise_config.reference_models:
                    config["reference_models"] = pairwise_config.reference_models
                if pairwise_config.position_flipping is not None:
                    config["position_flipping"] = pairwise_config.position_flipping
        else:
            config["eval_type"] = "direct_assessment"

        # Apply temperature if specified
        if eval_config.temperature is not None:
            config["temperature"] = eval_config.temperature

        # Apply self-grading exclusion
        if eval_config.exclude_self_grading is not None:
            config["exclude_self_grading"] = eval_config.exclude_self_grading

        return config

    def _rubric_to_config(self, rubric: list[RubricCriteria]) -> dict[str, Any]:
        """Convert our rubric format to lm-council config format.

        Args:
            rubric: List of RubricCriteria

        Returns:
            Config dict with rubric criteria
        """
        criteria = []
        for item in rubric:
            criteria.append(
                {
                    "name": item.name,
                    "statement": item.statement,
                    "weight": item.weight,
                }
            )
        return {"rubric": criteria}

    def configure(
        self,
        models: list[str],
        judge_models: list[str] | None = None,
        eval_config: EvalConfig | None = None,
        pairwise_config: PairwiseConfig | None = None,
    ) -> dict[str, Any]:
        """Configure the council with models and evaluation settings.

        Args:
            models: List of model IDs for respondents
            judge_models: Optional separate judge models
            eval_config: Evaluation configuration
            pairwise_config: Pairwise comparison settings

        Returns:
            Dict with configuration confirmation
        """
        try:
            # Build the evaluation config
            lmc_eval_config = self._build_eval_config(eval_config, pairwise_config)

            # Create new council instance
            self._council = LanguageModelCouncil(
                models=models,
                judge_models=judge_models,
                eval_config=lmc_eval_config,
            )

            # Store current config for reference
            self._current_config = {
                "models": models,
                "judge_models": judge_models or models,
                "eval_type": eval_config.type.value if eval_config else "direct_assessment",
                "preset": eval_config.preset.value if eval_config and eval_config.preset else "default_rubric",
            }
            self._has_results = False

            return {
                "configured": True,
                "models": models,
                "judge_models": judge_models or models,
                "eval_config": self._current_config,
            }

        except Exception as e:
            raise ConfigurationError(f"Failed to configure council: {e}") from e

    async def execute_council(
        self,
        prompts: str | list[str],
        models: list[str] | None = None,
        parallel: bool = True,
        timeout: int = 30,
    ) -> dict[str, Any]:
        """Execute council evaluation on prompts.

        Args:
            prompts: Single prompt or list of prompts
            models: Optional model override for this execution
            parallel: Run completions in parallel
            timeout: Timeout per model in seconds

        Returns:
            Dict with completions and judgments
        """
        try:
            # Ensure we have a configured council
            if self._council is None:
                if models:
                    # Auto-configure with provided models
                    self.configure(models)
                else:
                    raise ConfigurationError(
                        "Council not configured. Call configure_council first or provide models."
                    )

            # Normalize prompts to list
            if isinstance(prompts, str):
                prompts = [prompts]

            # Execute the council
            completions_df, judgments_df = await self._council.execute(prompts)

            self._has_results = True

            return {
                "prompts": prompts,
                "completions": df_to_json(completions_df),
                "judgments": df_to_json(judgments_df),
                "num_prompts": len(prompts),
                "num_models": len(self._current_config["models"]) if self._current_config else 0,
            }

        except Exception as e:
            raise ExecutionError(f"Council execution failed: {e}") from e

    def get_leaderboard(self, session_name: str | None = None) -> dict[str, Any]:
        """Get leaderboard data for current or specified session.

        Args:
            session_name: Optional session to get leaderboard for

        Returns:
            Dict with leaderboard data
        """
        try:
            council = self._get_council(session_name)

            if not self._has_results and session_name is None:
                raise AnalysisError("No council executions yet. Run execute_council first.")

            # Get leaderboard without saving to file
            leaderboard_df = council.leaderboard(outfile=None)

            return {
                "leaderboard": df_to_json(leaderboard_df),
                "eval_type": self._current_config.get("eval_type") if self._current_config else "unknown",
            }

        except AnalysisError:
            raise
        except Exception as e:
            raise AnalysisError(f"Failed to get leaderboard: {e}") from e

    def save_session(self, session_name: str, overwrite: bool = False) -> dict[str, Any]:
        """Save current council session to storage.

        Args:
            session_name: Name for the session
            overwrite: Whether to overwrite existing session

        Returns:
            Dict with save status
        """
        try:
            if self._council is None:
                raise SessionError("No council to save. Configure and execute first.")

            session_dir = self._storage_path / session_name

            if session_dir.exists() and not overwrite:
                raise SessionError(
                    f"Session '{session_name}' already exists. Use overwrite=true to replace."
                )

            # Create session directory
            session_dir.mkdir(parents=True, exist_ok=True)

            # Save council data
            self._council.save(str(session_dir))

            # Save our config metadata
            config_file = session_dir / "mcp_config.json"
            with open(config_file, "w") as f:
                json.dump(self._current_config, f, indent=2)

            return {
                "saved": True,
                "session_name": session_name,
                "path": str(session_dir),
            }

        except SessionError:
            raise
        except Exception as e:
            raise SessionError(f"Failed to save session: {e}") from e

    def load_session(self, session_name: str) -> dict[str, Any]:
        """Load a saved council session.

        Args:
            session_name: Name of the session to load

        Returns:
            Dict with session metadata
        """
        try:
            session_dir = self._storage_path / session_name

            if not session_dir.exists():
                raise SessionError(f"Session '{session_name}' not found.")

            # Load council from saved data
            self._council = LanguageModelCouncil.load(str(session_dir))

            # Load our config metadata if available
            config_file = session_dir / "mcp_config.json"
            if config_file.exists():
                with open(config_file) as f:
                    self._current_config = json.load(f)
            else:
                self._current_config = {"models": [], "eval_type": "unknown"}

            self._has_results = True

            return {
                "loaded": True,
                "session_name": session_name,
                "config": self._current_config,
            }

        except SessionError:
            raise
        except Exception as e:
            raise SessionError(f"Failed to load session: {e}") from e

    def get_judge_agreement(self, session_name: str | None = None) -> dict[str, Any]:
        """Analyze inter-rater agreement between judges.

        Args:
            session_name: Optional session to analyze

        Returns:
            Dict with agreement metrics
        """
        try:
            council = self._get_council(session_name)

            if not self._has_results and session_name is None:
                raise AnalysisError("No council executions yet. Run execute_council first.")

            # Get agreement analysis (suppress plots)
            agreement_result = council.judge_agreement(show_plots=False)

            # Convert any DataFrames in the result
            if isinstance(agreement_result, tuple):
                return {
                    "agreement_matrix": df_to_json(agreement_result[0]) if hasattr(agreement_result[0], "to_dict") else agreement_result[0],
                    "mean_agreement": agreement_result[1] if len(agreement_result) > 1 else None,
                }
            elif hasattr(agreement_result, "to_dict"):
                return {"agreement": df_to_json(agreement_result)}
            else:
                return {"agreement": agreement_result}

        except AnalysisError:
            raise
        except Exception as e:
            raise AnalysisError(f"Failed to get judge agreement: {e}") from e

    def get_affinity_analysis(self, session_name: str | None = None) -> dict[str, Any]:
        """Analyze judge preference patterns and biases.

        Args:
            session_name: Optional session to analyze

        Returns:
            Dict with affinity analysis results
        """
        try:
            council = self._get_council(session_name)

            if not self._has_results and session_name is None:
                raise AnalysisError("No council executions yet. Run execute_council first.")

            # Get affinity analysis (suppress plots)
            affinity_result = council.affinity(show_plots=False)

            # Convert any DataFrames in the result
            if hasattr(affinity_result, "to_dict"):
                return {"affinity": df_to_json(affinity_result)}
            else:
                return {"affinity": affinity_result}

        except AnalysisError:
            raise
        except Exception as e:
            raise AnalysisError(f"Failed to get affinity analysis: {e}") from e

    def get_win_rate_heatmap(
        self, session_name: str | None = None, analysis_type: str = "explicit"
    ) -> dict[str, Any]:
        """Get pairwise win rates between models.

        Only available in pairwise_comparison mode.

        Args:
            session_name: Optional session to analyze
            analysis_type: 'explicit' for raw win rates, 'bradley_terry' for model strength

        Returns:
            Dict with win rate matrix
        """
        try:
            council = self._get_council(session_name)

            if not self._has_results and session_name is None:
                raise AnalysisError("No council executions yet. Run execute_council first.")

            # Check if in pairwise mode
            eval_type = self._current_config.get("eval_type") if self._current_config else None
            if eval_type != "pairwise_comparison":
                raise AnalysisError(
                    "Win rate heatmap is only available in pairwise_comparison mode."
                )

            if analysis_type == "bradley_terry":
                heatmap_df = council.win_rate_heatmap()
            else:
                council.explicit_win_rate_heatmap()
                # explicit_win_rate_heatmap doesn't return data, try to get it another way
                heatmap_df = council.win_rate_heatmap()

            return {"win_rates": df_to_json(heatmap_df)}

        except AnalysisError:
            raise
        except Exception as e:
            raise AnalysisError(f"Failed to get win rate heatmap: {e}") from e

    async def upload_to_huggingface(
        self, repo_id: str, session_name: str | None = None, private: bool = False
    ) -> dict[str, Any]:
        """Upload council results to HuggingFace Hub.

        Args:
            repo_id: HuggingFace repository ID (e.g., 'username/dataset-name')
            session_name: Optional session to upload
            private: Make the dataset private

        Returns:
            Dict with upload status and URL
        """
        try:
            # Validate HF token
            validate_hf_token()

            council = self._get_council(session_name)

            if not self._has_results and session_name is None:
                raise AnalysisError("No council executions yet. Run execute_council first.")

            # Upload to HuggingFace
            council.upload_to_hf(repo_id)

            return {
                "uploaded": True,
                "repo_id": repo_id,
                "url": f"https://huggingface.co/datasets/{repo_id}",
                "private": private,
            }

        except ConfigurationError:
            raise
        except Exception as e:
            raise ExecutionError(f"Failed to upload to HuggingFace: {e}") from e

    def _get_council(self, session_name: str | None = None) -> LanguageModelCouncil:
        """Get council instance, loading from session if specified.

        Args:
            session_name: Optional session to load

        Returns:
            LanguageModelCouncil instance
        """
        if session_name:
            self.load_session(session_name)

        if self._council is None:
            raise ConfigurationError("No council configured. Call configure_council first.")

        return self._council

    def list_sessions(self) -> list[str]:
        """List all saved sessions.

        Returns:
            List of session names
        """
        if not self._storage_path.exists():
            return []

        return [
            d.name
            for d in self._storage_path.iterdir()
            if d.is_dir() and (d / "mcp_config.json").exists()
        ]
