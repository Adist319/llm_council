"""Pydantic input models for MCP tool validation."""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class EvalType(str, Enum):
    """Evaluation type for council judgments."""

    DIRECT_ASSESSMENT = "direct_assessment"
    PAIRWISE_COMPARISON = "pairwise_comparison"


class PairwiseAlgorithm(str, Enum):
    """Algorithm for pairwise comparison mode."""

    ALL_PAIRS = "all_pairs"
    RANDOM_PAIRS = "random_pairs"
    FIXED_REFERENCE_MODELS = "fixed_reference_models"


class RubricPreset(str, Enum):
    """Built-in rubric presets."""

    DEFAULT_RUBRIC = "default_rubric"
    LINKEDIN_CONTENT = "linkedin_content"
    CODE_REVIEW = "code_review"


class WinRateAnalysisType(str, Enum):
    """Type of win rate analysis."""

    EXPLICIT = "explicit"
    BRADLEY_TERRY = "bradley_terry"


class EvalConfig(BaseModel):
    """Evaluation configuration for direct assessment mode."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    type: EvalType = Field(
        default=EvalType.DIRECT_ASSESSMENT,
        description="Evaluation type: 'direct_assessment' or 'pairwise_comparison'",
    )
    preset: Optional[RubricPreset] = Field(
        default=RubricPreset.DEFAULT_RUBRIC,
        description="Rubric preset: 'default_rubric', 'linkedin_content', or 'code_review'",
    )
    temperature: Optional[float] = Field(
        default=0.7, description="Temperature for judge completions", ge=0.0, le=2.0
    )
    exclude_self_grading: Optional[bool] = Field(
        default=True, description="Whether judges should skip grading their own responses"
    )


class PairwiseConfig(BaseModel):
    """Configuration for pairwise comparison mode."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    algorithm: PairwiseAlgorithm = Field(
        default=PairwiseAlgorithm.ALL_PAIRS,
        description="Pairing algorithm: 'all_pairs', 'random_pairs', or 'fixed_reference_models'",
    )
    reference_models: Optional[list[str]] = Field(
        default=None,
        description="Reference models for fixed_reference_models algorithm",
    )
    position_flipping: Optional[bool] = Field(
        default=True, description="Whether to flip positions to reduce order bias"
    )


class ExecutionConfig(BaseModel):
    """Execution configuration for council runs."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    parallel: Optional[bool] = Field(default=True, description="Run model completions in parallel")
    timeout: Optional[int] = Field(
        default=30, description="Timeout in seconds for each model", ge=1, le=300
    )


# Tool Input Models


class ExecuteCouncilInput(BaseModel):
    """Input for execute_council tool."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    prompts: str | list[str] = Field(
        ...,
        description="Single prompt or list of prompts to evaluate (e.g., 'Write a haiku about AI')",
    )
    models: Optional[list[str]] = Field(
        default=None,
        description="Override models for this execution (e.g., ['openai/gpt-4', 'anthropic/claude-3'])",
    )
    config: Optional[ExecutionConfig] = Field(
        default=None, description="Execution configuration (parallel, timeout)"
    )


class ConfigureCouncilInput(BaseModel):
    """Input for configure_council tool."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    models: list[str] = Field(
        ...,
        description="List of model IDs to use as respondents (e.g., ['openai/gpt-4', 'anthropic/claude-3'])",
        min_length=1,
    )
    judge_models: Optional[list[str]] = Field(
        default=None,
        description="Separate judge models (defaults to respondent models if not specified)",
    )
    eval_config: Optional[EvalConfig] = Field(
        default=None, description="Evaluation configuration (type, preset, temperature)"
    )
    pairwise_config: Optional[PairwiseConfig] = Field(
        default=None, description="Pairwise comparison configuration (algorithm, reference models)"
    )


class GetLeaderboardInput(BaseModel):
    """Input for get_leaderboard tool."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    session_name: Optional[str] = Field(
        default=None,
        description="Session name to get leaderboard for (uses current session if not specified)",
    )


class SaveSessionInput(BaseModel):
    """Input for save_session tool."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    session_name: str = Field(
        ...,
        description="Name for the saved session (e.g., 'linkedin_eval_2024')",
        min_length=1,
        max_length=100,
    )
    overwrite: Optional[bool] = Field(
        default=False, description="Overwrite existing session with same name"
    )


class LoadSessionInput(BaseModel):
    """Input for load_session tool."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    session_name: str = Field(
        ..., description="Name of the session to load", min_length=1, max_length=100
    )


class GetJudgeAgreementInput(BaseModel):
    """Input for get_judge_agreement tool."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    session_name: Optional[str] = Field(
        default=None, description="Session name (uses current session if not specified)"
    )


class GetAffinityAnalysisInput(BaseModel):
    """Input for get_affinity_analysis tool."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    session_name: Optional[str] = Field(
        default=None, description="Session name (uses current session if not specified)"
    )


class GetWinRateHeatmapInput(BaseModel):
    """Input for get_win_rate_heatmap tool."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    session_name: Optional[str] = Field(
        default=None, description="Session name (uses current session if not specified)"
    )
    analysis_type: Optional[WinRateAnalysisType] = Field(
        default=WinRateAnalysisType.EXPLICIT,
        description="Analysis type: 'explicit' for raw win rates, 'bradley_terry' for model strength estimation",
    )


class UploadToHuggingFaceInput(BaseModel):
    """Input for upload_to_huggingface tool."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    repo_id: str = Field(
        ...,
        description="HuggingFace repository ID (e.g., 'username/dataset-name')",
        min_length=3,
    )
    session_name: Optional[str] = Field(
        default=None, description="Session name to upload (uses current session if not specified)"
    )
    private: Optional[bool] = Field(default=False, description="Make the dataset private")
