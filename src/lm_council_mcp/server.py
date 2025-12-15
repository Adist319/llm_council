"""FastMCP server for LM Council - Democratic LLM evaluation."""

from mcp.server.fastmcp import FastMCP

from .council_manager import CouncilManager
from .models import (
    ConfigureCouncilInput,
    ExecuteCouncilInput,
    GetAffinityAnalysisInput,
    GetJudgeAgreementInput,
    GetLeaderboardInput,
    GetWinRateHeatmapInput,
    LoadSessionInput,
    SaveSessionInput,
    UploadToHuggingFaceInput,
)
from .utils import format_success_response, handle_error

# Initialize FastMCP server
mcp = FastMCP("lm_council_mcp")

# Global council manager instance (maintains state between tool calls)
_manager: CouncilManager | None = None


def get_manager() -> CouncilManager:
    """Get or create the global CouncilManager instance."""
    global _manager
    if _manager is None:
        _manager = CouncilManager()
    return _manager


# ============================================================================
# MCP Tools
# ============================================================================


@mcp.tool(
    name="council_execute",
    annotations={
        "title": "Execute Council Evaluation",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True,
    },
)
async def execute_council(params: ExecuteCouncilInput) -> str:
    """Run a Language Model Council evaluation on one or more prompts.

    Executes the configured council on the provided prompts, generating
    completions from all respondent models and having judge models
    evaluate the responses.

    Args:
        params: ExecuteCouncilInput containing:
            - prompts: Single prompt or list of prompts to evaluate
            - models: Optional model override for this execution
            - config: Optional execution config (parallel, timeout)

    Returns:
        JSON with completions, judgments, and execution metadata.
        Includes responses from all models and peer evaluations.
    """
    try:
        manager = get_manager()

        config = params.config
        result = await manager.execute_council(
            prompts=params.prompts,
            models=params.models,
            parallel=config.parallel if config else True,
            timeout=config.timeout if config else 30,
        )

        return format_success_response(result, "Council execution completed successfully")

    except Exception as e:
        return handle_error(e)


@mcp.tool(
    name="council_configure",
    annotations={
        "title": "Configure Council",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
def configure_council(params: ConfigureCouncilInput) -> str:
    """Configure the Language Model Council with models and evaluation settings.

    Sets up the council with specified respondent models, optional separate
    judge models, and evaluation configuration (direct assessment or pairwise
    comparison mode).

    Args:
        params: ConfigureCouncilInput containing:
            - models: List of model IDs for respondents (required)
            - judge_models: Optional separate judge models
            - eval_config: Evaluation settings (type, preset, temperature)
            - pairwise_config: Pairwise comparison settings

    Returns:
        JSON confirmation with active configuration settings.
    """
    try:
        manager = get_manager()

        result = manager.configure(
            models=params.models,
            judge_models=params.judge_models,
            eval_config=params.eval_config,
            pairwise_config=params.pairwise_config,
        )

        return format_success_response(result, "Council configured successfully")

    except Exception as e:
        return handle_error(e)


@mcp.tool(
    name="council_get_leaderboard",
    annotations={
        "title": "Get Leaderboard",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
def get_leaderboard(params: GetLeaderboardInput) -> str:
    """Get the current leaderboard rankings from council evaluations.

    Returns model rankings based on peer evaluations. Format depends on
    evaluation mode:
    - direct_assessment: Mean scores per model across rubric criteria
    - pairwise_comparison: ELO rankings and win rate statistics

    Args:
        params: GetLeaderboardInput containing:
            - session_name: Optional session to get leaderboard for

    Returns:
        JSON with leaderboard data including rankings and scores.
    """
    try:
        manager = get_manager()
        result = manager.get_leaderboard(session_name=params.session_name)
        return format_success_response(result)

    except Exception as e:
        return handle_error(e)


@mcp.tool(
    name="council_save_session",
    annotations={
        "title": "Save Council Session",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False,
    },
)
def save_session(params: SaveSessionInput) -> str:
    """Save the current council session to persistent storage.

    Persists all council state including configurations, completions,
    judgments, and metadata. Sessions can be loaded later to continue
    evaluation or analysis.

    Args:
        params: SaveSessionInput containing:
            - session_name: Name for the session (required)
            - overwrite: Whether to overwrite existing session

    Returns:
        JSON with save status and session path.
    """
    try:
        manager = get_manager()
        result = manager.save_session(
            session_name=params.session_name,
            overwrite=params.overwrite or False,
        )
        return format_success_response(result, f"Session '{params.session_name}' saved")

    except Exception as e:
        return handle_error(e)


@mcp.tool(
    name="council_load_session",
    annotations={
        "title": "Load Council Session",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
def load_session(params: LoadSessionInput) -> str:
    """Load a previously saved council session.

    Restores a saved session including all configurations, results,
    and metadata. The loaded session becomes the active council.

    Args:
        params: LoadSessionInput containing:
            - session_name: Name of the session to load (required)

    Returns:
        JSON with session metadata and configuration.
    """
    try:
        manager = get_manager()
        result = manager.load_session(session_name=params.session_name)
        return format_success_response(result, f"Session '{params.session_name}' loaded")

    except Exception as e:
        return handle_error(e)


@mcp.tool(
    name="council_get_judge_agreement",
    annotations={
        "title": "Get Judge Agreement Analysis",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
def get_judge_agreement(params: GetJudgeAgreementInput) -> str:
    """Analyze inter-rater agreement between judge models.

    Calculates metrics showing how consistently different judge models
    rate the same responses. Higher agreement indicates more reliable
    evaluation consensus.

    Args:
        params: GetJudgeAgreementInput containing:
            - session_name: Optional session to analyze

    Returns:
        JSON with agreement matrices and mean agreement scores.
    """
    try:
        manager = get_manager()
        result = manager.get_judge_agreement(session_name=params.session_name)
        return format_success_response(result)

    except Exception as e:
        return handle_error(e)


@mcp.tool(
    name="council_get_affinity_analysis",
    annotations={
        "title": "Get Affinity Analysis",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
def get_affinity_analysis(params: GetAffinityAnalysisInput) -> str:
    """Analyze judge preference patterns and potential biases.

    Identifies which judges tend to favor which respondent models,
    helping detect systematic biases in the evaluation process.

    Args:
        params: GetAffinityAnalysisInput containing:
            - session_name: Optional session to analyze

    Returns:
        JSON with affinity matrices showing judge-model preferences.
    """
    try:
        manager = get_manager()
        result = manager.get_affinity_analysis(session_name=params.session_name)
        return format_success_response(result)

    except Exception as e:
        return handle_error(e)


@mcp.tool(
    name="council_get_win_rate_heatmap",
    annotations={
        "title": "Get Win Rate Heatmap",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
def get_win_rate_heatmap(params: GetWinRateHeatmapInput) -> str:
    """Get pairwise win rates between all models.

    Only available in pairwise_comparison evaluation mode. Shows
    head-to-head performance between model pairs.

    Args:
        params: GetWinRateHeatmapInput containing:
            - session_name: Optional session to analyze
            - analysis_type: 'explicit' for raw win rates,
              'bradley_terry' for model strength estimation

    Returns:
        JSON with win rate matrix showing pairwise model performance.
    """
    try:
        manager = get_manager()
        result = manager.get_win_rate_heatmap(
            session_name=params.session_name,
            analysis_type=params.analysis_type.value if params.analysis_type else "explicit",
        )
        return format_success_response(result)

    except Exception as e:
        return handle_error(e)


@mcp.tool(
    name="council_upload_to_huggingface",
    annotations={
        "title": "Upload to HuggingFace Hub",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True,
    },
)
async def upload_to_huggingface(params: UploadToHuggingFaceInput) -> str:
    """Upload council results to HuggingFace Hub.

    Publishes completions, judgments, leaderboard data, and auto-generated
    README to a HuggingFace dataset repository. Requires HF_TOKEN env var.

    Args:
        params: UploadToHuggingFaceInput containing:
            - repo_id: HuggingFace repository ID (e.g., 'username/dataset-name')
            - session_name: Optional session to upload
            - private: Whether to make the dataset private

    Returns:
        JSON with upload status and dataset URL.
    """
    try:
        manager = get_manager()
        result = await manager.upload_to_huggingface(
            repo_id=params.repo_id,
            session_name=params.session_name,
            private=params.private or False,
        )
        return format_success_response(result, f"Uploaded to {params.repo_id}")

    except Exception as e:
        return handle_error(e)


# ============================================================================
# Entry Point
# ============================================================================


def main():
    """Main entry point for the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
