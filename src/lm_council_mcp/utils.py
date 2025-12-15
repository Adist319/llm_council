"""Utility functions for the LM Council MCP server."""

import json
import os
from typing import Any

import pandas as pd


class CouncilError(Exception):
    """Base exception for council operations."""

    pass


class ConfigurationError(CouncilError):
    """Error in council configuration."""

    pass


class SessionError(CouncilError):
    """Error in session management."""

    pass


class ExecutionError(CouncilError):
    """Error during council execution."""

    pass


class AnalysisError(CouncilError):
    """Error during analysis operations."""

    pass


def validate_api_key() -> str:
    """Validate that OPENROUTER_API_KEY is set.

    Returns:
        The API key value

    Raises:
        ConfigurationError: If API key is not set
    """
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ConfigurationError(
            "OPENROUTER_API_KEY environment variable is not set. "
            "Please set it to your OpenRouter API key."
        )
    return api_key


def validate_hf_token() -> str:
    """Validate that HF_TOKEN is set for HuggingFace operations.

    Returns:
        The HuggingFace token

    Raises:
        ConfigurationError: If HF_TOKEN is not set
    """
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ConfigurationError(
            "HF_TOKEN environment variable is not set. "
            "Please set it to your HuggingFace API token for upload operations."
        )
    return token


def df_to_json(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a pandas DataFrame to a JSON-serializable list of dicts.

    Args:
        df: The DataFrame to convert

    Returns:
        List of dictionaries representing the DataFrame rows
    """
    if df is None or df.empty:
        return []
    return df.to_dict(orient="records")


def format_success_response(data: dict[str, Any], message: str | None = None) -> str:
    """Format a successful response as JSON.

    Args:
        data: Response data
        message: Optional success message

    Returns:
        JSON-formatted string
    """
    response = {"success": True, "data": data}
    if message:
        response["message"] = message
    return json.dumps(response, indent=2, default=str)


def format_error_response(error: Exception, suggestions: list[str] | None = None) -> str:
    """Format an error response as JSON.

    Args:
        error: The exception that occurred
        suggestions: Optional list of suggestions for resolution

    Returns:
        JSON-formatted error string
    """
    error_type = type(error).__name__
    error_mapping = {
        "ConfigurationError": "configuration_error",
        "SessionError": "session_error",
        "ExecutionError": "execution_error",
        "AnalysisError": "analysis_error",
        "ValueError": "validation_error",
        "FileNotFoundError": "not_found_error",
    }

    response = {
        "success": False,
        "error": {
            "type": error_mapping.get(error_type, "internal_error"),
            "message": str(error),
        },
    }

    if suggestions:
        response["error"]["suggestions"] = suggestions

    return json.dumps(response, indent=2)


def handle_error(error: Exception) -> str:
    """Handle an exception and return a formatted error response.

    Args:
        error: The exception to handle

    Returns:
        JSON-formatted error string with appropriate suggestions
    """
    suggestions = []

    if isinstance(error, ConfigurationError):
        if "OPENROUTER_API_KEY" in str(error):
            suggestions = [
                "Set the OPENROUTER_API_KEY environment variable",
                "Get an API key from https://openrouter.ai/",
            ]
        elif "HF_TOKEN" in str(error):
            suggestions = [
                "Set the HF_TOKEN environment variable",
                "Get a token from https://huggingface.co/settings/tokens",
            ]

    elif isinstance(error, SessionError):
        if "not found" in str(error).lower():
            suggestions = [
                "Check the session name spelling",
                "List available sessions to find the correct name",
            ]
        elif "already exists" in str(error).lower():
            suggestions = [
                "Use overwrite=true to replace the existing session",
                "Choose a different session name",
            ]

    elif isinstance(error, ExecutionError):
        suggestions = [
            "Check that the model names are valid OpenRouter model IDs",
            "Verify your API key has access to the requested models",
            "Try reducing the number of models or prompts",
        ]

    elif isinstance(error, AnalysisError):
        suggestions = [
            "Ensure you have run execute_council first",
            "Check that the council has judgment data available",
        ]

    return format_error_response(error, suggestions if suggestions else None)


def get_storage_path() -> str:
    """Get the storage path for council sessions.

    Returns:
        Path to the session storage directory
    """
    return os.environ.get("COUNCIL_STORAGE_PATH", ".council_sessions")
