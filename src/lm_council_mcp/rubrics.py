"""Rubric presets for council evaluation."""

from dataclasses import dataclass


@dataclass
class RubricCriteria:
    """A single evaluation criterion for direct assessment."""

    name: str
    statement: str
    weight: float = 1.0


# Default general-purpose rubric
DEFAULT_RUBRIC = [
    RubricCriteria(
        name="relevance",
        statement="The response directly addresses the question or prompt and stays on topic.",
    ),
    RubricCriteria(
        name="accuracy",
        statement="The information provided is factually correct and well-reasoned.",
    ),
    RubricCriteria(
        name="clarity",
        statement="The response is clear, well-organized, and easy to understand.",
    ),
    RubricCriteria(
        name="completeness",
        statement="The response thoroughly addresses all aspects of the prompt without unnecessary verbosity.",
    ),
]

# LinkedIn content creation rubric
LINKEDIN_RUBRIC = [
    RubricCriteria(
        name="hook_strength",
        statement=(
            "The opening line immediately captures attention and creates curiosity. "
            "Strong hooks use questions, bold claims, or unexpected statements."
        ),
    ),
    RubricCriteria(
        name="value_delivery",
        statement=(
            "The post delivers actionable insights, unique perspectives, or valuable "
            "information that the reader can apply. Avoids generic platitudes."
        ),
    ),
    RubricCriteria(
        name="authenticity",
        statement=(
            "The voice feels genuine and personal rather than corporate or AI-generated. "
            "Shows personality and real experience."
        ),
    ),
    RubricCriteria(
        name="engagement_potential",
        statement=(
            "The post encourages interaction through questions, relatable scenarios, "
            "or thought-provoking statements. Easy to comment on."
        ),
    ),
    RubricCriteria(
        name="formatting_readability",
        statement=(
            "Uses appropriate line breaks, whitespace, and structure for mobile reading. "
            "Scannable with clear visual hierarchy."
        ),
    ),
    RubricCriteria(
        name="professional_tone",
        statement=(
            "Maintains professional credibility while being approachable. "
            "Appropriate for the target audience and industry."
        ),
    ),
]

# Code review rubric
CODE_REVIEW_RUBRIC = [
    RubricCriteria(
        name="technical_accuracy",
        statement="The code/explanation is technically correct and follows best practices.",
    ),
    RubricCriteria(
        name="clarity",
        statement="The explanation is clear and understandable to the target audience.",
    ),
    RubricCriteria(
        name="completeness",
        statement="Covers all relevant aspects without unnecessary verbosity.",
    ),
    RubricCriteria(
        name="practical_applicability",
        statement="The solution/advice can be readily applied in real-world scenarios.",
    ),
]

# Mapping from preset names to rubric definitions
RUBRIC_PRESETS = {
    "default_rubric": DEFAULT_RUBRIC,
    "linkedin_content": LINKEDIN_RUBRIC,
    "code_review": CODE_REVIEW_RUBRIC,
}


def get_rubric_preset(preset_name: str) -> list[RubricCriteria]:
    """Get a rubric preset by name.

    Args:
        preset_name: Name of the preset ('default_rubric', 'linkedin_content', 'code_review')

    Returns:
        List of RubricCriteria for the preset

    Raises:
        ValueError: If preset name is not recognized
    """
    if preset_name not in RUBRIC_PRESETS:
        available = ", ".join(RUBRIC_PRESETS.keys())
        raise ValueError(f"Unknown rubric preset '{preset_name}'. Available presets: {available}")
    return RUBRIC_PRESETS[preset_name]
