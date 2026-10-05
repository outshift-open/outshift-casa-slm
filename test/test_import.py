"""Testing package import."""

import re


def test_import():
    """Test package import. Makes sure imports do not raise exceptions."""
    import casa_slm.prompt_optimizer  # noqa: F401

    assert True


def test_version_string():
    """Test package version string."""

    def is_canonical(version: str) -> bool:
        """Check if version string follows the canonical format.

        Source: https://packaging.python.org/en/latest/specifications/version-specifiers/#version-specifiers-regex
        """
        return (
            re.match(
                r"^([1-9][0-9]*!)?(0|[1-9][0-9]*)(\.(0|[1-9][0-9]*))*((a|b|rc)(0|[1-9][0-9]*))?(\.post(0|[1-9][0-9]*))?(\.dev(0|[1-9][0-9]*))?$",
                version,
            )
            is not None
        )

    """Test package version string."""
    import casa_slm.prompt_optimizer

    assert isinstance(casa_slm.prompt_optimizer.__version__, str)
    assert is_canonical(casa_slm.prompt_optimizer.__version__)
