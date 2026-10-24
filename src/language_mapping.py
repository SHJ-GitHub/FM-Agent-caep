"""Shared validation for phase-plan language and extension metadata."""

from __future__ import annotations

from collections.abc import Sequence


def extension_language_map(
    languages: Sequence[str],
    file_extensions: Sequence[str],
) -> dict[str, str]:
    """Return a validated extension-to-language mapping.

    The standard software phase-plan contract uses one extension per language.
    Hardware profiles intentionally use one language with several extensions,
    so that unambiguous one-to-many form remains supported.
    """
    if (
        not isinstance(languages, (list, tuple))
        or not languages
        or not all(
            isinstance(language, str) and language.strip()
            for language in languages
        )
    ):
        raise ValueError(
            "phases.json languages must be a non-empty array of non-empty strings"
        )
    if (
        not isinstance(file_extensions, (list, tuple))
        or not file_extensions
        or not all(
            isinstance(extension, str) and extension.lstrip(".").strip()
            for extension in file_extensions
        )
    ):
        raise ValueError(
            "phases.json file_extensions must be a non-empty array of "
            "non-empty extensions"
        )

    normalized_languages = [language.strip() for language in languages]
    normalized_extensions = [
        extension.strip().lower().lstrip(".")
        for extension in file_extensions
    ]
    if len(set(normalized_extensions)) != len(normalized_extensions):
        raise ValueError("phases.json file_extensions must not contain duplicates")

    if len(normalized_languages) == 1:
        return {
            extension: normalized_languages[0]
            for extension in normalized_extensions
        }
    if len(normalized_languages) != len(normalized_extensions):
        raise ValueError(
            "phases.json languages and file_extensions must have equal lengths "
            "when more than one language is configured"
        )
    return dict(zip(normalized_extensions, normalized_languages))
