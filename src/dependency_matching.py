"""Match dependency-info callee names to extracted-unit FQNs."""

from __future__ import annotations

import logging
import re
from typing import List, Optional, Sequence, Tuple

_QUALIFIED_NAME_SEPARATOR_RE = re.compile(r"::|\.")
_ERLANG_NAME_RE = re.compile(
    r"^(?P<module>[A-Za-z_][\w@]*)\s*:\s*"
    r"(?P<function>[A-Za-z_][\w@]*)/(?P<arity>\d+)$"
)

logger = logging.getLogger(__name__)


def extract_callee_spec_from_info(
    info_dict: dict,
    callee_fqn: str,
    aliases: Optional[Sequence[str]] = None,
) -> Optional[dict]:
    """Return the callee object matching the FQN or trusted edge aliases.

    ``aliases`` must contain only names supplied by a supplemental edge's
    ``callee.info_names``.  The matcher deliberately does not infer aliases
    from the candidate entries in ``info_dict``.
    """
    names = _callee_match_names(callee_fqn, aliases or ())
    callees = info_dict.get("callees", [])
    if not isinstance(callees, list):
        logger.warning(
            "No callee expectation matched FQN %s (aliases=%s, entries=invalid)",
            callee_fqn,
            list(aliases or ()),
        )
        return None

    named_callees = []
    for callee in callees:
        if not isinstance(callee, dict):
            continue
        name = callee.get("name", "")
        if not isinstance(name, str):
            continue
        normalized_name = _name_without_call_decoration(name)
        bare_name = _bare_name_for_matching(normalized_name)
        named_callees.append((callee, name, normalized_name, bare_name))

    for callee, name, _normalized_name, _bare_name in named_callees:
        if _info_name_matches_fqn(name, callee_fqn):
            return callee

    for callee, name, _normalized_name, _bare_name in named_callees:
        if any(
            _info_name_matches_alias(name, alias)
            for alias in aliases or ()
            if alias
        ):
            return callee

    bare_names = [name for name in names if not _is_qualified_name(name)]
    bare_name_counts = {}
    for _callee, _name, _normalized_name, bare_name in named_callees:
        if bare_name:
            bare_name_counts[bare_name.casefold()] = (
                bare_name_counts.get(bare_name.casefold(), 0) + 1
            )
    for callee, _name, normalized_name, bare_name in named_callees:
        if not bare_name or bare_name_counts.get(bare_name.casefold(), 0) != 1:
            continue
        if _is_qualified_name(normalized_name):
            # A source-qualified entry that failed the explicit FQN/alias
            # passes must not be reduced to its final component: doing so can
            # attach ``wrong::Class::method`` to ``src::Class::method``.
            # ``self``/``this`` are receiver syntax, not namespaces; dotted
            # member syntax is eligible only when its receiver names target.
            if not _qualified_source_name_compatible(normalized_name, callee_fqn):
                continue
        if _erlang_name_parts(normalized_name) and not _erlang_name_matches_fqn(
            normalized_name, callee_fqn
        ):
            continue
        if any(
            _info_line_mentions_name(bare_name, candidate)
            for candidate in bare_names
        ):
            return callee
    logger.warning(
        "No callee expectation matched FQN %s (aliases=%s, entries=%d)",
        callee_fqn,
        list(aliases or ()),
        len(named_callees),
    )
    return None


def _callee_match_names(callee_fqn: str, aliases: Sequence[str]) -> List[str]:
    names = [callee_fqn, callee_fqn.split("::")[-1]]
    for alias in aliases:
        if not alias:
            continue
        names.append(alias)
        alias_parts = _qualified_name_parts(alias)
        if len(alias_parts) > 1:
            names.append(alias_parts[-1])
    return list(dict.fromkeys(names))


def _qualified_name_parts(name: str) -> List[str]:
    """Split C++- and dot-qualified source names into components."""
    return [part for part in _QUALIFIED_NAME_SEPARATOR_RE.split(name) if part]


def _is_qualified_name(name: str) -> bool:
    """Return whether a source-level name contains a known qualifier."""
    angle_depth = 0
    index = 0
    while index < len(name):
        char = name[index]
        if char == "<":
            angle_depth += 1
        elif char == ">" and angle_depth:
            angle_depth -= 1
        elif angle_depth == 0:
            if char == "." or name.startswith("::", index):
                return True
        index += 1
    return False


def _trailing_parenthesized_group(name: str) -> Optional[Tuple[str, str]]:
    """Return the prefix and contents of one balanced trailing ``(...)`` group."""
    stripped = name.rstrip()
    if not stripped.endswith(")"):
        return None

    depth = 0
    for index in range(len(stripped) - 1, -1, -1):
        char = stripped[index]
        if char == ")":
            depth += 1
        elif char == "(":
            depth -= 1
            if depth == 0:
                return stripped[:index].rstrip(), stripped[index + 1 : -1].strip()
    return None


def _name_without_call_decoration(name: str) -> str:
    """Remove one balanced trailing call expression from a source name."""
    group = _trailing_parenthesized_group(name)
    if not group:
        return name
    undecorated, contents = group
    # ``operator()`` is a function name; only a following pair of parentheses
    # represents a call decoration for that operator.
    components = _QUALIFIED_NAME_SEPARATOR_RE.split(undecorated)
    if components[-1].strip() == "operator" and not contents:
        return name
    return undecorated


def _is_operator_call_name(name: str) -> bool:
    group = _trailing_parenthesized_group(name)
    if not group or group[1]:
        return False
    components = _QUALIFIED_NAME_SEPARATOR_RE.split(group[0])
    return components[-1].strip() == "operator"


def _erlang_name_parts(name: str) -> Optional[Tuple[str, str, int]]:
    match = _ERLANG_NAME_RE.fullmatch(name.strip())
    if not match:
        return None
    return match.group("module"), match.group("function"), int(match.group("arity"))


def _erlang_name_matches_fqn(info_name: str, callee_fqn: str) -> bool:
    parts = _erlang_name_parts(info_name)
    fqn_parts = [part for part in callee_fqn.split("::") if part]
    if not parts or len(fqn_parts) < 2:
        return False
    module, function, arity = parts

    # Erlang functions produced by the ELP adapter use an encoded final
    # component: ``module__function__arity``.  Older metadata may instead
    # expose ``module::function`` without an arity.  Compare arity whenever
    # the FQN carries it, while retaining compatibility for the old form.
    encoded = re.fullmatch(
        r"(?P<module>[A-Za-z_][\w@]*)__(?P<function>[A-Za-z_][\w@]*)__(?P<arity>\d+)",
        fqn_parts[-1],
    )
    if encoded:
        return (
            module.casefold() == encoded.group("module").casefold()
            and function.casefold() == encoded.group("function").casefold()
            and arity == int(encoded.group("arity"))
        )

    return (
        module.casefold() == fqn_parts[-2].casefold()
        and function.casefold() == fqn_parts[-1].casefold()
    )


def _bare_name_for_matching(name: str) -> str:
    """Return the final source-level component used by safe bare fallback."""
    erlang_parts = _erlang_name_parts(name)
    if erlang_parts:
        return erlang_parts[1]
    source_candidates = _source_name_fqn_candidates(name)
    if source_candidates:
        return source_candidates[0].split("::")[-1]
    if _is_qualified_name(name):
        parts = _qualified_name_parts(name)
        return _codegraph_source_component(parts[-1]) if parts else ""
    bare = name.strip().split()[0] if name.strip() else ""
    # Template arguments decorate a function name but do not identify its
    # callee for the safe bare fallback.  Canonicalizing here makes
    # ``run<int>()`` and ``run<float>()`` collide instead of selecting the
    # first entry based on incidental ordering.
    if "<" in name:
        # Look at the original spelling so spaces inside template arguments
        # do not truncate the token before we canonicalize it.
        template_start = name.find("<")
        prefix = name[:template_start].strip()
        if re.fullmatch(r"[A-Za-z_]\w*", prefix):
            bare = prefix
            return bare
    if "<" in bare:
        angle_depth = 0
        end = None
        for index, char in enumerate(bare):
            if char == "<":
                if angle_depth == 0:
                    end = index
                angle_depth += 1
            elif char == ">" and angle_depth:
                angle_depth -= 1
        if end is not None and angle_depth == 0:
            bare = bare[:end]
    return bare


def _codegraph_source_component(component: str) -> str:
    """Mirror CodeGraph's bare-name and FQN-safe normalization for one part."""
    name = component.strip()
    if not name:
        return ""

    tail = name
    if "::" in tail:
        tail = tail.rsplit("::", 1)[1].lstrip()
    elif "." in tail:
        tail = tail.rsplit(".", 1)[1].lstrip()

    if tail.startswith("operator"):
        rest = tail[len("operator") :].lstrip()
        if rest.startswith("[]"):
            return "operator[]"
        if rest.startswith("()"):
            return "operator()"
        if re.fullmatch(r"new(?:\s*\[\s*\])?", rest):
            return "operator new[]" if "[" in rest else "operator new"
        if re.fullmatch(r"delete(?:\s*\[\s*\])?", rest):
            return "operator delete[]" if "[" in rest else "operator delete"

        symbol = []
        for char in rest:
            if char in "+-*/%&|^~!=<>,":
                symbol.append(char)
            else:
                break
        if symbol:
            return ("operator" + "".join(symbol)).replace("/", "_")

    match = re.search(r"(?:^|::|\.)(\w+)$", name)
    if match:
        return match.group(1)
    match = re.match(r"\(\s*\*\s*(\w+)\s*\)", name)
    if match:
        return match.group(1)
    match = re.match(r"\*\s*(\w+)", name)
    if match:
        return match.group(1)
    match = re.match(r"^(\w+)", name)
    if match:
        return match.group(1)
    return name.replace("/", "_")


def _has_top_level_hyphen_before_final_component(name: str) -> bool:
    """Distinguish a literal ``base-ext`` FQN prefix from template arguments."""
    angle_depth = 0
    top_level_hyphens: List[int] = []
    last_separator = -1
    index = 0
    while index < len(name):
        char = name[index]
        if char == "<":
            angle_depth += 1
        elif char == ">" and angle_depth:
            angle_depth -= 1
        elif angle_depth == 0:
            if name.startswith("::", index):
                last_separator = index
                index += 2
                continue
            if char == ".":
                last_separator = index
            elif char == "-":
                top_level_hyphens.append(index)
        index += 1
    return any(position < last_separator for position in top_level_hyphens)


def _normalized_source_name_parts(name: str) -> List[str]:
    """Return CodeGraph-shaped parts for one source-qualified spelling."""
    # FM-Agent's file component uses ``base-ext`` and is not source syntax.
    # It must stay on the literal-prefix side of a mixed candidate boundary.
    # A hyphen inside a template argument (for example, ``Widget<-1>``) is
    # source decoration and must not be mistaken for that file marker.
    if _has_top_level_hyphen_before_final_component(name):
        return []
    # Namespace separators inside template arguments do not qualify the
    # function itself (for example, ``run<std::pair<int, float>>``).
    angle_depth = 0
    has_top_level_separator = False
    index = 0
    while index < len(name):
        char = name[index]
        if char == "<":
            angle_depth += 1
        elif char == ">" and angle_depth:
            angle_depth -= 1
        elif angle_depth == 0 and (char == "." or name.startswith("::", index)):
            has_top_level_separator = True
            break
        index += 1
    if not has_top_level_separator:
        return []
    parts = _QUALIFIED_NAME_SEPARATOR_RE.split(name)
    if len(parts) <= 1 or any(not part.strip() for part in parts):
        return []
    normalized_parts = [_codegraph_source_component(part) for part in parts]
    if any(not part for part in normalized_parts):
        return []
    return normalized_parts


def _source_name_fqn_candidates(name: str) -> List[str]:
    """Return complete source and literal-FQN-prefix interpretations of a name."""
    candidates: List[str] = []

    # Treat the complete spelling as a source-qualified name. This covers pure
    # ``::``, pure dot, and mixed source scopes, including template arguments
    # whose namespaces become separate CodeGraph components.
    normalized_parts = _normalized_source_name_parts(name)
    if normalized_parts:
        candidates.append("::".join(normalized_parts))

    # A leading FM-Agent FQN component may contain literal dots (for example,
    # ``foo.test-go``). At every explicit FQN boundary, preserve the prefix and
    # independently interpret the remaining qualified tail as source spelling.
    for boundary in re.finditer(r"::", name):
        literal_prefix = name[: boundary.start()].strip()
        source_tail = name[boundary.end() :].strip()
        if not literal_prefix or not source_tail:
            continue
        if any(not part.strip() for part in literal_prefix.split("::")):
            continue
        normalized_tail_parts = _normalized_source_name_parts(source_tail)
        if normalized_tail_parts:
            normalized_tail = "::".join(normalized_tail_parts)
            candidates.append(f"{literal_prefix}::{normalized_tail}")
    return list(dict.fromkeys(candidates))


def _qualified_source_name_compatible(name: str, callee_fqn: str) -> bool:
    """Allow only receiver/member spellings safe for bare fallback."""
    parts = _qualified_name_parts(name)
    if len(parts) == 2 and parts[0].strip().casefold() in {"self", "this"}:
        return True

    # Preserve the existing receiver/member spelling used by languages that
    # write ``Engine.start`` while requiring its receiver to identify the same
    # class as the target FQN.  This avoids treating arbitrary dotted names
    # such as ``myself.run`` as pseudo-receivers.
    fqn_parts = [part for part in callee_fqn.split("::") if part]
    if (
        len(parts) == 2
        and len(fqn_parts) >= 2
        and parts[0].strip().casefold() == fqn_parts[-2].casefold()
        and parts[1].strip().casefold() == fqn_parts[-1].casefold()
    ):
        return True

    # Qualified names have already had an exact/normalized suffix comparison
    # opportunity.  Never trim arbitrary namespace components during the bare
    # fallback: a shared tail is not evidence that the qualifiers agree.
    return False


def _fqn_has_component_suffix(callee_fqn: str, candidate: str) -> bool:
    """Return whether candidate equals a complete ``::``-delimited FQN suffix."""
    callee_parts = callee_fqn.split("::")
    candidate_parts = candidate.split("::")
    return (
        bool(candidate_parts)
        and all(callee_parts)
        and all(candidate_parts)
        and len(candidate_parts) <= len(callee_parts)
        and callee_parts[-len(candidate_parts) :] == candidate_parts
    )


def _info_name_matches_fqn(info_name: str, callee_fqn: str) -> bool:
    """Match a complete FQN or a source-language-qualified suffix."""
    variants = [info_name]
    undecorated = _name_without_call_decoration(info_name)
    if undecorated != info_name:
        variants.append(undecorated)
    for variant in variants:
        if variant == callee_fqn:
            return True
        # Preserve literal FM-Agent FQN components, which may contain dots.
        if "::" in variant and _fqn_has_component_suffix(callee_fqn, variant):
            return True
        if _erlang_name_matches_fqn(variant, callee_fqn):
            return True
        # Do not let the component normalizer erase a remaining call layer.
        # Literal suffix matching above still supports ``operator()`` and one
        # preserved call decoration when the FQN contains it.
        if _trailing_parenthesized_group(variant) and not _is_operator_call_name(variant):
            continue
        if any(
            _fqn_has_component_suffix(callee_fqn, candidate)
            for candidate in _source_name_fqn_candidates(variant)
        ):
            return True

    group = _trailing_parenthesized_group(info_name)
    if group:
        prefix, label = group
        fqn_parts = [part for part in callee_fqn.split("::") if part]
        if (
            label
            and len(fqn_parts) >= 2
            and label.casefold() == fqn_parts[-1].casefold()
            and _bare_name_for_matching(prefix).casefold()
            == fqn_parts[-2].casefold()
        ):
            return True
    return False


def _info_name_matches_alias(info_name: str, alias: str) -> bool:
    """Match an exact alias while accepting equivalent source qualifiers."""
    variants = [info_name]
    undecorated = _name_without_call_decoration(info_name)
    if undecorated != info_name:
        variants.append(undecorated)
    for variant in variants:
        if variant == alias:
            return True
        if _erlang_name_parts(variant) and _erlang_name_parts(alias):
            left = _erlang_name_parts(variant)
            right = _erlang_name_parts(alias)
            if left and right and left == right:
                return True
        if _is_qualified_name(variant) and _is_qualified_name(alias):
            if _qualified_name_parts(variant) == _qualified_name_parts(alias):
                return True

    group = _trailing_parenthesized_group(info_name)
    if group:
        prefix, label = group
        alias_parts = alias.split("::") if "::" in alias else _qualified_name_parts(alias)
        if (
            label
            and len(alias_parts) >= 2
            and label.casefold() == alias_parts[-1].casefold()
            and _bare_name_for_matching(prefix).casefold()
            == alias_parts[-2].casefold()
        ):
            return True
    return False


def _info_line_mentions_name(first_line: str, name: str) -> bool:
    if not name:
        return False
    if "::" in name:
        return name in first_line
    return bool(re.search(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?:\s*\(|\b)", first_line))
