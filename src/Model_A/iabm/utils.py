"""Internationalization helpers for the Model_A command-line interface.

The CLI in :mod:`iabm.main` wraps every user-facing message in a translator
callable. This module builds that callable from the human-maintained GNU
gettext catalogs shipped with the package, without requiring compiled
``.mo`` files or the :mod:`gettext` machinery.

Main public object:
    :func:`setup_i18n` -- returns a ``str -> str`` translation function.

Catalog contract:
    Catalogs live at ``<Model_A>/locales/<lang>/LC_MESSAGES/messages.po``,
    i.e. one directory above the ``iabm`` package. Only singular ``msgid`` /
    ``msgstr`` pairs (optionally split across several quoted lines) are
    understood. Plural forms (``msgid_plural``/``msgstr[n]``) and contexts
    (``msgctxt``) are not supported.

.. mermaid::

    flowchart LR
        L{lang == en?} -- yes --> ID[identity function]
        L -- no --> PO[locales/lang/LC_MESSAGES/messages.po]
        PO --> P[_load_po_catalog]
        P --> M[dict msgid to msgstr]
        M --> F[lookup with fallback to msgid]

Maintenance notes:
    English message identifiers are the source of truth: any missing or
    untranslated entry falls back to the original English text, so adding a
    new CLI message never breaks other languages. When adding messages to
    :mod:`iabm.main`, update the ``locales/<lang>/LC_MESSAGES/messages.po``
    files (``pyproject.toml`` also lists an optional ``locales/messages.pot``
    template); do not change the ``msgid`` strings in code without updating
    catalogs.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Callable


def setup_i18n(lang: str = "en") -> Callable[[str], str]:
    """Return a translation function for the requested interface language.

    The project stores human-maintained translations in ``locales/*/LC_MESSAGES``
    as ``.po`` files. This helper reads those catalogs directly so the CLI can be
    translated even when ``.mo`` files have not been compiled yet.

    Args:
        lang (str): ISO language code requested by the user (the CLI offers
            ``"en"`` and ``"es"``). ``"en"`` short-circuits to the identity
            function without touching the file system. Any other code is
            mapped to ``locales/<lang>/LC_MESSAGES/messages.po``.

    Returns:
        Callable[[str], str]: A callable compatible with ``gettext`` usage that
        translates a message identifier into the configured language. English
        falls back to the original message identifiers. For non-English
        languages, identifiers missing from the catalog (or a missing catalog
        file) are also returned unchanged.

    Examples:
        >>> _ = setup_i18n("en")
        >>> _("Interface language.")
        'Interface language.'

    See Also:
        :func:`_load_po_catalog`
    """
    if lang == "en":
        return lambda message: message

    # Note: parents[1] of this file is the Model_A package root, where the
    # ``locales`` directory lives next to ``iabm/``.
    locale_dir = Path(__file__).resolve().parents[1] / "locales" / lang / "LC_MESSAGES"
    catalog_path = locale_dir / "messages.po"
    catalog = _load_po_catalog(catalog_path)
    return lambda message: catalog.get(message, message)


def _load_po_catalog(path: Path) -> dict[str, str]:
    """Parse a minimal ``.po`` catalog into an in-memory translation mapping.

    The parser intentionally supports the subset of the GNU gettext format used
    by this repository: singular ``msgid`` and ``msgstr`` entries with optional
    multiline string fragments.

    Args:
        path (pathlib.Path): Path to the ``.po`` file to parse (UTF-8 encoded).

    Returns:
        dict[str, str]: A dictionary keyed by source ``msgid`` values. An empty
        ``msgstr`` is stored as the ``msgid`` itself (untranslated entries fall
        back to English). The header entry (empty ``msgid``) is skipped. A
        missing file yields an empty dictionary.

    Raises:
        ValueError: Propagated from :func:`_decode_po_string` if a quoted
            fragment is not a valid Python string literal.
        SyntaxError: Propagated from :func:`_decode_po_string` for malformed
            fragments.

    Notes:
        The parser is a small state machine: ``current_field`` tracks whether
        continuation lines (lines starting with ``"``) belong to the current
        ``msgid`` or ``msgstr``. An entry is flushed when the next ``msgid``
        starts and once more at end of file. Blank lines and ``#`` comments
        (including ``#,`` flags and ``#:`` references) are ignored.
    """
    if not path.exists():
        return {}

    catalog: dict[str, str] = {}
    current_field: str | None = None
    current_msgid: list[str] = []
    current_msgstr: list[str] = []

    def flush_entry() -> None:
        """Store the accumulated ``msgid``/``msgstr`` pair in ``catalog``.

        Concatenates the collected fragments of the current entry. Entries
        with an empty ``msgid`` (the catalog header) are ignored, and an empty
        ``msgstr`` falls back to the ``msgid``.
        """
        if not current_msgid:
            return
        msgid = "".join(current_msgid)
        msgstr = "".join(current_msgstr)
        if msgid:
            catalog[msgid] = msgstr or msgid

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("msgid "):
            flush_entry()
            current_msgid = [_decode_po_string(line[6:])]
            current_msgstr = []
            current_field = "msgid"
            continue
        if line.startswith("msgstr "):
            current_msgstr = [_decode_po_string(line[7:])]
            current_field = "msgstr"
            continue
        if line.startswith('"'):
            # Note: continuation fragment; append to whichever field was
            # opened last.
            if current_field == "msgid":
                current_msgid.append(_decode_po_string(line))
            elif current_field == "msgstr":
                current_msgstr.append(_decode_po_string(line))

    flush_entry()
    return catalog


def _decode_po_string(value: str) -> str:
    """Decode a quoted gettext string fragment into plain Python text.

    Args:
        value (str): One quoted fragment from a gettext catalog, including its
            surrounding double quotes (for example ``"Fold {}"``).

    Returns:
        str: The decoded Unicode string represented by the fragment, with
        backslash escape sequences resolved.

    Raises:
        ValueError: If ``value`` is not a valid Python literal.
        SyntaxError: If ``value`` is malformed (e.g. unbalanced quotes).

    Notes:
        :func:`ast.literal_eval` is used because gettext escapes are a subset
        of Python string-literal escapes; it never executes code.
    """
    return ast.literal_eval(value)
