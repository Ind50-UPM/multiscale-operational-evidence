"""Internationalization helpers for the Model_B command-line interface.

This module provides the minimal translation machinery used by
:mod:`iabm_behavior.main` to localise argparse help strings and the
"report saved to" console messages. It does not depend on the GNU gettext
toolchain: message catalogs are plain-text ``.po`` files parsed by a small
in-house reader.

Main public objects:

* :func:`setup_i18n` -- returns a ``str -> str`` translation callable for a
  language code.

Data contract:

* Catalogs are looked up at
  ``<package root>/locales/<lang>/LC_MESSAGES/messages.po``, where
  ``<package root>`` is the parent directory of the ``iabm_behavior`` package
  (i.e. ``src/Model_B``). Only ``msgid``/``msgstr`` pairs (including
  multi-line continuation strings) are interpreted; plural forms, contexts
  (``msgctxt``) and flags are ignored.
* English (``"en"``) is the source language and is served by an identity
  function, without touching the file system.

.. mermaid::

   flowchart LR
       A["setup_i18n(lang)"] -->|"lang == 'en'"| B[identity lambda]
       A -->|other lang| C[_load_po_catalog]
       C --> D[_decode_po_string]
       C --> E["dict msgid -> msgstr"]
       E --> F["lambda m: catalog.get(m, m)"]

Maintenance notes:
    The code of this module is identical to ``iabm_semantics.utils`` in
    Model_C (only the module docstring differs); keep both copies in sync when
    fixing the parser. The ``.po`` files themselves are user-facing artefacts
    and must not be edited as part of code refactors. Missing catalogs are not
    an error: an unknown language silently degrades to untranslated (English)
    messages.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Callable


def setup_i18n(lang: str = "en") -> Callable[[str], str]:
    """Return a translation function for the requested interface language.

    For ``"en"`` an identity function is returned. For any other code the
    ``messages.po`` catalog under ``locales/<lang>/LC_MESSAGES`` is parsed once
    and captured in a closure.

    Args:
        lang: ISO 639-1 language code requested by the caller. The CLI only
            offers ``"en"`` and ``"es"``, but any string is accepted here; an
            unknown code simply resolves to a non-existent catalog.

    Returns:
        A callable that translates user-facing CLI strings into the requested
        language, falling back to the original message when no translation
        exists (or when the catalog file is missing).

    Raises:
        ValueError: Propagated from :func:`_load_po_catalog` when the catalog
            contains a malformed quoted fragment.
        SyntaxError: Propagated from :func:`_load_po_catalog` for the same
            reason.

    Examples:
        >>> _ = setup_i18n("en")
        >>> _("Interface language.")
        'Interface language.'

    See Also:
        :func:`_load_po_catalog`
    """
    if lang == "en":
        return lambda message: message

    # Note: parents[1] of this file is the package root (src/Model_B), where
    # the ``locales`` directory is shipped alongside the Python package.
    catalog_path = (
        Path(__file__).resolve().parents[1] / "locales" / lang / "LC_MESSAGES" / "messages.po"
    )
    catalog = _load_po_catalog(catalog_path)
    return lambda message: catalog.get(message, message)


def _load_po_catalog(path: Path) -> dict[str, str]:
    """Parse a minimal gettext catalog into a dictionary.

    The parser is a small line-oriented state machine: a ``msgid`` line opens a
    new entry (flushing the previous one), a ``msgstr`` line switches the
    active field, and bare quoted lines are appended to whichever field is
    active. Blank lines and ``#`` comments are skipped.

    Args:
        path: Path to the UTF-8 encoded ``.po`` file to parse.

    Returns:
        A dictionary mapping ``msgid`` values to translated strings. An empty
        dictionary is returned when ``path`` does not exist. Entries with an
        empty ``msgstr`` map to their own ``msgid``; the catalog header (empty
        ``msgid``) is discarded.

    Raises:
        ValueError: Propagated from :func:`ast.literal_eval` when a quoted
            fragment is not a valid literal.
        SyntaxError: Propagated from :func:`ast.literal_eval` for malformed
            fragments.

    Notes:
        The project deliberately uses a tiny parser instead of binding the full
        gettext toolchain so tests and local CLI usage remain lightweight. The
        parser only implements the subset of ``.po`` syntax required by the
        repository message catalogs.
    """
    if not path.exists():
        return {}

    catalog: dict[str, str] = {}
    current_field: str | None = None
    current_msgid: list[str] = []
    current_msgstr: list[str] = []

    def flush_entry() -> None:
        """Commit the entry currently being accumulated into ``catalog``.

        Reads the enclosing ``current_msgid``/``current_msgstr`` buffers via
        closure. Nothing is stored when no ``msgid`` has been seen yet or when
        the joined ``msgid`` is empty (the ``.po`` header entry).

        Returns:
            ``None``. ``catalog`` is mutated in place.
        """
        if not current_msgid:
            return
        msgid = "".join(current_msgid)
        msgstr = "".join(current_msgstr)
        if msgid:
            # Why: untranslated entries (empty msgstr) fall back to the source text.
            catalog[msgid] = msgstr or msgid

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("msgid "):
            flush_entry()
            # Note: rebinding (not mutating) the buffers is safe because
            # flush_entry reads them by closure at call time.
            current_msgid = [_decode_po_string(line[6:])]
            current_msgstr = []
            current_field = "msgid"
            continue
        if line.startswith("msgstr "):
            current_msgstr = [_decode_po_string(line[7:])]
            current_field = "msgstr"
            continue
        if line.startswith('"'):
            # Note: continuation line of a multi-line msgid/msgstr.
            if current_field == "msgid":
                current_msgid.append(_decode_po_string(line))
            elif current_field == "msgstr":
                current_msgstr.append(_decode_po_string(line))

    # Invariant: the last entry has no following ``msgid`` to trigger a flush.
    flush_entry()
    return catalog


def _decode_po_string(value: str) -> str:
    """Decode a quoted gettext fragment into plain Python text.

    Args:
        value: One quoted fragment from a gettext catalog, including the
            surrounding double quotes.

    Returns:
        The decoded Unicode string represented by the fragment, with backslash
        escape sequences resolved.

    Raises:
        ValueError: If ``value`` is not a literal expression.
        SyntaxError: If ``value`` is not syntactically valid Python.

    Notes:
        ``ast.literal_eval`` is sufficient here because gettext fragments are
        serialized as Python-compatible quoted string literals. Using it keeps
        escaping behavior consistent across languages without hand-written
        unescaping logic. The result type is not checked: a fragment holding a
        non-string literal would be returned unchanged.
    """
    return ast.literal_eval(value)
