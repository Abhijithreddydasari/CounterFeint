"""
DEPRECATED — moved to :mod:`counterfeint.diagnostics.verify_fraudster`.

Kept as a thin shim so older docs / commands like
``python -m counterfeint.verify_fraudster`` and
``python counterfeint/verify_fraudster.py`` keep working. The script
now lives next to its sibling diagnostics
(:mod:`counterfeint.diagnostics.verify_investigator`,
:mod:`counterfeint.diagnostics.replay_match`).
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    _REPO_ROOT = Path(__file__).resolve().parent.parent
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

from counterfeint.diagnostics.verify_fraudster import main


if __name__ == "__main__":
    print(
        "[NOTE] counterfeint/verify_fraudster.py has moved to "
        "counterfeint/diagnostics/verify_fraudster.py — this shim still works.",
        file=sys.stderr,
    )
    sys.exit(main())
