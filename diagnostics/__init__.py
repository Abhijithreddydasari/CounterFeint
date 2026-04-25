"""
Live diagnostic scripts for the CounterFeint multi-agent stack.

These are **runnable scripts** (each has a ``main()`` and a
``if __name__ == "__main__":`` shim), not library code, but they live
inside the package so they can use relative imports and ship as part
of the wheel.

Modules:

  * :mod:`.verify_fraudster`    — Live diagnostic for the LLM Fraudster
                                   against Ollama. Confirms it receives the
                                   Fraudster observation (incl. the new
                                   ``my_proposal_signals`` ledger), produces
                                   schema-valid JSON, and reacts to the
                                   Investigator's verdicts on follow-up turns.

  * :mod:`.verify_investigator` — Symmetric live diagnostic for the
                                   ``HFInvestigator`` (the local-transformers
                                   policy GRPO trains). Confirms the new
                                   ``evidence_ledger`` reaches the model and
                                   that JSON output is well-formed.

  * :mod:`.replay_match`        — Full multi-agent episode replay with both
                                   LLMs talking to each other. Prints every
                                   prompt + raw completion + parsed action +
                                   reward in chronological order, and
                                   optionally writes a Markdown transcript
                                   into ``counterfeint/convo_logging/``.

Each module is invoked the same way::

    python -m counterfeint.diagnostics.verify_fraudster
    python -m counterfeint.diagnostics.verify_investigator
    python -m counterfeint.diagnostics.replay_match --task task_2 --seed 42

A direct ``python counterfeint/diagnostics/<script>.py`` invocation also
works (each script bootstraps ``sys.path`` when ``__package__`` is empty).
"""
