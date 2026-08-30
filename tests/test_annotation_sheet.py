from counterfeint.experiments.annotation_sheet import (
    normalize_label_row,
    render_case_markdown,
    types_agree,
)


def test_collapses_identical_fraudster_copies():
    md = "\n".join(
        render_case_markdown(
            "C001",
            {
                "task_id": "task_2",
                "n_ads": 17,
                "verdicts": [],
                "findings": {},
                "proposals": [
                    {
                        "ad_id": "ad_006",
                        "category": "ecommerce",
                        "ad_copy": "Buy now",
                        "targeting": "US",
                        "landing": "x.com",
                    },
                    {
                        "ad_id": "ad_007",
                        "category": "ecommerce",
                        "ad_copy": "Buy now",
                        "targeting": "US",
                        "landing": "x.com",
                    },
                ],
            },
        )
    )
    assert "medium marketplace" in md
    assert "17 ads" in md
    assert "×2 identical" in md
    assert md.count("Buy now") == 1
    assert md.index("Fraudster proposals") < md.index("Investigator verdicts")


def test_normalize_dual_track_labels():
    row = normalize_label_row(
        {
            "case_id": "C001",
            "track_a_present": "no",
            "track_a_type": "none",
            "track_b_present": "yes",
            "track_b_type": "clones",
        }
    )
    assert row["issue_present"] is True
    assert row["track_a_present"] is False
    assert row["track_b_present"] is True
    assert row["issue_type"] == "clones"


def test_coarse_types_match_fine_key():
    assert types_agree("template_repetition", "clones")
    assert types_agree("missing_citation", "unsupported")
    assert types_agree("incoherent_rationale", "incoherent")
    assert types_agree("parameter_mismatch", "mismatch")
    assert types_agree("branding_anomaly", "mismatch")
    assert not types_agree("gibberish", "clones")


def test_renders_full_landing_page():
    md = "\n".join(
        render_case_markdown(
            "C001",
            {
                "task_id": "task_1",
                "n_ads": 8,
                "verdicts": [],
                "findings": {
                    "ad_001": (
                        "Domain: store5085.site\n"
                        "Key claims on page:\n"
                        "  - Guaranteed returns\n"
                        "Content summary: wire transfer. Grammar errors."
                    )
                },
                "proposals": [],
            },
        )
    )
    assert "Guaranteed returns" in md
    assert "wire transfer" in md
    assert "partial" not in md.lower()
