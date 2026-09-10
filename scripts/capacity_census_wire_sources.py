"""Source-coordinate codec and local access observations, not AST admission."""

from capacity_census_wire_adapters import CodecCase, canonical


def source_cases():
    cases = []

    def add(carrier, codec, name, value, expected, error=None):
        cases.append(
            CodecCase(
                f"{carrier}/{codec}/{name}",
                "source-location",
                carrier,
                codec,
                canonical(value),
                expected,
                error,
            )
        )

    for offset, length in (
        (0, 0),
        (0, 2),
        (2, 1),
        (18446744073709551615, 0),
        (18446744073709551615, 1),
    ):
        value = {"offset": offset, "len": length}
        add("Span", "json", f"range-{offset}-{length}", value, value)
    for name, value in (
        ("negative-offset", {"offset": -1, "len": 0}),
        ("float-offset", {"offset": 0.0, "len": 0}),
        ("overflow-offset", {"offset": 18446744073709551616, "len": 0}),
        ("missing-extent", {"offset": 0}),
        ("extra-numeric", {"offset": 0, "len": 0, "value": 1.5}),
    ):
        add("Span", "json", name, value, None)
    for name, span, expected, error in (
        ("utf8-character", {"offset": 0, "len": 2}, {"slice": "é"}, None),
        ("empty-at-end", {"offset": 3, "len": 0}, {"slice": ""}, None),
        ("split-start", {"offset": 1, "len": 1}, None, "UTF-8"),
        ("split-end", {"offset": 0, "len": 1}, None, "UTF-8"),
        ("out-of-bounds", {"offset": 3, "len": 1}, None, "outside"),
        (
            "endpoint-overflow",
            {"offset": 18446744073709551615, "len": 1},
            None,
            "endpoint exceeds u64",
        ),
    ):
        add("Span", "slice", name, {"span": span, "source": "éx"}, expected, error)
    for name, value in (
        ("absent", None),
        ("point", {"span": "point", "offset": 0}),
        ("empty-range", {"span": "range", "offset": 0, "len": 0}),
        ("range", {"span": "range", "offset": 0, "len": 2}),
        ("foreign-point", {"span": "point", "offset": 18446744073709551615}),
    ):
        expected = {
            "json": value,
            "offset": None if value is None else value["offset"],
            "extent": None if value is None else value.get("len"),
        }
        add("DiagnosticLocation", "json", name, value, expected)
    for name, value in (
        ("point-fabricated-extent", {"span": "point", "offset": 0, "len": 0}),
        ("range-missing-extent", {"span": "range", "offset": 0}),
        ("unknown-tag", {"span": "numeric", "offset": 0}),
        ("float-offset", {"span": "point", "offset": 0.0}),
        ("negative-extent", {"span": "range", "offset": 0, "len": -1}),
    ):
        add("DiagnosticLocation", "json", name, value, None)
    return cases
