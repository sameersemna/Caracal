"""Word grouping of whisper sub-word tokens (docs/04 §10).

The Tagesschau fixture is real: whisper emits ``Tag`` + ``ess`` + ``chau`` and ``Sek`` +
``unden`` + ``.``, so a cue broken on tokens would split "Tagesschau" in the middle.
"""

from __future__ import annotations

from tests.fakes.whisper import load_fixture

from caracal.asr import Word, word_groups


def _word(
    text: str,
    *,
    start: float = 0.0,
    end: float = 0.1,
    speaker: str | None = None,
    space_before: bool = True,
) -> Word:
    return Word(text=text, start=start, end=end, prob=0.9, speaker=speaker, space_before=space_before)


def _tagesschau_words() -> list[Word]:
    payload = load_fixture("tagesschau_seg0")
    return [
        Word(
            text=item["text"],
            start=item["start"],
            end=item["end"],
            prob=item["prob"],
            speaker=item.get("speaker"),
            space_before=item["space_before"],
        )
        for item in payload["words"]
    ]


def test_real_tagesschau_tokens_form_whole_words() -> None:
    groups = word_groups(_tagesschau_words())

    assert [group.text for group in groups] == [
        "Guten",
        "Tag,",
        "hier",
        "ist",
        "die",
        "Tagesschau",
        "in",
        "100",
        "Sekunden.",
    ]
    assert [group.tokens for group in groups] == [
        (0, 0),
        (1, 2),
        (3, 3),
        (4, 4),
        (5, 5),
        (6, 8),
        (9, 9),
        (10, 10),
        (11, 13),
    ]


def test_group_times_span_their_first_and_last_token() -> None:
    groups = word_groups(_tagesschau_words())
    tagesschau = groups[5]
    sekunden = groups[8]

    assert (tagesschau.start, tagesschau.end) == (0.99, 1.47)
    assert (sekunden.start, sekunden.end) == (2.08, 2.58)
    assert groups[0].space_before is True
    assert tagesschau.space_before is True  # "Tag" carried the leading space


def test_punctuation_tokens_attach_to_the_preceding_word() -> None:
    words = [
        _word("Tag", space_before=False),
        _word("ess", space_before=False),
        _word("chau", space_before=False),
        _word(",", space_before=False),
        _word("now", space_before=True),
    ]
    groups = word_groups(words)

    assert [group.text for group in groups] == ["Tagesschau,", "now"]
    assert [group.tokens for group in groups] == [(0, 3), (4, 4)]


def test_first_token_always_starts_a_group_even_without_a_leading_space() -> None:
    groups = word_groups([_word("Tag", space_before=False), _word("ess", space_before=False)])

    assert [group.text for group in groups] == ["Tagess"]
    assert groups[0].tokens == (0, 1)


def test_empty_input_has_no_groups() -> None:
    assert word_groups([]) == []


def test_speaker_is_the_group_majority() -> None:
    words = [
        _word("Tag", speaker="S2", space_before=True),
        _word("ess", speaker="S1", space_before=False),
        _word("chau", speaker="S1", space_before=False),
    ]
    groups = word_groups(words)

    assert [group.speaker for group in groups] == ["S1"]


def test_speaker_tie_goes_to_the_first_token() -> None:
    words = [
        _word("Tag", speaker="S2", space_before=True),
        _word("ess", speaker="S1", space_before=False),
    ]
    groups = word_groups(words)

    assert [group.speaker for group in groups] == ["S2"]


def test_unassigned_tokens_do_not_outvote_a_speaker() -> None:
    words = [
        _word("Tag", speaker="S1", space_before=True),
        _word("ess", speaker=None, space_before=False),
        _word("chau", speaker=None, space_before=False),
    ]

    assert word_groups(words)[0].speaker == "S1"
    assert word_groups([_word("solo", speaker=None)])[0].speaker is None
