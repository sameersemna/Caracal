"""Word grouping of whisper sub-word tokens (docs/04 §10).

The Tagesschau fixture is real: whisper emits ``Tag`` + ``ess`` + ``chau`` and ``Sek`` +
``unden`` + ``.``, so a cue broken on tokens would split "Tagesschau" in the middle.
"""

from __future__ import annotations

import pytest
from tests.fakes.whisper import load_fixture

from caracal.asr import Word, word_groups
from caracal.asr.whisper import parse_whisper_response


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


def _parsed_words(name: str) -> list[Word]:
    return [word for segment in parse_whisper_response(load_fixture(name)).segments for word in segment.words]


def test_real_chinese_tokens_without_spaces_are_each_breakable() -> None:
    """zh_speech40 has 0 space-led tokens: grouping must still break between characters."""
    words = _parsed_words("zh_speech40")

    assert not any(word.space_before for word in words)
    groups = word_groups(words)

    assert [group.text for group in groups[:8]] == list("欧洲阿里亚纳火箭")
    assert "".join(group.text for group in groups) == "".join(word.text for word in words)
    # every character is breakable: no group is a whole sentence
    assert max(len(group.text) for group in groups) <= 5
    assert sum(len(group.text) for group in groups) / len(groups) < 2
    spans = [index for group in groups for index in range(group.tokens[0], group.tokens[1] + 1)]
    assert spans == list(range(len(words)))  # tokens are neither lost nor duplicated


def test_chinese_punctuation_attaches_to_the_previous_character() -> None:
    groups = word_groups(_parsed_words("zh_speech40"))
    texts = [group.text for group in groups]

    assert "水、" in texts
    assert "食、" in texts
    assert "气、" in texts
    water = next(group for group in groups if group.text == "水、")
    assert water.tokens[1] - water.tokens[0] == 1  # the 、 token rides with 水


def test_a_latin_sub_word_token_still_continues_its_word() -> None:
    """The no-space rule must not break Latin sub-word pieces apart (Tagesschau)."""
    groups = word_groups(_tagesschau_words())

    assert [group.text for group in groups][5] == "Tagesschau"


_NO_SPACE_PAIRS = (
    ("ส", "ว"),  # Thai
    ("こ", "ん"),  # Hiragana
    ("カ", "ナ"),  # Katakana
    ("ສ", "ວ"),  # Lao
    ("ក", "រ"),  # Khmer
    ("က", "ခ"),  # Myanmar
    ("ཀ", "ཁ"),  # Tibetan
)


@pytest.mark.parametrize(("first", "second"), _NO_SPACE_PAIRS)
def test_other_no_space_scripts_break_between_characters(first: str, second: str) -> None:
    groups = word_groups([_word(first, space_before=False), _word(second, space_before=False)])

    assert [group.text for group in groups] == [first, second]


def test_a_combining_mark_stays_with_its_base_character() -> None:
    words = [
        _word("ก", space_before=False),  # Thai base
        _word("\u0e48", space_before=False),  # Thai tone mark: not a word start
        _word("ว", space_before=False),
    ]

    assert [group.text for group in word_groups(words)] == ["ก\u0e48", "ว"]
