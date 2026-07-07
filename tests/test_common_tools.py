from common.tools import mock_search


def test_mock_search_is_deterministic():
    assert mock_search("photosynthesis") == mock_search("photosynthesis")


def test_mock_search_mentions_the_subtopic():
    result = mock_search("quantum tunneling")
    assert "quantum tunneling" in result


def test_mock_search_returns_nonempty_for_any_subtopic():
    assert mock_search("anything at all").strip() != ""
