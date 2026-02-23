# tests/unit/models/test_filter_models.py

import pytest


def test_news_item_for_filter():
    """Test NewsItemForFilter model."""
    from ashare_analyzer.models.results import NewsItemForFilter

    item = NewsItemForFilter(
        index=0,
        title="Test Title",
        snippet="Test snippet",
        source="example.com",
        url="https://example.com/test",
        published_date="2026-02-22",
    )
    assert item.index == 0
    assert item.title == "Test Title"
    assert item.published_date == "2026-02-22"


def test_news_filter_result():
    """Test NewsFilterResult model."""
    from ashare_analyzer.models.results import NewsFilterResult

    result = NewsFilterResult(index=0, is_relevant=True, freshness="fresh")
    assert result.index == 0
    assert result.is_relevant is True
    assert result.freshness == "fresh"


def test_news_filter_result_freshness_validation():
    """Test NewsFilterResult freshness must be valid."""
    from pydantic import ValidationError

    from ashare_analyzer.models.results import NewsFilterResult

    # Valid values - test each explicitly for type checker
    result_fresh = NewsFilterResult(index=0, is_relevant=True, freshness="fresh")
    assert result_fresh.freshness == "fresh"

    result_acceptable = NewsFilterResult(index=0, is_relevant=True, freshness="acceptable")
    assert result_acceptable.freshness == "acceptable"

    result_stale = NewsFilterResult(index=0, is_relevant=True, freshness="stale")
    assert result_stale.freshness == "stale"

    # Invalid value - should raise ValidationError
    # Use model_validate to test runtime validation with invalid literal
    with pytest.raises(ValidationError):
        NewsFilterResult.model_validate(
            {
                "index": 0,
                "is_relevant": True,
                "freshness": "invalid",
            }
        )
