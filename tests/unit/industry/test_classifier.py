"""Tests for industry classification functions."""

from ashare_analyzer.industry.classifier import (
    get_all_cyclical_industries,
    get_all_financial_industries,
    get_industry_category,
    is_cyclical_industry,
    is_financial_industry,
)
from ashare_analyzer.industry.constants import IndustryCategory


class TestIsCyclicalIndustry:
    """Tests for is_cyclical_industry function."""

    def test_exact_match_true(self):
        """Test exact match returns True for cyclical industries."""
        assert is_cyclical_industry("有色金属") is True
        assert is_cyclical_industry("煤炭") is True
        assert is_cyclical_industry("钢铁") is True
        assert is_cyclical_industry("化工") is True
        assert is_cyclical_industry("汽车") is True
        assert is_cyclical_industry("房地产") is True

    def test_exact_match_false(self):
        """Test exact match returns False for non-cyclical industries."""
        assert is_cyclical_industry("银行") is False
        assert is_cyclical_industry("计算机") is False
        assert is_cyclical_industry("医药生物") is False
        assert is_cyclical_industry("食品饮料") is False

    def test_alias_resolution(self):
        """Test alias resolution works for cyclical industries."""
        assert is_cyclical_industry("煤炭开采") is True
        assert is_cyclical_industry("钢铁冶炼") is True
        assert is_cyclical_industry("房地产开发") is True

    def test_keyword_matching(self):
        """Test keyword matching works for cyclical industries."""
        assert is_cyclical_industry("锂电池材料") is True
        assert is_cyclical_industry("稀土永磁") is True

    def test_empty_string_false(self):
        """Test empty string returns False."""
        assert is_cyclical_industry("") is False

    def test_none_returns_false(self):
        """Test None returns False."""
        assert is_cyclical_industry(None) is False  # type: ignore

    def test_normalized_names(self):
        """Test normalized industry names work."""
        assert is_cyclical_industry("有色金属行业") is True
        assert is_cyclical_industry("煤炭板块") is True
        assert is_cyclical_industry("  钢铁  ") is True


class TestIsFinancialIndustry:
    """Tests for is_financial_industry function."""

    def test_exact_match_true(self):
        """Test exact match returns True for financial industries."""
        assert is_financial_industry("银行") is True
        assert is_financial_industry("保险") is True
        assert is_financial_industry("证券") is True
        assert is_financial_industry("多元金融") is True
        assert is_financial_industry("信托") is True
        assert is_financial_industry("期货") is True

    def test_exact_match_false(self):
        """Test exact match returns False for non-financial industries."""
        assert is_financial_industry("有色金属") is False
        assert is_financial_industry("计算机") is False
        assert is_financial_industry("煤炭") is False

    def test_alias_resolution(self):
        """Test alias resolution works for financial industries."""
        assert is_financial_industry("券商") is True
        assert is_financial_industry("证券公司") is True
        assert is_financial_industry("商业银行") is True
        assert is_financial_industry("保险公司") is True

    def test_keyword_matching(self):
        """Test keyword matching works for financial industries."""
        # "券商投资" should match via keyword
        assert is_financial_industry("券商投资") is True

    def test_empty_string_false(self):
        """Test empty string returns False."""
        assert is_financial_industry("") is False

    def test_none_returns_false(self):
        """Test None returns False."""
        assert is_financial_industry(None) is False  # type: ignore

    def test_normalized_names(self):
        """Test normalized industry names work."""
        assert is_financial_industry("银行行业") is True
        assert is_financial_industry("证券板块") is True


class TestGetIndustryCategory:
    """Tests for get_industry_category function."""

    def test_cyclical_category(self):
        """Test cyclical industry returns CYCLICAL category."""
        assert get_industry_category("有色金属") == IndustryCategory.CYCLICAL
        assert get_industry_category("煤炭") == IndustryCategory.CYCLICAL
        assert get_industry_category("钢铁") == IndustryCategory.CYCLICAL

    def test_financial_category(self):
        """Test financial industry returns FINANCIAL category."""
        assert get_industry_category("银行") == IndustryCategory.FINANCIAL
        assert get_industry_category("保险") == IndustryCategory.FINANCIAL
        assert get_industry_category("证券") == IndustryCategory.FINANCIAL

    def test_other_category_for_unknown(self):
        """Test unknown industry returns OTHER category."""
        assert get_industry_category("计算机") == IndustryCategory.OTHER
        assert get_industry_category("医药生物") == IndustryCategory.OTHER
        assert get_industry_category("食品饮料") == IndustryCategory.OTHER

    def test_empty_string_returns_other(self):
        """Test empty string returns OTHER category."""
        assert get_industry_category("") == IndustryCategory.OTHER

    def test_none_returns_other(self):
        """Test None returns OTHER category."""
        assert get_industry_category(None) == IndustryCategory.OTHER  # type: ignore

    def test_category_with_alias(self):
        """Test category resolution with alias."""
        assert get_industry_category("券商") == IndustryCategory.FINANCIAL
        assert get_industry_category("煤炭开采") == IndustryCategory.CYCLICAL


class TestGetAllCyclicalIndustries:
    """Tests for get_all_cyclical_industries function."""

    def test_returns_set(self):
        """Test returns a set."""
        result = get_all_cyclical_industries()
        assert isinstance(result, set)

    def test_not_empty(self):
        """Test set is not empty."""
        result = get_all_cyclical_industries()
        assert len(result) > 0

    def test_contains_known_industries(self):
        """Test contains known cyclical industries."""
        result = get_all_cyclical_industries()
        assert "有色金属" in result
        assert "煤炭" in result
        assert "钢铁" in result
        assert "化工" in result
        assert "汽车" in result
        assert "房地产" in result

    def test_expansion_industries(self):
        """Test newly added industries from expansion."""
        result = get_all_cyclical_industries()
        # New industries added in expansion
        assert "水泥" in result
        assert "玻璃" in result
        assert "石油石化" in result
        assert "贵金属" in result
        assert "小金属" in result  # 东方财富命名（原稀有金属）
        assert "能源金属" in result  # 东方财富: 锂、钴、镍等
        assert "航空机场" in result  # 东方财富合并返回
        assert "航运港口" in result  # 东方财富合并返回

    def test_returns_copy(self):
        """Test returns a copy, not the original set."""
        result1 = get_all_cyclical_industries()
        result1.add("测试行业")
        result2 = get_all_cyclical_industries()
        assert "测试行业" not in result2


class TestGetAllFinancialIndustries:
    """Tests for get_all_financial_industries function."""

    def test_returns_set(self):
        """Test returns a set."""
        result = get_all_financial_industries()
        assert isinstance(result, set)

    def test_not_empty(self):
        """Test set is not empty."""
        result = get_all_financial_industries()
        assert len(result) > 0

    def test_contains_known_industries(self):
        """Test contains known financial industries."""
        result = get_all_financial_industries()
        assert "银行" in result
        assert "保险" in result
        assert "证券" in result
        assert "多元金融" in result

    def test_expansion_industries(self):
        """Test newly added industries from expansion."""
        result = get_all_financial_industries()
        # New industries added in expansion
        assert "信托" in result
        assert "期货" in result
        assert "金融科技" in result
        assert "互联网金融" in result
        assert "券商" in result  # Alias added as canonical

    def test_returns_copy(self):
        """Test returns a copy, not the original set."""
        result1 = get_all_financial_industries()
        result1.add("测试金融")
        result2 = get_all_financial_industries()
        assert "测试金融" not in result2


class TestDataSourceVariations:
    """Tests for industry names from different data sources."""

    def test_akshare_eastmoney_names(self):
        """Test industry names from Akshare (东方财富)."""
        # Common names from akshare's stock_board_industry_name_em()
        assert is_cyclical_industry("有色金属") is True
        assert is_cyclical_industry("煤炭") is True
        assert is_financial_industry("银行") is True
        assert is_financial_industry("证券") is True

    def test_tushare_names(self):
        """Test industry names from Tushare."""
        # Tushare returns similar names via stock_basic()
        assert is_cyclical_industry("有色金属") is True
        assert is_financial_industry("银行") is True

    def test_variant_names(self):
        """Test variant names that may appear from different sources."""
        # Aliases and variants
        assert is_financial_industry("券商") is True
        assert is_cyclical_industry("煤炭开采") is True
        assert is_cyclical_industry("房地产开发") is True

    def test_names_with_suffixes(self):
        """Test industry names with common suffixes."""
        # Names with "行业" or "板块" suffix
        assert is_cyclical_industry("有色金属行业") is True
        assert is_cyclical_industry("煤炭板块") is True
        assert is_financial_industry("银行行业") is True
        assert is_financial_industry("证券板块") is True
