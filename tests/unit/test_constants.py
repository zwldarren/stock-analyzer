"""Tests for A-share valuation constants."""

from ashare_analyzer.constants import (
    A_SHARE_MARKET_PREMIUM,
    A_SHARE_RISK_FREE_RATE,
    A_SHARE_RISK_PREMIUM,
    DEFAULT_PB,
    DEFAULT_PS,
)
from ashare_analyzer.industry.classifier import (
    is_cyclical_industry,
    is_financial_industry,
)
from ashare_analyzer.industry.constants import (
    CYCLICAL_INDUSTRIES,
    FINANCIAL_INDUSTRIES,
)


class TestIndustryConstants:
    """Tests for industry classification constants."""

    def test_cyclical_industries_not_empty(self):
        """Test cyclical industries set is not empty."""
        assert len(CYCLICAL_INDUSTRIES) > 0
        assert "有色金属" in CYCLICAL_INDUSTRIES
        assert "煤炭" in CYCLICAL_INDUSTRIES

    def test_financial_industries_not_empty(self):
        """Test financial industries set is not empty."""
        assert len(FINANCIAL_INDUSTRIES) > 0
        assert "银行" in FINANCIAL_INDUSTRIES
        assert "保险" in FINANCIAL_INDUSTRIES

    def test_expansion_cyclical_industries(self):
        """Test newly added cyclical industries from expansion."""
        # New industries added in the expansion
        assert "汽车" in CYCLICAL_INDUSTRIES
        assert "房地产" in CYCLICAL_INDUSTRIES
        assert "水泥" in CYCLICAL_INDUSTRIES
        assert "石油石化" in CYCLICAL_INDUSTRIES
        assert "贵金属" in CYCLICAL_INDUSTRIES
        # 东方财富实际名称
        assert "小金属" in CYCLICAL_INDUSTRIES  # 东方财富命名
        assert "能源金属" in CYCLICAL_INDUSTRIES  # 东方财富命名
        assert "航空机场" in CYCLICAL_INDUSTRIES  # 东方财富合并
        assert "航运港口" in CYCLICAL_INDUSTRIES  # 东方财富合并

    def test_expansion_financial_industries(self):
        """Test newly added financial industries from expansion."""
        # New industries added in the expansion
        assert "信托" in FINANCIAL_INDUSTRIES
        assert "期货" in FINANCIAL_INDUSTRIES
        assert "券商" in FINANCIAL_INDUSTRIES  # Alias added as canonical

    def test_is_cyclical_industry_true(self):
        """Test is_cyclical_industry returns True for cyclical industries."""
        assert is_cyclical_industry("有色金属") is True
        assert is_cyclical_industry("煤炭") is True
        assert is_cyclical_industry("汽车") is True
        assert is_cyclical_industry("房地产") is True

    def test_is_cyclical_industry_false(self):
        """Test is_cyclical_industry returns False for non-cyclical industries."""
        assert is_cyclical_industry("食品饮料") is False
        assert is_cyclical_industry("医药生物") is False

    def test_is_financial_industry_true(self):
        """Test is_financial_industry returns True for financial industries."""
        assert is_financial_industry("银行") is True
        assert is_financial_industry("保险") is True
        assert is_financial_industry("证券") is True

    def test_is_financial_industry_false(self):
        """Test is_financial_industry returns False for non-financial industries."""
        assert is_financial_industry("计算机") is False

    def test_alias_resolution_cyclical(self):
        """Test alias resolution works for cyclical industries."""
        assert is_cyclical_industry("煤炭开采") is True
        assert is_cyclical_industry("钢铁冶炼") is True
        assert is_cyclical_industry("房地产开发") is True
        # 东方财富命名测试
        assert is_cyclical_industry("稀有金属") is True  # 映射到小金属
        assert is_cyclical_industry("港口航运") is True  # 映射到航运港口

    def test_alias_resolution_financial(self):
        """Test alias resolution works for financial industries."""
        assert is_financial_industry("券商") is True
        assert is_financial_industry("证券公司") is True
        assert is_financial_industry("商业银行") is True

    def test_normalization(self):
        """Test industry name normalization works."""
        assert is_cyclical_industry("有色金属行业") is True
        assert is_cyclical_industry("煤炭板块") is True
        assert is_financial_industry("银行行业") is True
        assert is_financial_industry("  证券  ") is True

    def test_keyword_matching(self):
        """Test keyword matching works for industries."""
        assert is_cyclical_industry("锂电池材料") is True
        assert is_cyclical_industry("稀土永磁") is True

    def test_eastmoney_names(self):
        """Test 东方财富 actual industry names."""
        # 来自 stock_board_industry_name_em 的实际名称
        assert is_cyclical_industry("小金属") is True  # 东方财富用"小金属"
        assert is_cyclical_industry("能源金属") is True
        assert is_cyclical_industry("航运港口") is True
        assert is_cyclical_industry("航空机场") is True
        assert is_cyclical_industry("煤炭行业") is True  # 带后缀
        assert is_financial_industry("银行") is True


class TestValuationParameters:
    """Tests for A-share valuation parameters."""

    def test_market_premium_value(self):
        """Test market premium is reasonable for A-shares."""
        assert A_SHARE_MARKET_PREMIUM == 2.0

    def test_risk_free_rate_value(self):
        """Test risk-free rate for A-shares."""
        assert A_SHARE_RISK_FREE_RATE == 0.03

    def test_risk_premium_value(self):
        """Test risk premium for A-shares."""
        assert A_SHARE_RISK_PREMIUM == 0.05

    def test_default_pb_value(self):
        """Test default PB value."""
        assert DEFAULT_PB == 2.0

    def test_default_ps_value(self):
        """Test default PS value."""
        assert DEFAULT_PS == 2.0
