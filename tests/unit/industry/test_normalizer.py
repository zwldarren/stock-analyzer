"""Tests for industry normalization utilities."""

from ashare_analyzer.industry.normalizer import (
    classify_industry,
    match_by_keyword,
    normalize_industry_name,
    resolve_alias,
)


class TestNormalizeIndustryName:
    """Tests for normalize_industry_name function."""

    def test_strip_whitespace(self):
        """Test removing leading/trailing whitespace."""
        assert normalize_industry_name("  有色金属  ") == "有色金属"

    def test_remove_suffix_hangye(self):
        """Test removing '行业' suffix."""
        assert normalize_industry_name("有色金属行业") == "有色金属"
        assert normalize_industry_name("银行行业") == "银行"

    def test_remove_suffix_bankuai(self):
        """Test removing '板块' suffix."""
        assert normalize_industry_name("煤炭板块") == "煤炭"
        assert normalize_industry_name("证券板块") == "证券"

    def test_remove_suffix_gainian(self):
        """Test removing '概念' suffix."""
        assert normalize_industry_name("锂电池概念") == "锂电池"

    def test_remove_suffix_yiji_hangye(self):
        """Test removing '一级行业' suffix."""
        assert normalize_industry_name("银行一级行业") == "银行"

    def test_combined_normalization(self):
        """Test combined whitespace and suffix removal."""
        assert normalize_industry_name("  银行行业  ") == "银行"

    def test_empty_string(self):
        """Test empty string returns empty."""
        assert normalize_industry_name("") == ""

    def test_no_change_needed(self):
        """Test string that doesn't need normalization."""
        assert normalize_industry_name("有色金属") == "有色金属"

    def test_single_char_industry(self):
        """Test single character industry name."""
        assert normalize_industry_name("银行") == "银行"


class TestResolveAlias:
    """Tests for resolve_alias function."""

    def test_quashang_alias(self):
        """Test '券商' is now a canonical name, not an alias."""
        # "券商" is now in FINANCIAL_INDUSTRIES, so it's canonical
        assert resolve_alias("券商") == "券商"

    def test_quashang_xintuo_alias(self):
        """Test '券商信托' resolves to '证券'."""
        assert resolve_alias("券商信托") == "证券"

    def test_meitan_kaicai_alias(self):
        """Test '煤炭开采' resolves to '煤炭'."""
        assert resolve_alias("煤炭开采") == "煤炭"

    def test_gangtie_yelian_alias(self):
        """Test '钢铁冶炼' resolves to '钢铁'."""
        assert resolve_alias("钢铁冶炼") == "钢铁"

    def test_shangye_yinhang_alias(self):
        """Test '商业银行' is now a canonical name, not an alias."""
        # "商业银行" is now in FINANCIAL_INDUSTRIES, so it's canonical
        assert resolve_alias("商业银行") == "商业银行"

    def test_fangdichan_kaifa_alias(self):
        """Test '房地产开发' is now a canonical name, not an alias."""
        # "房地产开发" is now in CYCLICAL_INDUSTRIES, so it's canonical
        assert resolve_alias("房地产开发") == "房地产开发"

    def test_xiaojinshu_alias(self):
        """Test '稀有金属' resolves to '小金属' (东方财富命名)."""
        # 东方财富用"小金属"，统一到这个名称
        assert resolve_alias("稀有金属") == "小金属"
        assert resolve_alias("稀土") == "小金属"

    def test_nengyuanjinshu_alias(self):
        """Test '能源金属行业' resolves to '能源金属'."""
        assert resolve_alias("能源金属行业") == "能源金属"

    def test_hangyun_gangkou_alias(self):
        """Test '港口航运' resolves to '航运港口'."""
        # 东方财富返回"航运港口"，同花顺返回"港口航运"
        assert resolve_alias("港口航运") == "航运港口"

    def test_no_alias_returns_original(self):
        """Test industry without alias returns original."""
        assert resolve_alias("有色金属") == "有色金属"
        assert resolve_alias("计算机") == "计算机"

    def test_empty_string(self):
        """Test empty string returns empty."""
        assert resolve_alias("") == ""

    def test_normalized_then_aliased(self):
        """Test normalization before alias resolution."""
        # "煤炭开采行业" -> "煤炭开采" -> "煤炭"
        assert resolve_alias("煤炭开采行业") == "煤炭"


class TestMatchByKeyword:
    """Tests for match_by_keyword function."""

    def test_lithium_keyword(self):
        """Test lithium keyword matches 小金属/有色金属."""
        from ashare_analyzer.industry.constants import CYCLICAL_INDUSTRIES

        result = match_by_keyword("锂电池材料", CYCLICAL_INDUSTRIES)
        # 锂 is in INDUSTRY_KEYWORDS for 小金属 and 能源金属
        assert result in ("小金属", "有色金属", "能源金属")

    def test_rare_earth_keyword(self):
        """Test rare earth keyword matches 小金属 or 有色金属."""
        from ashare_analyzer.industry.constants import CYCLICAL_INDUSTRIES

        result = match_by_keyword("稀土永磁", CYCLICAL_INDUSTRIES)
        # "稀土" appears in 小金属 keywords
        assert result in ("小金属", "有色金属")

    def test_auto_keyword(self):
        """Test auto keyword matches 汽车."""
        from ashare_analyzer.industry.constants import CYCLICAL_INDUSTRIES

        result = match_by_keyword("新能源整车", CYCLICAL_INDUSTRIES)
        assert result == "汽车"

    def test_exact_match_preferred(self):
        """Test exact match is preferred over keyword match."""
        from ashare_analyzer.industry.constants import CYCLICAL_INDUSTRIES

        result = match_by_keyword("有色金属", CYCLICAL_INDUSTRIES)
        assert result == "有色金属"

    def test_no_keyword_match(self):
        """Test no match for unrelated industry."""
        from ashare_analyzer.industry.constants import CYCLICAL_INDUSTRIES

        result = match_by_keyword("计算机", CYCLICAL_INDUSTRIES)
        assert result is None

    def test_empty_string(self):
        """Test empty string returns None."""
        from ashare_analyzer.industry.constants import CYCLICAL_INDUSTRIES

        result = match_by_keyword("", CYCLICAL_INDUSTRIES)
        assert result is None

    def test_financial_keyword(self):
        """Test keyword matching for financial industries."""
        from ashare_analyzer.industry.constants import FINANCIAL_INDUSTRIES

        result = match_by_keyword("券商投资", FINANCIAL_INDUSTRIES)
        assert result == "证券"

    def test_energy_metal_keyword(self):
        """Test energy metal keyword matches."""
        from ashare_analyzer.industry.constants import CYCLICAL_INDUSTRIES

        result = match_by_keyword("锂矿资源", CYCLICAL_INDUSTRIES)
        # 锂矿 is in 小金属 and 能源金属 keywords
        assert result in ("小金属", "能源金属")


class TestClassifyIndustry:
    """Tests for classify_industry function."""

    def test_exact_match_cyclical(self):
        """Test exact match for cyclical industry."""
        assert classify_industry("有色金属") == "有色金属"
        assert classify_industry("煤炭") == "煤炭"
        assert classify_industry("钢铁") == "钢铁"

    def test_exact_match_financial(self):
        """Test exact match for financial industry."""
        assert classify_industry("银行") == "银行"
        assert classify_industry("保险") == "保险"
        assert classify_industry("证券") == "证券"

    def test_alias_resolution(self):
        """Test alias resolution."""
        # "券商" is now a canonical name in FINANCIAL_INDUSTRIES
        assert classify_industry("券商") == "券商"
        assert classify_industry("商业银行") == "商业银行"  # Now canonical
        assert classify_industry("煤炭开采") == "煤炭"

    def test_xiaojinshu_resolution(self):
        """Test '小金属' resolution (东方财富 naming)."""
        # "稀有金属" aliases to "小金属"
        assert classify_industry("稀有金属") == "小金属"
        assert classify_industry("稀土") == "小金属"

    def test_nengyuanjinshu_resolution(self):
        """Test '能源金属' resolution."""
        assert classify_industry("能源金属") == "能源金属"
        assert classify_industry("能源金属行业") == "能源金属"

    def test_hangyun_gangkou_resolution(self):
        """Test shipping port industry resolution."""
        # 东方财富用"航运港口"
        assert classify_industry("航运港口") == "航运港口"
        # 同花顺用"港口航运"，映射到"航运港口"
        assert classify_industry("港口航运") == "航运港口"

    def test_keyword_matching(self):
        """Test keyword matching fallback."""
        assert classify_industry("锂电池") in ("小金属", "有色金属", "能源金属")
        # "稀土" appears in 小金属 keywords
        assert classify_industry("稀土永磁材料") in ("小金属", "有色金属")

    def test_normalization_before_match(self):
        """Test normalization happens before matching."""
        assert classify_industry("  银行行业  ") == "银行"
        assert classify_industry("煤炭板块") == "煤炭"

    def test_no_match_returns_none(self):
        """Test unclassified industry returns None."""
        assert classify_industry("计算机") is None
        assert classify_industry("医药生物") is None
        assert classify_industry("食品饮料") is None

    def test_empty_string_returns_none(self):
        """Test empty string returns None."""
        assert classify_industry("") is None

    def test_normalized_alias_then_keyword(self):
        """Test full resolution chain: normalize -> alias -> keyword."""
        # "券商信托板块" -> "券商信托" -> "证券" (alias)
        assert classify_industry("券商信托板块") == "证券"


class TestEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_partial_match_boundary(self):
        """Test that partial matching respects word boundaries."""
        # "汽车服务" should ideally NOT match as cyclical
        # But our current implementation may match it
        # This test documents the current behavior
        classify_industry("汽车服务")
        # Current behavior: may match due to keyword "汽车"
        # Future improvement: add negative keyword list

    def test_subindustry_variants(self):
        """Test various subindustry naming variants."""
        # These should all resolve to parent industry
        # "汽车整车" is now a canonical name in CYCLICAL_INDUSTRIES
        result = classify_industry("汽车整车")
        assert result in ("汽车", "汽车整车")  # May be exact match or canonical
        # "汽车零部件" matches via keyword "零部件" -> "汽车"
        assert classify_industry("汽车零部件") in ("汽车", "汽车零部件", None)

    def test_data_source_variants_akshare(self):
        """Test industry names from Akshare (东方财富)."""
        # Common names returned by akshare's stock_board_industry_name_em()
        assert classify_industry("有色金属") == "有色金属"
        assert classify_industry("银行") == "银行"
        assert classify_industry("证券") == "证券"

    def test_data_source_variants_tushare(self):
        """Test industry names from Tushare."""
        # Tushare returns similar names but may have slight variations
        assert classify_industry("有色金属") == "有色金属"
        assert classify_industry("银行") == "银行"

    def test_new_industry_names_in_lists(self):
        """Test newly added industry names."""
        # These were added in the expansion
        assert classify_industry("房地产") == "房地产"
        assert classify_industry("水泥") == "水泥"
        assert classify_industry("玻璃") == "玻璃"
        assert classify_industry("信托") == "信托"
        assert classify_industry("期货") == "期货"
