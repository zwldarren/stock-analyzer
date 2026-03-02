"""
Industry classification constants for A-share market.

Contains expanded industry lists, aliases, and keyword mappings
to handle variations from different data sources (akshare, tushare, efinance).
"""

from enum import Enum
from typing import Final


class IndustryCategory(Enum):
    """Industry category classification."""

    CYCLICAL = "cyclical"
    FINANCIAL = "financial"
    GROWTH = "growth"
    DEFENSIVE = "defensive"
    OTHER = "other"


# ============================================================
# Cyclical Industries (周期性行业)
# ============================================================
# Industries whose performance is highly correlated with economic cycles.
# Characteristics: High earnings volatility, sensitive to GDP growth.

CYCLICAL_INDUSTRIES: Final[set[str]] = {
    # 有色金属系列 (Non-ferrous metals)
    # 东方财富返回: 有色金属、小金属、贵金属、能源金属
    "有色金属",
    "小金属",  # 东方财富实际名称 (稀有金属在东财叫小金属，其他数据源可能叫稀有金属)
    "贵金属",  # Gold, silver, platinum
    "能源金属",  # 东方财富: 锂、钴、镍等能源金属
    # 能源资源 (Energy resources)
    "煤炭",
    "石油石化",
    "油气开采",
    "油服工程",
    # 基础材料 (Basic materials)
    "钢铁",
    "化工",
    "化纤",
    "建材",
    "建筑材料",
    "水泥",
    "玻璃",
    "造纸",
    # 工业制造 (Industrial manufacturing)
    "工程机械",
    "重型机械",
    "船舶制造",
    "航运",
    "港口",
    "航运港口",  # 东方财富合并返回 (港口航运通过别名映射)
    "航空机场",  # 东方财富: 航空+机场合并
    "汽车",
    "汽车整车",
    "汽车零部件",
    # 房地产 (Real estate)
    "房地产",
    "房地产开发",
}

# ============================================================
# Financial Industries (金融行业)
# ============================================================
# Financial sector industries with special valuation considerations.
# Characteristics: Balance-sheet driven, regulated, use PB/ROE valuation.

FINANCIAL_INDUSTRIES: Final[set[str]] = {
    # 银行 (Banking)
    "银行",
    "商业银行",
    # 保险 (Insurance)
    "保险",
    "保险公司",
    # 证券 (Securities)
    "证券",
    "券商",
    "证券公司",
    "投资银行",
    # 多元金融 (Diversified financial)
    "多元金融",
    "金融控股",
    "信托",
    "期货",
    "租赁",
    "金融科技",
    "互联网金融",
}

# ============================================================
# Industry Aliases (行业别名映射)
# ============================================================
# Maps alternative names to canonical names.
# Used when data sources return different names for the same industry.

INDUSTRY_ALIASES: Final[dict[str, str]] = {
    # 券商别名 -> 证券 (券商 is now canonical, only aliases here)
    "证券公司": "证券",
    "投资银行": "证券",
    "券商信托": "证券",
    "证券投资": "证券",
    # 银行别名 -> 银行 (商业银行 is now canonical, only aliases here)
    "银行业": "银行",
    # 保险别名 -> 保险 (保险公司 is now canonical, only aliases here)
    "保险业": "保险",
    # 有色金属别名 -> 有色金属
    "有色": "有色金属",
    "有色金属冶炼": "有色金属",
    "有色金属加工": "有色金属",
    # 稀有金属/小金属别名 -> 小金属 (统一到东方财富命名)
    "稀有金属": "小金属",
    "稀土": "小金属",
    "稀土永磁": "小金属",
    "锂电": "小金属",
    "锂": "小金属",
    "锂矿": "小金属",
    "钴": "小金属",
    # 能源金属别名
    "能源金属行业": "能源金属",
    "锂资源": "能源金属",
    # 煤炭别名 -> 煤炭
    "煤炭开采": "煤炭",
    "煤炭采选": "煤炭",
    "煤炭行业": "煤炭",
    # 钢铁别名 -> 钢铁
    "黑色金属": "钢铁",
    "钢铁冶炼": "钢铁",
    "钢铁行业": "钢铁",
    "普钢": "钢铁",
    "特钢": "钢铁",
    # 化工别名 -> 化工
    "化学制品": "化工",
    "化学原料": "化工",
    "化工行业": "化工",
    "基础化工": "化工",
    "精细化工": "化工",
    # 化纤别名
    "化学纤维": "化纤",
    # 建材别名 -> 建筑材料
    "建材": "建筑材料",
    "建材行业": "建筑材料",
    # 水泥别名 -> 水泥
    "水泥制造": "水泥",
    "水泥行业": "水泥",
    # 汽车别名 -> 汽车 (汽车整车 is now canonical, only aliases here)
    "汽车制造": "汽车",
    "汽车行业": "汽车",
    "乘用车": "汽车",
    "商用车": "汽车",
    # 房地产别名 -> 房地产 (房地产开发 is now canonical, only aliases here)
    "地产": "房地产",
    "地产行业": "房地产",
    "房地产开发经营": "房地产",
    # 石油石化别名 -> 石油石化
    "石油": "石油石化",
    "石油开采": "石油石化",
    "石油行业": "石油石化",
    "石化": "石油石化",
    # 贵金属别名
    "黄金": "贵金属",
    "白银": "贵金属",
    # 航运港口别名 (东方财富合并)
    "港口航运": "航运港口",
    # 航空机场别名
    "航空运输": "航空机场",
    "机场航运": "航空机场",
}

# ============================================================
# Industry Keywords (行业关键词)
# ============================================================
# Used for partial matching when exact match and alias resolution fail.
# Maps canonical industry name to list of keywords that indicate membership.

INDUSTRY_KEYWORDS: Final[dict[str, list[str]]] = {
    # 有色金属子行业关键词
    "有色金属": ["铜", "铝", "锌", "铅", "镍", "锡", "钨", "钼"],
    # 小金属关键词 (东方财富命名，原稀有金属)
    "小金属": ["稀土", "锂", "锂矿", "钴", "锂电材料", "钴矿", "锗", "钨", "钼", "镁"],
    # 能源金属关键词
    "能源金属": ["锂", "钴", "镍", "锂矿", "锂盐"],
    # 贵金属关键词
    "贵金属": ["黄金", "白银", "铂金", "钯金"],
    # 煤炭子行业关键词
    "煤炭": ["焦煤", "动力煤", "无烟煤", "焦炭"],
    # 钢铁子行业关键词
    "钢铁": ["特钢", "不锈钢", "普钢", "钢材", "铁矿石"],
    # 化工子行业关键词
    "化工": ["化肥", "农药", "聚氨酯", "氯碱", "纯碱", "钛白粉", "染料", "涂料", "有机硅"],
    # 建筑材料关键词
    "建筑材料": ["水泥", "玻璃", "管材", "防水材料", "保温材料"],
    # 汽车子行业关键词
    "汽车": ["整车", "零部件", "新能源车", "电动车", "汽车电子", "轮胎"],
    # 房地产关键词
    "房地产": ["地产", "物业", "园区开发", "商业地产"],
    # 石油石化关键词
    "石油石化": ["油气", "炼化", "油服", "石化"],
    # 航运港口关键词
    "航运港口": ["航运", "港口", "海运", "集装箱", "散货"],
    # 航空机场关键词
    "航空机场": ["航空", "机场", "民航", "空运"],
    # 证券关键词
    "证券": ["券商", "投行", "证券投资"],
    # 银行关键词
    "银行": ["商业银行", "股份制银行", "城商行"],
}

# Common suffixes to remove when normalizing industry names
# These are commonly added by data sources but don't affect classification
COMMON_INDUSTRY_SUFFIXES: Final[list[str]] = [
    "行业",
    "板块",
    "概念",
    "指数",
    "一级行业",
    "二级行业",
    "三级行业",
]
