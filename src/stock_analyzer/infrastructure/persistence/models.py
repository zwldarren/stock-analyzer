"""SQLAlchemy ORM"""

from datetime import datetime
from typing import Any

from sqlalchemy import (
    Column,
    Date,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class StockDaily(Base):
    """
    股票日线数据模型

    存储每日行情数据和计算的技术指标
    支持多股票、多日期的唯一约束
    """

    __tablename__ = "stock_daily"

    # 主键
    id = Column(Integer, primary_key=True, autoincrement=True)

    # 股票代码（如 600519, 000001）
    code = Column(String(10), nullable=False, index=True)

    # 交易日期
    date = Column(Date, nullable=False, index=True)

    # OHLC 数据
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)

    # 成交数据
    volume = Column(Float)  # 成交量（股）
    amount = Column(Float)  # 成交额（元）
    pct_chg = Column(Float)  # 涨跌幅（%）

    # 技术指标
    ma5 = Column(Float)
    ma10 = Column(Float)
    ma20 = Column(Float)
    volume_ratio = Column(Float)  # 量比

    # 数据来源
    data_source = Column(String(50))  # 记录数据来源（如 AkshareFetcher）

    # 更新时间
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

    # 唯一约束：同一股票同一日期只能有一条数据
    __table_args__ = (
        UniqueConstraint("code", "date", name="uix_code_date"),
        Index("ix_code_date", "code", "date"),
    )

    def __repr__(self):
        return f"<StockDaily(code={self.code}, date={self.date}, close={self.close})>"

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "code": self.code,
            "date": self.date,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "amount": self.amount,
            "pct_chg": self.pct_chg,
            "ma5": self.ma5,
            "ma10": self.ma10,
            "ma20": self.ma20,
            "volume_ratio": self.volume_ratio,
            "data_source": self.data_source,
        }


class NewsIntel(Base):
    """
    新闻情报数据模型

    存储搜索到的新闻情报条目，用于后续分析与查询
    """

    __tablename__ = "news_intel"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # 关联用户查询操作
    query_id = Column(String(64), index=True)

    # 股票信息
    code = Column(String(10), nullable=False, index=True)
    name = Column(String(50))

    # 搜索上下文
    dimension = Column(String(32), index=True)  # latest_news / risk_check / earnings / market_analysis / industry
    query = Column(String(255))
    provider = Column(String(32), index=True)

    # 新闻内容
    title = Column(String(300), nullable=False)
    snippet = Column(Text)
    url = Column(String(1000), nullable=False)
    source = Column(String(100))
    published_date = Column(DateTime, index=True)

    # 入库时间
    fetched_at = Column(DateTime, default=datetime.now, index=True)
    query_source = Column(String(32), index=True)  # bot/web/cli/system
    requester_platform = Column(String(20))
    requester_user_id = Column(String(64))
    requester_user_name = Column(String(64))
    requester_chat_id = Column(String(64))
    requester_message_id = Column(String(64))
    requester_query = Column(String(255))

    __table_args__ = (
        UniqueConstraint("url", name="uix_news_url"),
        Index("ix_news_code_pub", "code", "published_date"),
    )

    def __repr__(self) -> str:
        return f"<NewsIntel(code={self.code}, title={self.title[:20]}...)>"


class AnalysisHistory(Base):
    """
    分析结果历史记录模型

    保存每次分析结果，支持按 query_id/股票代码检索
    """

    __tablename__ = "analysis_history"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # 关联查询链路
    query_id = Column(String(64), index=True)

    # 股票信息
    code = Column(String(10), nullable=False, index=True)
    name = Column(String(50))

    # 核心结论
    sentiment_score = Column(Integer)
    operation_advice = Column(String(20))
    trend_prediction = Column(String(50))
    analysis_summary = Column(Text)

    # 详细数据
    raw_result = Column(Text)
    news_content = Column(Text)

    created_at = Column(DateTime, default=datetime.now, index=True)

    __table_args__ = (Index("ix_analysis_code_time", "code", "created_at"),)

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "query_id": self.query_id,
            "code": self.code,
            "name": self.name,
            "sentiment_score": self.sentiment_score,
            "operation_advice": self.operation_advice,
            "trend_prediction": self.trend_prediction,
            "analysis_summary": self.analysis_summary,
            "raw_result": self.raw_result,
            "news_content": self.news_content,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
