"""
Technical Analysis Agent Module

Technical analysis agent using LLM for trend and pattern recognition.

This agent analyzes technical indicators and uses LLM to:
- Interpret complex technical patterns
- Assess trend strength and direction
- Generate trading signals with reasoning
- Identify support/resistance and entry points

Combines algorithmic calculations with AI-driven interpretation.
"""

import logging
from typing import TYPE_CHECKING, Any

from ashare_analyzer.ai.clients import get_llm_client
from ashare_analyzer.ai.tools import ANALYZE_SIGNAL_TOOL
from ashare_analyzer.models import AgentSignal, SignalType

from .base import BaseAgent

if TYPE_CHECKING:
    from ashare_analyzer.ai.clients import LiteLLMClient

logger = logging.getLogger(__name__)

# System prompt for technical analysis
TECHNICAL_SYSTEM_PROMPT = """You are a professional technical analyst for A-share market.

Your task: Analyze technical indicators and price action to generate trading signals.

=== Checklist for Signal Generation ===
- [ ] Trend direction confirmed (MA alignment or ADX > 25)
- [ ] Momentum aligned (MACD histogram direction)
- [ ] RSI not extreme (< 70 for buy, > 30 for sell)
- [ ] Volume confirms (volume_ratio > 1.2 for buy)
- [ ] Price within Bollinger bounds

=== Signal Rules with Thresholds ===
BUY conditions:
- 3+ checklist items passed
- bias_ma5 < 5% (not chasing)
- RSI < 70 (not overbought)
- ADX > 20 (trend present)

SELL conditions:
- 3+ checklist items failed
- bias_ma5 > 8% (overextended)
- OR RSI > 80 (extreme overbought)
- OR breakdown below MA20 with bearish alignment

HOLD conditions:
- Mixed checklist results
- bias 5-8% (wait for pullback)
- Low ADX (< 20, no clear trend)

=== Confidence Levels ===
- 90-100%: All indicators aligned, clear trend + volume confirmation
- 70-89%: Most indicators aligned, minor uncertainties
- 50-69%: Trend present but mixed signals
- 30-49%: Conflicting indicators, high volatility
- 10-29%: No clear direction, extreme volatility

=== Trading Philosophy ===
- Follow the trend (顺势而为)
- Never chase high prices (不追高)
- Buy on pullbacks to support (回踩买点)

Use the analyze_signal function to return your analysis."""


class TechnicalAgent(BaseAgent):
    """
    Technical Analysis Agent using LLM for pattern recognition.

    This agent combines algorithmic technical calculations with LLM-driven
    interpretation to generate trading signals.

    Example:
        agent = TechnicalAgent()
        signal = agent.analyze({
            "code": "600519",
            "today": {"close": 1800, "ma5": 1790, "ma10": 1780, "ma20": 1770},
            "ma_status": "多头排列"
        })
    """

    def __init__(self):
        """Initialize the Technical Agent."""
        super().__init__("TechnicalAgent")
        self._logger = logging.getLogger(__name__)
        self._llm_client: LiteLLMClient | None = None
        self._init_llm_client()

    def _init_llm_client(self) -> None:
        """Initialize LLM client for analysis using shared factory."""
        self._llm_client = get_llm_client()
        if self._llm_client:
            self._logger.debug("TechnicalAgent LLM client initialized successfully")
        else:
            self._logger.warning("No LLM API key configured, TechnicalAgent will use rule-based fallback")

    def is_available(self) -> bool:
        """Technical agent is always available with fallback."""
        return True

    async def analyze(self, context: dict[str, Any]) -> AgentSignal:
        """
        Execute technical analysis using LLM (async).

        Args:
            context: Analysis context containing:
                - code: Stock code
                - today: Current day's data with OHLCV and MAs
                - ma_status: Moving average alignment status

        Returns:
            AgentSignal with LLM-generated trading signal
        """
        stock_code = context.get("code", "")
        today = context.get("today", {})
        ma_status = context.get("ma_status", "")

        self._logger.debug(f"[{stock_code}] TechnicalAgent开始技术面分析")

        try:
            # Extract price data
            close = today.get("close", 0)
            ma5 = today.get("ma5", 0)

            if close <= 0 or ma5 <= 0:
                return AgentSignal(
                    agent_name=self.name,
                    signal=SignalType.HOLD,
                    confidence=0,
                    reasoning="价格数据不完整，无法分析",
                    metadata={"error": "missing_price_data"},
                )

            # Calculate technical metrics
            technical_data = self._calculate_technical_metrics(today, ma_status)

            # Use LLM if available for sophisticated analysis
            if self._llm_client and self._llm_client.is_available():
                llm_analysis = await self._analyze_with_llm(stock_code, technical_data, context)
                if llm_analysis:
                    return self._build_signal_from_llm(llm_analysis, technical_data)

            # Fallback to rule-based analysis
            return self._rule_based_analysis(stock_code, technical_data, context)

        except Exception as e:
            self._logger.error(f"[{stock_code}] TechnicalAgent分析失败: {e}")
            return AgentSignal(
                agent_name=self.name,
                signal=SignalType.HOLD,
                confidence=0,
                reasoning=f"技术分析失败: {str(e)}",
                metadata={"error": str(e)},
            )

    def _calculate_technical_metrics(self, today: dict[str, Any], ma_status: str) -> dict[str, Any]:
        """Calculate all technical metrics from raw data, including advanced indicators."""
        close = today.get("close", 0)
        ma5 = today.get("ma5", 0)
        ma10 = today.get("ma10", 0)
        ma20 = today.get("ma20", 0)
        high = today.get("high", close)
        low = today.get("low", close)
        open_price = today.get("open", close)
        volume = today.get("volume", 0)
        volume_ratio = today.get("volume_ratio", 1.0)

        # Calculate bias rates
        bias_ma5 = ((close - ma5) / ma5 * 100) if ma5 > 0 else 0
        bias_ma10 = ((close - ma10) / ma10 * 100) if ma10 > 0 else 0
        bias_ma20 = ((close - ma20) / ma20 * 100) if ma20 > 0 else 0

        # Determine trend
        is_bullish = close > ma5 > ma10 > ma20 if ma20 > 0 else False
        is_bearish = close < ma5 < ma10 < ma20 if ma20 > 0 else False
        is_mixed = not is_bullish and not is_bearish

        # Calculate support and resistance
        support = min(ma5, ma10, ma20) if ma20 > 0 else ma5
        resistance = max(close * 1.05, max(ma5, ma10, ma20) * 1.02 if ma20 > 0 else close * 1.03)

        # Volume analysis
        volume_status = self._analyze_volume(volume_ratio)

        # Price change
        price_change = today.get("pct_chg", 0)

        # Get advanced technical indicators from context (if available)
        technical_data = {
            "close": close,
            "open": open_price,
            "high": high,
            "low": low,
            "ma5": ma5,
            "ma10": ma10,
            "ma20": ma20,
            "bias_ma5": bias_ma5,
            "bias_ma10": bias_ma10,
            "bias_ma20": bias_ma20,
            "is_bullish": is_bullish,
            "is_bearish": is_bearish,
            "is_mixed": is_mixed,
            "ma_status": ma_status,
            "support": support,
            "resistance": resistance,
            "volume": volume,
            "volume_ratio": volume_ratio,
            "volume_status": volume_status,
            "price_change": price_change,
            # Advanced indicators (from context_builders)
            "rsi_14": today.get("rsi_14", 50.0),
            "rsi_28": today.get("rsi_28", 50.0),
            "macd": today.get("macd", 0.0),
            "macd_signal": today.get("macd_signal", 0.0),
            "macd_hist": today.get("macd_hist", 0.0),
            "bb_upper": today.get("bb_upper", 0.0),
            "bb_middle": today.get("bb_middle", close),
            "bb_lower": today.get("bb_lower", 0.0),
            "bb_position": today.get("bb_position", 0.5),
            "atr": today.get("atr", 0.0),
            "atr_ratio": today.get("atr_ratio", 0.0),
            "adx": today.get("adx", 20.0),
            "stochastic_k": today.get("stochastic_k", 50.0),
            "stochastic_d": today.get("stochastic_d", 50.0),
        }

        # Calculate additional derived metrics
        technical_data["rsi_status"] = self._interpret_rsi(technical_data["rsi_14"])
        technical_data["macd_status"] = self._interpret_macd(
            technical_data["macd"], technical_data["macd_signal"], technical_data["macd_hist"]
        )
        technical_data["adx_status"] = self._interpret_adx(technical_data["adx"])
        technical_data["stochastic_status"] = self._interpret_stochastic(
            technical_data["stochastic_k"], technical_data["stochastic_d"]
        )

        return technical_data

    async def _analyze_with_llm(
        self, stock_code: str, technical_data: dict[str, Any], context: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Use LLM for technical analysis with Function Call (async)."""
        if not self._llm_client:
            return None

        try:
            prompt = self._build_technical_prompt(stock_code, technical_data)

            self._logger.debug(f"[{stock_code}] TechnicalAgent调用LLM进行技术分析...")
            result = await self._llm_client.generate_with_tool(
                prompt=prompt,
                tool=ANALYZE_SIGNAL_TOOL,
                generation_config={"temperature": 0.2, "max_output_tokens": 2048},
                system_prompt=TECHNICAL_SYSTEM_PROMPT,
            )

            if result and "signal" in result:
                self._logger.debug(f"[{stock_code}] LLM技术分析成功: {result}")
                return result
            else:
                self._logger.warning(f"[{stock_code}] LLM技术分析返回格式无效")
                return None

        except Exception as e:
            self._logger.error(f"[{stock_code}] LLM技术分析失败: {e}")
            return None

    def _build_technical_prompt(self, stock_code: str, data: dict[str, Any]) -> str:
        """Build technical analysis prompt for LLM with advanced indicators."""

        return f"""请作为专业的技术面分析师，分析以下股票的技术指标并生成交易信号。

股票代码: {stock_code}

=== 价格数据 ===
- 收盘价: {data["close"]:.2f}
- 开盘价: {data["open"]:.2f}
- 最高价: {data["high"]:.2f}
- 最低价: {data["low"]:.2f}
- 涨跌幅: {data["price_change"]:.2f}%

=== 移动平均线 ===
- MA5: {data["ma5"]:.2f}
- MA10: {data["ma10"]:.2f}
- MA20: {data["ma20"]:.2f}
- 均线状态: {data["ma_status"]}

=== 乖离率 ===
- 与MA5乖离率: {data["bias_ma5"]:.2f}%
- 与MA10乖离率: {data["bias_ma10"]:.2f}%
- 与MA20乖离率: {data["bias_ma20"]:.2f}%

=== 趋势指标 ===
- ADX (趋势强度): {data["adx"]:.1f} ({data["adx_status"]})
- 趋势方向: {"多头" if data["is_bullish"] else "空头" if data["is_bearish"] else "震荡"}

=== 动量指标 ===
- RSI(14): {data["rsi_14"]:.1f} ({data["rsi_status"]})
- RSI(28): {data["rsi_28"]:.1f}
- MACD: {data["macd"]:.4f}, 信号线: {data["macd_signal"]:.4f}
- MACD柱状图: {data["macd_hist"]:.4f} ({data["macd_status"]})

=== 布林带 ===
- 上轨: {data["bb_upper"]:.2f}
- 中轨: {data["bb_middle"]:.2f}
- 下轨: {data["bb_lower"]:.2f}
- 价格位置: {data["bb_position"] * 100:.1f}% (0%=下轨, 100%=上轨)

=== 随机指标 ===
- K值: {data["stochastic_k"]:.1f}
- D值: {data["stochastic_d"]:.1f}
- 状态: {data["stochastic_status"]}

=== 波动率 ===
- ATR: {data["atr"]:.4f}
- ATR比率: {data["atr_ratio"] * 100:.2f}%

=== 量能 ===
- 量比: {data["volume_ratio"]:.2f}
- 量能状态: {data["volume_status"]}

=== 关键价位 ===
- 支撑位: {data["support"]:.2f}
- 阻力位: {data["resistance"]:.2f}

请根据以上所有技术指标综合分析，使用 analyze_signal 函数返回交易信号。"""

    def _build_signal_from_llm(self, llm_analysis: dict[str, Any], technical_data: dict[str, Any]) -> AgentSignal:
        """Build AgentSignal from LLM analysis."""
        signal_str = llm_analysis.get("signal", "hold")
        signal = SignalType.from_string(signal_str)
        confidence = llm_analysis.get("confidence", 50)
        reasoning = llm_analysis.get("reasoning", "无详细分析")

        key_levels = llm_analysis.get("key_levels", {})

        return AgentSignal(
            agent_name=self.name,
            signal=signal,
            confidence=confidence,
            reasoning=reasoning,
            metadata={
                "trend_assessment": llm_analysis.get("trend_assessment", "hold"),
                "trend_strength": llm_analysis.get("trend_strength", 50),
                "key_levels": key_levels,
                "technical_factors": llm_analysis.get("technical_factors", []),
                "risk_factors": llm_analysis.get("risk_factors", []),
                "recommendation": llm_analysis.get("recommendation", ""),
                # Raw data for reference
                "ma_alignment": technical_data["ma_status"],
                "bias_ma5": round(technical_data["bias_ma5"], 2),
                "trend_direction": "bullish"
                if technical_data["is_bullish"]
                else "bearish"
                if technical_data["is_bearish"]
                else "hold",
                "volume_status": technical_data["volume_status"],
                "support_level": round(technical_data["support"], 2),
                "resistance_level": round(technical_data["resistance"], 2),
                # Advanced indicators
                "rsi_14": round(technical_data["rsi_14"], 1),
                "rsi_status": technical_data["rsi_status"],
                "adx": round(technical_data["adx"], 1),
                "adx_status": technical_data["adx_status"],
                "macd_hist": round(technical_data["macd_hist"], 4),
                "macd_status": technical_data["macd_status"],
                "bb_position": round(technical_data["bb_position"], 2),
                "stochastic_k": round(technical_data["stochastic_k"], 1),
                "stochastic_status": technical_data["stochastic_status"],
            },
        )

    def _rule_based_analysis(self, stock_code: str, data: dict[str, Any], context: dict[str, Any]) -> AgentSignal:
        """Fallback rule-based technical analysis."""
        # Bias rate thresholds
        BIAS_BEST_BUY = 2.0
        BIAS_ACCEPTABLE = 5.0
        BIAS_DANGEROUS = 8.0

        close = data["close"]
        bias_ma5 = data["bias_ma5"]
        is_bullish = data["is_bullish"]
        is_bearish = data["is_bearish"]
        ma_status = data["ma_status"]
        volume_status = data["volume_status"]

        # Generate signal based on rules
        if is_bullish:
            if abs(bias_ma5) < BIAS_BEST_BUY:
                signal, confidence = SignalType.BUY, 85
                reasoning = f"多头排列，乖离率{bias_ma5:.1f}%处于最佳买点区间，{volume_status}"
            elif abs(bias_ma5) < BIAS_ACCEPTABLE:
                signal, confidence = SignalType.BUY, 70
                reasoning = f"多头排列，乖离率{bias_ma5:.1f}%可小仓介入，{volume_status}"
            elif abs(bias_ma5) < BIAS_DANGEROUS:
                signal, confidence = SignalType.HOLD, 60
                reasoning = f"多头排列但乖离率{bias_ma5:.1f}%偏高，不宜追高，等待回踩"
            else:
                signal, confidence = SignalType.SELL, 65
                reasoning = f"乖离率{bias_ma5:.1f}%过高，短期回调风险大，建议减仓"
        elif is_bearish:
            if close < data["ma20"]:
                if bias_ma5 > 5:
                    signal, confidence = SignalType.SELL, 75
                    reasoning = f"空头排列，价格跌破MA20且乖离率{bias_ma5:.1f}%偏高，建议减仓，{volume_status}"
                else:
                    signal, confidence = SignalType.SELL, 80
                    reasoning = f"空头排列，价格跌破MA20，趋势向下，{volume_status}"
            elif bias_ma5 > 3:
                signal, confidence = SignalType.HOLD, 60
                reasoning = f"空头排列，价格反弹至MA5上方，但趋势仍弱，建议观望，{volume_status}"
            else:
                signal, confidence = SignalType.SELL, 70
                reasoning = f"空头排列，{ma_status}，建议观望"
        else:
            signal, confidence = SignalType.HOLD, 50
            reasoning = f"技术面信号不明，{ma_status}"

        self._logger.debug(f"[{stock_code}] TechnicalAgent规则分析完成: {signal} (置信度{confidence}%)")

        return AgentSignal(
            agent_name=self.name,
            signal=signal,
            confidence=confidence,
            reasoning=reasoning,
            metadata={
                "ma_alignment": ma_status,
                "bias_ma5": round(bias_ma5, 2),
                "trend_direction": "bullish" if is_bullish else "bearish" if is_bearish else "neutral",
                "support_level": round(data["support"], 2),
                "resistance_level": round(data["resistance"], 2),
                "volume_status": volume_status,
                "analysis_method": "rule_based",
            },
        )

    def _analyze_volume(self, volume_ratio: float) -> str:
        """Analyze volume status."""
        if volume_ratio > 2.0:
            return "显著放量"
        elif volume_ratio > 1.5:
            return "温和放量"
        elif volume_ratio > 0.8:
            return "量能正常"
        elif volume_ratio > 0.5:
            return "轻度缩量"
        else:
            return "明显缩量"

    def _interpret_rsi(self, rsi: float) -> str:
        """Interpret RSI value."""
        if rsi >= 80:
            return "严重超买"
        elif rsi >= 70:
            return "超买"
        elif rsi >= 50:
            return "偏强"
        elif rsi >= 30:
            return "偏弱"
        elif rsi >= 20:
            return "超卖"
        else:
            return "严重超卖"

    def _interpret_macd(self, macd: float, signal: float, histogram: float) -> str:
        """Interpret MACD indicator."""
        if histogram > 0:
            if macd > signal:
                return "多头增强" if histogram > 0 else "多头减弱"
            else:
                return "多头减弱"
        else:
            if macd < signal:
                return "空头增强" if histogram < 0 else "空头减弱"
            else:
                return "空头减弱"

    def _interpret_adx(self, adx: float) -> str:
        """Interpret ADX trend strength."""
        if adx >= 50:
            return "极强趋势"
        elif adx >= 40:
            return "很强趋势"
        elif adx >= 25:
            return "明显趋势"
        elif adx >= 20:
            return "趋势形成"
        else:
            return "无趋势"

    def _interpret_stochastic(self, k: float, d: float) -> str:
        """Interpret Stochastic oscillator."""
        if k >= 80 and d >= 80:
            return "超买区"
        elif k <= 20 and d <= 20:
            return "超卖区"
        elif k > d:
            return "金叉看涨"
        elif k < d:
            return "死叉看跌"
        else:
            return "中性"
