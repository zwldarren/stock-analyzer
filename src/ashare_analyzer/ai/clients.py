"""
AI客户端模块 - 基于 litellm 的统一实现 (async)

使用 litellm 支持 100+ LLM providers，统一接口格式
用于股票分析等需要LLM生成的场景
"""

import asyncio
import json
import logging
import os
from typing import Any

import litellm
from litellm import acompletion

from ashare_analyzer.config import get_config
from ashare_analyzer.exceptions import AnalysisError
from ashare_analyzer.utils import calculate_backoff_delay

logger = logging.getLogger(__name__)

os.environ.setdefault("LITELLM_LOG", "WARNING")
litellm.set_verbose = False
litellm.drop_params = True


DEFAULT_SYSTEM_PROMPT = """你是一位专业的A股投资分析师，擅长市场分析和投资研究。
请基于提供的数据生成专业、客观的分析报告。
注意：
1. 分析要基于事实和数据
2. 避免过度乐观或悲观的偏见
3. 提供具体可操作的建议
4. 风险提示要明确"""


class LiteLLMClient:
    """
    基于 litellm 的统一 LLM 客户端 (async)

    使用方式：
        client = LiteLLMClient("deepseek/deepseek-reasoner", api_key="sk-...")
        response = await client.generate("分析这只股票", {"temperature": 0.7})
    """

    def __init__(
        self,
        model: str,
        api_key: str | None,
        base_url: str | None = None,
    ):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self._available = self._validate_config()

        if not self._available:
            logger.warning(f"LiteLLM 客户端初始化失败 (模型: {model}) - 配置无效")

    def _validate_config(self) -> bool:
        if not self.api_key:
            return False
        if not self.model or "/" not in self.model:
            logger.warning(f"模型名称格式错误，应为 'provider/model-name' 格式: {self.model}")
            return False
        return True

    def is_available(self) -> bool:
        return self._available

    async def generate(self, prompt: str, generation_config: dict, system_prompt: str | None = None) -> str:
        """
        生成内容，带重试机制 (async)

        Args:
            prompt: 用户提示词
            generation_config: 生成配置（temperature, max_tokens 等）
            system_prompt: 系统提示词（可选，默认使用DEFAULT_SYSTEM_PROMPT）

        Returns:
            生成的文本

        Raises:
            Exception: 当 API 调用失败且重试次数用尽时
        """
        if not self._available:
            raise AnalysisError(f"LiteLLM 客户端未初始化或配置无效 (模型: {self.model})")

        config = get_config()
        max_retries = config.ai.llm_max_retries
        base_delay = config.ai.llm_retry_delay

        temperature = generation_config.get("temperature", config.ai.llm_temperature)
        max_tokens = generation_config.get("max_output_tokens", config.ai.llm_max_tokens)

        sys_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt},
        ]

        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    delay = calculate_backoff_delay(attempt, base_delay, max_delay=60.0)
                    logger.info(f"[{self.model}] 第 {attempt + 1} 次重试，等待 {delay:.1f} 秒...")
                    await asyncio.sleep(delay)

                kwargs: dict[str, Any] = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }

                if self.api_key:
                    kwargs["api_key"] = self.api_key

                if self.base_url:
                    kwargs["api_base"] = self.base_url

                response = await acompletion(**kwargs)

                if response and response.choices and len(response.choices) > 0:
                    content = response.choices[0].message.content
                    if content:
                        logger.debug(f"[{self.model}] LLM响应: {content[:500]}...")
                        return content.strip()

                raise AnalysisError(f"{self.model} 返回空响应")

            except AnalysisError as e:
                error_str = str(e)
                is_rate_limit = "429" in error_str or "rate" in error_str.lower() or "quota" in error_str.lower()

                if is_rate_limit:
                    logger.warning(f"[{self.model}] API 限流，第 {attempt + 1}/{max_retries} 次尝试: {error_str[:100]}")
                else:
                    logger.warning(
                        f"[{self.model}] API 调用失败，第 {attempt + 1}/{max_retries} 次尝试: {error_str[:100]}"
                    )

                if attempt == max_retries - 1:
                    raise AnalysisError(f"{self.model} API 调用失败，已达最大重试次数: {error_str}") from e

        raise AnalysisError(f"{self.model} API 调用失败，已达最大重试次数") from None

    async def generate_with_tool(
        self,
        prompt: str,
        tool: dict[str, Any],
        generation_config: dict[str, Any],
        system_prompt: str | None = None,
    ) -> dict[str, Any] | None:
        """
        Generate structured output using Function Call (async).

        Uses litellm's tool calling support to get guaranteed structured output.
        This eliminates the need for JSON parsing and repair.

        Args:
            prompt: User prompt for the LLM
            tool: Function tool schema dict
            generation_config: Generation config (temperature, max_tokens)
            system_prompt: Optional system prompt override

        Returns:
            Parsed tool call arguments as dict, or None on failure
        """
        if not self._available:
            logger.warning(f"[{self.model}] Client not available, cannot use function call")
            return None

        config = get_config()
        temperature = generation_config.get("temperature", config.ai.llm_temperature)
        max_tokens = generation_config.get("max_output_tokens", config.ai.llm_max_tokens)

        sys_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt},
        ]

        try:
            kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "tools": [tool],
                "tool_choice": {"type": "function", "function": {"name": tool["function"]["name"]}},
                "temperature": temperature,
                "max_tokens": max_tokens,
            }

            if self.api_key:
                kwargs["api_key"] = self.api_key

            if self.base_url:
                kwargs["api_base"] = self.base_url

            response = await acompletion(**kwargs)

            if response and response.choices and len(response.choices) > 0:
                choice = response.choices[0]
                if choice.message.tool_calls and len(choice.message.tool_calls) > 0:
                    tool_call = choice.message.tool_calls[0]
                    arguments_str = tool_call.function.arguments
                    result = json.loads(arguments_str)
                    logger.debug(f"[{self.model}] Function call result: {result}")
                    return result

            logger.warning(f"[{self.model}] No tool calls in response")
            return None

        except json.JSONDecodeError as e:
            logger.error(f"[{self.model}] Failed to parse tool call arguments: {e}")
            return None
        except Exception as e:
            logger.error(f"[{self.model}] Function call failed: {e}")
            return None


def get_llm_client() -> LiteLLMClient | None:
    """
    获取 LLM 客户端实例（简单工厂函数）

    Returns:
        LiteLLMClient 实例，如果未配置则返回 None
    """
    try:
        config = get_config()
        if not config.ai.llm_api_key:
            logger.warning("未配置 LLM API Key")
            return None

        client = LiteLLMClient(
            model=config.ai.llm_model,
            api_key=config.ai.llm_api_key,
            base_url=config.ai.llm_base_url,
        )

        if client.is_available():
            logger.debug("LLM client created successfully")
            return client
        else:
            logger.warning("LLM 客户端初始化失败")
            return None

    except AnalysisError as e:
        logger.error(f"创建 LLM 客户端失败: {e}")
        return None
