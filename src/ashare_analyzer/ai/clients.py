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

    支持主模型和备用模型，当主模型失败时自动切换到备用模型。

    使用方式：
        client = LiteLLMClient("deepseek/deepseek-reasoner", api_key="sk-...")
        response = await client.generate("分析这只股票", {"temperature": 0.7})
    """

    def __init__(
        self,
        model: str,
        api_key: str | None,
        base_url: str | None = None,
        fallback_model: str | None = None,
        fallback_api_key: str | None = None,
        fallback_base_url: str | None = None,
    ):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url

        # Fallback model configuration
        self.fallback_model = fallback_model
        self.fallback_api_key = fallback_api_key
        self.fallback_base_url = fallback_base_url

        self._available = self._validate_config()

        if not self._available:
            logger.warning(f"LiteLLM 客户端初始化失败 (模型: {model}) - 配置无效")

        if self.fallback_model:
            logger.info(f"LiteLLM 客户端已配置备用模型: {self.fallback_model}")

    def _validate_config(self) -> bool:
        if not self.api_key:
            return False
        if not self.model or "/" not in self.model:
            logger.warning(f"模型名称格式错误，应为 'provider/model-name' 格式: {self.model}")
            return False
        return True

    def is_available(self) -> bool:
        return self._available

    def has_fallback(self) -> bool:
        """Check if fallback model is configured."""
        return bool(self.fallback_model and self.fallback_api_key)

    async def generate(self, prompt: str, generation_config: dict, system_prompt: str | None = None) -> str:
        """
        生成内容，带重试机制和备用模型支持 (async)

        Args:
            prompt: 用户提示词
            generation_config: 生成配置（temperature, max_tokens 等）
            system_prompt: 系统提示词（可选，默认使用DEFAULT_SYSTEM_PROMPT）

        Returns:
            生成的文本

        Raises:
            Exception: 当主模型和备用模型都失败时
        """
        if not self._available:
            raise AnalysisError(f"LiteLLM 客户端未初始化或配置无效 (模型: {self.model})")

        config = get_config()
        max_retries = config.ai.llm_max_retries
        base_delay = config.ai.llm_retry_delay
        timeout = generation_config.get("timeout", config.ai.llm_timeout)

        temperature = generation_config.get("temperature", config.ai.llm_temperature)
        max_tokens = generation_config.get("max_output_tokens", config.ai.llm_max_tokens)

        sys_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt},
        ]

        # Try primary model first, then fallback
        models_to_try = [(self.model, self.api_key, self.base_url)]
        if self.has_fallback():
            models_to_try.append((self.fallback_model, self.fallback_api_key, self.fallback_base_url))

        last_error = None
        for model_idx, (model, api_key, base_url) in enumerate(models_to_try):
            is_fallback = model_idx > 0
            model_name = model or "unknown"

            if is_fallback:
                logger.info(f"切换到备用模型: {model_name}")

            for attempt in range(max_retries):
                try:
                    if attempt > 0:
                        delay = calculate_backoff_delay(attempt, base_delay, max_delay=60.0)
                        logger.info(f"[{model_name}] 第 {attempt + 1} 次重试，等待 {delay:.1f} 秒...")
                        await asyncio.sleep(delay)

                    kwargs: dict[str, Any] = {
                        "model": model,
                        "messages": messages,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "timeout": timeout,
                    }

                    if api_key:
                        kwargs["api_key"] = api_key

                    if base_url:
                        kwargs["api_base"] = base_url

                    response = await acompletion(**kwargs)

                    if response and response.choices and len(response.choices) > 0:
                        content = response.choices[0].message.content
                        if content:
                            if is_fallback:
                                logger.info(f"[{model_name}] 备用模型响应成功")
                            else:
                                logger.debug(f"[{model_name}] LLM响应: {content[:500]}...")
                            return content.strip()

                    raise AnalysisError(f"{model_name} 返回空响应")

                except Exception as e:
                    error_str = str(e)
                    is_rate_limit = "429" in error_str or "rate" in error_str.lower()
                    is_rate_limit = is_rate_limit or "quota" in error_str.lower()
                    is_server_error = "524" in error_str or "500" in error_str
                    is_server_error = is_server_error or "502" in error_str or "503" in error_str

                    if is_rate_limit:
                        logger.warning(
                            f"[{model_name}] API 限流，第 {attempt + 1}/{max_retries} 次尝试: {error_str[:100]}"
                        )
                    elif is_server_error:
                        logger.warning(
                            f"[{model_name}] 服务器错误，第 {attempt + 1}/{max_retries} 次尝试: {error_str[:100]}"
                        )
                    else:
                        logger.warning(
                            f"[{model_name}] API 调用失败，第 {attempt + 1}/{max_retries} 次尝试: {error_str[:100]}"
                        )

                    last_error = e

                    # Only retry on same model if we haven't exhausted retries
                    if attempt == max_retries - 1 and not is_fallback and self.has_fallback():
                        logger.warning(f"[{model_name}] 主模型失败，尝试备用模型...")
                        break  # Break inner loop to try fallback model
                    elif attempt == max_retries - 1:
                        raise AnalysisError(f"{model_name} API 调用失败，已达最大重试次数: {error_str}") from e

        raise AnalysisError(f"所有模型调用失败: {last_error}") from last_error

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

        Includes retry logic and fallback model support for resilience.

        Args:
            prompt: User prompt for the LLM
            tool: Function tool schema dict
            generation_config: Generation config (temperature, max_tokens, timeout)
            system_prompt: Optional system prompt override

        Returns:
            Parsed tool call arguments as dict, or None on failure
        """
        if not self._available:
            logger.warning(f"[{self.model}] Client not available, cannot use function call")
            return None

        config = get_config()
        max_retries = config.ai.llm_max_retries
        base_delay = config.ai.llm_retry_delay
        temperature = generation_config.get("temperature", config.ai.llm_temperature)
        max_tokens = generation_config.get("max_output_tokens", config.ai.llm_max_tokens)
        timeout = generation_config.get("timeout", config.ai.llm_timeout)

        sys_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt},
        ]

        # Try primary model first, then fallback
        models_to_try = [(self.model, self.api_key, self.base_url)]
        if self.has_fallback():
            models_to_try.append((self.fallback_model, self.fallback_api_key, self.fallback_base_url))

        last_error = None
        for model_idx, (model, api_key, base_url) in enumerate(models_to_try):
            is_fallback = model_idx > 0
            model_name = model or "unknown"

            if is_fallback:
                logger.info(f"切换到备用模型: {model_name}")

            for attempt in range(max_retries):
                try:
                    if attempt > 0:
                        delay = calculate_backoff_delay(attempt, base_delay, max_delay=60.0)
                        logger.info(f"[{model_name}] 第 {attempt + 1} 次重试，等待 {delay:.1f} 秒...")
                        await asyncio.sleep(delay)

                    kwargs: dict[str, Any] = {
                        "model": model,
                        "messages": messages,
                        "tools": [tool],
                        "tool_choice": {"type": "function", "function": {"name": tool["function"]["name"]}},
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "timeout": timeout,
                    }

                    if api_key:
                        kwargs["api_key"] = api_key

                    if base_url:
                        kwargs["api_base"] = base_url

                    response = await acompletion(**kwargs)

                    if response and response.choices and len(response.choices) > 0:
                        choice = response.choices[0]
                        if choice.message.tool_calls and len(choice.message.tool_calls) > 0:
                            tool_call = choice.message.tool_calls[0]
                            arguments_str = tool_call.function.arguments
                            result = json.loads(arguments_str)
                            if is_fallback:
                                logger.info(f"[{model_name}] 备用模型 Function call 成功")
                            else:
                                logger.debug(f"[{model_name}] Function call result: {result}")
                            return result
                        else:
                            # Log when model returns text instead of tool call
                            content_preview = ""
                            if choice.message.content:
                                content_preview = choice.message.content[:200]
                            logger.warning(
                                f"[{model_name}] No tool calls in response. "
                                f"Model returned text instead: {content_preview}..."
                            )

                    logger.warning(f"[{model_name}] No tool calls in response")
                    last_error = Exception("No tool calls in response")

                except json.JSONDecodeError as e:
                    logger.error(f"[{model_name}] Failed to parse tool call arguments: {e}")
                    last_error = e
                except Exception as e:
                    error_str = str(e)
                    is_rate_limit = "429" in error_str or "rate" in error_str.lower()
                    is_rate_limit = is_rate_limit or "quota" in error_str.lower()
                    is_server_error = "524" in error_str or "500" in error_str
                    is_server_error = is_server_error or "502" in error_str or "503" in error_str

                    if is_rate_limit:
                        logger.warning(
                            f"[{model_name}] API 限流，第 {attempt + 1}/{max_retries} 次尝试: {error_str[:100]}"
                        )
                    elif is_server_error:
                        logger.warning(
                            f"[{model_name}] 服务器错误，第 {attempt + 1}/{max_retries} 次尝试: {error_str[:100]}"
                        )
                    else:
                        logger.error(f"[{model_name}] Function call failed: {e}")
                    last_error = e

                # Only retry on same model if we haven't exhausted retries
                if attempt == max_retries - 1 and not is_fallback and self.has_fallback():
                    logger.warning(f"[{model_name}] 主模型失败，尝试备用模型...")
                    break  # Break inner loop to try fallback model

        logger.error(f"所有模型 generate_with_tool 调用失败: {last_error}")
        return None


def get_llm_client() -> LiteLLMClient | None:
    """
    获取 LLM 客户端实例（简单工厂函数）

    支持主模型和备用模型配置，当主模型失败时自动切换到备用模型。

    Returns:
        LiteLLMClient 实例，如果未配置则返回 None
    """
    try:
        config = get_config()
        if not config.ai.llm_api_key:
            logger.warning("未配置 LLM API Key")
            return None

        if not config.ai.llm_model:
            logger.warning("未配置 LLM 模型名称")
            return None

        client = LiteLLMClient(
            model=config.ai.llm_model,
            api_key=config.ai.llm_api_key,
            base_url=config.ai.llm_base_url,
            fallback_model=config.ai.llm_fallback_model,
            fallback_api_key=config.ai.llm_fallback_api_key,
            fallback_base_url=config.ai.llm_fallback_base_url,
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


def get_filter_llm_client() -> LiteLLMClient | None:
    """
    Get LLM client for news filtering.

    Uses a smaller/faster model if configured, otherwise falls back to main model.

    Returns:
        LiteLLMClient instance for filtering, or None if not configured
    """
    try:
        config = get_config()
        if not config.ai.llm_api_key:
            logger.warning("未配置 LLM API Key")
            return None

        # Use filter-specific model if configured, otherwise use main model
        model = config.news_filter.news_filter_model or config.ai.llm_model
        if not model:
            logger.warning("未配置 LLM 模型名称")
            return None

        client = LiteLLMClient(
            model=model,
            api_key=config.ai.llm_api_key,
            base_url=config.ai.llm_base_url,
            fallback_model=config.ai.llm_fallback_model,
            fallback_api_key=config.ai.llm_fallback_api_key,
            fallback_base_url=config.ai.llm_fallback_base_url,
        )

        if client.is_available():
            logger.debug(f"Filter LLM client created with model: {model}")
            return client
        else:
            logger.warning("Filter LLM 客户端初始化失败")
            return None

    except Exception as e:
        logger.error(f"创建 Filter LLM 客户端失败: {e}")
        return None
