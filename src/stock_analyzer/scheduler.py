"""
定时任务调度模块

提供定时执行分析任务的功能。
"""

import logging
import time
from collections.abc import Callable
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


def run_with_schedule(
    task: Callable,
    schedule_time: str = "09:00",
    run_immediately: bool = True,
) -> None:
    """
    按指定时间定时运行任务

    Args:
        task: 要执行的任务函数
        schedule_time: 执行时间（HH:MM格式）
        run_immediately: 是否立即执行一次
    """
    if run_immediately:
        logger.info("立即执行首次任务...")
        try:
            task()
        except Exception as e:
            logger.error(f"首次任务执行失败: {e}")

    logger.info(f"进入定时模式，每日 {schedule_time} 执行")

    while True:
        try:
            now = datetime.now()
            target_time = datetime.strptime(schedule_time, "%H:%M").time()
            target_datetime = datetime.combine(now.date(), target_time)

            # 如果今天的时间已过，设置为明天
            if target_datetime <= now:
                target_datetime += timedelta(days=1)

            wait_seconds = (target_datetime - now).total_seconds()
            logger.info(f"下次执行时间: {target_datetime}, 等待 {wait_seconds / 60:.1f} 分钟")

            # 等待到目标时间
            time.sleep(wait_seconds)

            # 执行任务
            logger.info(f"执行定时任务: {datetime.now()}")
            try:
                task()
            except Exception as e:
                logger.error(f"定时任务执行失败: {e}")

        except KeyboardInterrupt:
            logger.info("定时任务被中断")
            break
        except Exception as e:
            logger.error(f"调度器错误: {e}")
            time.sleep(60)  # 出错后等待1分钟再试
