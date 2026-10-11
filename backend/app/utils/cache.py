"""缓存管理"""

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional
import logging
import copy
import tempfile
from functools import wraps, lru_cache
from threading import RLock

logger = logging.getLogger(__name__)


def synchronized(method):
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)
    return wrapped


class CacheManager:
    """缓存管理器"""

    def __init__(self, cache_dir: str = "./cache", enabled: bool = True):
        self._lock = RLock()
        self.cache_dir = Path(cache_dir)
        self.enabled = enabled
        self._cache: Dict[str, Any] = {}

        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._load_cache()

    def _get_cache_file(self) -> Path:
        """获取缓存文件路径"""
        return self.cache_dir / "extraction_cache.json"

    def _load_cache(self) -> None:
        """加载缓存"""
        cache_file = self._get_cache_file()
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    self._cache = json.load(f)
                logger.info(f"Loaded {len(self._cache)} cached items")
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
                self._cache = {}

    def _save_cache(self) -> None:
        """保存缓存"""
        if not self.enabled:
            return

        try:
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.cache_dir, delete=False) as f:
                    temporary = f.name
                    json.dump(self._cache, f, ensure_ascii=False, indent=2)
                os.replace(temporary, self._get_cache_file())
            finally:
                if temporary and os.path.exists(temporary):
                    os.unlink(temporary)
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

    def _make_key(self, text: str, scenario: str) -> str:
        """生成缓存键"""
        content = f"{scenario}:{text}"
        return hashlib.md5(content.encode()).hexdigest()

    @synchronized
    def get(self, text: str, scenario: str) -> Optional[Dict[str, Any]]:
        """获取缓存"""
        if not self.enabled:
            return None

        key = self._make_key(text, scenario)
        result = self._cache.get(key)

        if result:
            logger.debug(f"Cache hit for key: {key[:8]}...")

        return copy.deepcopy(result)

    @synchronized
    def set(self, text: str, scenario: str, value: Dict[str, Any]) -> None:
        """设置缓存"""
        if not self.enabled:
            return

        key = self._make_key(text, scenario)
        self._cache[key] = copy.deepcopy(value)
        self._save_cache()
        logger.debug(f"Cached result for key: {key[:8]}...")

    @synchronized
    def clear(self) -> None:
        """清空缓存"""
        self._cache = {}
        cache_file = self._get_cache_file()
        if cache_file.exists():
            cache_file.unlink()
        logger.info("Cache cleared")

    @synchronized
    def stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        cache_file = self._get_cache_file()
        return {
            "enabled": self.enabled,
            "total_items": len(self._cache),
            "cache_file": str(cache_file),
            "cache_size_bytes": cache_file.stat().st_size if cache_file.exists() else 0,
        }


_cache_factory_lock = RLock()


@lru_cache(maxsize=16)
def _cached_manager(path: str, enabled: bool) -> CacheManager:
    return CacheManager(path, enabled)


def get_cache_manager(cache_dir: str, enabled: bool) -> CacheManager:
    with _cache_factory_lock:
        return _cached_manager(str(Path(cache_dir).resolve()), enabled)
