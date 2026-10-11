"""pytest 全局配置

作用：
1. 把 `backend/` 目录加入 sys.path，让测试可以用 `from app.services... import ...`。
   项目是扁平布局（`backend/app/`），pytest 默认只把测试文件所在目录加入 sys.path，
   因此不配置的话 `import app` 会失败。
2. 提供 `workspace_tmp` fixture。

为什么不用 pytest 自带的 `tmp_path`：它落在系统临时目录（`%TEMP%`），
而本项目的运行沙箱只允许写入工作区内部，写系统临时目录会抛
`PermissionError: [WinError 5] 拒绝访问`。这里统一把临时文件放在
`backend/.pytest_tmp/` 下，并且清理失败也不影响测试结果。
"""

import os
import shutil
import sys
import uuid

import pytest

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

TMP_ROOT = os.path.join(BACKEND_DIR, ".pytest_tmp")


@pytest.fixture
def workspace_tmp():
    """在工作区内提供一个唯一的临时目录，测试结束后尽力清理。"""
    path = os.path.join(TMP_ROOT, uuid.uuid4().hex[:12])
    os.makedirs(path, exist_ok=True)
    yield path
    shutil.rmtree(path, ignore_errors=True)
