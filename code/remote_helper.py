#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
密码认证的 SSH 助手（paramiko），用于迁移到新服务器。

用法：
    python remote_helper.py test                       # 测试连通性
    python remote_helper.py survey                     # 勘察硬件与环境
    python remote_helper.py install-key                # 安装本地公钥（之后可免密 ssh/scp）
    python remote_helper.py run "<命令>"                # 执行命令
    python remote_helper.py put <本地> <远端>            # 上传文件
    python remote_helper.py putdir <本地目录> <远端目录>  # 上传目录（递归）

连接信息必须通过环境变量提供（不再内置任何默认值）：
    NEW_HOST NEW_PORT NEW_USER NEW_PASS
示例：
    NEW_HOST=1.2.3.4 NEW_USER=xxx NEW_PASS=yyy python remote_helper.py test
"""

from __future__ import annotations

import os
import stat
import sys
import time
from pathlib import Path

import paramiko

HOST = os.environ.get("NEW_HOST", "")
PORT = int(os.environ.get("NEW_PORT", "22"))
USER = os.environ.get("NEW_USER", "")
PASS = os.environ.get("NEW_PASS", "")


def _require_env() -> None:
    missing = [k for k, v in (("NEW_HOST", HOST), ("NEW_USER", USER),
                              ("NEW_PASS", PASS)) if not v]
    if missing:
        sys.exit("错误：缺少环境变量 %s —— 本脚本不再内置默认连接信息，"
                 "请以 NEW_HOST=.. NEW_USER=.. NEW_PASS=.. 方式提供。" % ", ".join(missing))


def connect(timeout: int = 20) -> paramiko.SSHClient:
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username=USER, password=PASS,
              timeout=timeout, banner_timeout=timeout, auth_timeout=timeout,
              look_for_keys=False, allow_agent=False)
    return c


def run(c: paramiko.SSHClient, cmd: str, timeout: int = 300) -> tuple:
    stdin, stdout, stderr = c.exec_command(cmd, timeout=timeout, get_pty=False)
    out = stdout.read().decode("utf-8", "replace")
    err = stderr.read().decode("utf-8", "replace")
    rc = stdout.channel.recv_exit_status()
    return rc, out, err


def sftp_put_dir(sftp: paramiko.SFTPClient, local: Path, remote: str) -> int:
    """递归上传目录，返回文件数。"""
    n = 0
    try:
        sftp.mkdir(remote)
    except IOError:
        pass
    for p in sorted(local.rglob("*")):
        rel = p.relative_to(local).as_posix()
        target = "%s/%s" % (remote.rstrip("/"), rel)
        if p.is_dir():
            try:
                sftp.mkdir(target)
            except IOError:
                pass
        else:
            sftp.put(str(p), target)
            n += 1
    return n


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    action = sys.argv[1]
    _require_env()

    if action == "test":
        try:
            c = connect()
            rc, out, err = run(c, "echo CONNECTED; hostname; whoami; uname -sr")
            print(out.strip())
            c.close()
            return 0
        except Exception as e:
            print("连接失败:", type(e).__name__, e)
            return 1

    if action == "survey":
        c = connect()
        script = r"""
echo "=== 身份 ==="; whoami; hostname; uname -sr
echo; echo "=== GPU ==="
nvidia-smi --query-gpu=index,name,memory.total,temperature.gpu,utilization.gpu --format=csv 2>&1
echo; echo "=== 驱动/CUDA ==="
nvidia-smi 2>/dev/null | grep -E "Driver Version|CUDA Version"
echo; echo "=== CPU/内存 ==="
nproc; free -g | head -2
echo; echo "=== 磁盘 ==="
df -h ~ / 2>/dev/null | grep -v tmpfs
echo; echo "=== conda ==="
ls -d ~/miniconda3 ~/anaconda3 ~/miniconda /opt/conda 2>/dev/null || echo "(未发现常见 conda 路径)"
which conda 2>/dev/null || echo "(conda 不在 PATH)"
echo; echo "=== python ==="
python3 --version 2>&1
echo; echo "=== 网络（国内源可达性）==="
for h in modelscope.cn hf-mirror.com pypi.tuna.tsinghua.edu.cn repo.huaweicloud.com github.com; do
  printf "%-32s " "$h"
  curl -s -o /dev/null -w "%{http_code}\n" --max-time 8 "https://$h" 2>/dev/null || echo FAIL
done
"""
        rc, out, err = run(c, script, timeout=180)
        print(out)
        if err.strip():
            print("--- stderr ---")
            print(err[:1500])
        c.close()
        return 0

    if action == "install-key":
        pub = Path.home() / ".ssh" / "id_rsa.pub"
        if not pub.exists():
            print("未找到公钥:", pub)
            return 1
        key = pub.read_text(encoding="utf-8").strip()
        c = connect()
        cmd = (
            'mkdir -p ~/.ssh && chmod 700 ~/.ssh && touch ~/.ssh/authorized_keys && '
            'chmod 600 ~/.ssh/authorized_keys && '
            'grep -qF "%s" ~/.ssh/authorized_keys || echo "%s" >> ~/.ssh/authorized_keys; '
            'echo KEY_INSTALLED; wc -l < ~/.ssh/authorized_keys' % (key, key)
        )
        rc, out, err = run(c, cmd)
        print(out.strip() or err.strip())
        c.close()
        return 0

    if action == "run":
        c = connect()
        rc, out, err = run(c, sys.argv[2], timeout=900)
        print(out)
        if err.strip():
            print("--- stderr ---")
            print(err[:2000])
        c.close()
        return rc

    if action == "put":
        c = connect()
        sftp = c.open_sftp()
        sftp.put(sys.argv[2], sys.argv[3])
        print("已上传:", sys.argv[3])
        sftp.close()
        c.close()
        return 0

    if action == "putdir":
        c = connect()
        sftp = c.open_sftp()
        n = sftp_put_dir(sftp, Path(sys.argv[2]), sys.argv[3])
        print("已上传 %d 个文件到 %s" % (n, sys.argv[3]))
        sftp.close()
        c.close()
        return 0

    print("未知操作:", action)
    return 2


if __name__ == "__main__":
    sys.exit(main())
