#!/usr/bin/env python3
"""
read_shm_image_mmap_only.py

仅使用 mmap 读取共享内存的 reader：
- 共享内存 /dev/shm/debug_frame 前 4 bytes = uint32 jpg_size (little endian)
  紧随其后的是 jpg_bytes (jpg_size bytes)
- 不再回退到文件读取；若 mmap 不可用则按 MMAP_RETRY_SECONDS 重试打开
- 原子写入 out_path.tmp -> out_path
- 支持环境变量覆盖： SHM_PATH, SHM_SIZE, OUT_PATH, FPS, DEBUG, MMAP_RETRY_SECONDS
"""
import os
import sys
import time
import struct
import hashlib
import signal

try:
    import mmap as _mmap
except Exception:
    _mmap = None

from io import BytesIO

# ---------- 配置（环境变量覆盖） ----------
SHM_PATH = os.environ.get("SHM_PATH", "/dev/shm/debug_frame")
SHM_SIZE = int(os.environ.get("SHM_SIZE", str(2 * 1024 * 1024)))  # bytes
OUT_PATH = os.environ.get("OUT_PATH", "/dev/shm/frame_preview.jpg")
FPS = float(os.environ.get("FPS", "10.0"))
DEBUG = os.environ.get("DEBUG", "0") != "0"
MMAP_RETRY_SECONDS = float(os.environ.get("MMAP_RETRY_SECONDS", "5.0"))
# -------------------------------------------------------------------

sleep_interval = 1.0 / max(1.0, FPS)

def log(*a, **k):
    if DEBUG:
        print(*a, **k)

def is_valid_image_bytes(b: bytes) -> bool:
    """尝试用 Pillow 验证图像是否完整（若 Pillow 不存在，返回 True 以不阻塞）"""
    try:
        from PIL import Image
        im = Image.open(BytesIO(b))
        im.verify()
        return True
    except ImportError:
        log("Pillow not installed; skipping image verify")
        return True
    except Exception as e:
        log("image verify failed:", e)
        return False

def open_mmap():
    """尝试打开并返回 (fd, mmap_obj) 或 (None, None)"""
    if _mmap is None:
        return None, None
    try:
        fd = os.open(SHM_PATH, os.O_RDONLY)
        mm = _mmap.mmap(fd, SHM_SIZE, access=_mmap.ACCESS_READ)
        log("mmap opened:", SHM_PATH, "size", SHM_SIZE)
        return fd, mm
    except Exception as e:
        log("open_mmap failed:", e)
        try:
            if 'fd' in locals() and fd:
                os.close(fd)
        except:
            pass
        return None, None

def close_mmap(fd, mm):
    try:
        if mm:
            mm.close()
    except:
        pass
    try:
        if fd:
            os.close(fd)
    except:
        pass

def read_from_mmap(mm):
    """
    从 mmap 读取一帧：
    结构: [uint32 little-endian jpg_size][jpg_bytes...]
    返回 bytes 或 None
    """
    try:
        mm.seek(0)
        hdr = mm.read(4)
        if len(hdr) < 4:
            return None
        jpg_size = struct.unpack("<I", hdr)[0]
        if jpg_size <= 0 or jpg_size > (SHM_SIZE - 4):
            log("invalid jpg_size:", jpg_size)
            return None
        jpg_bytes = mm.read(jpg_size)
        if len(jpg_bytes) != jpg_size:
            log(f"mmap truncated: expected {jpg_size} got {len(jpg_bytes)}")
            return None
        # quick header check
        if not (jpg_bytes.startswith(b"\xff\xd8\xff")):
            log("jpg header mismatch")
            return None
        return jpg_bytes
    except ValueError as e:
        log("mmap read ValueError:", e)
        return None
    except Exception as e:
        log("mmap read exception:", e)
        raise

def atomic_write(out_path, data):
    tmp = out_path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, out_path)

def loop():
    if _mmap is None:
        print("ERROR: mmap module not available in this Python environment. Exiting.", file=sys.stderr)
        sys.exit(2)

    fd = None
    mm = None
    last_hash = None
    last_mmap_try = 0.0

    # 初始尝试打开 mmap（若失败则进入重试循环）
    fd, mm = open_mmap()

    print(f"watching (mmap_only) '{SHM_PATH}' -> '{OUT_PATH}' @ {FPS} fps")
    while True:
        try:
            jpg_bytes = None

            # 尝试 mmap 读取（若 mm 为 None 则尝试重建）
            if mm is not None:
                try:
                    jpg_bytes = read_from_mmap(mm)
                    if jpg_bytes is None:
                        # 写端可能尚未写入完整帧，短暂等待
                        pass
                    else:
                        log("read frame from mmap, len=", len(jpg_bytes))
                except Exception as e:
                    log("mmap error, closing and will retry:", e)
                    close_mmap(fd, mm)
                    fd, mm = None, None
                    last_mmap_try = time.time()

            # 如果没有读到数据并且 mmap 不可用或不可读，则重试打开 mmap（按间隔）
            if jpg_bytes is None:
                now = time.time()
                if mm is None and (now - last_mmap_try) >= MMAP_RETRY_SECONDS:
                    fd, mm = open_mmap()
                    last_mmap_try = now
                time.sleep(sleep_interval)
                continue

            # 校验完整性（PIL verify）
            if not is_valid_image_bytes(jpg_bytes):
                log("image bytes failed verify, skipping")
                time.sleep(sleep_interval)
                continue

            # 内容哈希，避免重复写磁盘
            h = hashlib.sha256(jpg_bytes).hexdigest()
            if h != last_hash:
                atomic_write(OUT_PATH, jpg_bytes)
                last_hash = h
                log("wrote", OUT_PATH, "len=", len(jpg_bytes))
            # else: skip (no change)

            time.sleep(sleep_interval)

        except KeyboardInterrupt:
            print("terminated by user")
            break
        except Exception as e:
            log("unexpected error:", e)
            try:
                if mm is not None:
                    close_mmap(fd, mm)
            except:
                pass
            fd, mm = None, None
            time.sleep(1.0)

    # 退出前清理
    close_mmap(fd, mm)

def _handle_sigterm(signum, frame):
    print("terminated", file=sys.stderr)
    sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, _handle_sigterm)
    signal.signal(signal.SIGTERM, _handle_sigterm)
    loop()
