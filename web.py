from flask import Flask, render_template, Response, jsonify
import time, json, socket, os, logging, struct
import mmap
import threading
import subprocess
import atexit
import fcntl

app = Flask(__name__)

# 通信参数
shared_memory_path = "/dev/shm/debug_frame"
shared_size = 2 * 1024 * 1024  # 2MB
shared_frame_path = "/dev/shm/debug_frame.jpg"

# 初始化通信模式
use_shared_memory = False
mapfile = None
fd = None

# 权限修复锁
permission_lock = threading.Lock()


def ensure_shared_memory_permissions():
    """确保共享内存文件存在且权限正确"""
    with permission_lock:
        try:
            # 1. 确保文件存在
            if not os.path.exists(shared_memory_path):
                print(f"创建共享内存文件: {shared_memory_path}")
                with open(shared_memory_path, "wb") as f:
                    f.write(b"\0" * shared_size)

            # 2. 检查并修复权限
            current_mode = oct(os.stat(shared_memory_path).st_mode & 0o777)
            if current_mode != "0o666":
                print(f"修复权限 (当前: {current_mode} -> 目标: 666)")

                # 使用 sudo 修复权限
                result = subprocess.run(
                    ["sudo", "chmod", "666", shared_memory_path],
                    capture_output=True,
                    text=True,
                )

                if result.returncode == 0:
                    print("权限修复成功")
                    return True
                else:
                    print(f"权限修复失败: {result.stderr.strip()}")
                    return False
            return True
        except Exception as e:
            print(f"权限修复异常: {str(e)}")
            return False


def init_shared_memory():
    """初始化共享内存连接"""
    global use_shared_memory, mapfile, fd

    # 确保权限正确
    if not ensure_shared_memory_permissions():
        print("[WARN] 权限修复失败，回退到文件模式")
        use_shared_memory = False
        return False

    try:
        # 使用直接文件访问方式
        fd = os.open(shared_memory_path, os.O_RDONLY)
        mapfile = mmap.mmap(fd, shared_size, mmap.MAP_SHARED, mmap.PROT_READ)

        # 设置非阻塞锁
        fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)

        use_shared_memory = True
        print("[INFO] 共享内存初始化成功")
        return True
    except Exception as e:
        print(f"[WARN] 共享内存初始化失败: {e}")
        use_shared_memory = False

        # 清理资源
        if mapfile:
            try:
                mapfile.close()
            except:
                pass
            mapfile = None

        if fd:
            try:
                os.close(fd)
            except:
                pass
            fd = None

        return False


# 初始连接尝试
if init_shared_memory():
    print("✅ 使用共享内存模式")
else:
    print("⚠️ 使用文件回退模式")


# 清理函数
@atexit.register
def cleanup():
    if mapfile:
        try:
            mapfile.close()
        except:
            pass

    if fd:
        try:
            os.close(fd)
        except:
            pass


# MJPEG 流生成器
def mjpeg_stream():
    global use_shared_memory, mapfile

    last_fix_attempt = 0

    while True:
        try:
            if use_shared_memory and mapfile:
                try:
                    # 重置到文件开头
                    mapfile.seek(0)

                    # 读取JPEG大小 (前4字节)
                    size_bytes = mapfile.read(4)
                    if len(size_bytes) < 4:
                        print("[WARN] 未读取到完整的大小头")
                        time.sleep(0.03)
                        continue

                    jpg_size = struct.unpack("I", size_bytes)[0]

                    # 验证大小有效性
                    if jpg_size <= 0 or jpg_size > shared_size - 4:
                        print(f"[WARN] 无效的JPEG大小: {jpg_size}")
                        time.sleep(0.03)
                        continue

                    # 读取JPEG数据
                    jpg_bytes = mapfile.read(jpg_size)

                    # 验证数据完整性
                    if len(jpg_bytes) != jpg_size:
                        print(f"[WARN] 数据不完整: {len(jpg_bytes)}/{jpg_size} 字节")
                        time.sleep(0.03)
                        continue

                    # 检查JPEG头部
                    if jpg_bytes[0:3] != b"\xff\xd8\xff":
                        print("[WARN] 无效的JPEG头部")
                        time.sleep(0.03)
                        continue

                except (OSError, ValueError) as e:
                    print(f"[ERROR] 共享内存访问错误: {e}")

                    # 限流：每分钟最多尝试修复一次
                    current_time = time.time()
                    if current_time - last_fix_attempt > 60:
                        print("尝试重新初始化共享内存...")
                        if init_shared_memory():
                            continue
                        last_fix_attempt = current_time

                    # 暂时回退到文件模式
                    use_shared_memory = False
                    continue

            # 文件模式回退
            if not use_shared_memory or not mapfile:
                try:
                    with open(shared_frame_path, "rb") as f:
                        jpg_bytes = f.read()

                    # 验证JPEG文件
                    if jpg_bytes[0:3] != b"\xff\xd8\xff":
                        print("[WARN] 文件模式: 无效的JPEG头部")
                        time.sleep(0.03)
                        continue
                except FileNotFoundError:
                    print(f"[WARN] 文件未找到: {shared_frame_path}")
                    time.sleep(0.1)
                    continue
                except Exception as e:
                    print(f"[ERROR] 文件读取失败: {e}")
                    time.sleep(0.1)
                    continue

            # 生成MJPEG帧
            yield (
                b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + jpg_bytes + b"\r\n"
            )
            time.sleep(0.03)
        except Exception as e:
            print(f"[CRITICAL] MJPEG 生成器异常: {e}")
            time.sleep(0.5)


@app.route("/")
def index():
    def get_local_ip():
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("10.255.255.255", 1))
            IP = s.getsockname()[0]
        except:
            IP = "127.0.0.1"
        finally:
            s.close()
        return IP

    url = f"http://{get_local_ip()}:5000"
    return render_template("index.html", server_url=url)


@app.route("/video")
def video_feed():
    return Response(
        mjpeg_stream(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/data")
def get_data():
    try:
        with open("/dev/shm/cmd_log.json", "r") as f:
            return jsonify(json.load(f))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/serial_log")
def serial_log():
    try:
        with open("/dev/shm/serial_log.json", "r") as f:
            return jsonify(json.load(f))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/target_log")
def target_log():
    try:
        with open("/dev/shm/target_log.json", "r") as f:
            return jsonify(json.load(f))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("werkzeug").setLevel(logging.ERROR)

    # 获取本地IP
    def get_local_ip():
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("10.255.255.255", 1))
            IP = s.getsockname()[0]
        except:
            IP = "127.0.0.1"
        finally:
            s.close()
        return IP

    url = f"http://{get_local_ip()}:5000"
    print(f"✅ Web 调试器已启动: {url}")
    print(f"   - 共享内存模式: {'是' if use_shared_memory else '否'}")
    print(f"   - 访问地址: {url}")

    app.run(host="0.0.0.0", port=5000, threaded=True)
