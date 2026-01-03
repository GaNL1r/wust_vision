from flask import Flask, render_template, Response, jsonify
import time, json, socket, os, logging, struct
import mmap
import threading
import subprocess
import atexit
import fcntl
import setproctitle
setproctitle.setproctitle("wust_vision_web")

app = Flask(__name__)

# ===============================
# 参数设置：选择模式
# ===============================
# True -> 强制共享内存模式
# False -> 文件模式
USE_SHARED_MEMORY_MODE = True
STREAM_FPS = 60          
FRAME_INTERVAL = 1.0 / STREAM_FPS
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
            if not os.path.exists(shared_memory_path):
                print(f"创建共享内存文件: {shared_memory_path}")
                with open(shared_memory_path, "wb") as f:
                    f.write(b"\0" * shared_size)

            current_mode = oct(os.stat(shared_memory_path).st_mode & 0o777)
            if current_mode != "777":
                print(f"修复权限 (当前: {current_mode} -> 目标: 777)")
                result = subprocess.run(
                    ["sudo", "chmod", "777", shared_memory_path],
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

    if not ensure_shared_memory_permissions():
        print("[WARN] 权限修复失败")
        use_shared_memory = False
        return False

    try:
        fd = os.open(shared_memory_path, os.O_RDONLY)
        mapfile = mmap.mmap(fd, shared_size, mmap.MAP_SHARED, mmap.PROT_READ)
        fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        use_shared_memory = True
        print("[INFO] 共享内存初始化成功")
        return True
    except Exception as e:
        print(f"[WARN] 共享内存初始化失败: {e}")
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
        use_shared_memory = False
        return False


# ===============================
# 初始化模式
# ===============================
if USE_SHARED_MEMORY_MODE:
    if init_shared_memory():
        print("✅ 使用共享内存模式")
    else:
        print("⚠️ 强制共享内存模式失败，回退到文件模式")
        use_shared_memory = False
else:
    use_shared_memory = False
    print("ℹ️ 使用文件模式")


# ===============================
# 清理函数
# ===============================
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


# ===============================
# MJPEG 流生成器
# ===============================
def mjpeg_stream():
    global use_shared_memory, mapfile
    last_fix_attempt = 0

    while True:
        try:
            if use_shared_memory and mapfile:
                try:
                    mapfile.seek(0)
                    size_bytes = mapfile.read(4)
                    if len(size_bytes) < 4:
                        time.sleep(FRAME_INTERVAL)
                        continue
                    jpg_size = struct.unpack("I", size_bytes)[0]
                    if jpg_size <= 0 or jpg_size > shared_size - 4:
                        time.sleep(FRAME_INTERVAL)
                        continue
                    jpg_bytes = mapfile.read(jpg_size)
                    if len(jpg_bytes) != jpg_size:
                        time.sleep(FRAME_INTERVAL)
                        continue
                    if jpg_bytes[0:3] != b"\xff\xd8\xff":
                        time.sleep(FRAME_INTERVAL)
                        continue
                except (OSError, ValueError) as e:
                    current_time = time.time()
                    if current_time - last_fix_attempt > 60:
                        print("尝试重新初始化共享内存...")
                        if init_shared_memory():
                            continue
                        last_fix_attempt = current_time
                    use_shared_memory = False
                    continue

            if not use_shared_memory or not mapfile:
                try:
                    with open(shared_frame_path, "rb") as f:
                        jpg_bytes = f.read()
                    if jpg_bytes[0:3] != b"\xff\xd8\xff":
                        time.sleep(FRAME_INTERVAL)
                        continue
                except FileNotFoundError:
                    time.sleep(0.1)
                    continue
                except Exception:
                    time.sleep(0.1)
                    continue

            yield b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + jpg_bytes + b"\r\n"
            time.sleep(FRAME_INTERVAL)
        except Exception:
            time.sleep(0.5)


# ===============================
# Flask 路由
# ===============================
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


# ===============================
# 主函数
# ===============================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("werkzeug").setLevel(logging.ERROR)

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
