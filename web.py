from flask import Flask, render_template, Response, jsonify
import time, json, socket, os, logging, struct
import mmap, posix_ipc

app = Flask(__name__)

# 通信参数
shared_name = "/debug_frame"
shared_size = 2 * 1024 * 1024
shared_frame_path = "/dev/shm/debug_frame.jpg"

# 初始化通信模式
use_shared_memory = False
mapfile = None

try:
    shm = posix_ipc.SharedMemory(shared_name)
    mapfile = mmap.mmap(shm.fd, shared_size, mmap.MAP_SHARED, mmap.PROT_READ)
    shm.close_fd()
    use_shared_memory = True
    print("[INFO] 使用共享内存模式")
except Exception as e:
    print(f"[WARN] 共享内存不可用: {e}")
    use_shared_memory = False


# MJPEG 流生成器（共享内存或文件）
def mjpeg_stream():
    global mapfile, use_shared_memory
    while True:
        try:
            if use_shared_memory:
                mapfile.seek(0)
                size_bytes = mapfile.read(4)
                jpg_size = struct.unpack("I", size_bytes)[0]
                if 0 < jpg_size < shared_size - 4:
                    jpg_bytes = mapfile.read(jpg_size)
                else:
                    print(f"[WARN] jpg_size 超出范围: {jpg_size}")
                    time.sleep(0.03)
                    continue
            else:
                with open(shared_frame_path, "rb") as f:
                    jpg_bytes = f.read()

            yield (
                b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + jpg_bytes + b"\r\n"
            )
        except FileNotFoundError:
            time.sleep(0.01)
        except Exception as e:
            print(f"[ERROR] MJPEG 读取失败: {e}")
        time.sleep(0.03)


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


@app.route("/aim_log")
def aim_log():
    try:
        with open("/dev/shm/aim_log.json", "r") as f:
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
    app.run(host="0.0.0.0", port=5000, threaded=True)
