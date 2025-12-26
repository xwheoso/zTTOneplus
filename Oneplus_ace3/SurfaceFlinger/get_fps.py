import time
import subprocess
import threading
import re

nanoseconds_per_second = 1e9

class SurfaceFlingerFPS:
    def __init__(self, view, ip):
        self.view = view
        self.ip = ip
        self.fps = 0.0
        self.running = False
        self.lock = threading.Lock()
        self.thread = None

    def __frame_data__(self):
        try:
            # 获取最近的帧数据
            cmd = ['adb', '-s', self.ip, 'shell', 'dumpsys', 'SurfaceFlinger', '--latency', self.view]
            # 设置超时防止卡死
            out = subprocess.check_output(cmd, timeout=2).decode('utf-8')
            results = out.splitlines()
            
            if not results: return []

            timestamps = []
            # 第一行是刷新周期，跳过
            for line in results[1:]:
                fields = line.split()
                if len(fields) != 3: continue
                # index 1 是提交时间 (submitting timestamp)
                timestamp = int(fields[1])
                if timestamp == 0: continue
                timestamps.append(timestamp)
            
            return timestamps
        except Exception as e:
            # print(f"FPS read error: {e}")
            return []

    def _monitor_loop(self):
        while self.running:
            start_time = time.time()
            
            # 获取当前 SurfaceFlinger 缓冲区里的所有帧时间戳
            current_timestamps = self.__frame_data__()
            
            if current_timestamps:
                # 逻辑：计算最近 1 秒内产生的帧数
                # 也就是：(最新帧时间 - 某帧时间) < 1秒
                if len(current_timestamps) > 1:
                    newest = current_timestamps[-1]
                    count = 0
                    # 倒序遍历，统计1秒窗口内的帧
                    for ts in reversed(current_timestamps):
                        if (newest - ts) < 1e9: # 1e9 ns = 1 second
                            count += 1
                        else:
                            break
                    
                    with self.lock:
                        self.fps = float(count)
            
            # 控制采样频率，大约每秒更新一次 FPS
            elapsed = time.time() - start_time
            sleep_time = max(0.1, 1.0 - elapsed) 
            time.sleep(sleep_time)

    def start(self):
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._monitor_loop)
            self.thread.daemon = True # 主程序退出时自动结束线程
            self.thread.start()
            print("[FPS Driver] Background thread started.")

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()

    def getFPS(self):
        # 瞬间返回，不阻塞
        with self.lock:
            return self.fps