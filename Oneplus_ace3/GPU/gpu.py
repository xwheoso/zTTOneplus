import subprocess
import time

gpu_clock_list = [
    124800000, 220000000, 295000000, 348000000, 
    401000000, 475000000, 550000000, 615000000, 680000000
]

temp_path = '/sys/class/thermal/thermal_zone63/temp'

class GPU:
    def __init__(self, ip):
        self.ip = ip
        self.clock_data = []
        self.temp_data = []
        self.clk_idx = 0
        
        # 传感器路径
        self.temp_path = temp_path
        
        # 可用频率列表
        self.gpu_clock_list = gpu_clock_list
        
        # 初始化：解除限制，设为最高频
        if self.gpu_clock_list:
            self.setGPUclock(len(self.gpu_clock_list) - 1)

    def setGPUclock(self, i):
        # 边界检查
        if not self.gpu_clock_list: return
        if i < 0: i = 0
        if i >= len(self.gpu_clock_list): i = len(self.gpu_clock_list) - 1
        
        self.clk_idx = i
        target_freq = self.gpu_clock_list[i]
        lowest_freq = self.gpu_clock_list[0]

        # 路径配置
        base = '/sys/class/kgsl/kgsl-3d0/devfreq'
        path_max = f'{base}/max_freq'
        path_min = f'{base}/min_freq'

        # 锁定逻辑：降Min -> 设Max -> 提Min
        cmd = f'echo {lowest_freq} > {path_min} && echo {target_freq} > {path_max} && echo {target_freq} > {path_min}'
        
        try:
            subprocess.check_output(['adb', '-s', self.ip, 'shell', 'su -c', '\"' + cmd + '\"'])
        except Exception as e:
            print(f"Set clock error: {e}")

    def getGPUtemp(self):
        try:
            output = subprocess.check_output(['adb', '-s', self.ip, 'shell', 'su -c', '\"cat ' + self.temp_path + '\"'])
            return int(output.decode('utf-8').strip()) / 1000
        except:
            return 0.0

    def getGPUclock(self):
        fname = '/sys/class/kgsl/kgsl-3d0/devfreq/cur_freq'
        try:
            output = subprocess.check_output(['adb', '-s', self.ip, 'shell', 'su -c', '\"cat ' + fname + '\"'])
            return int(output.decode('utf-8').strip())
        except:
            return 0

    def collectdata(self):
        self.clock_data.append(self.getGPUclock())
        self.temp_data.append(self.getGPUtemp())

    def setUserspace(self):
        pass
        
    def setdefault(self):
        # 恢复默认：解除锁定
        if not self.gpu_clock_list: return
        
        base = '/sys/class/kgsl/kgsl-3d0/devfreq'
        path_max = f'{base}/max_freq'
        path_min = f'{base}/min_freq'
        
        max_f = self.gpu_clock_list[-1]
        min_f = self.gpu_clock_list[0]
        
        cmd = f'echo {min_f} > {path_min} && echo {max_f} > {path_max}'
        try:
            subprocess.check_output(['adb', '-s', self.ip, 'shell', 'su -c', '\"' + cmd + '\"'])
            print("[GPU] Defaults restored")
        except:
            print("[GPU] Restore failed")

if __name__ == "__main__":
    # 填入设备IP
    IP = "dbff6874" 
    
    # 连接ADB
    print(f"Connecting to {IP}...")
    subprocess.run(f"adb connect {IP}", shell=True)

    gpu = GPU(IP)
    
    print(f"Temp: {gpu.getGPUtemp()} C")
    print(f"Freq (Start): {gpu.getGPUclock()} Hz")
    
    # 测试：锁定到最低频 (索引0: 124800000 Hz)
    target_idx = 0 
    print(f"Locking to freq index {target_idx} ({gpu.gpu_clock_list[target_idx]} Hz) for 5s...")
    gpu.setGPUclock(target_idx)
    
    # 保持5秒
    time.sleep(5)
    
    # 验证频率
    current_freq = gpu.getGPUclock()
    print(f"Freq (After 5s): {current_freq} Hz")
    
    if current_freq == gpu.gpu_clock_list[target_idx]:
        print("Status: Success")
    else:
        print("Status: Failed (Freq mismatch)")

    # 恢复默认
    gpu.setdefault()