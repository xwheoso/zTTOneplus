import subprocess


# Cluster 0 (CPU 0-2)
little_cpu_clock_list = [
    307200, 441600, 556800, 672000, 787200, 902400, 1017600, 1113600, 
    1228800, 1344000, 1459200, 1555200, 1670400, 1785600, 1900800, 2016000
]

# Cluster 1 (CPU 3-6)
big_cpu_clock_list = [
    499200, 614400, 729600, 844800, 940800, 1056000, 1171200, 1286400, 
    1401600, 1536000, 1651200, 1785600, 1920000, 2054400, 2188800, 2323200, 
    2457600, 2592000, 2707200, 2803200
]

# Cluster 2 (CPU 7)
prime_cpu_clock_list = [
    595200, 729600, 864000, 998400, 1132800, 1248000, 1363200, 1478400, 
    1593600, 1708800, 1843200, 1977600, 2092800, 2227200, 2342400, 2476800, 
    2592000, 2726400, 2841600, 2956800
]

dir_thermal = '/sys/devices/virtual/thermal'

class CPU:
    def __init__(self, idx, cpu_type, ip):
        """
        初始化 CPU 对象
        :param idx: CPU 核心编号 (建议只使用代表核心: 0, 3, 或 7)
        :param cpu_type: 'l' (little), 'b' (big), 'p' (prime)
        :param ip: 设备 IP 地址 (用于 adb -s 连接)
        """
        self.idx = idx
        self.cpu_type = cpu_type
        self.ip = ip
        self.clock_data = []
        self.temp_data = []

        # 根据类型加载对应的频率表
        if self.cpu_type == 'l':
            self.cpu_clock_list = little_cpu_clock_list
        elif self.cpu_type == 'b':
            self.cpu_clock_list = big_cpu_clock_list
        elif self.cpu_type == 'p':
            self.cpu_clock_list = prime_cpu_clock_list
        else:
            print(f"Warning: Unknown CPU type {cpu_type}, defaulting to little")
            self.cpu_clock_list = little_cpu_clock_list

        # 初始化时，默认设置为最高频 (解锁性能)
        self.max_freq_idx = len(self.cpu_clock_list) - 1
        self.setCPUclock(self.max_freq_idx)

    def setCPUclock(self, i):
        """
        使用 Min-Max 锁定法
        """
        # 边界检查
        if i < 0: i = 0
        if i >= len(self.cpu_clock_list): i = len(self.cpu_clock_list) - 1
        
        target_freq = self.cpu_clock_list[i]
        
        # 文件路径
        path_max = '/sys/devices/system/cpu/cpu%s/cpufreq/scaling_max_freq' % (self.idx)
        path_min = '/sys/devices/system/cpu/cpu%s/cpufreq/scaling_min_freq' % (self.idx)
        
        # 获取列表中的最小频率，用于临时解锁下限
        lowest_freq = self.cpu_clock_list[0]
        
        # 锁定逻辑：先降下限(解锁) -> 设上限 -> 提下限(锁定)
        # 这一步防止 target < current_min 导致写入失败
        cmd = 'echo {} > {} && echo {} > {} && echo {} > {}'.format(
            lowest_freq, path_min,   # 1. Unlock min
            target_freq, path_max,   # 2. Set max
            target_freq, path_min    # 3. Lock min
        )
        
        try:
            subprocess.check_output(['adb', '-s', self.ip, 'shell', 'su -c', '\"' + cmd + '\"'])
            # print(f"DEBUG: CPU{self.idx} locked to {target_freq}") 
        except subprocess.CalledProcessError as e:
            print(f"Error setting CPU{self.idx} clock: {e}")

    def getCPUtemp(self):
        # cpuss-0 位于 thermal_zone31
        fname = '{}/thermal_zone31/temp'.format(dir_thermal)
        
        try:
            # 读取温度值
            output = subprocess.check_output(['adb', '-s', self.ip, 'shell', 'su -c', '\"cat', fname+"\""])
            return int(output.decode('utf-8').strip())/1000
        except Exception as e:
            print(f"Error reading CPU temp: {e}")
            return 0.0

    def getCPUclock(self, idx):
        fname = '/sys/devices/system/cpu/cpu%s/cpufreq/cpuinfo_cur_freq' % idx
        try:
            output = subprocess.check_output(['adb', '-s', self.ip, 'shell', 'su -c', '\"cat', fname+"\""])
            return int(output.decode('utf-8').strip())/1000
        except:
            return 0.0

    def collectdata(self):
        self.clock_data.append(self.getCPUclock(self.idx))
        self.temp_data.append(self.getCPUtemp())

        
    def setdefault(self, mode=None):
        # 恢复默认频率范围 (Min=最低, Max=最高)
        path_max = '/sys/devices/system/cpu/cpu%s/cpufreq/scaling_max_freq' % (self.idx)
        path_min = '/sys/devices/system/cpu/cpu%s/cpufreq/scaling_min_freq' % (self.idx)
        
        max_limit = self.cpu_clock_list[-1]
        min_limit = self.cpu_clock_list[0]
        
        cmd = 'echo {} > {} && echo {} > {}'.format(min_limit, path_min, max_limit, path_max)
        try:
            subprocess.check_output(['adb', '-s', self.ip, 'shell', 'su -c', '\"' + cmd + '\"'])
            print(f'[cpu{self.idx}] Restored bounds to {min_limit}-{max_limit}')
        except Exception as e:
            print(f"Error restoring defaults: {e}")

# ==========================================
# 如何在主程序中使用
# ==========================================
if __name__ == "__main__":
    # 如果是WiFi连接的，这里换成手机的ip地址
    DEVICE_IP = "dbff6874"
    
    # 只需要实例化三个代表，因为同 Cluster 频率同步
    cpu_little = CPU(0, 'l', DEVICE_IP)
    cpu_big    = CPU(3, 'b', DEVICE_IP)
    cpu_prime  = CPU(7, 'p', DEVICE_IP)
    
    print("正在测试小核频率设置...")
    cpu_little.setCPUclock(5)# 应该输出902.4
    print(f"当前频率: {cpu_little.getCPUclock(0)}")
    
    print("正在测试超大核频率设置...")
    cpu_prime.setCPUclock(10)# 应该输出1843.2
    print(f"当前频率: {cpu_prime.getCPUclock(7)}")