import subprocess
import time

battery_path = "/sys/class/power_supply/battery"

class PowerLogger:
    def __init__(self, ip):
        self.ip = ip
        self.power = 0
        self.voltage = 0
        self.current = 0
        
        self.battery_path = battery_path
        self._connect()

    def _connect(self):
        print(f"Connecting to {self.ip}...")
        subprocess.run(f"adb connect {self.ip}", shell=True)

    def _read_file(self, filename):
        cmd = f"cat {self.battery_path}/{filename}"
        try:
            # timeout 设置短一点，防止卡顿
            output = subprocess.check_output(
                ['adb', '-s', self.ip, 'shell', 'su -c', f'"{cmd}"'], 
                timeout=1
            )
            val = float(output.decode('utf-8').strip())
            return val
        except:
            return None # 返回 None 表示读取失败

    def getVoltage(self):
        # 电压单位通常是 uV (10^-6)
        val = self._read_file("voltage_now")
        if val is not None:
            self.voltage = val / 1_000_000 
        return self.voltage

    def getCurrent(self):
        val = self._read_file("current_now")
        if val is not None:
            # 这里的单位是 mA，所以除以 1000
            self.current = abs(val) / 1000.0
        return self.current

    def getPower(self):
        v = self.getVoltage()
        i = self.getCurrent()
        
        # 简单的数据清洗：如果电压或电流为0，说明读取失败，保持上一次的值不更新
        if v > 0 and i >= 0:
            self.power = v * i
        
        return self.power

if __name__ == "__main__":
    IP = "192.168.3.38"
    logger = PowerLogger(IP)
    
    print("功耗监控已启动 (单位: mA -> W)")
    try:
        while True:
            p = logger.getPower()
            print(f"Power: {p:.4f} W | Current: {logger.current:.4f} A")
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass