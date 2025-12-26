#!/usr/bin/env python3

import os
import time
import socket
import argparse
import matplotlib.pyplot as plt
import csv
import numpy as np

# 导入你的模块
from SurfaceFlinger.get_fps import SurfaceFlingerFPS
from PowerLogger.powerlogger import PowerLogger
from CPU.cpu import CPU
from GPU.gpu import GPU

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--app', type=str, required=True, choices=['showroom', 'skype', 'call'], help="Application name")
    parser.add_argument('--exp_time', type=int, default=300, help="Steps")
    parser.add_argument('--server_ip', type=str, required=True, help="Agent IP")
    parser.add_argument('--server_port', type=int, default=8702)
    parser.add_argument('--target_fps', type=int, required=True)
    parser.add_argument('--pixel_ip', type=str, required=True)

    args = parser.parse_args()
    
    # 提取参数
    app = args.app
    experiment_time = args.exp_time
    server_ip = args.server_ip
    server_port = args.server_port
    target_fps = args.target_fps
    pixel_ip = args.pixel_ip
    
    t = 0
    ts = []
    fps_data = []

    print(f"Connecting to Agent Server {server_ip}:{server_port}...")
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        client_socket.connect((server_ip, server_port))
        print("Connected!")
    except ConnectionRefusedError:
        print("Error: Could not connect to Agent Server. Please make sure agent.py is running first.")
        exit(1)

    ''' 
    === 初始化三个 CPU 核心对象 ===
    OnePlus Ace3: 
    c0 -> Little (小核)
    c3 -> Big (大核)
    c7 -> Prime (超大核)
    '''
    c0 = CPU(0, ip=pixel_ip, cpu_type='l')
    c3 = CPU(3, ip=pixel_ip, cpu_type='b') 
    c7 = CPU(7, ip=pixel_ip, cpu_type='p') 
    
    g = GPU(ip=pixel_ip)
    pl = PowerLogger(ip=pixel_ip)

    ''' 设置初始频率 (全部最高) '''
    c0.setCPUclock(len(c0.cpu_clock_list)-1)
    c3.setCPUclock(len(c3.cpu_clock_list)-1)
    c7.setCPUclock(len(c7.cpu_clock_list)-1) 
    g.setGPUclock(len(g.gpu_clock_list)-1)

    ''' 启动 FPS 监控 '''
    view = ""
    if app == 'showroom':
        view = "SurfaceView - com.android.chrome/com.google.android.apps.chrome.Main#0"
    elif app == 'skype':
        view = "com.skype.raider/com.skype4life.MainActivity#0"
    elif app == 'call':
        view = "SurfaceView - com.tencent.tmgp.kr.codm/com.tencent.tmgp.cod.CODMainActivity#0"  
    
    print(f"Monitoring FPS for view: {view}")
    sf_fps_driver = SurfaceFlingerFPS(view, ip=pixel_ip)
    sf_fps_driver.start()

    ''' 初始化状态变量 '''
    # 我们用大核的频率索引作为代表
    c_c = len(c3.cpu_clock_list)-1
    g_c = len(g.gpu_clock_list)-1
    
    print("Waiting for sensors to stabilize...")
    time.sleep(2)
    
    # === 功耗读取 (W -> mW) ===
    p_reading = pl.getPower()
    c_p = int(p_reading * 1000) 
    g_p = 0 
    
    # 温度取大核
    c_t = float(c3.getCPUtemp()) 
    g_t = float(g.getGPUtemp())
    
    state = (c_c, g_c, c_p, g_p, c_t, g_t, 60.0)
    
    print("Start Learning Loop...")

    while t < experiment_time:
        # 1. 获取 FPS
        fps = float(sf_fps_driver.getFPS())
        if fps > 60: fps = 60.0
        
        # 2. 采集数据 (包含 c7)
        c0.collectdata()
        c3.collectdata()
        c7.collectdata() 
        g.collectdata()
        
        # 3. 获取功耗
        p_reading = pl.getPower()
        c_p = int(p_reading * 1000)
        
        # 简单的数据清洗：如果读数为0且不是第一帧，沿用上一次的值
        if c_p == 0 and t > 0: 
            c_p = state[2]
        
        c_t = float(c3.getCPUtemp())
        g_t = float(g.getGPUtemp())
        
        # 4. 组装状态
        next_state = (c_c, g_c, c_p, g_p, c_t, g_t, fps)
        
        # 5. 发送数据
        send_msg = f"{c_c},{g_c},{c_p},{g_p},{c_t},{g_t},{fps}"
        try:
            client_socket.send(send_msg.encode())
        except BrokenPipeError:
            print("Server disconnected.")
            break
        
        print(f'[{t}] FPS:{fps:.1f} Pwr:{c_p}mW Temp:{c_t}/{g_t}')
        
        # 更新本地状态
        state = next_state
        fps_data.append(fps)
        ts.append(t)

        # 6. 接收并执行动作
        try:
            recv_msg = client_socket.recv(1024).decode()
            if not recv_msg: break
            
            clk = recv_msg.split(',')
            new_c_c = int(clk[0]) # CPU 频率索引
            new_g_c = int(clk[1]) # GPU 频率索引
            
            # === 执行动作逻辑 ===
            # 同时应用到 c0, c3, c7
            if new_c_c != c_c:
                c0.setCPUclock(new_c_c) 
                c3.setCPUclock(new_c_c)
                c7.setCPUclock(new_c_c) 
                c_c = new_c_c
            
            if new_g_c != g_c:
                g.setGPUclock(new_g_c)
                g_c = new_g_c
                
        except Exception as e:
            print(f"Error: {e}")
            break

        t += 1
        time.sleep(1.0) # 控制采样频率

    sf_fps_driver.stop()
    client_socket.close()

    # Logging results
    if len(pl.power_data) > 0:
        print('Average Total power={} W'.format(sum(pl.power_data)/len(pl.power_data)))
    if len(fps_data) > 0:
        print('Average fps = {} fps'.format(sum(fps_data)/len(fps_data)))

    # === 保存数据 (文件名动态化) ===
    # 使用 app 名字作为文件名，防止覆盖
    file_suffix = f"{app}_zTT.csv"

    with open(f'power_{file_suffix}', 'w', encoding='utf-8', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(pl.power_data[1:])

    with open(f'temp_{file_suffix}', 'w', encoding='utf-8', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(c0.temp_data)
        wr.writerow(c3.temp_data)
        wr.writerow(c7.temp_data) 
        wr.writerow(g.temp_data)

    with open(f'clock_{file_suffix}', 'w', encoding='utf-8', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(c0.clock_data)
        wr.writerow(c3.clock_data)
        wr.writerow(c7.clock_data) 
        wr.writerow(g.clock_data)

    with open(f'fps_{file_suffix}', 'w', encoding='utf-8', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(fps_data)

    print(f"Data saved to *_{file_suffix}")

    # Plot results
    fig = plt.figure(figsize=(12, 14))
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)

    # Power (确保数据长度匹配，防止绘图报错)
    min_len = min(len(ts), len(pl.power_data))
    ax1.set_title('Total Power')
    ax1.plot(ts[:min_len], pl.power_data[:min_len], label='Total(W)')
    ax1.legend()

    # Temp
    min_len = min(len(ts), len(c3.temp_data))
    ax2.set_title('Temperature')
    ax2.plot(ts[:min_len], c0.temp_data[:min_len], label='Little')
    ax2.plot(ts[:min_len], c3.temp_data[:min_len], label='Big')
    ax2.plot(ts[:min_len], c7.temp_data[:min_len], label='Prime') 
    ax2.plot(ts[:min_len], g.temp_data[:min_len], label='GPU')
    ax2.legend()

    # Clock
    min_len = min(len(ts), len(c3.clock_data))
    ax3.set_title('Clock Freq (Hz)')
    ax3.plot(ts[:min_len], c0.clock_data[:min_len], label='Little')
    ax3.plot(ts[:min_len], c3.clock_data[:min_len], label='Big')
    ax3.plot(ts[:min_len], c7.clock_data[:min_len], label='Prime') 
    ax3.plot(ts[:min_len], g.clock_data[:min_len], label='GPU')
    ax3.legend()

    # FPS
    ax4.set_title('FPS')
    ax4.plot(ts, fps_data, label='FPS')
    ax4.axhline(y=target_fps, color='r', linewidth=1)
    ax4.legend()
    
    plt.show()