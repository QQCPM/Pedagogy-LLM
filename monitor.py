"""
GPU monitoring and system resource tracking
"""
import torch
import psutil
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt
import logging
from config import config

logger = logging.getLogger(__name__)

class SystemMonitor:
    """Monitor system resources during training and inference"""
    
    def __init__(self, log_interval: int = 30):
        self.log_interval = log_interval  # seconds
        self.monitoring = False
        self.stats = []
        self.log_file = Path(config.logs_dir) / f"system_monitor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    def get_gpu_stats(self) -> Dict:
        """Get current GPU statistics"""
        if not torch.cuda.is_available():
            return {"gpu_available": False}
        
        gpu_stats = {
            "gpu_available": True,
            "device_count": torch.cuda.device_count(),
            "current_device": torch.cuda.current_device(),
        }
        
        # Get stats for each GPU
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            allocated = torch.cuda.memory_allocated(i) / 1e9
            cached = torch.cuda.memory_reserved(i) / 1e9
            total = props.total_memory / 1e9
            
            gpu_stats[f"gpu_{i}"] = {
                "name": props.name,
                "memory_total_gb": total,
                "memory_allocated_gb": allocated,
                "memory_cached_gb": cached,
                "memory_free_gb": total - cached,
                "utilization_percent": (allocated / total) * 100 if total > 0 else 0,
                "temperature": self._get_gpu_temperature(i)
            }
        
        return gpu_stats
    
    def _get_gpu_temperature(self, device_id: int) -> float:
        """Get GPU temperature (requires nvidia-smi)"""
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits", f"--id={device_id}"],
                capture_output=True, text=True, timeout=5
            )
            return float(result.stdout.strip())
        except:
            return -1  # Temperature not available
    
    def get_cpu_stats(self) -> Dict:
        """Get current CPU statistics"""
        return {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "cpu_count": psutil.cpu_count(),
            "cpu_count_physical": psutil.cpu_count(logical=False),
            "load_average": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None,
        }
    
    def get_memory_stats(self) -> Dict:
        """Get current memory statistics"""
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        return {
            "ram_total_gb": memory.total / 1e9,
            "ram_available_gb": memory.available / 1e9,
            "ram_used_gb": memory.used / 1e9,
            "ram_percent": memory.percent,
            "swap_total_gb": swap.total / 1e9,
            "swap_used_gb": swap.used / 1e9,
            "swap_percent": swap.percent,
        }
    
    def get_disk_stats(self) -> Dict:
        """Get current disk statistics"""
        disk = psutil.disk_usage('/')
        
        return {
            "disk_total_gb": disk.total / 1e9,
            "disk_used_gb": disk.used / 1e9,
            "disk_free_gb": disk.free / 1e9,
            "disk_percent": (disk.used / disk.total) * 100,
        }
    
    def take_snapshot(self) -> Dict:
        """Take a complete system snapshot"""
        snapshot = {
            "timestamp": datetime.now().isoformat(),
            "unix_timestamp": time.time(),
            "gpu": self.get_gpu_stats(),
            "cpu": self.get_cpu_stats(),
            "memory": self.get_memory_stats(),
            "disk": self.get_disk_stats(),
        }
        
        return snapshot
    
    def start_monitoring(self):
        """Start continuous monitoring"""
        self.monitoring = True
        logger.info(f"Started system monitoring (interval: {self.log_interval}s)")
        
        while self.monitoring:
            snapshot = self.take_snapshot()
            self.stats.append(snapshot)
            
            # Log GPU memory if available
            if snapshot["gpu"]["gpu_available"]:
                gpu_0 = snapshot["gpu"]["gpu_0"]
                logger.info(f"GPU Memory: {gpu_0['memory_allocated_gb']:.1f}GB / {gpu_0['memory_total_gb']:.1f}GB "
                           f"({gpu_0['utilization_percent']:.1f}%)")
            
            # Save stats periodically
            if len(self.stats) % 10 == 0:
                self.save_stats()
            
            time.sleep(self.log_interval)
    
    def stop_monitoring(self):
        """Stop monitoring and save final stats"""
        self.monitoring = False
        self.save_stats()
        logger.info("Stopped system monitoring")
    
    def save_stats(self):
        """Save monitoring stats to file"""
        with open(self.log_file, 'w') as f:
            json.dump(self.stats, f, indent=2)
    
    def generate_report(self, output_file: str = None) -> str:
        """Generate monitoring report with plots"""
        if not self.stats:
            logger.warning("No monitoring data available")
            return None
        
        output_file = output_file or f"monitoring_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        output_path = Path(config.output_dir) / output_file
        
        # Extract time series data
        timestamps = [stat["unix_timestamp"] for stat in self.stats]
        start_time = timestamps[0]
        relative_times = [(t - start_time) / 60 for t in timestamps]  # Convert to minutes
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('System Monitoring Report', fontsize=16)
        
        # GPU Memory Usage
        if self.stats[0]["gpu"]["gpu_available"]:
            gpu_memory = [stat["gpu"]["gpu_0"]["memory_allocated_gb"] for stat in self.stats]
            gpu_total = self.stats[0]["gpu"]["gpu_0"]["memory_total_gb"]
            
            axes[0, 0].plot(relative_times, gpu_memory, 'b-', linewidth=2)
            axes[0, 0].axhline(y=gpu_total, color='r', linestyle='--', alpha=0.7, label=f'Total: {gpu_total:.1f}GB')
            axes[0, 0].set_title('GPU Memory Usage')
            axes[0, 0].set_xlabel('Time (minutes)')
            axes[0, 0].set_ylabel('Memory (GB)')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        else:
            axes[0, 0].text(0.5, 0.5, 'No GPU Available', ha='center', va='center', transform=axes[0, 0].transAxes)
            axes[0, 0].set_title('GPU Memory Usage')
        
        # RAM Usage
        ram_used = [stat["memory"]["ram_used_gb"] for stat in self.stats]
        ram_total = self.stats[0]["memory"]["ram_total_gb"]
        
        axes[0, 1].plot(relative_times, ram_used, 'g-', linewidth=2)
        axes[0, 1].axhline(y=ram_total, color='r', linestyle='--', alpha=0.7, label=f'Total: {ram_total:.1f}GB')
        axes[0, 1].set_title('RAM Usage')
        axes[0, 1].set_xlabel('Time (minutes)')
        axes[0, 1].set_ylabel('Memory (GB)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # CPU Usage
        cpu_percent = [stat["cpu"]["cpu_percent"] for stat in self.stats]
        
        axes[1, 0].plot(relative_times, cpu_percent, 'orange', linewidth=2)
        axes[1, 0].set_title('CPU Usage')
        axes[1, 0].set_xlabel('Time (minutes)')
        axes[1, 0].set_ylabel('CPU (%)')
        axes[1, 0].set_ylim(0, 100)
        axes[1, 0].grid(True, alpha=0.3)
        
        # GPU Temperature (if available)
        if self.stats[0]["gpu"]["gpu_available"]:
            gpu_temps = [stat["gpu"]["gpu_0"]["temperature"] for stat in self.stats if stat["gpu"]["gpu_0"]["temperature"] > 0]
            if gpu_temps:
                temp_times = [relative_times[i] for i, stat in enumerate(self.stats) if stat["gpu"]["gpu_0"]["temperature"] > 0]
                axes[1, 1].plot(temp_times, gpu_temps, 'r-', linewidth=2)
                axes[1, 1].set_title('GPU Temperature')
                axes[1, 1].set_xlabel('Time (minutes)')
                axes[1, 1].set_ylabel('Temperature (°C)')
                axes[1, 1].grid(True, alpha=0.3)
            else:
                axes[1, 1].text(0.5, 0.5, 'Temperature data not available', ha='center', va='center', transform=axes[1, 1].transAxes)
        else:
            axes[1, 1].text(0.5, 0.5, 'No GPU Available', ha='center', va='center', transform=axes[1, 1].transAxes)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Monitoring report saved: {output_path}")
        return str(output_path)

def main():
    """Test monitoring functionality"""
    monitor = SystemMonitor(log_interval=5)
    
    print("Taking system snapshot...")
    snapshot = monitor.take_snapshot()
    
    print(f"GPU Available: {snapshot['gpu']['gpu_available']}")
    if snapshot['gpu']['gpu_available']:
        gpu_0 = snapshot['gpu']['gpu_0']
        print(f"GPU: {gpu_0['name']}")
        print(f"GPU Memory: {gpu_0['memory_allocated_gb']:.1f}GB / {gpu_0['memory_total_gb']:.1f}GB")
    
    print(f"RAM: {snapshot['memory']['ram_used_gb']:.1f}GB / {snapshot['memory']['ram_total_gb']:.1f}GB "
          f"({snapshot['memory']['ram_percent']:.1f}%)")
    print(f"CPU: {snapshot['cpu']['cpu_percent']:.1f}%")
    
    print("\nStarting 30-second monitoring test...")
    try:
        monitor.monitoring = True
        for i in range(6):  # 30 seconds of monitoring
            snapshot = monitor.take_snapshot()
            monitor.stats.append(snapshot)
            print(f"Sample {i+1}: RAM {snapshot['memory']['ram_percent']:.1f}%, CPU {snapshot['cpu']['cpu_percent']:.1f}%")
            time.sleep(5)
    except KeyboardInterrupt:
        pass
    finally:
        monitor.stop_monitoring()
        report_path = monitor.generate_report()
        print(f"Report generated: {report_path}")

if __name__ == "__main__":
    main()
