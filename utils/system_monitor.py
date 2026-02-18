"""
System Health Monitor - Real-time monitoring and diagnostics
"""

import asyncio
import psutil
import json
from typing import Dict, Any, List
from datetime import datetime
import logging

class SystemMonitor:
    """Monitor system health and performance"""
    
    def __init__(self):
        self.logger = logging.getLogger("system_monitor")
        self.metrics_history = []
        
    async def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health status"""
        health = {
            "timestamp": datetime.now().isoformat(),
            "cpu": self._get_cpu_info(),
            "memory": self._get_memory_info(),
            "disk": self._get_disk_info(),
            "network": self._get_network_info(),
            "process": self._get_process_info(),
            "ai_components": await self._get_ai_component_status()
        }
        
        # Store metrics for trend analysis
        self.metrics_history.append(health)
        if len(self.metrics_history) > 100:  # Keep last 100 readings
            self.metrics_history.pop(0)
        
        return health
    
    def _get_cpu_info(self) -> Dict[str, Any]:
        """Get CPU usage and information"""
        try:
            return {
                "usage_percent": psutil.cpu_percent(interval=1),
                "cores": psutil.cpu_count(),
                "logical_cores": psutil.cpu_count(logical=True),
                "frequency": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else {},
                "load_average": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else []
            }
        except Exception as e:
            return {"error": str(e)}
    
    def _get_memory_info(self) -> Dict[str, Any]:
        """Get memory usage information"""
        try:
            virtual_memory = psutil.virtual_memory()
            swap_memory = psutil.swap_memory()
            
            return {
                "virtual": {
                    "total": virtual_memory.total,
                    "available": virtual_memory.available,
                    "used": virtual_memory.used,
                    "percent": virtual_memory.percent,
                    "free": virtual_memory.free
                },
                "swap": {
                    "total": swap_memory.total,
                    "used": swap_memory.used,
                    "percent": swap_memory.percent
                }
            }
        except Exception as e:
            return {"error": str(e)}
    
    def _get_disk_info(self) -> Dict[str, Any]:
        """Get disk usage information"""
        try:
            disk_usage = psutil.disk_usage('/')
            disk_io = psutil.disk_io_counters()
            
            return {
                "usage": {
                    "total": disk_usage.total,
                    "used": disk_usage.used,
                    "free": disk_usage.free,
                    "percent": disk_usage.percent
                },
                "io": disk_io._asdict() if disk_io else {}
            }
        except Exception as e:
            return {"error": str(e)}
    
    def _get_network_info(self) -> Dict[str, Any]:
        """Get network statistics"""
        try:
            net_io = psutil.net_io_counters()
            net_connections = len(psutil.net_connections())
            
            return {
                "io": net_io._asdict() if net_io else {},
                "active_connections": net_connections,
                "interfaces": list(psutil.net_if_addrs().keys())
            }
        except Exception as e:
            return {"error": str(e)}
    
    def _get_process_info(self) -> Dict[str, Any]:
        """Get current process information"""
        try:
            current_process = psutil.Process()
            return {
                "pid": current_process.pid,
                "name": current_process.name(),
                "status": current_process.status(),
                "cpu_percent": current_process.cpu_percent(),
                "memory_percent": current_process.memory_percent(),
                "threads": current_process.num_threads(),
                "open_files": len(current_process.open_files())
            }
        except Exception as e:
            return {"error": str(e)}
    
    async def _get_ai_component_status(self) -> Dict[str, Any]:
        """Get AI component status (stub for now)"""
        return {
            "reasoning_core": "unknown",
            "decision_engine": "unknown", 
            "memory_system": "unknown",
            "tool_manager": "unknown",
            "learning_engine": "unknown"
        }
    
    def get_health_summary(self) -> Dict[str, str]:
        """Get simple health status summary"""
        health = asyncio.run(self.get_system_health())
        
        # Simple health assessment
        issues = []
        
        if "error" not in health["cpu"] and health["cpu"]["usage_percent"] > 80:
            issues.append("High CPU usage")
            
        if "error" not in health["memory"] and health["memory"]["virtual"]["percent"] > 85:
            issues.append("High memory usage")
            
        if "error" not in health["disk"] and health["disk"]["usage"]["percent"] > 90:
            issues.append("Low disk space")
        
        status = "healthy" if not issues else "warning"
        
        return {
            "status": status,
            "issues": issues,
            "timestamp": health["timestamp"]
        }
    
    def get_performance_trends(self) -> Dict[str, Any]:
        """Get performance trends from historical data"""
        if len(self.metrics_history) < 10:
            return {"message": "Insufficient data for trend analysis"}
        
        recent = self.metrics_history[-10:]
        
        trends = {
            "cpu_usage_trend": self._calculate_trend([m["cpu"].get("usage_percent", 0) for m in recent]),
            "memory_usage_trend": self._calculate_trend([m["memory"]["virtual"].get("percent", 0) for m in recent]),
            "response": "trend_analysis_complete"
        }
        
        return trends
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate simple trend direction"""
        if len(values) < 2:
            return "insufficient_data"
        
        recent_avg = sum(values[-3:]) / 3 if len(values) >= 3 else values[-1]
        older_avg = sum(values[:3]) / 3 if len(values) >= 3 else values[0]
        
        diff = recent_avg - older_avg
        
        if diff > 5:
            return "increasing"
        elif diff < -5:
            return "decreasing"
        else:
            return "stable"

# Requirements: pip install psutil