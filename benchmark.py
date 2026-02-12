"""
Benchmarking utilities for comparing model performance.

Measures inference time, throughput, and accuracy metrics.
"""

import time
import torch
import numpy as np
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
import cv2


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    model_name: str
    total_time: float
    avg_time_per_frame: float
    fps: float
    num_frames: int
    memory_mb: Optional[float] = None


class PerformanceBenchmark:
    """Benchmarks model performance for video inference."""
    
    def __init__(self, warmup_frames: int = 5):
        self.warmup_frames = warmup_frames
        self.results: Dict[str, BenchmarkResult] = {}
    
    def benchmark_detector(
        self,
        detector,
        video_path: str,
        model_name: str = "model",
        num_frames: int = 50,
        sample_rate: int = 1
    ) -> BenchmarkResult:
        """
        Benchmark a detector on a video.
        
        Args:
            detector: SemanticEventDetector instance
            video_path: Path to test video
            model_name: Name for this benchmark
            num_frames: Number of frames to process
            sample_rate: Sample every Nth frame
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        frames = []
        while len(frames) < num_frames + self.warmup_frames:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop video
                continue
            frames.append(frame)
        cap.release()
        
        # Warmup
        print(f"Warming up ({self.warmup_frames} frames)...")
        for i in range(self.warmup_frames):
            _ = detector.detect_frame(frames[i])
        
        # Benchmark
        print(f"Benchmarking {model_name} ({num_frames} frames)...")
        times = []
        
        for i in range(num_frames):
            frame = frames[self.warmup_frames + i]
            
            start = time.perf_counter()
            _ = detector.detect_frame(frame)
            end = time.perf_counter()
            
            times.append(end - start)
        
        total_time = sum(times)
        avg_time = np.mean(times)
        fps = 1.0 / avg_time
        
        # Memory usage (if CUDA)
        memory_mb = None
        if torch.cuda.is_available():
            memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        
        result = BenchmarkResult(
            model_name=model_name,
            total_time=total_time,
            avg_time_per_frame=avg_time,
            fps=fps,
            num_frames=num_frames,
            memory_mb=memory_mb
        )
        
        self.results[model_name] = result
        return result
    
    def benchmark_inference_only(
        self,
        inference_fn: Callable,
        test_input: torch.Tensor,
        model_name: str = "model",
        num_iterations: int = 100
    ) -> BenchmarkResult:
        """
        Benchmark raw inference without video I/O.
        
        Args:
            inference_fn: Function that takes input and returns output
            test_input: Input tensor for inference
            model_name: Name for this benchmark
            num_iterations: Number of inference iterations
        """
        # Warmup
        print(f"Warming up ({self.warmup_frames} iterations)...")
        for _ in range(self.warmup_frames):
            with torch.no_grad():
                _ = inference_fn(test_input)
        
        # Benchmark
        print(f"Benchmarking {model_name} ({num_iterations} iterations)...")
        times = []
        
        for _ in range(num_iterations):
            start = time.perf_counter()
            with torch.no_grad():
                _ = inference_fn(test_input)
            end = time.perf_counter()
            times.append(end - start)
        
        total_time = sum(times)
        avg_time = np.mean(times)
        fps = 1.0 / avg_time
        
        result = BenchmarkResult(
            model_name=model_name,
            total_time=total_time,
            avg_time_per_frame=avg_time,
            fps=fps,
            num_frames=num_iterations
        )
        
        self.results[model_name] = result
        return result
    
    def compare_results(self) -> Dict:
        """Compare all benchmark results."""
        if len(self.results) < 2:
            return {"error": "Need at least 2 results to compare"}
        
        baseline_name = list(self.results.keys())[0]
        baseline = self.results[baseline_name]
        
        comparison = {
            "baseline": baseline_name,
            "baseline_fps": baseline.fps,
            "comparisons": {}
        }
        
        for name, result in self.results.items():
            if name == baseline_name:
                continue
            
            speedup = result.fps / baseline.fps
            time_reduction = (1 - result.avg_time_per_frame / baseline.avg_time_per_frame) * 100
            
            comparison["comparisons"][name] = {
                "fps": result.fps,
                "speedup": speedup,
                "time_reduction_pct": time_reduction
            }
        
        return comparison
    
    def generate_comparison_table(self) -> str:
        """Generate a markdown comparison table."""
        if not self.results:
            return "No benchmark results available."
        
        lines = [
            "| Model | Avg Time (ms) | FPS | Speedup |",
            "|-------|---------------|-----|---------|"
        ]
        
        baseline = list(self.results.values())[0]
        
        for name, result in self.results.items():
            speedup = result.fps / baseline.fps
            time_ms = result.avg_time_per_frame * 1000
            lines.append(
                f"| {name} | {time_ms:.2f} | {result.fps:.1f} | {speedup:.2f}x |"
            )
        
        return "\n".join(lines)


def print_benchmark_results(results: Dict[str, BenchmarkResult]):
    """Pretty print benchmark results."""
    print("\n" + "="*60)
    print("PERFORMANCE BENCHMARK RESULTS")
    print("="*60)
    
    for name, result in results.items():
        print(f"\n{name}:")
        print(f"  Frames processed: {result.num_frames}")
        print(f"  Total time: {result.total_time:.2f}s")
        print(f"  Avg time/frame: {result.avg_time_per_frame*1000:.2f}ms")
        print(f"  Throughput: {result.fps:.1f} FPS")
        if result.memory_mb:
            print(f"  Peak memory: {result.memory_mb:.1f} MB")


def print_comparison(comparison: Dict):
    """Pretty print comparison results."""
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON")
    print("="*60)
    
    print(f"\nBaseline: {comparison['baseline']} ({comparison['baseline_fps']:.1f} FPS)")
    
    print("\nOptimized Models:")
    for name, data in comparison['comparisons'].items():
        print(f"\n  {name}:")
        print(f"    FPS: {data['fps']:.1f}")
        print(f"    Speedup: {data['speedup']:.2f}x")
        print(f"    Time reduction: {data['time_reduction_pct']:.1f}%")
