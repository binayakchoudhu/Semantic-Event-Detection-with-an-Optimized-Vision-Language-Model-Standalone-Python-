#!/usr/bin/env python3
"""
Semantic Event Detection Pipeline - Main Entry Point

Usage:
    python main.py detect <video_path> [--sample-rate N] [--max-frames N]
    python main.py optimize [--output-dir PATH] [--prune-amount FLOAT]
    python main.py benchmark <video_path> [--num-frames N]
    python main.py full <video_path> [--output-dir PATH]
"""

import argparse
import sys
import os
import json

from event_detector import SemanticEventDetector, print_results
from model_optimizer import ModelOptimizer, print_optimization_stats
from benchmark import PerformanceBenchmark, print_benchmark_results, print_comparison


def detect_events(args):
    """Run event detection on a video."""
    print(f"\n{'='*60}")
    print("SEMANTIC EVENT DETECTION")
    print(f"{'='*60}")
    print(f"Video: {args.video}")
    print(f"Sample rate: every {args.sample_rate} frames")
    print(f"Max frames: {args.max_frames or 'all'}")
    
    detector = SemanticEventDetector()
    results = detector.process_video(
        args.video,
        sample_rate=args.sample_rate,
        max_frames=args.max_frames
    )
    summary = detector.summarize_results(results)
    print_results(results, summary)
    
    # Save results to JSON
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nResults saved to: {args.output}")
    
    return results, summary


def optimize_models(args):
    """Optimize the model using quantization and pruning."""
    print(f"\n{'='*60}")
    print("MODEL OPTIMIZATION")
    print(f"{'='*60}")
    
    optimizer = ModelOptimizer()
    
    # Load and analyze original
    optimizer.load_model()
    
    # Apply optimizations
    optimizer.quantize_dynamic()
    optimizer.prune_model(amount=args.prune_amount)
    
    # Get stats
    stats = optimizer.get_optimization_stats()
    print_optimization_stats(stats)
    
    # Save models
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Save pruned model (quantized models need special handling)
    pruned_path = optimizer.save_optimized_model(
        optimizer.pruned_model,
        output_dir,
        f"pruned_{int(args.prune_amount*100)}pct"
    )
    
    # Save stats
    stats_path = os.path.join(output_dir, "optimization_stats.json")
    
    # Convert stats for JSON serialization
    json_stats = {}
    for key, value in stats.items():
        json_stats[key] = {k: float(v) if isinstance(v, (int, float)) else v 
                          for k, v in value.items()}
    
    with open(stats_path, 'w') as f:
        json.dump(json_stats, f, indent=2)
    print(f"\nStats saved to: {stats_path}")
    
    return optimizer, stats


def run_benchmark(args):
    """Benchmark original vs optimized models."""
    print(f"\n{'='*60}")
    print("PERFORMANCE BENCHMARK")
    print(f"{'='*60}")
    print(f"Video: {args.video}")
    print(f"Frames: {args.num_frames}")
    
    benchmark = PerformanceBenchmark(warmup_frames=5)
    
    # Benchmark original model
    print("\n--- Original Model ---")
    detector_original = SemanticEventDetector()
    benchmark.benchmark_detector(
        detector_original,
        args.video,
        model_name="Original (FP32)",
        num_frames=args.num_frames
    )
    
    # Create and benchmark pruned model
    print("\n--- Pruned Model (30%) ---")
    optimizer = ModelOptimizer()
    optimizer.load_model()
    pruned_model = optimizer.prune_model(amount=0.3)
    
    detector_pruned = SemanticEventDetector()
    detector_pruned.set_model(pruned_model)
    benchmark.benchmark_detector(
        detector_pruned,
        args.video,
        model_name="Pruned (30%)",
        num_frames=args.num_frames
    )
    
    # Print results
    print_benchmark_results(benchmark.results)
    comparison = benchmark.compare_results()
    print_comparison(comparison)
    
    # Print comparison table
    print("\n" + "="*60)
    print("COMPARISON TABLE")
    print("="*60)
    print(benchmark.generate_comparison_table())
    
    return benchmark


def run_full_pipeline(args):
    """Run complete pipeline: optimize, benchmark, detect."""
    print("\n" + "="*60)
    print("FULL PIPELINE EXECUTION")
    print("="*60)
    
    # Step 1: Optimize
    print("\n[1/3] Optimizing models...")
    args.prune_amount = 0.3
    optimizer, stats = optimize_models(args)
    
    # Step 2: Benchmark
    print("\n[2/3] Running benchmarks...")
    args.num_frames = 30
    benchmark = run_benchmark(args)
    
    # Step 3: Detect with optimized model
    print("\n[3/3] Running detection with pruned model...")
    detector = SemanticEventDetector()
    detector.set_model(optimizer.pruned_model)
    
    results = detector.process_video(
        args.video,
        sample_rate=args.sample_rate,
        max_frames=args.max_frames or 50
    )
    summary = detector.summarize_results(results)
    print_results(results, summary)
    
    # Save comprehensive results
    output_path = os.path.join(args.output_dir, "full_results.json")
    full_results = {
        "optimization": {k: {kk: float(vv) if isinstance(vv, (int, float)) else vv 
                            for kk, vv in v.items()} 
                        for k, v in stats.items()},
        "benchmark": {
            name: {
                "fps": result.fps,
                "avg_time_ms": result.avg_time_per_frame * 1000
            }
            for name, result in benchmark.results.items()
        },
        "detection_summary": {
            "total_frames": summary["total_frames"],
            "dominant_event": summary["dominant_event"],
            "event_counts": summary["event_counts"]
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(full_results, f, indent=2)
    print(f"\nFull results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Semantic Event Detection with Optimized Vision-Language Model",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Detect command
    detect_parser = subparsers.add_parser('detect', help='Detect events in video')
    detect_parser.add_argument('video', help='Path to video file')
    detect_parser.add_argument('--sample-rate', type=int, default=5,
                               help='Process every Nth frame (default: 5)')
    detect_parser.add_argument('--max-frames', type=int, default=None,
                               help='Maximum frames to process')
    detect_parser.add_argument('--output', '-o', type=str, default=None,
                               help='Output JSON file for results')
    
    # Optimize command
    optimize_parser = subparsers.add_parser('optimize', help='Optimize the model')
    optimize_parser.add_argument('--output-dir', type=str, default='./optimized_models',
                                 help='Directory to save optimized models')
    optimize_parser.add_argument('--prune-amount', type=float, default=0.3,
                                 help='Fraction of weights to prune (default: 0.3)')
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser('benchmark', help='Benchmark models')
    benchmark_parser.add_argument('video', help='Path to video file')
    benchmark_parser.add_argument('--num-frames', type=int, default=50,
                                  help='Number of frames to benchmark (default: 50)')
    
    # Full pipeline command
    full_parser = subparsers.add_parser('full', help='Run full pipeline')
    full_parser.add_argument('video', help='Path to video file')
    full_parser.add_argument('--output-dir', type=str, default='./optimized_models',
                             help='Directory for outputs')
    full_parser.add_argument('--sample-rate', type=int, default=5,
                             help='Process every Nth frame')
    full_parser.add_argument('--max-frames', type=int, default=None,
                             help='Maximum frames for detection')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    if args.command == 'detect':
        detect_events(args)
    elif args.command == 'optimize':
        optimize_models(args)
    elif args.command == 'benchmark':
        run_benchmark(args)
    elif args.command == 'full':
        run_full_pipeline(args)


if __name__ == "__main__":
    main()
