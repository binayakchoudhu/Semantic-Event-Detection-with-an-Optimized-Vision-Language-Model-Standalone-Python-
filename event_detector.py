"""
Semantic Event Detection using CLIP Vision-Language Model.

Detects events like: person walking, vehicle stopping, crowded scene.
"""

import cv2
import torch
import numpy as np
from PIL import Image
from typing import List, Dict, Optional
from transformers import CLIPProcessor, CLIPModel
from dataclasses import dataclass
from tqdm import tqdm


@dataclass
class DetectionResult:
    """Result for a single frame."""
    frame_idx: int
    timestamp: float
    events: Dict[str, float]  # event_name -> confidence
    top_event: str
    top_confidence: float


class SemanticEventDetector:
    """
    Detects semantic events in video using CLIP.
    
    Uses zero-shot classification by comparing frame embeddings
    with text descriptions of target events.
    """
    
    DEFAULT_EVENTS = [
        "a person walking",
        "a vehicle stopping",
        "a crowded scene with many people",
        "an empty scene",
        "a person running",
        "a car moving",
        "people standing still",
    ]
    
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: Optional[str] = None,
        events: Optional[List[str]] = None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.events = events or self.DEFAULT_EVENTS
        
        print(f"Loading CLIP model: {model_name}")
        print(f"Using device: {self.device}")
        
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        self._precompute_text_embeddings()
    
    def _precompute_text_embeddings(self):
        """Pre-compute text embeddings for all target events."""
        with torch.no_grad():
            inputs = self.processor(
                text=self.events,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            text_features = self.model.get_text_features(**inputs)
            # Handle both tensor and object outputs (API compatibility)
            if hasattr(text_features, 'pooler_output'):
                text_features = text_features.pooler_output
            elif hasattr(text_features, 'last_hidden_state'):
                text_features = text_features.last_hidden_state[:, 0, :]
            self.text_embeddings = text_features / text_features.norm(
                dim=-1, keepdim=True
            )
    
    def set_model(self, model):
        """Replace the model (used for optimized models)."""
        self.model = model.to(self.device)
        self.model.eval()
        self._precompute_text_embeddings()
    
    def detect_frame(self, frame: np.ndarray) -> Dict[str, float]:
        """Detect events in a single frame."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        
        with torch.no_grad():
            inputs = self.processor(
                images=pil_image,
                return_tensors="pt"
            ).to(self.device)
            image_features = self.model.get_image_features(**inputs)
            # Handle both tensor and object outputs (API compatibility)
            if hasattr(image_features, 'pooler_output'):
                image_features = image_features.pooler_output
            elif hasattr(image_features, 'last_hidden_state'):
                image_features = image_features.last_hidden_state[:, 0, :]
            image_embedding = image_features / image_features.norm(
                dim=-1, keepdim=True
            )
            
            similarities = (image_embedding @ self.text_embeddings.T).squeeze(0)
            probs = torch.softmax(similarities * 100, dim=0)
            
        return {
            event: prob.item() 
            for event, prob in zip(self.events, probs)
        }
    
    def process_video(
        self,
        video_path: str,
        sample_rate: int = 1,
        confidence_threshold: float = 0.3,
        max_frames: Optional[int] = None
    ) -> List[DetectionResult]:
        """Process a video file and detect events."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if max_frames:
            total_frames = min(total_frames, max_frames * sample_rate)
        
        results = []
        frame_idx = 0
        processed = 0
        
        pbar = tqdm(total=total_frames // sample_rate, desc="Processing video")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if max_frames and processed >= max_frames:
                break
            
            if frame_idx % sample_rate == 0:
                timestamp = frame_idx / fps
                events = self.detect_frame(frame)
                
                top_event = max(events, key=events.get)
                top_confidence = events[top_event]
                
                filtered_events = {
                    k: v for k, v in events.items() 
                    if v >= confidence_threshold
                }
                
                results.append(DetectionResult(
                    frame_idx=frame_idx,
                    timestamp=timestamp,
                    events=filtered_events,
                    top_event=top_event,
                    top_confidence=top_confidence
                ))
                
                processed += 1
                pbar.update(1)
            
            frame_idx += 1
        
        pbar.close()
        cap.release()
        
        return results
    
    def summarize_results(self, results: List[DetectionResult]) -> Dict:
        """Summarize detection results."""
        if not results:
            return {"error": "No results to summarize"}
        
        event_counts = {}
        event_confidences = {}
        
        for result in results:
            event = result.top_event
            event_counts[event] = event_counts.get(event, 0) + 1
            
            if event not in event_confidences:
                event_confidences[event] = []
            event_confidences[event].append(result.top_confidence)
        
        avg_confidences = {
            event: np.mean(confs) 
            for event, confs in event_confidences.items()
        }
        
        return {
            "total_frames": len(results),
            "event_counts": event_counts,
            "average_confidences": avg_confidences,
            "dominant_event": max(event_counts, key=event_counts.get),
            "timeline": [
                {
                    "timestamp": r.timestamp,
                    "event": r.top_event,
                    "confidence": r.top_confidence
                }
                for r in results
            ]
        }


def print_results(results: List[DetectionResult], summary: Dict):
    """Pretty print detection results."""
    print("\n" + "="*60)
    print("SEMANTIC EVENT DETECTION RESULTS")
    print("="*60)
    
    print(f"\nTotal frames analyzed: {summary['total_frames']}")
    print(f"Dominant event: {summary['dominant_event']}")
    
    print("\nEvent Distribution:")
    print("-"*40)
    for event, count in sorted(
        summary['event_counts'].items(), 
        key=lambda x: -x[1]
    ):
        pct = count / summary['total_frames'] * 100
        avg_conf = summary['average_confidences'][event]
        print(f"  {event}: {count} frames ({pct:.1f}%) | avg conf: {avg_conf:.3f}")
    
    print("\nTimeline (first 10 detections):")
    print("-"*40)
    for item in summary['timeline'][:10]:
        print(f"  {item['timestamp']:.2f}s: {item['event']} ({item['confidence']:.3f})")
