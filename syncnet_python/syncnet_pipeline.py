import json
import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import ffmpeg
import numpy as np
import torch
from scipy import signal
from scipy.interpolate import interp1d
from scenedetect import ContentDetector, SceneManager, StatsManager
from scenedetect.video_manager import VideoManager

try:
    from .detectors.s3fd import S3FD
    from .detectors.s3fd.nets import S3FDNet
    from .SyncNetInstance import SyncNetInstance
    from .SyncNetModel import S
except ImportError:
    # Fallback for direct script execution
    from detectors.s3fd import S3FD
    from detectors.s3fd.nets import S3FDNet
    from SyncNetInstance import SyncNetInstance
    from SyncNetModel import S

# ---------------------------------------------------------------------- #
# Configuration                                                          #
# ---------------------------------------------------------------------- #
@dataclass
class PipelineConfig:
    # Face-detection / tracking
    facedet_scale: float = 0.25
    crop_scale: float = 0.40
    min_track: int = 50
    frame_rate: int = 25
    num_failed_det: int = 25
    min_face_size: int = 100

    # SyncNet
    batch_size: int = 20
    vshift: int = 15

    # Local weight paths
    s3fd_weights: str = "sfd_face.pth"
    syncnet_weights: str = "syncnet_v2.model"

    # Tools
    ffmpeg_bin: str = "ffmpeg"  # assumes ffmpeg in $PATH
    audio_sample_rate: int = 16000  # resample rate for speech

    # Optimization flags
    enable_scene_detection: bool = False  # Disable by default for performance
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.ffmpeg_bin is None:
            self.ffmpeg_bin = "ffmpeg"

    @classmethod
    def from_dict(cls, d: Dict[str, Any]):
        return cls(**{k: v for k, v in d.items() if k in cls.__annotations__})


# ---------------------------------------------------------------------- #
# Pipeline                                                               #
# ---------------------------------------------------------------------- #
class SyncNetPipeline:
    def __init__(
        self,
        cfg: Union[PipelineConfig, Dict[str, Any], None] = None,
        *,
        device: str = "cuda",
        **override,
    ):
        base = cfg if isinstance(cfg, PipelineConfig) else PipelineConfig.from_dict(cfg or {})
        for k, v in override.items():
            if hasattr(base, k):
                setattr(base, k, v)
        self.cfg = base
        self.device = device

        self.s3fd = self._load_s3fd(self.cfg.s3fd_weights)
        self.syncnet = self._load_syncnet(self.cfg.syncnet_weights)

    # ---------------------------- model loading ---------------------------- #
    def _load_s3fd(self, path: str) -> S3FD:
        logging.info(f"Loading S3FD from {path}")
        net = S3FDNet(device=self.device)
        net.load_state_dict(torch.load(path, map_location=self.device))
        net.eval()
        return S3FD(net=net, device=self.device)

    def _load_syncnet(self, path: str) -> SyncNetInstance:
        logging.info(f"Loading SyncNet from {path}")
        model = S()
        model.load_state_dict(torch.load(path, map_location=self.device))
        model.eval()
        return SyncNetInstance(net=model, device=self.device)

    # ---------------------------- helpers ---------------------------------- #
    @staticmethod
    def _iou(a, b):
        xA, yA = max(a[0], b[0]), max(a[1], b[1])
        xB, yB = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0, xB - xA) * max(0, yB - yA)
        areaA = (a[2] - a[0]) * (a[3] - a[1])
        areaB = (b[2] - b[0]) * (b[3] - b[1])
        return inter / (areaA + areaB - inter + 1e-8)

    def _calculate_optimal_batch_size(self, frame_shape, target_memory_gb=4.0):
        """
        Calculate optimal batch size based on available GPU memory

        Args:
            frame_shape: (H, W, C) of detection frames
            target_memory_gb: Max GPU memory to use for batch

        Returns:
            Optimal batch size
        """
        if not torch.cuda.is_available():
            return 32  # Default for CPU

        # Get available GPU memory
        gpu_props = torch.cuda.get_device_properties(0)
        total_memory = gpu_props.total_memory / (1024**3)  # GB
        allocated = torch.cuda.memory_allocated(0) / (1024**3)
        available = total_memory - allocated

        # Estimate memory per frame (input + feature maps)
        h, w, c = frame_shape
        bytes_per_frame = h * w * c * 4  # float32
        # S3FD feature maps are ~5-8x input size
        estimated_per_frame = bytes_per_frame * 8 / (1024**3)  # GB

        # Use 70% of available memory as safety margin
        safe_memory = min(available * 0.7, target_memory_gb)
        optimal_batch = int(safe_memory / estimated_per_frame)

        # Clamp between reasonable bounds
        return max(8, min(optimal_batch, 200))

    def _track(self, dets):
        cfg = self.cfg
        tracks = []
        while True:
            t = []
            for faces in dets:
                for f in faces:
                    if not t:
                        t.append(f)
                        faces.remove(f)
                    elif (
                        f["frame"] - t[-1]["frame"] <= cfg.num_failed_det
                        and self._iou(f["bbox"], t[-1]["bbox"]) > 0.5
                    ):
                        t.append(f)
                        faces.remove(f)
                        continue
                    else:
                        break
            if not t:
                break
            if len(t) > cfg.min_track:
                fr = np.array([d["frame"] for d in t])
                bb = np.array([d["bbox"] for d in t])
                full_f = np.arange(fr[0], fr[-1] + 1)
                bb_i = np.stack([interp1d(fr, bb[:, i])(full_f) for i in range(4)], 1)
                if max(
                    np.mean(bb_i[:, 2] - bb_i[:, 0]),
                    np.mean(bb_i[:, 3] - bb_i[:, 1]),
                ) > cfg.min_face_size:
                    tracks.append({"frame": full_f, "bbox": bb_i})
        return tracks

    def _smooth_track_boxes(self, track):
        """
        Extract and smooth bounding box parameters from track

        Args:
            track: Track dict with 'bbox' array

        Returns:
            Tuple of (s, x, y) smoothed arrays for size, x-center, y-center
        """
        s, x, y = [], [], []
        for b in track["bbox"]:
            s.append(max(b[3] - b[1], b[2] - b[0]) / 2)
            x.append((b[0] + b[2]) / 2)
            y.append((b[1] + b[3]) / 2)
        # Apply median filter for smoothing
        s, x, y = map(lambda v: signal.medfilt(v, 13), (s, x, y))
        return s, x, y

    def _crop(self, track, frames, audio_wav, base):
        cfg = self.cfg
        base.parent.mkdir(parents=True, exist_ok=True)
        tmp_avi = f"{base}t.avi"
        vw = cv2.VideoWriter(tmp_avi, cv2.VideoWriter_fourcc(*"XVID"), cfg.frame_rate, (224, 224))

        # Use helper for smoothing
        s, x, y = self._smooth_track_boxes(track)

        for i, fidx in enumerate(track["frame"]):
            img = cv2.imread(frames[fidx])
            if img is None:
                continue
            bs = s[i]
            cs = cfg.crop_scale
            pad = int(bs * (1 + 2 * cs))
            img_p = cv2.copyMakeBorder(
                img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(110, 110, 110)
            )
            my, mx = y[i] + pad, x[i] + pad
            y1, y2 = int(my - bs), int(my + bs * (1 + 2 * cs))
            x1, x2 = int(mx - bs * (1 + cs)), int(mx + bs * (1 + cs))
            crop = cv2.resize(img_p[y1:y2, x1:x2], (224, 224))
            vw.write(crop)
        vw.release()

        slice_wav = f"{base}.wav"
        ss = track["frame"][0] / cfg.frame_rate
        to = (track["frame"][-1] + 1) / cfg.frame_rate
        
        # Ensure ffmpeg_bin is not None
        ffmpeg_bin = cfg.ffmpeg_bin if cfg.ffmpeg_bin is not None else "ffmpeg"
        
        cmd = [
            ffmpeg_bin, "-y", "-i", str(audio_wav), 
            "-ss", f"{ss:.3f}", "-to", f"{to:.3f}", 
            str(slice_wav)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            logging.error(f"FFmpeg audio slicing failed: {e.stderr}")
            raise RuntimeError(f"FFmpeg audio slicing failed: {e.stderr}")
        except FileNotFoundError:
            logging.error(f"FFmpeg not found at: {ffmpeg_bin}")
            raise RuntimeError(f"FFmpeg not found. Please ensure ffmpeg is installed and in PATH.")

        final_avi = f"{base}.avi"
        
        cmd = [
            ffmpeg_bin, "-y", "-i", str(tmp_avi), "-i", str(slice_wav),
            "-c:v", "copy", "-c:a", "copy", str(final_avi)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            logging.error(f"FFmpeg video/audio merge failed: {e.stderr}")
            raise RuntimeError(f"FFmpeg video/audio merge failed: {e.stderr}")
        except FileNotFoundError:
            logging.error(f"FFmpeg not found at: {ffmpeg_bin}")
            raise RuntimeError(f"FFmpeg not found. Please ensure ffmpeg is installed and in PATH.")
        
        os.remove(tmp_avi)
        return final_avi

    def _crop_streaming(self, track, video_path, audio_wav):
        """
        Create cropped frames in memory without writing intermediate files
        Optimized version that avoids disk I/O

        Args:
            track: Track dict with 'frame' and 'bbox' arrays
            video_path: Path to source video
            audio_wav: Path to audio file

        Returns:
            Dict with 'frames' (list of 224x224 numpy arrays) and 'audio' (numpy array)
        """
        from scipy.io import wavfile

        cfg = self.cfg

        # Smooth bounding boxes using helper
        s, x, y = self._smooth_track_boxes(track)

        # Stream video and crop frames in memory
        cap = cv2.VideoCapture(str(video_path))
        cropped_frames = []

        for i, fidx in enumerate(track["frame"]):
            # Seek to specific frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
            ret, img = cap.read()
            if not ret:
                continue

            # Crop using smoothed bounding box (same logic as _crop)
            bs = s[i]
            cs = cfg.crop_scale
            pad = int(bs * (1 + 2 * cs))
            img_p = cv2.copyMakeBorder(
                img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(110, 110, 110)
            )
            my, mx = y[i] + pad, x[i] + pad
            y1, y2 = int(my - bs), int(my + bs * (1 + 2 * cs))
            x1, x2 = int(mx - bs * (1 + cs)), int(mx + bs * (1 + cs))
            crop = cv2.resize(img_p[y1:y2, x1:x2], (224, 224))

            cropped_frames.append(crop)

        cap.release()

        # Slice audio once in memory (no temp file)
        ss = track["frame"][0] / cfg.frame_rate
        to = (track["frame"][-1] + 1) / cfg.frame_rate

        # Load audio directly into memory
        sample_rate, full_audio = wavfile.read(str(audio_wav))
        start_sample = int(ss * sample_rate)
        end_sample = int(to * sample_rate)
        sliced_audio = full_audio[start_sample:end_sample]

        return {
            'frames': cropped_frames,
            'audio': sliced_audio,
            'sample_rate': sample_rate
        }

    def _detect_faces_streaming(self, video_path):
        """
        Stream video and detect faces without writing frames to disk
        Uses GPU batch processing for improved performance

        Args:
            video_path: Path to input video

        Returns:
            List of detections per frame (same format as file-based method)
        """
        cfg = self.cfg
        cap = cv2.VideoCapture(str(video_path))

        # Get video dimensions for batch size calculation
        ret, first_frame = cap.read()
        if not ret:
            cap.release()
            return []

        # Calculate downscaled dimensions
        h, w = first_frame.shape[:2]
        detection_h = int(h * cfg.facedet_scale)
        detection_w = int(w * cfg.facedet_scale)

        # Calculate optimal batch size based on GPU memory
        batch_size = self._calculate_optimal_batch_size((detection_h, detection_w, 3))
        logging.info(f"Using batch size {batch_size} for face detection")

        # Reset to beginning
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        detections = []
        frame_buffer = []
        frame_indices = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Resize to detection scale
            detection_frame = cv2.resize(
                frame,
                (detection_w, detection_h),
                interpolation=cv2.INTER_LINEAR
            )

            # Convert BGR to RGB
            detection_frame_rgb = cv2.cvtColor(detection_frame, cv2.COLOR_BGR2RGB)

            frame_buffer.append(detection_frame_rgb)
            frame_indices.append(frame_idx)
            frame_idx += 1

            # Process batch when full or at end
            if len(frame_buffer) >= batch_size:
                # Batch detection
                if hasattr(self.s3fd, 'detect_faces_batch'):
                    boxes_batch = self.s3fd.detect_faces_batch(
                        frame_buffer,
                        conf_th=0.9,
                        scale=1.0  # Already downscaled
                    )
                else:
                    # Fallback to sequential detection
                    boxes_batch = [
                        self.s3fd.detect_faces(img, conf_th=0.9, scales=[1.0])
                        for img in frame_buffer
                    ]

                # Scale boxes back to original resolution and store
                for fidx, boxes in zip(frame_indices, boxes_batch):
                    if len(boxes) > 0:
                        boxes_scaled = boxes.copy()
                        boxes_scaled[:, :4] /= cfg.facedet_scale

                        detections.append([
                            {"frame": fidx, "bbox": b[:-1].tolist(), "conf": float(b[-1])}
                            for b in boxes_scaled
                        ])
                    else:
                        detections.append([])

                frame_buffer = []
                frame_indices = []

        # Process remaining frames
        if frame_buffer:
            if hasattr(self.s3fd, 'detect_faces_batch'):
                boxes_batch = self.s3fd.detect_faces_batch(
                    frame_buffer,
                    conf_th=0.9,
                    scale=1.0
                )
            else:
                boxes_batch = [
                    self.s3fd.detect_faces(img, conf_th=0.9, scales=[1.0])
                    for img in frame_buffer
                ]

            for fidx, boxes in zip(frame_indices, boxes_batch):
                if len(boxes) > 0:
                    boxes_scaled = boxes.copy()
                    boxes_scaled[:, :4] /= cfg.facedet_scale

                    detections.append([
                        {"frame": fidx, "bbox": b[:-1].tolist(), "conf": float(b[-1])}
                        for b in boxes_scaled
                    ])
                else:
                    detections.append([])

        cap.release()
        return detections

    # ---------------------------- inference -------------------------------- #
    def inference(
        self,
        video_path: str,  # We do not extract audio from video_path!
        audio_path: str,
        *,
        cache_dir: Optional[str] = None,
    ) -> Tuple[List[int], List[float], List[float], float, float, str, bool]:
        cfg = self.cfg
        work = Path(cache_dir) if cache_dir else Path(tempfile.mkdtemp())
        if cache_dir:
            work.mkdir(parents=True, exist_ok=True)

        try:
            # 1-4) OPTIMIZED: Streaming face detection (replaces AVI conversion, frame extraction, and face detection)
            logging.info("Using optimized streaming pipeline (no disk I/O for frames)")
            detections = self._detect_faces_streaming(video_path)

            # Keep audio resampling (still needed, but fast ~2s)
            audio_wav = work / "speech.wav"
            (
                ffmpeg.input(audio_path)
                .output(str(audio_wav), ac=1, ar=cfg.audio_sample_rate, format="wav")
                .overwrite_output()
                .run()
            )

            flat = [f for fs in detections for f in fs]
            s3fd_json = json.dumps(flat) if flat else ""
            has_face = bool(flat)

            # 5) Scene detection (optional, disabled by default for performance)
            if self.cfg.enable_scene_detection:
                vm = VideoManager([str(avi)])
                sm = SceneManager(StatsManager())
                sm.add_detector(ContentDetector())
                vm.start()
                sm.detect_scenes(frame_source=vm)
                scenes = sm.get_scene_list(vm.get_base_timecode()) or [
                    (vm.get_base_timecode(), vm.get_current_timecode())
                ]
            else:
                # Treat entire video as single scene for faster processing
                # Face tracking will naturally segment at hard cuts via IoU threshold
                # Create simple frame range without using VideoManager
                scenes = None

            # 6) Track faces
            tracks = []
            if self.cfg.enable_scene_detection and scenes:
                for sc in scenes:
                    s, e = sc[0].frame_num, sc[1].frame_num
                    if e - s >= cfg.min_track:
                        tracks.extend(self._track([lst.copy() for lst in detections[s:e]]))
            else:
                # Process entire video as one scene
                if len(detections) >= cfg.min_track:
                    tracks.extend(self._track([lst.copy() for lst in detections]))

            # 7-8) OPTIMIZED: In-memory crop and evaluation (no disk I/O for cropped frames)
            offsets, confs, dists = [], [], []

            for i, track in enumerate(tracks):
                # Get in-memory crop data (no file writes)
                crop_data = self._crop_streaming(track, video_path, audio_wav)

                # Create opt object for SyncNet
                class Opt: ...
                opt = Opt()
                opt.batch_size = cfg.batch_size
                opt.vshift = cfg.vshift

                # Pass directly to SyncNet (no file I/O)
                off, conf, dist = self.syncnet.evaluate_from_memory(
                    frames=crop_data['frames'],
                    audio=crop_data['audio'],
                    sample_rate=crop_data['sample_rate'],
                    opt=opt
                )

                offsets.append(off)
                confs.append(conf)
                dists.append(dist)

            if not offsets:
                return ([], [], [], 0.0, 0.0, "", False)

            return offsets, confs, dists, max(confs), min(dists), s3fd_json, has_face

        finally:
            if not cache_dir:
                shutil.rmtree(work, ignore_errors=True)
