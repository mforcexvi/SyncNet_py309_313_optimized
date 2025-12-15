import logging
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .box_utils import nms_
from .nets import S3FDNet

img_mean = np.array([104.0, 117.0, 123.0])[:, np.newaxis, np.newaxis].astype("float32")


class S3FD:
    def __init__(self, net: S3FDNet, device="cuda"):
        """
        We now accept an *already-initialized* S3FDNet as `net`,
        instead of loading weights here.
        """
        tstamp = time.time()
        self.device = device
        self.net = net.to(self.device)
        self.net.eval()
        logging.info(
            f"[S3FD] S3FDNet instance is ready (initialized in {time.time()-tstamp:.4f} sec)."
        )

    def detect_faces(self, image, conf_th=0.8, scales=[1]):
        """
        Same detection code as before, but we no longer load the model here.
        """
        self.net.to(self.device)
        self.net.eval()
        w, h = image.shape[1], image.shape[0]
        bboxes = np.empty(shape=(0, 5))

        with torch.no_grad():
            for s in scales:
                scaled_img = cv2.resize(
                    image, dsize=(0, 0), fx=s, fy=s, interpolation=cv2.INTER_LINEAR
                )
                scaled_img = np.swapaxes(scaled_img, 1, 2)
                scaled_img = np.swapaxes(scaled_img, 1, 0)
                scaled_img = scaled_img[[2, 1, 0], :, :]
                scaled_img = scaled_img.astype("float32")
                scaled_img -= img_mean
                scaled_img = scaled_img[[2, 1, 0], :, :]
                x = torch.from_numpy(scaled_img).unsqueeze(0).to(self.device)

                y = self.net(x)  # forward pass
                detections = y.data.to(self.device)
                scale_tensor = torch.Tensor([w, h, w, h]).to(self.device)

                for i in range(detections.size(1)):
                    j = 0
                    while detections[0, i, j, 0] > conf_th:
                        score = detections[0, i, j, 0].item()
                        pt = (detections[0, i, j, 1:] * scale_tensor).cpu().numpy()
                        bbox = (pt[0], pt[1], pt[2], pt[3], score)
                        bboxes = np.vstack((bboxes, bbox))
                        j += 1

            # NMS, etc. (unchanged)
            keep = nms_(bboxes, 0.1)
            bboxes = bboxes[keep]
        return bboxes

    def detect_faces_batch(self, images_batch, conf_th=0.8, scale=1.0):
        """
        Batch face detection on GPU for improved performance

        Args:
            images_batch: List of numpy arrays [H, W, 3] in RGB
            conf_th: Confidence threshold
            scale: Image scale (1.0 if already downscaled)

        Returns:
            List of detection arrays [N, 5] for each image
        """
        self.net.to(self.device)
        self.net.eval()

        if not images_batch:
            return []

        original_shapes = []

        with torch.no_grad():
            # Batch preprocessing - process on CPU first to save GPU memory
            batch_tensors = []

            for img in images_batch:
                original_shapes.append((img.shape[1], img.shape[0]))  # (w, h)

                # Do ALL preprocessing in numpy (matching original detect_faces exactly)
                # CRITICAL: Original ALWAYS does cv2.resize(), even at scale=1.0
                # This normalizes array format (C-contiguous layout) for torch.from_numpy()
                if scale != 1.0:
                    new_h = int(img.shape[0] * scale)
                    new_w = int(img.shape[1] * scale)
                    processed = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                else:
                    # Even at scale 1.0, do resize to normalize array format (matches line 40-42 of detect_faces)
                    processed = cv2.resize(img, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)

                # Match original preprocessing exactly (lines 43-48 of detect_faces method)
                processed = np.swapaxes(processed, 1, 2)
                processed = np.swapaxes(processed, 1, 0)
                processed = processed[[2, 1, 0], :, :]  # RGB to BGR (numpy indexing)
                processed = processed.astype("float32")
                processed -= img_mean  # Subtract mean (numpy operation)
                processed = processed[[2, 1, 0], :, :]  # BGR back to RGB (numpy indexing)

                # NOW convert to torch (matching line 49 of detect_faces - no ascontiguousarray!)
                img_t = torch.from_numpy(processed)
                batch_tensors.append(img_t)

            # Stack on CPU then move to GPU (more memory efficient)
            x_batch = torch.stack(batch_tensors)
            del batch_tensors  # Free CPU memory

            # Move to GPU just before inference
            x_batch = x_batch.to(self.device)

            # Single forward pass for entire batch
            y = self.net(x_batch)

            # Free input batch memory immediately
            del x_batch

            # Extract detections for each image in batch
            all_detections = []
            detections_batch = y.data

            for i in range(len(images_batch)):
                w, h = original_shapes[i]
                scale_tensor = torch.Tensor([w, h, w, h]).to(self.device)
                bboxes = np.empty(shape=(0, 5))

                # Extract detections for this image
                for det_idx in range(detections_batch.size(1)):
                    j = 0
                    while j < detections_batch.size(2) and detections_batch[i, det_idx, j, 0] > conf_th:
                        score = detections_batch[i, det_idx, j, 0].item()
                        pt = (detections_batch[i, det_idx, j, 1:] * scale_tensor).cpu().numpy()
                        bbox = (pt[0], pt[1], pt[2], pt[3], score)
                        bboxes = np.vstack((bboxes, bbox))
                        j += 1

                # Apply NMS
                if len(bboxes) > 0:
                    keep = nms_(bboxes, 0.1)
                    bboxes = bboxes[keep]

                all_detections.append(bboxes)

            # Free detection results from GPU
            del detections_batch, y

        return all_detections
