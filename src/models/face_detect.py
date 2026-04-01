"""
Face detection using Apple's Vision framework.

Runs on the Neural Engine (ANE) for hardware acceleration.
"""

import io
import logging
from typing import Optional

import numpy as np
from PIL import Image
from Foundation import NSData, NSAutoreleasePool
import Vision

logger = logging.getLogger(__name__)


def detect_faces(image_bytes: bytes) -> tuple[list[dict], int, int]:
    """
    Detect faces using Apple's Vision framework.
    
    Args:
        image_bytes: Raw image data (JPEG, PNG, etc.)
        
    Returns:
        Tuple of (faces, image_width, image_height)
        Each face dict contains:
          - boundingBox: {x1, y1, x2, y2} in pixels
          - score: confidence score
          - landmarks: 5-point landmarks for alignment (if available)
    """
    try:
        pil_image = Image.open(io.BytesIO(image_bytes))
        img_width, img_height = pil_image.size
    except Exception as e:
        logger.error(f"Failed to load image: {e}")
        raise ValueError(f"Invalid image data: {e}") from e
    
    # Use autorelease pool to prevent memory accumulation in long-running service
    pool = NSAutoreleasePool.alloc().init()
    try:
        return _detect_faces_impl(image_bytes, img_width, img_height)
    finally:
        del pool


def _detect_faces_impl(
    image_bytes: bytes,
    img_width: int,
    img_height: int
) -> tuple[list[dict], int, int]:
    """Internal face detection implementation (assumes autorelease pool is active)."""
    try:
        from src.gpu_lock import metal_lock

        ns_data = NSData.dataWithBytes_length_(image_bytes, len(image_bytes))
        handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(ns_data, None)
        request = Vision.VNDetectFaceLandmarksRequest.alloc().init()

        with metal_lock:
            success, error = handler.performRequests_error_([request], None)
        
        if not success or error:
            logger.error(f"Vision framework error: {error}")
            return [], img_width, img_height
        
        faces = []
        results = request.results() or []
        
        for observation in results:
            bbox = observation.boundingBox()
            
            # Convert to pixel coordinates (flip Y axis - Vision uses bottom-left origin)
            x1 = bbox.origin.x * img_width
            y1 = (1.0 - bbox.origin.y - bbox.size.height) * img_height
            x2 = (bbox.origin.x + bbox.size.width) * img_width
            y2 = (1.0 - bbox.origin.y) * img_height
            
            face_data = {
                "boundingBox": {
                    "x1": int(x1),
                    "y1": int(y1),
                    "x2": int(x2),
                    "y2": int(y2)
                },
                "score": float(observation.confidence()),
            }
            
            landmarks = observation.landmarks()
            if landmarks:
                five_points = extract_five_point_landmarks(
                    landmarks, bbox, img_width, img_height
                )
                if five_points is not None:
                    face_data["landmarks"] = five_points
            
            faces.append(face_data)
        
        logger.debug(f"Detected {len(faces)} face(s) in {img_width}x{img_height} image")
        return faces, img_width, img_height
        
    except Exception as e:
        logger.error(f"Face detection failed: {e}", exc_info=True)
        return [], img_width, img_height


def extract_five_point_landmarks(
    landmarks: "Vision.VNFaceLandmarks2D",
    face_bbox,
    img_width: int,
    img_height: int
) -> Optional[list[list[float]]]:
    """
    Extract 5 landmark points for ArcFace alignment:
    - Left eye center
    - Right eye center
    - Nose tip
    - Left mouth corner
    - Right mouth corner

    IMPORTANT: Vision framework's normalizedPoints are in the coordinate space
    of the face's bounding box (0..1 within the bbox), NOT the full image.
    We must map them through the face bounding box to get image pixel coords.
    See: VNImagePointForFaceLandmarkPoint() in VNUtils.h

    Args:
        landmarks: VNFaceLandmarks2D from the face observation
        face_bbox: The face observation's boundingBox() (normalized, bottom-left origin)
        img_width: Full image width in pixels
        img_height: Full image height in pixels

    Returns list of [x, y] points in pixel coordinates, or None if not available.
    """
    # Extract the face bounding box in normalized coords (bottom-left origin)
    bbox_x = face_bbox.origin.x
    bbox_y = face_bbox.origin.y
    bbox_w = face_bbox.size.width
    bbox_h = face_bbox.size.height

    def landmark_to_image_coords(norm_x: float, norm_y: float) -> list[float]:
        """
        Convert a landmark point from face-bbox-relative normalized coords
        to full image pixel coords.

        normalizedPoints are in face bbox space (0..1), with bottom-left origin.
        Face bbox is in image normalized space (0..1), with bottom-left origin.
        Final image pixels use top-left origin.
        """
        # Map from face-bbox-relative to image-normalized coords
        img_norm_x = bbox_x + norm_x * bbox_w
        img_norm_y = bbox_y + norm_y * bbox_h

        # Convert to pixel coords (flip Y: Vision uses bottom-left origin)
        px_x = img_norm_x * img_width
        px_y = (1.0 - img_norm_y) * img_height

        return [px_x, px_y]

    def get_region_points(region) -> list:
        """Convert PyObjC varlist to Python list of points."""
        if region is None:
            return []
        point_count = region.pointCount()
        if point_count == 0:
            return []
        raw_points = region.normalizedPoints()
        return [raw_points[i] for i in range(point_count)]

    def get_region_center(region) -> Optional[list[float]]:
        """Get center point of a landmark region in image pixel coordinates."""
        points = get_region_points(region)
        if not points:
            return None

        x_sum = sum(p.x for p in points)
        y_sum = sum(p.y for p in points)
        n = len(points)

        return landmark_to_image_coords(x_sum / n, y_sum / n)

    try:
        # Left eye center
        left_eye = get_region_center(landmarks.leftEye())

        # Right eye center
        right_eye = get_region_center(landmarks.rightEye())

        # Nose tip - use last point of nose region
        nose = None
        nose_points = get_region_points(landmarks.nose())
        if nose_points:
            p = nose_points[-1]
            nose = landmark_to_image_coords(p.x, p.y)

        # Mouth corners - find leftmost and rightmost points by x-coordinate
        # (Vision framework doesn't guarantee point ordering in contours)
        left_mouth = None
        right_mouth = None
        outer_lips_points = get_region_points(landmarks.outerLips())
        if outer_lips_points:
            # Convert all points to image pixel coordinates
            lips_px = [
                landmark_to_image_coords(p.x, p.y)
                for p in outer_lips_points
            ]
            # Find extremes by x-coordinate
            left_mouth = min(lips_px, key=lambda p: p[0])
            right_mouth = max(lips_px, key=lambda p: p[0])

        # All 5 points must be present
        if all([left_eye, right_eye, nose, left_mouth, right_mouth]):
            return [left_eye, right_eye, nose, left_mouth, right_mouth]

        logger.debug("Could not extract all 5 landmark points")
        return None

    except Exception as e:
        logger.warning(f"Landmark extraction failed: {e}")
        return None


if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
    
    if len(sys.argv) < 2:
        logger.info("Usage: python -m src.models.face_detect <image_path>")
        logger.info("Creating test with a blank image...")
        
        test_img = Image.new("RGB", (640, 480), color=(200, 180, 170))
        buffer = io.BytesIO()
        test_img.save(buffer, format="JPEG")
        test_bytes = buffer.getvalue()
    else:
        with open(sys.argv[1], "rb") as f:
            test_bytes = f.read()
    
    logger.info("Testing Vision framework face detection...")
    faces, width, height = detect_faces(test_bytes)
    
    logger.info(f"Image size: {width}x{height}")
    logger.info(f"Faces detected: {len(faces)}")
    
    for i, face in enumerate(faces):
        logger.info(f"\nFace {i + 1}:")
        logger.info(f"  Bounding box: {face['boundingBox']}")
        logger.info(f"  Score: {face['score']:.3f}")
        if "landmarks" in face:
            logger.info(f"  Landmarks (5-point): ✓")
        else:
            logger.info(f"  Landmarks: not available")
    
    logger.info("\n✅ Face detection test complete!")