import cv2
import numpy as np
from paddleocr import PaddleOCR
from gtts import gTTS
import pygame
import tempfile
import os
import time

# -------------------------------
# Initialize TTS and OCR
# -------------------------------
# Initialize audio safely (handle missing/locked audio devices)
try:
    pygame.mixer.init()
    AUDIO_ENABLED = True
except Exception as _audio_init_err:
    print("⚠️ Audio not available, continuing without speech:", _audio_init_err)
    AUDIO_ENABLED = False

# Initialize OCR with updated parameters
ocr = PaddleOCR(lang='en', use_textline_orientation=True, text_det_limit_side_len=640)

# -------------------------------
# Setup webcam
# -------------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Could not open webcam.")
    exit()

print("🚀 PaddleOCR Live started. Press 'q' to exit.\n")

last_text = ""
last_spoken = 0
COOLDOWN = 5  # seconds between readings
CONFIDENCE_THRESHOLD = 0.9  # Only speak text with confidence > 90%

# Frame skipping for better FPS
frame_count = 0
OCR_SKIP_FRAMES = 2  # Process OCR every 3rd frame (0, 1, 2, then reset)

# Store last detected text and boxes for display on skipped frames
last_text_data = []  # List of (text, score, box_coords) tuples
last_text_list = []
last_high_confidence_text_list = []

# -------------------------------
# Main loop
# -------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize for faster inference
    frame = cv2.resize(frame, (640, 480))

    # Run OCR only every Nth frame to increase FPS
    result = None
    if frame_count % (OCR_SKIP_FRAMES + 1) == 0:
        result = ocr.ocr(frame)
    frame_count += 1

    text_list = []  # All detected text (for display)
    high_confidence_text_list = []  # Only high-confidence text (for speech)
    current_text_data = []  # Store text, score, and boxes for this frame
    all_text = ""
    
    # Handle different result formats
    # New format: [OCRResult, ...] where OCRResult has rec_texts, rec_scores, rec_polys
    # Old format: [[[box], (text, score)], ...]
    if result:
        # Check if result[0] is OCRResult object (new format)
        if isinstance(result, list) and len(result) > 0:
            ocr_result = result[0]
            
            # Check if it's an OCRResult object (supports dictionary-like access)
            if hasattr(ocr_result, '__getitem__') or hasattr(ocr_result, 'get'):
                # New OCRResult format - extract using dictionary access
                try:
                    # Try dictionary access first (OCRResult supports dict-like access)
                    if hasattr(ocr_result, 'get'):
                        rec_texts = ocr_result.get('rec_texts', [])
                        rec_scores = ocr_result.get('rec_scores', [])
                        rec_polys = ocr_result.get('rec_polys', ocr_result.get('dt_polys', []))
                    else:
                        rec_texts = ocr_result['rec_texts']
                        rec_scores = ocr_result['rec_scores']
                        rec_polys = ocr_result.get('rec_polys', ocr_result.get('dt_polys', [])) if 'rec_polys' in ocr_result else ocr_result.get('dt_polys', [])
                except (KeyError, TypeError, AttributeError):
                    # Try attribute access as fallback
                    try:
                        rec_texts = getattr(ocr_result, 'rec_texts', [])
                        rec_scores = getattr(ocr_result, 'rec_scores', [])
                        rec_polys = getattr(ocr_result, 'rec_polys', getattr(ocr_result, 'dt_polys', []))
                    except AttributeError:
                        rec_texts = []
                        rec_scores = []
                        rec_polys = []
                
                # Ensure lists
                if not isinstance(rec_texts, list):
                    rec_texts = [rec_texts] if rec_texts else []
                if not isinstance(rec_scores, list):
                    rec_scores = [rec_scores] if rec_scores else [1.0] * len(rec_texts)
                if not isinstance(rec_polys, list):
                    rec_polys = [rec_polys] if rec_polys else []
                
                # Extract text and draw boxes
                for i, text in enumerate(rec_texts):
                    if text and str(text).strip():
                        text = str(text).strip()
                        score = rec_scores[i] if i < len(rec_scores) else 1.0
                        
                        # Add to display list (all text)
                        text_list.append(text)
                        
                        # Add to speech list only if confidence > threshold
                        if score > CONFIDENCE_THRESHOLD:
                            high_confidence_text_list.append(text)
                        
                        box = rec_polys[i] if i < len(rec_polys) else None
                        
                        if box is not None:
                            try:
                                # Convert box to numpy array
                                box_array = np.array(box, dtype=np.float32)
                                if box_array.ndim == 2 and box_array.shape[1] == 2:
                                    box_coords = box_array.astype(int)
                                else:
                                    box_coords = box_array.reshape(-1, 2).astype(int)
                                
                                if box_coords.shape[0] >= 4:
                                    # Store text data for this frame
                                    current_text_data.append((text, score, box_coords))
                                    # Color: green for high confidence, yellow for low
                                    box_color = (0, 255, 0) if score > CONFIDENCE_THRESHOLD else (0, 255, 255)
                                    cv2.polylines(frame, [box_coords], True, box_color, 2)
                                    top_left = box_coords[0]
                                    cv2.putText(frame, f"{text} ({score:.2f})",
                                                (int(top_left[0]), int(top_left[1]) - 10),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                            except Exception as e:
                                # Skip invalid box
                                pass
            # Old format: list of detections
            elif isinstance(ocr_result, list) and ocr_result is not None:
                for line in ocr_result:
                    try:
                        # Validate line structure
                        if not isinstance(line, (list, tuple)) or len(line) < 2:
                            continue
                        
                        box = line[0]
                        text_info = line[1]
                        
                        # Extract box coordinates - ensure they're numeric
                        if isinstance(box, (list, tuple)) and len(box) >= 4:
                            # Validate box contains coordinates (not text)
                            try:
                                # Convert box to numpy array and ensure it's numeric
                                box_array = np.array(box, dtype=np.float32)
                                # Check if conversion worked (all values should be numeric)
                                if box_array.size >= 4:
                                    box_coords = box_array.reshape(-1, 2).astype(int)
                                else:
                                    continue  # Skip invalid box
                            except (ValueError, TypeError):
                                continue  # Skip if box is not numeric
                        else:
                            continue  # Skip invalid box structure
                        
                        # Handle both old format (text, score) and new format (just text)
                        if isinstance(text_info, tuple) and len(text_info) >= 2:
                            text, score = text_info[0], text_info[1]
                        elif isinstance(text_info, tuple) and len(text_info) == 1:
                            text, score = text_info[0], 1.0
                        elif isinstance(text_info, str):
                            text, score = text_info, 1.0
                        else:
                            text, score = str(text_info), 1.0
                        
                        # Add to display list (all text)
                        text_list.append(text)
                        
                        # Add to speech list only if confidence > threshold
                        if score > CONFIDENCE_THRESHOLD:
                            high_confidence_text_list.append(text)
                        
                        # Draw boxes + labels (only if we have valid coordinates)
                        if box_coords.shape[0] >= 4:
                            # Store text data for this frame
                            current_text_data.append((text, score, box_coords))
                            # Color: green for high confidence, yellow for low
                            box_color = (0, 255, 0) if score > CONFIDENCE_THRESHOLD else (0, 255, 255)
                            cv2.polylines(frame, [box_coords], True, box_color, 2)
                            # Get top-left corner for text placement
                            top_left = box_coords[0]
                            cv2.putText(frame, f"{text} ({score:.2f})",
                                        (int(top_left[0]), int(top_left[1]) - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    except Exception as e:
                        # Skip this line if there's any error
                        continue
        
        # Update last detected data if we have new results (even if empty)
        if result is not None:
            last_text_data = current_text_data.copy()
            last_text_list = text_list.copy()
            last_high_confidence_text_list = high_confidence_text_list.copy()
        
        # Use last detected text for display if no new result
        if not result and len(last_text_data) > 0:
            text_list = last_text_list.copy()
            # Draw last detected boxes on skipped frames
            for text, score, box_coords in last_text_data:
                try:
                    box_color = (0, 255, 0) if score > CONFIDENCE_THRESHOLD else (0, 255, 255)
                    cv2.polylines(frame, [box_coords], True, box_color, 2)
                    top_left = box_coords[0]
                    cv2.putText(frame, f"{text} ({score:.2f})",
                                (int(top_left[0]), int(top_left[1]) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                except Exception:
                    pass
        
        all_text = " ".join(text_list).strip()
        high_confidence_text = " ".join(high_confidence_text_list).strip()
        
        # Use last high confidence text for speech if no new result
        if not result and len(last_high_confidence_text_list) > 0:
            high_confidence_text = " ".join(last_high_confidence_text_list).strip()

        # Speak only high-confidence text if new or after cooldown
        if AUDIO_ENABLED and high_confidence_text and high_confidence_text != last_text and (time.time() - last_spoken > COOLDOWN):
            print(f"🗣️ Detected (confidence >90%): {high_confidence_text}")
            try:
                tts = gTTS(high_confidence_text, lang='en')
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
                    temp_path = tmp_file.name
                tts.save(temp_path)

                pygame.mixer.music.load(temp_path)
                pygame.mixer.music.play()
                last_text = high_confidence_text
                last_spoken = time.time()

                # remove after playback (non-blocking cleanup)
                while pygame.mixer.music.get_busy():
                    cv2.waitKey(100)
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

            except Exception as e:
                print("⚠️ Audio error:", e)

    # Add status overlay to frame (always visible)
    if len(text_list) > 0:
        status_text = f"Text detected: {len(text_list)} lines"
        status_color = (0, 255, 0)  # Green
    else:
        status_text = "No text detected - Show text to camera"
        status_color = (0, 255, 255)  # Yellow
    
    cv2.putText(frame, status_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Display output
    cv2.imshow("PaddleOCR Live", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# -------------------------------
# Cleanup
# -------------------------------
cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
print("✅ OCR stopped. Goodbye!")
