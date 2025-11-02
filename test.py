from ultralytics import YOLO
import cv2
model=YOLO('runs/detect/train/weights/best.pt')
#path= "objects_classroom/data/images/test/pen196.jpg"
results=model(source=0,show=True,conf=0.6)


# annotated_img = results[0].plot()  # Get annotated frame (NumPy array)
# cv2.imshow("Detection Result", annotated_img)
#
# print("\nPress any key to close the window...")
# cv2.waitKey(0)
# cv2.destroyAllWindows()