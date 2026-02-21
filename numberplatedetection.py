!pip install opencv-python pytesseract imutils
import cv2
import imutils
import pytesseract
from google.colab.patches import cv2_imshow
from google.colab import files
uploaded = files.upload()
image_path = list(uploaded.keys())[0]
img = cv2.imread(image_path)
img = imutils.resize(img, width=600)
cv2_imshow(img)
print(" Original Image Loaded")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2_imshow(gray)
gray = cv2.bilateralFilter(gray, 11, 17, 17)
edges = cv2.Canny(gray, 30, 200)
cv2_imshow(edges)
cnts, _ = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
screenCnt = None
for c in cnts:
peri = cv2.arcLength(c, True)
approx = cv2.approxPolyDP(c, 0.018 * peri, True)
if len(approx) == 4:
screenCnt = approx
break
if screenCnt is not None:
cv2.drawContours(img, [screenCnt], -1, (0, 255, 0), 3)
cv2_imshow(img)
print(" Number Plate Region Detected")
else:
print(" Could not detect number plate region!")
if screenCnt is not None:
mask = cv2.drawContours(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), [screenCnt], -1, (255, 255, 255), -1)
x, y, w, h = cv2.boundingRect(screenCnt)
cropped = gray[y:y + h, x:x + w]
cv2_imshow(cropped)
else:
cropped = gray
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'
text = pytesseract.image_to_string(cropped, config='--psm 8')
print(" Detected Number Plate Text:", text.strip())
print("\n Process Completed Successfully!")