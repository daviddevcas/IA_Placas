import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
import LPR

def plot_image(img, grayscale=True):
    plt.axis('off')
    if grayscale:
        plt.imshow(img, cmap='gray')
    else:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()

idx = 1
lpr = LPR.LPR()

img = cv2.imread(f"./imgs/{idx:03}.png")
plot_image(img, False)

gray = lpr.grayscale(img)
plot_image(gray)

thresh = lpr.apply_threshold(gray)
plot_image(thresh)

contours = lpr.find_contours(thresh)
canvas = np.zeros_like(img)
cv2.drawContours(canvas , contours, -1, (0, 255, 0), 2)
plt.axis('off')
plt.imshow(canvas);
plt.show();

candidates = lpr.filter_candidates(contours)
canvas = np.zeros_like(img)
cv2.drawContours(canvas , candidates, -1, (0, 255, 0), 2)
plt.axis('off')
plt.imshow(canvas);
plt.show();

license = lpr.get_lowest_candidate(candidates)
canvas = np.zeros_like(img)
cv2.drawContours(canvas , [license], -1, (0, 255, 0), 2)
plt.axis('off')
plt.imshow(canvas);
plt.show();

cropped = lpr.crop_license_plate(gray, license)
cropped2 = lpr.crop_license_plate(img, license)
plot_image(cropped2, False)

thresh_cropped = lpr.apply_adaptive_threshold(cropped)
plot_image(thresh_cropped)

clear_border = lpr.clear_border(thresh_cropped)
final = lpr.invert_image(clear_border)
plot_image(final)

psm = 7
alphanumeric = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
options = "-c tessedit_char_whitelist={}".format(alphanumeric)
options += " --psm {}".format(psm)
txt = pytesseract.image_to_string(final, config=options)
print(txt[:2], txt[2:5], txt[5:])

# img = np.zeros((512,512,3), np.uint8)

# # Write some Text

# font                   = cv2.FONT_HERSHEY_SIMPLEX
# bottomLeftCornerOfText = (10,50)
# fontScale              = 1
# fontColor              = (255,255,255)
# thickness              = 1
# lineType               = 2

# cv2.putText(img,txt[:2]+' '+txt[2:5]+' '+txt[5:], 
#     bottomLeftCornerOfText, 
#     font, 
#     fontScale,
#     fontColor,
#     thickness,
#     lineType)

# #Display the image
# cv2.imshow("img",img)
# cv2.waitKey(0)