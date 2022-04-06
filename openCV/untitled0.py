import cv2 as cv

def rescaleFrame(frame, scale=0.75):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    
    dimensions = (width,height)
    
    return cv.resize(frame,dimensions, interpolation=cv.INTER_AREA)


img = cv.imread('levi1000.jpg')
levi = rescaleFrame(img)
levi[200:300,200:300] = 0,0,0
cv.rectangle(levi, (198,298), (298,398),(0,255,0),thickness=1)
cv.imshow('Levi',levi)

grayLevi = cv.cvtColor(levi, cv.COLOR_BGR2GRAY)
cv.imshow('Gray Levi', grayLevi)

blurLevi = cv.GaussianBlur(levi, (3,3), cv.BORDER_DEFAULT) 
cv.imshow('Blur Levi', blurLevi)


#Edge Cascade
KEENNY =  cv.Canny(levi, 125,175)
cv.imshow('Keeeeennnyyyy', KEENNY)

# Dilation
dilated = cv.dilate(KEENNY, (3,3), iterations=3)
cv.imshow('Dilated', dilated)

# Erotion
eroded = cv.erode(dilated, (3,3), iterations=3)
cv.imshow('Eroded',eroded)

# Cropping
cropped = levi[200:300,300:400]
cv.imshow('Cropped',cropped)

# Resize 
resized = cv.resize(levi,(500,500),interpolation=cv.INTER_CUBIC)
cv.imshow('Resized',resized)


if cv.waitKey(0) and 0xFF == ord('d'):
    cv.destroyAllWindows()