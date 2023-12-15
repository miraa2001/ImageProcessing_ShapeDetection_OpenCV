import numpy as np
import cv2

# read image
img = cv2.imread('mainShapes1.jpg', cv2.IMREAD_COLOR)
# convert image to grayscale
imgGrey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# apply thresholding to greyscale image
_, thresholdedImg = cv2.threshold(imgGrey, 240, 255, cv2.THRESH_BINARY)

# find contours and hierarchy
contours, hierarchy = cv2.findContours(thresholdedImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
# initialize shapesArray
shapes = []
i = -1
b = 0
shapesCount = 0
# append shapes that have area bigger than 100
for shape in contours:
    if cv2.contourArea(shape) < 100:
        continue
    else:
        shapesCount += 1
shapesNames = ["" for y in range(shapesCount)]
for shape in contours:
    i += 1
    child = hierarchy[0][i][2]
    parent = np.intc(hierarchy[0][i][3])
    grandchild = np.intc(hierarchy[0][child][2])
    nCount = 0
    nextNeighbor = grandchild
    # find number of edges
    approx = cv2.approxPolyDP(shape, 0.035 * cv2.arcLength(shape, True), True)
    edgesCount = len(approx)

    # find number of children inside an object to determine if it's a face
    while True:
        nextNeighbor = np.intc(hierarchy[0][nextNeighbor][0])
        if nextNeighbor == -1:
            break
        else:
            if cv2.contourArea(contours[nextNeighbor]) > 100:
                nCount += 1

    # if area < 60 or it's an inner shape ignore
    if cv2.contourArea(shape) < 60 or parent != 0:
        continue
    else:
        if edgesCount == 3:
            shapesNames[b] = "Triangle"  # 3 edges -----> triangle
        elif edgesCount < 3:
            shapesNames[b] = "Line"      # less than 3 edges -----> line
        elif edgesCount == 4:
            # get 4 edges shape dimensions
            (x, y, w, h) = cv2.boundingRect(approx)
            # find the ratio between two edges
            ratio = w / float(h)
            if child != -1:
                if nCount <= 2:     # if nCount is less or equal two then it's not a face
                    if cv2.contourArea(shape) > 500:
                        shapesNames[b] = "Square" if 0.95 <= ratio <= 1.05 else "Rectangle"   # 4 edges -----> rectangle or square if ratio is close to 1
                else:
                    shapesNames[b] = "Face"
            else:
                shapesNames[b] = "curve"    # if the 4 edges shape does not have an inner edge then it's a curve
        elif edgesCount > 5:
            approxCircle = cv2.approxPolyDP(shape, 0.01 * cv2.arcLength(shape, True), True)
            if child != -1:
                if len(approxCircle) > 10:
                    if nCount <= 2:
                        shapesNames[b] = "Circle"
                    else:
                        shapesNames[b] = "Face"
            else:
                shapesNames[b] = "curve"    # if the 5 edges shape does not have an inner edge then it's a curve
        else:
            shapesNames[b] = "curve"        # if it's not a line triangle rectangle square or a circle then it's a curve
        xx, yy = approx[0][0]               # get location of object to write text
        cv2.putText(img, shapesNames[b], (xx, yy - 5), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
        b += 1
i = -1
for j in range(shapesCount):
    shapesNames[j] = ""
b = 0
eyeCount = 0

# find inner objects
for shape in contours:
    i += 1
    child = np.intc(hierarchy[0][i][2])
    parent = np.intc(hierarchy[0][i][3])
    grandparent = np.intc(hierarchy[0][parent][3])
    greatGrandParent = np.intc(hierarchy[0][grandparent][3])
    approx = cv2.approxPolyDP(shape, 0.035 * cv2.arcLength(shape, True), True)
    edgesCount = len(approx)
    # if area is less than 60 or is an outer edge ignore
    if cv2.contourArea(shape) < 60 or grandparent <= 0:
        continue
    else:
        if edgesCount == 3:
            if child != -1 and hierarchy[0][child][2] == -1:
                shapesNames[b] = "nose"     # if it has an inner edge and the inner edge has no children then it's a nose
        elif edgesCount < 3:
            if child == -1:
                shapesNames[b] = "mouth"    # if it has no children then it's a line which is a mouth
        elif edgesCount == 4:
            (x, y, w, h) = cv2.boundingRect(approx)
            ratio = w / float(h)
            if cv2.contourArea(shape) > 500:
                shapesNames[b] = "Square" if 0.95 <= ratio <= 1.05 else "Rectangle"
            else:
                shapesNames[b] = "mouth"   # if area is less than 500 then it's a mouth which is  a curve actually
        elif edgesCount > 5:
            approxCircle = cv2.approxPolyDP(shape, 0.01 * cv2.arcLength(shape, True), True)
            if len(approxCircle) > 9:
                if child != -1 and greatGrandParent == 0 and cv2.contourArea(shape) > 1150:
                    shapesNames[b] = "eye"
                elif child != -1 and greatGrandParent == 0 and  1000 < cv2.contourArea(shape) < 1150:
                    shapesNames[b] = "mouth"
                elif child != -1 and greatGrandParent == 0:
                    shapesNames[b] = "eye"
                elif child == -1 and greatGrandParent == 0:
                    shapesNames[b] = "eye"
        xx, yy = approx[0][0]
        cv2.putText(img, shapesNames[b], (xx, yy - 6), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
        b += 1
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
