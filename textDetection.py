import cv2, sys, os, tflearn
import numpy as np

# loads an image and straightens it
# returns a properly rotated image
# https://www.pyimagesearch.com/2017/02/20/text-skew-correction-opencv-python/
def loadAndRotateImage(imgPath, testing=False):
	# load as grayscale and then transform to binary using Otsu's method for obtaining the threshold
	image = cv2.imread(imgPath)

	gray = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
	gray = cv2.bitwise_not(gray)
	(_thresh, binImg) = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

	# find rotation angle of the minimum rotated rounding box
	coords = np.column_stack(np.where(binImg > 0))
	
	minArea = cv2.minAreaRect(coords)
	angle = minArea[-1]
 	
	if angle < -45:
		angle = -(90 + angle)
	else:
		angle = -angle

	# find image coordinates
	(h, w) = binImg.shape[:2]
	center = (w // 2, h // 2)

	# adjust image size if necessary because after the rotation part of the
	# rounding box might be "outside"	
	difs = int(abs(minArea[0][0] - minArea[1][0])), int(abs(minArea[0][1] - minArea[1][1]))
	if difs[0] > center[0]:
		w += difs[0] - center[0]
	if difs[1] > center[1]:
		h += difs[1] - center[1]

	# init the rotation matrix
	M = cv2.getRotationMatrix2D(center, angle, 1.0)
	
	# rotate the image
	rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

	# testing
	if testing:
		cv2.imshow("Input(Gray)", image)
		cv2.imshow("Rotated", rotated)
		cv2.waitKey(0)

	return rotated


# detects the words ROIs in the processed image
# based dhanushka response: https://stackoverflow.com/questions/23506105/extracting-text-opencv
# here's his license
"""
The MIT License (MIT)

Copyright (c) 2014 Dhanushka Dangampola

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""
def extractWordsROIs(img, testing=False, showRectangle=False, saveWords=False):
	rgb = img
	small = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)

	# applies morphological gradient on the grayscale image
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
	grad = cv2.morphologyEx(small, cv2.MORPH_GRADIENT, kernel)

	if testing:
		cv2.imshow("grad", grad)
		cv2.waitKey(0)

	# turns the image to binary with Otsu
	_, bw = cv2.threshold(grad, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

	if testing:
		cv2.imshow("bw", bw)
		cv2.waitKey(0)

	# applies morphological close to horizontally connect the words
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 2))
	connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
	
	if testing:
		cv2.imshow("connected", connected)
		cv2.waitKey(0)

	# obtain contours
	_, contours, hierarchy = cv2.findContours(connected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

	# mask used to compute the ratio of non-zero pixels in each ROI
	mask = np.zeros(bw.shape, dtype=np.uint8)

	# the list of regions of interest which will be returned
	ROIs = []

	if showRectangle:
		withRectangles = rgb.copy()
	for idx in range(len(contours)):
		# obtain ROI
		x, y, w, h = cv2.boundingRect(contours[idx])

		# restore the ROI on the mask to 0 (sometimes ROIs overlap?) and draw the contour
		mask[y:y+h, x:x+w] = 0
		cv2.drawContours(mask, contours, idx, (255, 255, 255), -1)

		# compute the ratio of non-zero pixels in the region
		r = float(cv2.countNonZero(mask[y:y+h, x:x+w])) / (w * h)

		# r > 0.45 				-> check for at least 45% filled area
		# w > 8 and h > 8		-> constraints on region size
		if r > 0.45 and w > 8 and h > 8:
			# add the region of interest in the list
			roi = rgb[y:y+h, x:x+w]
			ROIs.append(roi)	

			# save the ROI as .png file
			if saveWords:
				if not os.path.exists(".\\testing"):
					os.makedirs(".\\testing")
	
				cv2.imwrite(".\\testing\\word_%s.png" % idx, roi)

			# add rectangle
			if showRectangle:
				cv2.rectangle(withRectangles, (x, y), (x + w, y + h), (0, 255, 0), 1)


	if testing:
		cv2.imshow("mask", mask)
		cv2.waitKey(0)

	if testing and showRectangle:
		cv2.imshow("rects", withRectangles)
		cv2.waitKey(0)

	return ROIs


# extracts letters from a word region of interest
# returns the letter
# similar to words extraction
def extractLettersROIs(wordROI, testing=False, showRectangle=False, saveLetters=False):
	if testing:
		cv2.imshow("word", wordROI)
		cv2.waitKey(0)

	gray = cv2.cvtColor(wordROI, cv2.COLOR_BGR2GRAY)

	# turns the image to binary with a low threshold to disjoint letters
	_, bw = cv2.threshold(gray, 100, 255.0, cv2.THRESH_BINARY)

	bw = cv2.bitwise_not(bw)

	if testing:
		cv2.imshow("bw", bw)
		cv2.waitKey(0)

	_, contours, hierarchy = cv2.findContours(bw.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

	ROIs = []
	
	if showRectangle:
		withRectangles = wordROI.copy()
	
	for idx in range(len(contours)):
		x, y, w, h = cv2.boundingRect(contours[idx])

		roi = gray[y:y+h+1, x:x+w+1]

		# process the letter and add it to the list (resize to 28x28, and invert colors)
		roi = cv2.bitwise_not(roi)
		roi = cv2.resize(roi, (28, 28)) 

		# save the ROI as .png file
		if saveLetters:
			if not os.path.exists(".\\testing"):
				os.makedirs(".\\testing")
	
			cv2.imwrite(".\\testing\\letter_%s.png" % idx, roi)

		# resize to be able to be fed to the CNN
		roi = roi.reshape([-1, 28, 28, 1])
		ROIs.append(roi)	

		# add rectangle
		if showRectangle:
			cv2.rectangle(withRectangles, (x, y), (x + w, y + h), (0, 255, 0), 1)
	
	if testing and showRectangle:
		cv2.imshow("rects", withRectangles)
		cv2.waitKey(0)

	return ROIs


def main():
	
	if len(sys.argv) != 2:
		print ("usage: {} <file>".format(sys.argv[0]))
		return

	# rotate the image if necessary
	rotatedImage = loadAndRotateImage(sys.argv[1], testing=False)
	
	# obtain words regions of interest
	wordsROIs = extractWordsROIs(rotatedImage, testing=False, showRectangle=False, saveWords=True)

	if wordsROIs is None or len(wordsROIs) == 0:
		return

	# obtain letters
	lettersROIs = []
	for wordROI in wordsROIs:
		lettersROIs.extend(extractLettersROIs(wordROI, testing=False, showRectangle=False, saveLetters=True))
	
	# print intermediary results
	print ("There are {} words and {} letters.".format(len(wordsROIs), len(lettersROIs)))


if __name__ == "__main__":
	main()