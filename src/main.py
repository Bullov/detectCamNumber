import cv2 as cv
import os
import math
import argparse
import src.utils.text_exctraxtor as textExtractor
import pytesseract


def getFrameFromVideo(pathToVideo):
    capture = cv.VideoCapture(cv.samples.findFileOrKeep(pathToVideo))
    if not capture.isOpened():
        print('Unable to open: ' + pathToVideo)
        exit(0)

    # defaultWidth = capture.get(cv.CAP_PROP_FRAME_WIDTH)
    # defaultHeight = capture.get(cv.CAP_PROP_FRAME_HEIGHT)

    while True:
        ret, frame = capture.read()
        if frame is None:
            break
        # currentFrameTime = capture.get(cv.CAP_PROP_POS_MSEC)
        currentFrame = capture.get(cv.CAP_PROP_POS_FRAMES)

        if currentFrame > 1:
            return frame

        keyboard = cv.waitKey(30)
        if keyboard == 'q' or keyboard == 27:
            break


def getCleanImage(image):
    height, width, _ = image.shape
    hls = cv.cvtColor(image, cv.COLOR_BGR2HLS)
    hlsChannels = cv.split(hls)
    lightnessChannel = hlsChannels[1]
    for i in range(height):
        for j in range(width):
            lightnessChannel[i][j] = 0 if lightnessChannel[i][j] < 215 else 255

    hlsChannels[1] = lightnessChannel
    hls = cv.merge(hlsChannels)
    bgrImg = cv.cvtColor(hls, cv.COLOR_HLS2BGR)
    # cv.imwrite('../output/sig10.jpg', bgrImg)
    return bgrImg


# parser = argparse.ArgumentParser(description='Video processing')
# parser.add_argument('--i', type=str, help='Input file path')
# parser.add_argument('--o', nargs='?', type=str, default='../output/output.txt', help='Output file path')
# args = parser.parse_args()
pathToVideos = os.path.join('E:', 'wfs0.4 test', 'videosTest')
# pathToVideos = os.path.join('../output')
filesNameList = os.listdir(pathToVideos)

for fileName in filesNameList:
    filePath = os.path.join(pathToVideos, fileName)
    img = getCleanImage(getFrameFromVideo(filePath))
    # img = getCleanImage(cv.imread(filePath))
    extractor = textExtractor.PyTextractor()
    test = extractor.get_image_text(img, display=True)
    print(test)
    # newFilePath = os.path.join(pathToVideos, test[0])
    # try:
    #     os.mkdir(newFilePath)
    #     os.rename(filePath, os.path.join(newFilePath, fileName))
    # except FileExistsError:
    #     os.rename(filePath, os.path.join(newFilePath, fileName))
