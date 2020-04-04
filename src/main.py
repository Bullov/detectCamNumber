import cv2 as cv
import os
import re
import argparse
import utils.text_exctraxtor as text_extractor


def getFrameFromVideo(path_to_video):
    capture = cv.VideoCapture()
    capture.open(cv.samples.findFileOrKeep(path_to_video))
    if not capture.isOpened():
        print('Unable to open: ' + path_to_video)
        return None

    # defaultWidth = capture.get(cv.CAP_PROP_FRAME_WIDTH)
    # defaultHeight = capture.get(cv.CAP_PROP_FRAME_HEIGHT)

    while True:
        ret, frame = capture.read()
        if frame is None:
            break
        # current_frame_time = capture.get(cv.CAP_PROP_POS_MSEC)
        current_frame = capture.get(cv.CAP_PROP_POS_FRAMES)

        if current_frame > 1:
            gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            return getCleanImage(gray_frame)

        keyboard = cv.waitKey(30)
        if keyboard == 'q' or keyboard == 27:
            break


def getCleanImage(img):
    _, image = cv.threshold(img, 212, 255, cv.THRESH_BINARY_INV)
    image = cv.equalizeHist(image)
    image = cv.adaptiveThreshold(image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 7, 35)

    return cv.cvtColor(image, cv.COLOR_GRAY2BGR)


def getOnlyCamName(img):
    return img[520:560, 50:140]


def copyFile(current_file_path, new_file_path, file_name):
    try:
        os.mkdir(new_file_path)
        os.rename(current_file_path, os.path.join(new_file_path, file_name))
    except FileExistsError:
        os.rename(current_file_path, os.path.join(new_file_path, file_name))


def convertStringArrayToString(str_arr):
    result = '[\'' + str_arr[0] + '\''
    for string in str_arr[1:]:
        result += (', \'' + string + '\'')
    result += ']'
    return result.replace('\n', '\\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detecting cam name/number from raw h264 video file')
    parser.add_argument('--i', type=str, help='Input path to folder with videos')
    parser.add_argument('--o', nargs='?', type=str, default='../output/result.txt', help='Output log file path')
    args = parser.parse_args()

    pathToVideos = args.i
    filesNameList = os.listdir(pathToVideos)
    outputFile = open(args.o, 'wb')
    regex = re.compile('^[@a-zA-Z0-9]+$')

    for fileName in filesNameList:
        currentFilePath = os.path.join(pathToVideos, fileName)

        if os.path.isdir(currentFilePath):
            outputFile.write((fileName + ': DIRECTORY\n').encode('utf-8'))
            continue

        img = getFrameFromVideo(currentFilePath)

        if img is None:
            outputFile.write((fileName + ': unable to open\n').encode('utf-8'))
            continue

        extractor = text_extractor.PyTextExtractor()
        result_list = extractor.get_image_text(img)
        isCorrectFileName = False

        for text in result_list:
            if regex.match(text):
                isCorrectFileName = True
                newFilePath = os.path.join(pathToVideos, text)
                copyFile(currentFilePath, newFilePath, fileName)
                break

        if not isCorrectFileName:
            outputFile.write((fileName + ': ' + convertStringArrayToString(result_list) + '\n').encode('utf-8'))

    outputFile.close()
