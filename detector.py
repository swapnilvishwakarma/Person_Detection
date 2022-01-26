import cv2, time, os, tensorflow as tf
import numpy as np
from tensorflow.python.keras.utils.data_utils import get_file
import logging

np.random.seed(123)


class Detector:
    def __init__(self) -> None:
        pass

    def readClasses(self, classesFilePath):
        with open(classesFilePath, "r") as f:
            self.classesList = [line.rstrip() for line in f]

        # colors for drawing bounding boxes
        self.colorsList = np.random.uniform(0, 255, size=(len(self.classesList), 3))

    def downloadModel(self, modelURL):
        fileName = os.path.basename(modelURL)
        self.modelName = fileName[:fileName.index(".")]
        
        self.cacheDir = "./pretrained_models/"
        os.makedirs(self.cacheDir, exist_ok=True)
        get_file(fileName, modelURL, cache_dir=self.cacheDir, cache_subdir="checkpoints", extract=True)

    def loadModel(self):
        print("Loading model..." + self.modelName)
        tf.keras.backend.clear_session()
        self.model = tf.saved_model.load(os.path.join(self.cacheDir, "checkpoints", self.modelName, "saved_model"))
        print("Model " + self.modelName + " loaded successfully!")

    def createBoundingBox(self, image, threshold=0.5):
        inputTensor = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
        inputTensor = tf.convert_to_tensor(inputTensor, dtype=tf.uint8)
        inputTensor = inputTensor[tf.newaxis, ...]
        detections = self.model(inputTensor)
        bboxs = detections["detection_boxes"][0].numpy()
        classIndexes = detections["detection_classes"][0].numpy().astype(np.int32)
        classScores = detections["detection_scores"][0].numpy()

        imH, imW, imC = image.shape
        bboxIdx = tf.image.non_max_suppression(bboxs, classScores, max_output_size=50, iou_threshold=threshold, score_threshold=threshold)
        print(bboxIdx)
        if len(bboxIdx)!=0:
            for i in range(len(bboxIdx)):
                bbox = tuple(bboxs[i].tolist())
                classConfidence = round(100*classScores[i])
                classIndex = classIndexes[i]
                if classIndex == 1:
                    classLabelText = self.classesList[classIndex].upper()
                    classColor = self.colorsList[classIndex]

                    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%d-%m-%Y %H:%M:%S")
                    handler = logging.FileHandler(filename='logs/person_detection_logs.log')
                    handler.setFormatter(formatter)
                    logger = logging.getLogger()
                    logger.setLevel(logging.DEBUG)
                    logger.addHandler(handler)

                    displayText = '{} {:.2f}%'.format(classLabelText, classConfidence)
                    ymin, xmin, ymax, xmax = bbox
                    xmin, xmax, ymin, ymax = (xmin * imW, xmax * imW, ymin * imH, ymax * imH)
                    xmin, xmax, ymin, ymax = int(xmin), int(xmax), int(ymin), int(ymax)
                    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), classColor, 2)
                    cv2.putText(image, displayText, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, classColor, 2)

                    logger.info(displayText)

                    lineWidth = min(int((xmax - xmin)*0.2), int((ymax - ymin)*0.2))

                    cv2.line(image, (xmin, ymin), (xmin + lineWidth, ymin), classColor, thickness=5)
                    cv2.line(image, (xmin, ymin), (xmin, ymin + lineWidth), classColor, thickness=5)
                    cv2.line(image, (xmax, ymin), (xmax - lineWidth, ymin), classColor, thickness=5)
                    cv2.line(image, (xmax, ymin), (xmax, ymin + lineWidth), classColor, thickness=5)

                    cv2.line(image, (xmin, ymax), (xmin + lineWidth, ymax), classColor, thickness=5)
                    cv2.line(image, (xmin, ymax), (xmin, ymax - lineWidth), classColor, thickness=5)
                    cv2.line(image, (xmax, ymax), (xmax - lineWidth, ymax), classColor, thickness=5)
                    cv2.line(image, (xmax, ymax), (xmax, ymax - lineWidth), classColor, thickness=5)

        return image

    def predictImage(self, imagePath, threshold=0.5):
        image = cv2.imread(imagePath)

        bboxImage = self.createBoundingBox(image, threshold)
        cv2.imwrite(self.modelName + ".jpg", bboxImage)
        cv2.imshow("Result", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def predictVideo(self, videoPath, threshold=0.5):
        cap = cv2.VideoCapture(videoPath)
        if (cap.isOpened() == False):
            print("Error opening video stream or file")
            return
        
        (success, image) = cap.read()
        startTime = 0
        while success:
            currentTime = time.time()

            fps = 1/(currentTime - startTime)
            startTime = currentTime

            bboxImage = self.createBoundingBox(image, threshold)
            cv2.putText(bboxImage, "FPS: " + str(int(fps)), (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
            cv2.imshow("Result", bboxImage)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            (success, image) = cap.read()
        cv2.destroyAllWindows()