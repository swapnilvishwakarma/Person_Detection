from detector import *
import argparse
import logging

# model_URL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz" # ssd_mobilenet_v2_coco_2018_03_29 for quick predictions with compromised accuracy
# model_URL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d6_coco17_tpu-32.tar.gz" # EfficientDet D6 for better predictions with compromised speed
# model_URL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d5_coco17_tpu-32.tar.gz" # EfficientDet D5 for better predictions with compromised speed

# imagePath = "/Users/swapnilvishwakarma/Desktop/OurEye.ai/Object_Detection/test_images/img4.jpeg"
# videoPath = "/Users/swapnilvishwakarma/Desktop/OurEye.ai/Object_Detection/test_videos/video3.mp4" # 0 for webcam input
# videoPath = 0

classFile = "coco.names"


def arg_parse():
    parse = argparse.ArgumentParser(description="Person Detection", add_help=True)
    parse.add_argument("-m", "--model", help="Model Path. Link to any model can be passed from this url: 'https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md' ", type=str, default="http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d5_coco17_tpu-32.tar.gz")
    parse.add_argument("-i", "--image", help="path to image", default=None, type=str)
    parse.add_argument("-v", "--video", help="path to video, 0 for webcam", default=None, type=str)
    parse.add_argument("-t", "--threshold", help="threshold for prediction", default=0.5, type=float)
    args = parse.parse_args()
    return args

if __name__ == "__main__":
    args = arg_parse()
    threshold = args.threshold
    model_URL = args.model
    imagePath = args.image
    videoPath = args.video

    detector = Detector()
    detector.readClasses(classFile)
    detector.downloadModel(model_URL)
    detector.loadModel()
    
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler = logging.FileHandler(filename='logs/person_detection_logs.log')
    handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)

    if imagePath:
        logger.info("Running Person Detection on given Image...\n")
        detector.predictImage(imagePath, threshold)
    elif videoPath:
        if videoPath == "0":
            logger.info("Running Person Detection on given Webcam...\n")
            detector.predictVideo(0, threshold)
        logger.info("Running Person Detection on given Video...\n")
        detector.predictVideo(videoPath, threshold)
    else:
        print("Please specify image or video path, or 0 for web-cam. Use -h for help")