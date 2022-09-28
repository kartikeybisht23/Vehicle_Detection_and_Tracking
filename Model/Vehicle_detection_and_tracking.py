import cv2, time, os, tensorflow as tf
import numpy as np
from tensorflow.python.keras.utils.data_utils import get_file

np.random.seed(32)

class Detector():
    def __init__(self):
        pass
    def readClasses(self,classesFilePath):
        # going through all the classes
        with open (classesFilePath,'r') as f:
            self.classesList=f.read().splitlines()
            # color list
            self.colorList=np.random.uniform(low=0,high=255,size=(len(self.classesList),3))
            

    def downloadModel(self,modelUrl):
        # getting the base name from modelURL
        fileName=os.path.basename(modelURL)
        #getting the model name
        self.modelName=fileName[:fileName.index('.')]
        print(fileName)
        print(self.modelName)
        self.cacheDir='C://test_folder//pretrained_models'
        
        os.makedirs(self.cacheDir,exist_ok=True)
        
        #download the model using get_file
        get_file(fname=fileName,origin=modelURL,cache_dir=self.cacheDir,cache_subdir='check',extract=True)
        
        
    def loadModel(self):
        tf.keras.backend.clear_session()
        self.model=tf.saved_model.load(os.path.join(self.cacheDir,'check',self.modelName,'saved_model'))
        print(f'model: {self.modelName} is loaded successfully')
     
    def createBoundingBox(self,image):
        inputTensor=cv2.cvtColor(image.copy(),cv2.COLOR_BGR2RGB)
        inputTensor=tf.convert_to_tensor(inputTensor,dtype='uint8')
        #batch as input so as this is 1 image so we will exapnding the axis=0
        inputTensor=inputTensor[tf.newaxis,...]
        #detections=self.model(inputTensor)
        #detections = detect_fn(input_tensor) 
        detections=self.model.signatures['serving_default'](inputTensor)
        #detections = detect_fn.signatures['serving_default'](input_tensor)
        
        bboxs=detections['detection_boxes'][0].numpy()
        classIndexes=detections['detection_classes'][0].numpy().astype(np.int32)
        classScores=detections['detection_scores'][0].numpy()
        
        imH,imW,imC=image.shape
        required_class=['bicycle','car','motorbike','bus','train','truck']

        
        bboxIdx=tf.image.non_max_suppression(bboxs,classScores,max_output_size=50,iou_threshold=0.5,score_threshold=0.5)
        #print(bbox)
        count={'car':0,'bus':0,'truck':0,'motorbike':0}
        if len(bboxIdx)!=0:
            for i in bboxIdx:
                bbox=tuple(bboxs[i])
                classIndex=classIndexes[i]
                classConfidence=round(100*classScores[i])
            
                classLabel=self.classesList[classIndex]
                classColor=self.colorList[classIndex]
                if classLabel in count.keys():
                    count[classLabel]=count[classLabel]+1
            
                displayText=f'{classLabel}:{classConfidence}'

                ymin,xmin,ymax,xmax=bbox

                xmin,xmax,ymin,ymax=(xmin*imW,xmax*imW,ymin*imH,ymax*imH)
                xmin,xmax,ymin,ymax=int(xmin),int(xmax),int(ymin),int(ymax)
                if classLabel in required_class:
                    cv2.rectangle(image,(xmin,ymin),(xmax,ymax),color=classColor,thickness=1)
                    cv2.putText(image,displayText,(xmin,ymin-10),cv2.FONT_HERSHEY_PLAIN,2,classColor,2)
                    linewidth=min(int((xmax-xmin)*0.3),int((ymax-ymin)*0.3))
              
                    cv2.line(image,(xmin,ymin),(xmin+linewidth,ymin),classColor,5)
                    cv2.line(image,(xmin,ymin),(xmin,ymin+linewidth),classColor,5)

                    cv2.line(image,(xmax,ymin),(xmax-linewidth,ymin),classColor,5)           
                    cv2.line(image,(xmax,ymin),(xmax,ymin+linewidth),classColor,5)

                    cv2.line(image,(xmin,ymax),(xmin+linewidth,ymax),classColor,5)
                    cv2.line(image,(xmin,ymax),(xmin,ymax-linewidth),classColor,5)

                    cv2.line(image,(xmax,ymax),(xmax-linewidth,ymax),classColor,5)
                    cv2.line(image,(xmax,ymax),(xmax,ymax-linewidth),classColor,5)

              
                cv2.putText(image,'total car ='+str(count['car']),(20,70),cv2.FONT_HERSHEY_PLAIN,2,(0,0,255),2)
                cv2.putText(image,'total truck ='+str(count['truck']),(20,90),cv2.FONT_HERSHEY_PLAIN,2,(0,0,255),2)
                cv2.putText(image,'total bus ='+str(count['bus']),(20,110),cv2.FONT_HERSHEY_PLAIN,2,(0,0,255),2)
                cv2.putText(image,'total motorbike ='+str(count['motorbike']),(20,130),cv2.FONT_HERSHEY_PLAIN,2,(0,0,255),2)


        return image                                 

    def predictVideo(self,videoPath):
        print('inside video predict')
        cap=cv2.VideoCapture(videoPath)
        if (cap.isOpened()==False):
            print('Error check the path')
            return
        (success,image)=cap.read()  
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        fourcc=cv2.VideoWriter_fourcc(*'MP4V')
        output=cv2.VideoWriter('output.mp4',fourcc,20.0,(frame_width,frame_height))
        while success==True:
            print(success)

            bboxImage=self.createBoundingBox(image)
            output.write(bboxImage)
            if cv2.waitKey(1)==ord('q'):
                break
            (success,image)=cap.read()  
        cap.release()
        output.release()
        cv2.destroyAllWindows()  



if __name__=='__main__':
    classesFilePath='coco.names'
    videoPath='road_trafifc.mp4'
     modelURL='http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz

       '
    detector=Detector()
    detector.readClasses(classesFilePath)
    detector.downloadModel(modelURL)
    detector.loadModel()
    detector.predictVideo(videoPath)        
        
