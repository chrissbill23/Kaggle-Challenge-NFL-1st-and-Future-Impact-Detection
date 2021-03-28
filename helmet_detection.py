
import sys
sys.path.insert(1, './darknet')

import cv2
import argparse
from threading import Thread, enumerate
from queue import Queue
import random
from joblib import dump, load

app = argparse.ArgumentParser()
#app.add_argument("-v", "--video", help="Path to the video", default='')
app.add_argument("-i", "--image", help="Full Path to image or path relative to the darknet installation folder. NB: the path to darknet installation folder shoud be provided with command -d")
app.add_argument("-d", "--darknet", help="Path to Darknet installation. Attention: in this folder there must be the following files: data/obj.data; cfg/yolo-obj.cfg; backup/yolo-obj_last.weights", default='/darknet')
app.add_argument("-view", "--view", help="View of cam", default='Endzone')
app.add_argument("-t1", "--thresholcollision", type=float, help="Threshold of collision detection", default=0.9)
app.add_argument("-t2", "--thresholconfidence", type=float, help="Threshold of collision confidence", default=0.5)
app.add_argument("-t3", "--thresholvisibility", type=float, help="Threshold of collision visibility", default=0.5)
 
args = vars(app.parse_args())

#video_path = args['video']
#cap = cv2.VideoCapture(video_path)
#sample_rate = 1

labels= ['No-Collision', 'Collision']

rf = load('rf_col_predic.joblib') 
view = 1 if args['view'] == 'Endzone' else 0
WIDTH = 1280
HEIGHT = 720
thresh = args['thresholcollision']
thresh2 = args['thresholconfidence']
thresh3 = args['thresholvisibility']
def bbox2points(bbox):
    """
    From bounding box yolo format
    to corner points cv2 rectangle
    """
    l, x, y, w, h = bbox
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return (l, xmin, ymin, xmax, ymax)
    
    
def class_colors(names):
    """
    Create a dict with one random BGR color for each
    class name
    """
    return {name: (
        random.randint(0, 255),
        random.randint(0, 255),
        random.randint(0, 255)) for name in names}
        
 

def draw_boxes(detections, image, colors):
    import cv2
    for label, left, top, right, bottom in detections:
        cv2.rectangle(image, (left, top), (right, bottom), colors[labels[label]], 1)
        cv2.putText(image, "{}".format(labels[label]),(left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,colors[labels[label]], 2)
    return image

def bboxFromTxt(filepath):
    box = []
    with open(filepath, "r") as f:
        box = f.read().splitlines()
    boxes = []
    boxes2 = []
    for b in box:
        els = b.split(" ")
        boxes.append([0,float(els[1]),float(els[2]),float(els[3]),float(els[4])])
    for i in range(len(boxes)):
        b1 = boxes[i]
        for j in range(i,len(boxes)):
            b2 = boxes[j]
            #left 1, width1, top1, height1, left2, width2, top2, height2, view1, view2
            pred = rf.predict([[ b1[1], b1[2], b2[1], b2[2],view,1-view,view,1-view ]])
            print(pred[0],b1)
            if pred[0][0] >= thresh and pred[0][1] >= thresh2 and pred[0][2] >= thresh3:
                b1 = bbox2points((1,b1[1]*WIDTH,b1[2]*HEIGHT,b1[3]*WIDTH,b1[4]*HEIGHT))
                b2 = bbox2points((1,b2[1]*WIDTH,b2[2]*HEIGHT,b2[3]*WIDTH,b2[4]*HEIGHT))
                boxes2.append(b1)
                boxes2.append(b2)
                break
        boxes2.append(bbox2points((0,b1[1]*WIDTH,b1[2]*HEIGHT,b1[3]*WIDTH,b1[4]*HEIGHT)))
    return boxes2
    
if __name__ == '__main__':

    import subprocess
    import os
    wd = os.getcwd()
    os.chdir(args["darknet"])
    
    
    
    
    
    #success = cap.grab()
    #fno = 0
    #while success:
    #    _, img = cap.retrieve()
        #cv2.imshow('frame',img)
    #    filename = "../tmp_files/frame_tmp"+str(fno)+".jpg"
    #    cv2.imwrite(filename, img)
    #    subprocess.call(['darknet.exe','detector', 'test', 'data\\obj.data','cfg\\yolo-obj.cfg','backup\\yolo-obj_last.weights', '-ext_output',filename, '-dont_show', '-save_labels'])
    #    #with open('../tmp_files/{}.txt'.format('all_inputs'), 'a') as writefile:
        #  line = "{}\n".format(filename)
        #  writefile.write(line)
        
    #    if cv2.waitKey(1) & 0xFF == ord('q'):
    #        break
    #    fno+=1
        # read next frame
    #    success = cap.grab()
    
    image = cv2.imread(args['image'])
    subprocess.call(['darknet.exe','detector', 'test', 'data\\obj.data','cfg\\yolo-obj.cfg','backup\\yolo-obj_last.weights', '-ext_output',args['image'], '-dont_show', '-save_labels'])
    colors = class_colors(labels)
    tmp = os.path.splitext(args['image'])[0]
    print(tmp)
    cv2.imshow('image', draw_boxes(bboxFromTxt(tmp+".txt"), image, colors))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('col_'+str(thresh)+'_conf_'+str(thresh2)+'_vis_'+str(thresh3)+'.jpg',image)  




    