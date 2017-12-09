# -*- coding: UTF-8 -*-
import os
import cv2
import progressbar
import copy
import Utils_Image
import Utils_Imagenet
import Utils_Tensorbox
import frame
import multiclass_rectangle
import vid_classes
from PIL import Image,ImageDraw
import sys
import re

### Fucntions to mount the video from frames

def draw_rectangles(path_video_folder, labeled_video_frames, flag):

    # width = 1080
    # height = 1920

    labeled_frames =[]
    if flag:
        folder_path=path_video_folder+"/deted_frames/"

    else:
        folder_path=path_video_folder+"/labeled_frames/"
    
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print("created folder: %s"%folder_path)
    
    maxID = -1
    for frame in labeled_video_frames:
        for bb_rect in frame.rects:
            if bb_rect.trackID > maxID:
                maxID = bb_rect.trackID
    
            
    for frame in labeled_video_frames:
        #bb_img = image.open(frame.filename)
        dr = cv2.imread(frame.filename)

        new_img = folder_path + os.path.splitext(os.path.basename(frame.filename))[0]+ "_labeled" + os.path.splitext(os.path.basename(frame.filename))[1]
        print "original filename:%s"%frame.filename
        print "new filename:%s"%new_img
        i = 0
        j = 0
        for bb_rect in frame.rects:
        ################ adding rectangle ###################
            ## dr = imagedraw.draw(bb_img)
            ## cor = (bb_rect.x1,bb_rect.y1,bb_rect.x2 ,bb_rect.y2) # DA VERIFICARE Try_2 (x1,y1, x2,y2) cor = (bb_rect.left() ,bb_rect.right(),bb_rect.bottom(),bb_rect.top()) Try_1
            ## ## fixed idごとに色分け
            ## ## if bb_rect.label_code is 'Not Set':
            ## ##     outline_class=(240,255,240)
            ## ## else :  
            ## ##     outline_class=vid_classes.code_to_color(bb_rect.label_chall)
            ## if bb_rect.trackid is -1:
            ##     outline_class=(240,255,240)
            ## else :  
            ##     outline_class=vid_classes.code_to_color(bb_rect.trackID)

            ## dr.rectangle(cor, outline=outline_class)

            #bb_rect.x1,bb_rect.y1,bb_rect.x2,bb_rect.y2=Utils_Image.get_orig_rect(width, height, 640, 480, bb_rect.x1,bb_rect.y1,bb_rect.x2 ,bb_rect.y2)


            if bb_rect.trackID == -1:
                #outline_class=(240,255,240)
                # add
                j += 1
                continue
            else :  
                top_left = (int(bb_rect.x1),int(bb_rect.y1)) # DA VERIFICARE Try_2 (x1,y1, x2,y2) cor = (bb_rect.left() ,bb_rect.right(),bb_rect.bottom(),bb_rect.top()) Try_1
                under_right = (int(bb_rect.x2) ,int(bb_rect.y2))

                outline_class=vid_classes.code_to_color(bb_rect.trackID)
                #outline_class=vid_classes.randam_color_generate(maxID, bb_rect.trackID)

                cv2.rectangle(dr, top_left, under_right, outline_class, 4, 4)

                i+=1

            ## top_left = (int(bb_rect.x1),int(bb_rect.y1)) # DA VERIFICARE Try_2 (x1,y1, x2,y2) cor = (bb_rect.left() ,bb_rect.right(),bb_rect.bottom(),bb_rect.top()) Try_1
            ## under_right = (int(bb_rect.x2) ,int(bb_rect.y2))
            ## cv2.rectangle(dr, top_left, under_right, (255,255,255), 4, 4)
                
            #dr = cv2.rectangle(dr, top_left, under_right, (240,255,240), 4, 4)
        # print save_img  
        # change
        print(frame.filename)
        print(len(frame.rects))
        print(i)
        print(j)
        cv2.imwrite(new_img, dr)
        labeled_frames.append(new_img)
    return labeled_frames

def draw_rectangle(image_path, rect_box):

    bb_img = image.open(image_path)
    ################ adding rectangle ###################
    dr = imagedraw.draw(bb_img)
    cor = (rect_box[0],rect_box[1],rect_box[2],rect_box[3]) # DA VERIFICARE Try_2 (x1,y1, x2,y2) cor = (bb_rect.left() ,bb_rect.right(),bb_rect.bottom(),bb_rect.top()) Try_1
    outline_class=(240,255,240)
    dr.rectangle(cor, outline=outline_class)
    # print save_img  
    bb_img.save(image_path)


def make_tracked_video(out_vid_path, labeled_video_frames):

    if labeled_video_frames[0] is not None:

        img = cv2.imread(labeled_video_frames[0], True)
        print "reading filename: %s"%labeled_video_frames[0]
        h, w = img.shape[:2]
        print "video size: width: %d height: %d"%(h, w)
        #fourcc = cv2.cv.CV_FOURCC('m', 'p', '4', 'v')
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        out = cv2.VideoWriter(out_vid_path,fourcc, 20.0, (w, h), True)
        print("Start Making File Video:%s " % out_vid_path)
        print("%d Frames to Compress"%len(labeled_video_frames))
        progress = progressbar.ProgressBar(widgets=[progressbar.Bar('=', '[', ']'), ' ',progressbar.Percentage(), ' ',progressbar.ETA()])
        for i in progress(range(0,len(labeled_video_frames))):
            if Utils_Image.check_image_with_pil(labeled_video_frames[i]):
                out.write(img)
                img = cv2.imread(labeled_video_frames[i], True)
        out.release()
        print("Finished Making File Video:%s " % out_vid_path)


def make_video_from_list(out_vid_path, frames_list):
    if frames_list[0] is not None:
        img = cv2.imread(frames_list[0], True)
        print frames_list[0]
        h, w = img.shape[:2]
        #fourcc = cv2.cv.CV_FOURCC('m', 'p', '4', 'v')
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        out = cv2.VideoWriter(out_vid_path,fourcc, 20.0, (w, h), True)
        print("Start Making File Video:%s " % out_vid_path)
        print("%d Frames to Compress"%len(frames_list))
        progress = progressbar.ProgressBar(widgets=[progressbar.Bar('=', '[', ']'), ' ',progressbar.Percentage(), ' ',progressbar.ETA()])
        for i in progress(range(0,len(frames_list))):
            if Utils_Image.check_image_with_pil(frames_list[i]):
                out.write(img)
                img = cv2.imread(frames_list[i], True)
        out.release()
        print("Finished Making File Video:%s " % out_vid_path)


def make_video_from_frames(out_vid_path, frames):
    if frames[0] is not None:
        h, w = frames[0].shape[:2]
        fourcc = cv2.FOURCC('m', 'p', '4', 'v')
        out = cv2.VideoWriter(out_vid_path,fourcc, 20.0, (w, h), True)
        print("Start Making File Video:%s " % out_vid_path)
        print("%d Frames to Compress"%len(frames))
        progress = progressbar.ProgressBar(widgets=[progressbar.Bar('=', '[', ']'), ' ',progressbar.Percentage(), ' ',progressbar.ETA()])
        for i in progress(range(0,len(frames))):
            out.write(frames[i])
        out.release()
        print("Finished Making File Video:%s " % out_vid_path)


####### FOR TENSORBOX ###########

def extract_idl_from_frames(vid_path, video_perc, path_video_folder, folder_path_frames, idl_filename):
    
    ####### Creating Folder for the video frames and the idl file for the list
    
    if not os.path.exists(path_video_folder):
        os.makedirs(path_video_folder)
        print("Created Folder: %s"%path_video_folder)
    if not os.path.exists(path_video_folder+'/'+folder_path_frames):
        os.makedirs(path_video_folder+'/'+folder_path_frames)
        print("Created Folder: %s"% (path_video_folder+'/'+folder_path_frames))
    if not os.path.exists(idl_filename):
        open(idl_filename, 'a')
        print "Created File: "+ idl_filename
    list=[]
    # Opening & Reading the Video

    print("Opening File Video:%s " % vid_path)
    vidcap = cv2.VideoCapture(vid_path)
    if not vidcap.isOpened():
        print "could Not Open :",vid_path
        return
    print("Opened File Video:%s " % vid_path)
    print("Start Reading File Video:%s " % vid_path)
    
    total = int((vidcap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)/100)*video_perc)
    
    print("%d Frames to Read"%total)
    progress = progressbar.ProgressBar(widgets=[progressbar.Bar('=', '[', ']'), ' ',progressbar.Percentage(), ' ',progressbar.ETA()])
    image = vidcap.read()
    with open(idl_filename, 'w') as f:
        for i in progress(range(0,total)):
            #frame_name="%s/%s/fram%d.jpeg"%(path_video_folder,folder_path_frames,i)
            list.append("%s/%sframe%d.jpeg"%(path_video_folder,folder_path_frames,i))
            cv2.imwrite("%s/%sframe%d.jpeg"%(path_video_folder,folder_path_frames,i), image[1])     # save frame as JPEG file
            image = vidcap.read()

    print("Finish Reading File Video:%s " % vid_path)
    return list

#def extract_frames_incten(vid_path, video_perc, path_video_folder, idl_filename):
def extract_frames_incten(input_dir,video_perc, idl_filename):
    
    ####### Creating Folder for the video frames and the idl file for the list
    
    ## if not os.path.exists(path_video_folder):
    ##     os.makedirs(path_video_folder)
    ##     print("Created Folder: %s"%path_video_folder)
    ## if not os.path.exists(path_video_folder+'/frames_tensorbox/'):
    ##     os.makedirs(path_video_folder+'/frames_tensorbox/')
    ##     print("Created Folder: %s"% (path_video_folder+'/frames_tensorbox/'))
    ## if not os.path.exists(path_video_folder+'/frames_inception/'):
    ##     os.makedirs(path_video_folder+'/frames_inception/')
    ##     print("Created Folder: %s"% (path_video_folder+'/frames_inception/'))
    if not os.path.exists(idl_filename):
        open(idl_filename, 'a')
        print "Created File: "+ idl_filename
        
    list_tensorbox=[]
    list_inception=[]
    # Opening & Reading the Video

    ## print("Opening File Video:%s " % vid_path)
    ## vidcap = cv2.VideoCapture(vid_path)
    ## if not vidcap.isOpened():
    ##     print "could Not Open :",vid_path
    ##     return
    ## print("Opened File Video:%s " % vid_path)
    ## print("Start Reading File Video:%s " % vid_path)
    
    # total = int((vidcap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)/100)*video_perc)
    # Fixed
    # 枚数数える
    image_dir = os.getcwd()# カレントディレクトリのパスを取得
    image_dir = image_dir + "/" + input_dir +"/frames_tensorbox"
     
    files = os.listdir(image_dir)# ファイルのリストを取得
    count = 0# カウンタの初期化
    for file in files:# ファイルの数だけループ
        index = re.search('.jpg', file)# 拡張子がjpgのものを検出
        if index:# jpgの時だけ（今回の場合は）カウンタをカウントアップ
            count = count + 1
     
    print(count)# ファイル数の表示

    total = int((count/100)*video_perc)
    
    print("%d Frames to Read"%total)
    progress = progressbar.ProgressBar(widgets=[progressbar.Bar('=', '[', ']'), ' ',progressbar.Percentage(), ' ',progressbar.ETA()])
    #image = vidcap.read()
    with open(idl_filename, 'w') as f:
        for i in progress(range(1,total+1)):
            #frame_name="%s/%s/fram%d.jpeg"%(path_video_folder,folder_path_frames,i)
            list_tensorbox.append("%s/%s%06d.jpg"%(input_dir,"frames_tensorbox/",i))
            #cv2.imwrite("%s/%sframe%d.jpeg"%(path_video_folder,"frames_tensorbox/",i), image[1])     # save frame as JPEG file
            list_inception.append("%s/%s%06d.jpg"%(input_dir,"frames_inception/",i))
            #cv2.imwrite("%s/%sframe%d.jpeg"%(path_video_folder,"frames_inception/",i), image[1])     # save frame as JPEG file
            #image = vidcap.read()

    #print("Finish Reading File Video:%s " % vid_path)
    return list_tensorbox, list_inception


### Function to track objects and spread informations between frames

def recurrent_track_objects(video_info):

    previous_frame= None
    previous_num_obj=-1

    tracked_video=[]
    deltas_video=[]
    deltas_frame=[]
    dx1,dx2,dy1,dy2=0,0,0,0

    # Add
    trackID=1
    #Add
    tmp_trackID = -1

    file = open("hikaku.txt",'a')
# フレームごとの処理
    for frame_info in video_info:
        print "Tracking Frame Nr: %d"%frame_info.frame # correct
        print "Len Rects Frame: %d"%len(frame_info.rects)
        current_frame = frame.Frame_Info()
        # インスタンスコピー
        current_frame=frame_info.duplicate()
        # 現在のフレームrectsは初期化
        current_frame.rects=[]
        # Add
        # poped_rects = []

        # 2フレーム目以降
        if previous_frame is not None:
            deltas_frame=[]
            ## if frame_info.frame>1:
            if frame_info.frame>0:
                print "Len Previous Rects Frame: %d"%len(previous_frame.rects)
                rect_idx=0 # < tmp_trackID
                    
                for rect in previous_frame.rects:
                    # Add
                    print("TOP::rect.trackID:{0},tmp_trackID:{1}".format(rect.trackID,tmp_trackID))
                    if rect.trackID >= tmp_trackID :
                    #for rect in previous_frame.rects:
                        print "Before"+str(len(current_frame.rects))
                        # IOU高いやつ取り出す frame_info.rectsはlen-1
                        current_rect = multiclass_rectangle.pop_max_iou(frame_info.rects,rect)
                        # Add
                        if current_rect is not None:
                            dx1=current_rect.x1-rect.x1
                            dx2=current_rect.x2-rect.x2
                            dy1=current_rect.y1-rect.y1
                            dy2=current_rect.y2-rect.y2
                            deltas_frame.append((dx1,dx2,dy1,dy2))
                            current_rect.load_trackID(rect.trackID)
                            current_frame.append_labeled_rect(current_rect)
                        else: break
                        deltas_frame.append((dx1,dx2,dy1,dy2))
                        continue

                    print('curent_rect:'+str(len(current_frame.rects)))
                    # previousのrectに差分タス
                    # (dx1,dx2,dy1,dy2)
                    ## rectそのまま書き換えるのはダメか
                    ## 一つ前のrectは書き換えても問題なしか
                    pred_rect = rect.duplicate()
                    pred_rect.add_delta(deltas_video[frame_info.frame-2][rect_idx][0],deltas_video[frame_info.frame-2][rect_idx][1],deltas_video[frame_info.frame-2][rect_idx][2],deltas_video[frame_info.frame-2][rect_idx][3])

                    # max_iou rect
                    current_rect = multiclass_rectangle.pop_max_iou(frame_info.rects,pred_rect)
                    #poped_rects.append(current_rect)

                    if current_rect is not None:
                        # 以前のフレームのrectを代入
                        current_rect.load_trackID(rect.trackID)

                        # current rectとprevious rectの差が閾値以上
                        # current rectを描写し,current_rectの値を更新
                        # print(str(frame_info.filename)) # 12dpm_libx264/frames_tensorbox/frame~~
                        # 上記画像に一色で描写

                        ## ここ考える
                        ## 1,2つ前dx,dyに1.2かけたものより、1つ前と現在の移動距離のほうが大きかったら、一つ前の座標に1つ前と現在の差を足して現在の座標を更新
                        #current_rect.check_rects_motion(frame_info.filename, rect, deltas_video[frame_info.frame-2][rect_idx][0],deltas_video[frame_info.frame-2][rect_idx][1],deltas_video[frame_info.frame-2][rect_idx][2],deltas_video[frame_info.frame-2][rect_idx][3])
                        # current_rectをcurrent_frameのrectsに追加
                        current_frame.append_labeled_rect(current_rect)
                        dx1=current_rect.x1-rect.x1
                        dx2=current_rect.x2-rect.x2
                        dy1=current_rect.y1-rect.y1
                        dy2=current_rect.y2-rect.y2

                        #print("x1,y1,x2,y2")
                        print("trackID:{0}".format(current_rect.trackID))
                        print("current_rect:" + str(current_rect.x1) + ',' +str(current_rect.y1) + ','+str(current_rect.x2) + ',' +str(current_rect.y2))
                        print("previous_rect:" + str(rect.x1) + ',' +str(rect.y1) + ','+str(rect.x2) + ',' +str(rect.y2) + '\n')
                        file.write("current_rect:" + str(current_rect.x1) + ',' +str(current_rect.y1) + ','+str(current_rect.x2) + ',' +str(current_rect.y2) + ' ' +"previous_rect:" + str(rect.x1) + ',' +str(rect.y1) + ','+str(rect.x2) + ',' +str(rect.y2) + '\n')
                        # 枠外に行ったら消す      
                        if current_rect.x1 < 0 or current_rect.y1 < 0 or current_rect.x2 < 0 or current_rect.y2 < 0 :
                            current_rect.load_trackID(-1)

                        deltas_frame.append((dx1,dx2,dy1,dy2))
                    else: break
                    rect_idx += 1
                
                # add 
                # 新たに出現した人を検出
                ## picked_rects=Utils_Tensorbox.FixedNMS(frame_info.rects,poped_rects)
                ## picked_rects=Utils_Tensorbox.NMS(picked_rects)
                ## Utils_Tensorbox.rectsToText(picked_rects,'picked_rects'+str(frame_info.frame))
                #print("ADD:frame_info.rects:{0}".format(len(frame_info.rects)))
                tmp_trackID = trackID
                # この時点ではtmp_trackIDは使われてないIDの先頭

                print("tmp_trackID:{0}".format(tmp_trackID))
                # rect: Rectangle_Multiclass Object
                # bboxにIDを付加
                # for rect in picked_rects :       
                # この時点でrectsはpopされている
                for rect in frame_info.rects :       
                    # コピー
                    current_rect = rect.duplicate()
                    # rectにidを与える
                    current_rect.load_trackID(trackID)
                    # current_rectを追加
                    current_frame.append_labeled_rect(current_rect)
                    trackID=trackID+1
                print("new_trackID:{0}".format(trackID))
                # 次のループのCurrent Frame objとnew_trackID-1は同じ


            deltas_video.append(deltas_frame)
        # 1フレーム目    
        else:
            #trackID=1
            ## picked_rect=Utils_Tensorbox.NMS(frame_info.rects)
            # rect: Rectangle_Multiclass Object
            # bboxにIDを付加
            ##for rect in picked_rect:       
            for rect in frame_info.rects:       
                # コピー
                current_rect = rect.duplicate()
                # rectにidを与える
                current_rect.load_trackID(trackID)
                # current_rectを追加
                current_frame.append_labeled_rect(current_rect)
                trackID=trackID+1


        previous_frame=current_frame.duplicate()
        # rectsのみコピー 上のコピーではrectsはコピーしないので
        previous_frame.rects= multiclass_rectangle.duplicate_rects(current_frame.rects)

        print "Current Frame obj:%d"%len(current_frame.rects)
        tracked_video.insert(len(tracked_video), current_frame)
    file.close()
    return tracked_video

def track_objects(video_info):

    previous_frame= None
    previous_num_obj=-1

    tracked_video=[]


    for frame_info in video_info:
        print "Tracking Frame Nr: %d"%frame_info.frame
        print "Len Rects Frame: %d"%len(frame_info.rects)
        current_frame = frame.Frame_Info()
        current_frame=frame_info.duplicate()
        current_frame.rects=[]
        if previous_frame is not None:
            print "Len Previous Rects Frame: %d"%len(previous_frame.rects)
            for rect in previous_frame.rects:
                print len(current_frame.rects)
                current_rect = multiclass_rectangle.pop_max_iou(frame_info.rects,rect)
                current_rect.load_trackID(rect.trackID)
                current_frame.append_labeled_rect(current_rect)
        else:
            trackID=1
            picked_rect=Utils_Tensorbox.NMS(frame_info.rects)
            for rect in picked_rect:       
                current_rect = rect.duplicate()
                current_rect.load_trackID(trackID)
                current_frame.append_labeled_rect(current_rect)
                trackID=trackID+1

        previous_frame=current_frame.duplicate()
        previous_frame.rects= multiclass_rectangle.duplicate_rects(current_frame.rects)

        print "Current Frame obj:%d"%len(current_frame.rects)
        tracked_video.insert(len(tracked_video), current_frame)

    return tracked_video

def track_min_objects(video_info):

    previous_frame= None
    previous_num_obj=-1

    tracked_video=[]

    frame_id=0
    min_rects=[]
    min_frame_id=None
    min_num_obj=None
    for frame_info in video_info:
        if (min_num_obj is None) & (len(frame_info.rects) >0):
            min_num_obj = len(frame_info.rects)
            min_frame_id=frame_id
        if (len(frame_info.rects) < min_num_obj ) & (len(frame_info.rects) >0):
            min_num_obj = len(frame_info.rects)
            min_frame_id=frame_id
        frame_id=frame_id+1
    min_rects = multiclass_rectangle.duplicate_rects(video_info[min_frame_id].rects)
    print "Min num object video:%d"%min_num_obj

    for frame_info in video_info:
        print "Tracking Frame Nr: %d"%frame_info.frame
        print "Len Rects Frame: %d"%len(frame_info.rects)
        current_frame = frame.Frame_Info()
        current_frame=frame_info.duplicate()
        current_frame.rects=[]
        if previous_frame is not None:
            print "Min num object video:%d"%min_num_obj
            print "Len Previous Rects Frame: %d"%len(previous_frame.rects)
            for rect in previous_frame.rects:
                print len(current_frame.rects)
                if len(current_frame.rects)<=min_num_obj:               
                    current_rect = multiclass_rectangle.pop_max_iou(frame_info.rects,rect)
                    current_rect.load_trackID(rect.trackID)
                    current_frame.append_labeled_rect(current_rect)
        else:
            trackID=1
            for rect in min_rects:
                if len(current_frame.rects)<min_num_obj:               
                    current_rect = multiclass_rectangle.pop_max_iou(frame_info.rects,rect)
                    current_rect.load_trackID(trackID)
                    current_frame.append_labeled_rect(current_rect)
                    trackID=trackID+1

        previous_frame=current_frame.duplicate()
        previous_frame.rects= multiclass_rectangle.duplicate_rects(current_frame.rects)

        print "Current Frame obj:%d"%len(current_frame.rects)
        tracked_video.insert(len(tracked_video), current_frame)

    return tracked_video


def track_and_label_objects(video_info):

    previous_frame= None
    previous_num_obj=-1

    cropped_img_array=[]
    tracked_video=[]

    for frame_info in video_info:
        print "Tracking Frame Nr: %d"%frame_info.frame
        print len(frame_info.rects)
        current_frame = frame.Frame_Info()
        current_frame=frame_info.duplicate()
        current_frame.rects=[]
        print len(frame_info.rects)
        if previous_frame is not None:
            print "Previous Frame obj:%d"%previous_num_obj
            for rect in frame_info.rects:
                print "Entered into the rect check"
                max_rect=None
                max_iou=0
                current_rect= Rectangle_Multiclass()
                trackID=-1
                if previous_num_obj >0: ### If i come here means that there's the same number of object between the previous and the current frame
                    print "Entered into the rect check with :%d objects"%previous_num_obj
                    id_rect=0
                    max_id=0
                    for prev_rect in previous_frame.rects:
                        print "Entered"
                        if rect.iou(prev_rect)>max_iou:
                            max_iou=rect.iou(prev_rect)
                            max_id=id_rect
                        id_rect=id_rect+1
                    print "Lenght previous rects array: %d"%len(previous_frame.rects)
                    print "max_rect track ID: %d"%previous_frame.rects[max_id].trackID
                    print "max_rect label: %s"%previous_frame.rects[max_id].label
                    current_rect.load_labeled_rect(previous_frame.rects[max_id].trackID, previous_frame.rects[max_id].confidence, previous_frame.rects[max_id].label_confidence, previous_frame.rects[max_id].x1,previous_frame.rects[max_id].y1,previous_frame.rects[max_id].x2 ,previous_frame.rects[max_id].y2, previous_frame.rects[max_id].label, previous_frame.rects[max_id].label_chall, previous_frame.rects[max_id].label_code)
                    current_frame.append_labeled_rect(current_rect)
                    rect.load_label(previous_frame.rects[max_id].trackID,previous_frame.rects[max_id].label_confidence, previous_frame.rects[max_id].label, previous_frame.rects[max_id].label_chall, previous_frame.rects[max_id].label_code)
                    previous_frame.rects.pop(max_id)
                    previous_num_obj=previous_num_obj-1
                else:
                    ### If i come here means that there's more objects in the current frame respect to che previous
                    if previous_num_obj == 0:
                        trackID = len(frame_info.rects)
                        previous_num_obj = -1
                    current_rect= Rectangle_Multiclass()

                    img= Image.open(frame_info.filename)
                    cor = (rect.x1,rect.y1,rect.x2 ,rect.y2)

                    cropped_img=img.crop(cor)
                    cropped_img_name="cropped_frame_%d.JPEG"%(frame_info.frame)
                    cropped_img.save(cropped_img_name)
                    cropped_img_array.append(cropped_img_name)

                    label, confidence = Utils_Imagenet.run_inception_once(cropped_img_name)
                    rect.load_label(trackID,confidence, vid_classes.code_to_class_string(label), vid_classes.code_to_code_chall(vid_classes), label)
                    current_rect.load_labeled_rect(trackID, rect.confidence, confidence, rect.x1,rect.y1,rect.x2 ,rect.y2, vid_classes.code_to_class_string(label), vid_classes.code_to_code_chall(vid_classes), label)
                    print "current_rect track ID: %d"%current_rect.trackID
                    print "current_rect label: %s"%current_rect.label
                    current_frame.append_labeled_rect(current_rect)
        else:
            trackID=1

            for rect in frame_info.rects:
                
                current_rect= Rectangle_Multiclass()

                img= Image.open(frame_info.filename)
                cor = (rect.x1,rect.y1,rect.x2 ,rect.y2)

                cropped_img=img.crop(cor)
                cropped_img_name="cropped_frame_%d.JPEG"%(frame_info.frame)
                cropped_img.save(cropped_img_name)
                cropped_img_array.append(cropped_img_name)

                label, confidence = Utils_Imagenet.run_inception_once(cropped_img_name)
                rect.load_label(trackID,confidence, vid_classes.code_to_class_string(label), vid_classes.code_to_code_chall(vid_classes), label)
                current_rect.load_labeled_rect(trackID, rect.confidence, confidence, rect.x1,rect.y1,rect.x2 ,rect.y2, vid_classes.code_to_class_string(label), vid_classes.code_to_code_chall(vid_classes), label)
                current_frame.append_labeled_rect(current_rect)
                
                trackID=trackID+1

        previous_num_obj=len(frame_info.rects)
        previous_frame=frame_info.duplicate()
        previous_frame.duplicate_rects(frame_info.rects)

        print previous_frame
        print "Previous Frame obj:%d"%previous_num_obj
        print "prev_rect 0 track ID: %d"%previous_frame.rects[0].trackID
        print "prev_rect 0 label: %s"%previous_frame.rects[0].label
        tracked_video.insert(len(tracked_video), current_frame)

    return tracked_video


####### FOR YOLO ###########

def extract_frames(vid_path, video_perc):
    list=[]
    frames=[]
    # Opening & Reading the Video
    print("Opening File Video:%s " % vid_path)
    vidcap = cv2.VideoCapture(vid_path) # 動画を扱うためのインスタンス生成
    if not vidcap.isOpened():
        print "could Not Open :",vid_path
        return
    print("Opened File Video:%s " % vid_path)
    print("Start Reading File Video:%s " % vid_path)
    image = vidcap.read() # Convert Image (tuple)
    #total = int((vidcap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)/100)*video_perc)
    total = int((vidcap.get(7)/100)*video_perc) # default 30
    print("%d Frames to Read"%total)
    progress = progressbar.ProgressBar(widgets=[progressbar.Bar('=', '[', ']'), ' ',progressbar.Percentage(), ' ',progressbar.ETA()])
    for i in progress(range(0,total)):
        list.append("frame%d.jpg" % i)
        frames.append(image)
        image = vidcap.read()
    print("Finish Reading File Video:%s " % vid_path)
    return frames, list # percのFrame imageとList(frame1,2....jpg) どちらもList
