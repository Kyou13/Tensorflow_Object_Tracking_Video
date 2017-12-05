# -*- coding: UTF-8 -*- 
import tensorflow as tf
import os
import json
import subprocess
from scipy.misc import imread, imresize
from scipy import misc

from train import build_forward
from utils.annolist import AnnotationLib as al
from utils.train_utils import add_rectangles, rescale_boxes

import cv2
import argparse

# add
import vid_classes
import frame
import multiclass_rectangle
import Utils_Image
import Utils_Video
import progressbar
import numpy as np
import time

# VID_tensorbox
import Utils_Imagenet

#import Utils_Tensorbox
# Utils_Tensorboxもここに記載

def get_image_dir(args):
    weights_iteration = int(args.weights.split('-')[-1])
    expname = '_' + args.expname if args.expname else ''
    image_dir = '%s/images_%s_%d%s' % (os.path.dirname(args.weights), os.path.basename(args.test_boxes)[:-5], weights_iteration, expname)
    return image_dir

## def get_results(args, H):
##     tf.reset_default_graph()
## 
##     #
##     H["grid_width"] = H["image_width"] / H["region_size"]
##     H["grid_height"] = H["image_height"] / H["region_size"]
##     x_in = tf.placeholder(tf.float32, name='x_in', shape=[H['image_height'], H['image_width'], 3])
##     if H['use_rezoom']:
##         pred_boxes, pred_logits, pred_confidences, pred_confs_deltas, pred_boxes_deltas = build_forward(H, tf.expand_dims(x_in, 0), 'test', reuse=None)
##         grid_area = H['grid_height'] * H['grid_width']
##         pred_confidences = tf.reshape(tf.nn.softmax(tf.reshape(pred_confs_deltas, [grid_area * H['rnn_len'], 2])), [grid_area, H['rnn_len'], 2])
##         if H['reregress']:
##             pred_boxes = pred_boxes + pred_boxes_deltas
##     else:
##         pred_boxes, pred_logits, pred_confidences = build_forward(H, tf.expand_dims(x_in, 0), 'test', reuse=None)
##     saver = tf.train.Saver()
## 
##     with tf.Session() as sess:
## 
##         sess.run(tf.global_variables_initializer())
##         saver.restore(sess, args.weights)
## # ここまで同じ
## 
##         pred_annolist = al.AnnoList()
## 
##         true_annolist = al.parse(args.test_boxes)
##         data_dir = os.path.dirname(args.test_boxes)
##         image_dir = get_image_dir(args)
##         subprocess.call('mkdir -p %s' % image_dir, shell=True)
##         for i in range(len(true_annolist)):
##             true_anno = true_annolist[i]
##             orig_img = imread('%s/%s' % (data_dir, true_anno.imageName))[:,:,:3]
##             img = imresize(orig_img, (H["image_height"], H["image_width"]), interp='cubic')
##             feed = {x_in: img}
##             (np_pred_boxes, np_pred_confidences) = sess.run([pred_boxes, pred_confidences], feed_dict=feed)
##             pred_anno = al.Annotation()
##             pred_anno.imageName = true_anno.imageName
##             new_img, rects = add_rectangles(H, [img], np_pred_confidences, np_pred_boxes,
##                                             use_stitching=True, rnn_len=H['rnn_len'], min_conf=args.min_conf, tau=args.tau, show_suppressed=args.show_suppressed)
##         
##             pred_anno.rects = rects
##             pred_anno.imagePath = os.path.abspath(data_dir)
##             pred_anno = rescale_boxes((H["image_height"], H["image_width"]), pred_anno, orig_img.shape[0], orig_img.shape[1])
import Utils_Imagenet

##             pred_annolist.append(pred_anno)
##                 
##             imname = '%s/%s' % (image_dir, os.path.basename(true_anno.imageName))
##             misc.imsave(imname, new_img)
##             if i % 25 == 0:
##                 print(i)
##     return pred_annolist, true_annolist
def NMS(rects,overlapThresh=0.3):
    # if there are no boxes, return an empty list
    if len(rects) == 0:
        print "WARNING: Passed Empty Boxes Array"
        return []
 
    # initialize the list of picked indexes
    pick = []
    x1, x2, y1, y2, conf=[],[],[],[], []
    for rect in rects:
        x1.append(rect.x1)
        x2.append(rect.x2)
        y1.append(rect.y1)
        y2.append(rect.y2)
        conf.append(rect.confidence)
    # grab the coordinates of the bounding boxes
    x1 = np.array(x1)
    y1 = np.array(y1)
    x2 = np.array(x2)
    y2 = np.array(y2)
    conf = np.array(conf)
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(conf)
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list, add the index
        # value to the list of picked indexes, then initialize
        # the suppression list (i.e. indexes that will be deleted)
        # using the last index
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        suppress = [last]
        # loop over all indexes in the indexes list
        for pos in xrange(0, last):
            # grab the current index
            j = idxs[pos]

            # find the largest (x, y) coordinates for the start of
            # the bounding box and the smallest (x, y) coordinates
            # for the end of the bounding box
            xx1 = max(x1[i], x1[j])
            yy1 = max(y1[i], y1[j])
            xx2 = min(x2[i], x2[j])
            yy2 = min(y2[i], y2[j])

            # compute the width and height of the bounding box
            w = max(0, xx2 - xx1 + 1)
            h = max(0, yy2 - yy1 + 1)
 
            # compute the ratio of overlap between the computed
            # bounding box and the bounding box in the area list
            overlap = float(w * h) / area[j]
            # union = area[j] + float(w * h) - overlap

            # iou = overlap/union

            # if there is sufficient overlap, suppress the
            # current bounding box
            if (overlap > overlapThresh):
                suppress.append(pos)
 
        # delete all indexes from the index list that are in the
        # suppression list
        idxs = np.delete(idxs, suppress)
 
    # return only the bounding boxes that were picked
    picked =[]
    for i in pick: picked.append(rects[i])
    return picked

def getTextIDL(annotations):

	frame = -1
	conf=0
	silhouette=-1
	xmin,ymin,xmax,ymax=0,0,0,0

	detections_array=[]

	if annotations.frameNr is not -1:
		frame=annotations.frameNr
	for rect in annotations.rects:
		if rect.silhouetteID is not -1:
			silhouette=rect.silhouetteID
		conf = rect.score
		xmin,ymin,xmax,ymax = rect.x1,rect.y1,rect.x2 ,rect.y2
		detections_array.append(str(frame)+' '+str(silhouette)+' '+str(conf)+' '+str(xmin)+' '+str(ymin)+' '+str(xmax)+' '+str(ymax))
	return detections_array

def writeText(annotations, file):
	detections= getTextIDL(annotations)
	for detection in detections:
		file.write(detection + os.linesep)

def saveTextResults(filename, annotations):
    if not os.path.exists(filename):
        print "Created File: "+ filename
    file = open(filename, 'w')
    for annotation in annotations:
        writeText(annotation,file)
    file.close()

def get_silhouette_confidence(silhouettes_confidence):
    higher=0.0
    index=0
    # print "conf_sil : " + str(silhouettes_confidence)
    # print "conf_sil LEN : " + str(len(silhouettes_confidence))

    for i in range(0,len(silhouettes_confidence)):
        # print "conf_sil I : " + str(silhouettes_confidence[i])
        if silhouettes_confidence[i]>higher:
            higher = silhouettes_confidence[i]
            index = i
    # print str(index+1),str(higher)
    return index+1 , higher

def get_higher_confidence(rectangles):
    higher=0.0
    index=0
    # print "conf_sil : " + str(silhouettes_confidence)
    # print "conf_sil LEN : " + str(len(silhouettes_confidence))

    for rect in rectangles:
        # print "conf_sil I : " + str(silhouettes_confidence[i])
        if rect.confidence>higher:
            higher = rect.confidence
    # print str(index+1),str(higher)
    # print "higher: %.2f"%higher
    higher=higher*10
    # print "higher: %.1f"%higher
    higher=int(higher)
    # print "higher: %.d"%higher
    higher=float(higher)/10.0
    # print "rounded max: %.1f"%(higher)
    if(higher>0.5):
        return  higher-0.3
    if(higher<0.3):
        return  higher-0.1
    else: return  higher-0.2

def get_multiclass_rectangles(H, confidences, boxes, rnn_len):
    boxes_r = np.reshape(boxes, (-1,
                                 H["grid_height"],
                                 H["grid_width"],
                                 rnn_len,
                                 4))
    # reshape 後ろから分割
    confidences_r = np.reshape(confidences, (-1,
                                             H["grid_height"],
                                             H["grid_width"],
                                             rnn_len,
                                             H['num_classes']))
    # print "boxes_r shape" + str(boxes_r.shape)
    # print "confidences" + str(confidences.shape)
    cell_pix_size = H['region_size']
    all_rects = [[[] for _ in range(H["grid_width"])] for _ in range(H["grid_height"])]
    for n in range(rnn_len):
        for y in range(H["grid_height"]):
            for x in range(H["grid_width"]):
                bbox = boxes_r[0, y, x, n, :]
                abs_cx = int(bbox[0]) + cell_pix_size/2 + cell_pix_size * x
                abs_cy = int(bbox[1]) + cell_pix_size/2 + cell_pix_size * y
                w = bbox[2]
                h = bbox[3]
                # conf = np.max(confidences_r[0, y, x, n, 1:])
                # max を取得
                index, conf = get_silhouette_confidence(confidences_r[0, y, x, n, 1:])
                # print index, conf
                # print np.max(confidences_r[0, y, x, n, 1:])
                # print "conf" + str(conf)
                # print "conf" + str(confidences_r[0, y, x, n, 1:])
                new_rect=multiclass_rectangle.Rectangle_Multiclass()
                new_rect.set_unlabeled_rect(abs_cx,abs_cy,w,h,conf)
                all_rects[y][x].append(new_rect)

    # print "confidences_r" + str(confidences_r.shape)
    # all_rects_r: [r,]のリスト作成 各要素はRectangle_multiclass
    # Rectangle_Multiclass:
    #   cx, cy, width, height, confidence, x1, x2, y1, y2
    all_rects_r = [r for row in all_rects for cell in row for r in cell]
    #print("len(all_rects):{0}".format(len(all_rects_r)))
    # confidencesの値によって引く
    #min_conf = get_higher_confidence(all_rects_r)
    min_conf = 0.2
    # 一定のconfidencesを超えるものを代入
    # tensorboxの赤枠にあたるもの？
    # add 2 sentence
    from origin_utils.stitch_wrapper import stitch_rects
    acc_rects = stitch_rects(all_rects,0.25)
    #Rect to multiclass_rectangle
    acc_rects=[rect for rect in acc_rects if rect.confidence > min_conf]
    print("first_len(acc_rects):{0}".format(len(acc_rects)))
    #acc_rects=[rect for rect in all_rects_r if rect.confidence>min_conf]
#    rects = []
#    #for rect in all_rects_r:
#    # add
#    for rect in acc_rects:
#    	if rect.confidence>min_conf:
#	        r = al.AnnoRect()
#	        r.x1 = rect.cx - rect.width/2.
#	        r.x2 = rect.cx + rect.width/2.
#	        r.y1 = rect.cy - rect.height/2.
#	        r.y2 = rect.cy + rect.height/2.
#	        r.score = rect.confidence
#                # label は "Not Set"
#	        r.silhouetteID=rect.label
#	        rects.append(r)
#    print len(rects),len(acc_rects)
    print len(acc_rects)
    #return rects, acc_rects
    return acc_rects

def bbox_det_TENSORBOX_multiclass(frames_list,H,args,pred_idl):
    # add
    video_info = []
    
    tf.reset_default_graph()

    H["grid_width"] = H["image_width"] / H["region_size"]
    H["grid_height"] = H["image_height"] / H["region_size"]
    x_in = tf.placeholder(tf.float32, name='x_in', shape=[H['image_height'], H['image_width'], 3])
    if H['use_rezoom']:
        pred_boxes, pred_logits, pred_confidences, pred_confs_deltas, pred_boxes_deltas = build_forward(H, tf.expand_dims(x_in, 0), 'test', reuse=None)
        grid_area = H['grid_height'] * H['grid_width']
        pred_confidences = tf.reshape(tf.nn.softmax(tf.reshape(pred_confs_deltas, [grid_area * H['rnn_len'], 2])), [grid_area, H['rnn_len'], 2])
        # add
        # pred_logits = tf.reshape(tf.nn.softmax(tf.reshape(pred_logits, [grid_area * H['rnn_len'], 2])), [grid_area, H['rnn_len'], 2])
        if H['reregress']:
            pred_boxes = pred_boxes + pred_boxes_deltas
    else:
        pred_boxes, pred_logits, pred_confidences = build_forward(H, tf.expand_dims(x_in, 0), 'test', reuse=None)
    saver = tf.train.Saver()

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        saver.restore(sess, args.weights)
# ここまで同じ
        print("%d Frames to DET"%len(frames_list))
        progress = progressbar.ProgressBar(widgets=[progressbar.Bar('=', '[', ']'), ' ',progressbar.Percentage(), ' ',progressbar.ETA()])
        frameNr=0
        skipped=0

        ## pred_annolist = al.AnnoList()

        ## true_annolist = al.parse(args.test_boxes)
        ## data_dir = os.path.dirname(args.test_boxes)
        ## image_dir = get_image_dir(args)
        ## subprocess.call('mkdir -p %s' % image_dir, shell=True)
        ## for i in range(len(true_annolist)):
        for i in progress(range(0, len(frames_list))):
            current_frame = frame.Frame_Info()
            current_frame.frame=frameNr
            current_frame.filename=frames_list[i]

            if Utils_Image.isnotBlack(frames_list[i]) & Utils_Image.check_image_with_pil(frames_list[i]):

                ## true_anno = true_annolist[i]
                ## orig_img = imread('%s/%s' % (data_dir, true_anno.imageName))[:,:,:3]
                ## img = imresize(orig_img, (H["image_height"], H["image_width"]), interp='cubic')
                orig_img  = imread(frames_list[i])
                img = imresize(orig_img, (H["image_height"], H["image_width"]), interp='cubic')

                feed = {x_in: img}
                # (np_pred_boxes, np_pred_confidences) = sess.run([pred_boxes, pred_confidences], feed_dict=feed)
                (np_pred_boxes, np_pred_confidences) = sess.run([pred_boxes, pred_confidences], feed_dict=feed)
                # fix
                #_,rects = get_multiclass_rectangles(H, np_pred_confidences, np_pred_boxes, rnn_len=H['rnn_len'])
                rects = get_multiclass_rectangles(H, np_pred_confidences, np_pred_boxes, rnn_len=H['rnn_len'])
                if len(rects)>0:
                    # pick = NMS(rects)
                    pick = rects
                    # Fixed
                    #print len(rects),len(pick)
                    current_frame.rects=pick
                    frameNr=frameNr+1
                    video_info.insert(len(video_info), current_frame)
                    print ("current_frame_rects_length:"+str(len(current_frame.rects)))
                else: skipped=skipped+1 
            else: skipped=skipped+1 

        print("Skipped %d Black Frames"%skipped)
    ##         pred_anno = al.Annotation()
    ##         pred_anno.imageName = true_anno.imageName
    ##         new_img, rects = add_rectangles(H, [img], np_pred_confidences, np_pred_boxes,
    ##                                         use_stitching=True, rnn_len=H['rnn_len'], min_conf=args.min_conf, tau=args.tau, show_suppressed=args.show_suppressed)
    ##     
    ##         pred_anno.rects = rects
    ##         pred_anno.imagePath = os.path.abspath(data_dir)
    ##         pred_anno = rescale_boxes((H["image_height"], H["image_width"]), pred_anno, orig_img.shape[0], orig_img.shape[1])
    ##         pred_annolist.append(pred_anno)
    ##             
    ##         imname = '%s/%s' % (image_dir, os.path.basename(true_anno.imageName))
    ##         misc.imsave(imname, new_img)
    ##         if i % 25 == 0:
    ##             print(i)
    ## return pred_annolist, true_annolist

    #### END TENSORBOX CODE ###

    return video_info


def main():
    # add
    start = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', required=True) #True
    parser.add_argument('--expname', default='')
    #parser.add_argument('--test_boxes', required=True) #True
    parser.add_argument('--gpu', default=0)
    parser.add_argument('--logdir', default='output')
    parser.add_argument('--iou_threshold', default=0.5, type=float)
    parser.add_argument('--tau', default=0.25, type=float)
    parser.add_argument('--min_conf', default=0.2, type=float)
    parser.add_argument('--show_suppressed', default=True, type=bool)
    parser.add_argument('--path_video', default='12dpm_libx264.mp4', type=str) # add
    parser.add_argument('--perc',default=100,type=int)
    parser.add_argument('--output_name', default='output.mp4', type=str)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    hypes_file = '%s/hypes.json' % os.path.dirname(args.weights)
    with open(hypes_file, 'r') as f:
        H = json.load(f)
    expname = args.expname + '_' if args.expname else ''
    # pred_boxes = '%s.%s%s' % (args.weights, expname, os.path.basename(args.test_boxes))
    # true_boxes = '%s.gt_%s%s' % (args.weights, expname, os.path.basename(args.test_boxes))
    
    # add
    path_video_folder = os.path.splitext(os.path.basename(args.path_video))[0]
    pred_idl = './%s/%s_val.idl' % (path_video_folder, path_video_folder)
    idl_filename=path_video_folder+'/'+path_video_folder+'.idl'
    frame_tensorbox = []
    frame_inception = []
    frame_tensorbox, frame_inception = Utils_Video.extract_frames_incten(args.path_video, args.perc, path_video_folder, idl_filename)
    progress = progressbar.ProgressBar(widgets=[progressbar.Bar('=', '[', ']'), ' ',progressbar.Percentage(), ' ',progressbar.ETA()])
    #for image_path in progress(frame_tensorbox):
    #    Utils_Image.resizeImage(image_path)
    #Utils_Image.resizeImage(-1)
    # add
    video_info=bbox_det_TENSORBOX_multiclass(frame_tensorbox, H, args, pred_idl)

    # Trackking
    tracked_video=Utils_Video.recurrent_track_objects(video_info)
    # tracked_video=utils_video.track_objects(video_info)
    # labeled_video=Utils_Imagenet.label_video(tracked_video, frame_inception)
    labeled_video=Utils_Imagenet.recurrent_label_video(tracked_video, frame_inception)
    # tracked_video=utils_video.track_objects(video_info)

    # tracked_video=utils_video.track_and_label_objects(video_info)
    labeled_frames=Utils_Video.draw_rectangles(path_video_folder, labeled_video)
    Utils_Video.make_tracked_video(args.output_name, labeled_frames)
    frame.saveVideoResults(idl_filename,labeled_video)

    # utils_video.make_tracked_video(args.output_name, labeled_video)
    end = time.time()

    print("Elapsed Time:%d Seconds"%(end-start))
    print("Running Completed with Success!!!")

    #pred_annolist, true_annolist = get_results(args, H)
    #pred_annolist.save(pred_boxes)
    #true_annolist.save(true_boxes)

    
    #try:
    #    rpc_cmd = './utils/annolist/doRPC.py --minOverlap %f %s %s' % (args.iou_threshold, true_boxes, pred_boxes)
    #    print('$ %s' % rpc_cmd)
    #    rpc_output = subprocess.check_output(rpc_cmd, shell=True)
    #    print(rpc_output)
    #    txt_file = [line for line in rpc_output.split('\n') if line.strip()][-1]
    #    output_png = '%s/results.png' % get_image_dir(args)
    #    plot_cmd = './utils/annolist/plotSimple.py %s --output %s' % (txt_file, output_png)
    #    print('$ %s' % plot_cmd)
    #    plot_output = subprocess.check_output(plot_cmd, shell=True)
    #    print('output results at: %s' % plot_output)
    #except Exception as e:
    #    print(e)

if __name__ == '__main__':
    main()
