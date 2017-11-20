# -*- coding: UTF-8 -*-
#### My import

import argparse
import Utils_Image
import Utils_Video
import Utils_Tensorbox
import Utils_Imagenet
import frame
import vid_classes
import progressbar
import time
import os

######### MAIN ###############

def main():
    '''
    Parse command line arguments and execute the code 

    '''

    ######### TENSORBOX PARAMETERS


    start = time.time()

    parser = argparse.ArgumentParser()
    # parser.add_argument('--result_folder', default='summary_result/', type=str)
    # parser.add_argument('--summary_file', default='results.txt', type=str)
    parser.add_argument('--output_name', default='output.mp4', type=str)
    parser.add_argument('--hypes', default='./TENSORBOX/hypes/overfeat_rezoom.json', type=str)
    parser.add_argument('--weights', default='./TENSORBOX/data/save.ckpt-1250000', type=str)
    parser.add_argument('--perc', default=100, type=int)
    parser.add_argument('--path_video', default='ILSVRC2015_val_00004000.mp4', type=str)# required=True, type=str)

    args = parser.parse_args()

    # hypes_file = './hypes/overfeat_rezoom.json'
    # weights_file= './output/save.ckpt-1090000'

    path_video_folder = os.path.splitext(os.path.basename(args.path_video))[0]
    pred_idl = './%s/%s_val.idl' % (path_video_folder, path_video_folder)
    idl_filename=path_video_folder+'/'+path_video_folder+'.idl'
    frame_tensorbox=[]
    frame_inception=[]
    # 画像生成 tensorbox(640*480)とInception(original)
    frame_tensorbox, frame_inception = Utils_Video.extract_frames_incten(args.path_video, args.perc, path_video_folder, idl_filename )

    progress = progressbar.ProgressBar(widgets=[progressbar.Bar('=', '[', ']'), ' ',progressbar.Percentage(), ' ',progressbar.ETA()])

    # ReSize tensorbox(640*480)
    for image_path in progress(frame_tensorbox):
        Utils_Image.resizeImage(image_path)
    Utils_Image.resizeImage(-1)
# ----------------------------------------------------------------------#
    video_info=Utils_Tensorbox.bbox_det_TENSORBOX_multiclass(frame_tensorbox, path_video_folder, args.hypes, args.weights, pred_idl)
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

if __name__ == '__main__':
    main()

