def recurrent_track_objects(video_info):

    previous_frame= None
    previous_num_obj=-1

    tracked_video=[]
    deltas_video=[]
    deltas_frame=[]
    dx1,dx2,dy1,dy2=0,0,0,0

# フレームごとの処理
    for frame_info in video_info:
        print "Tracking Frame Nr: %d"%frame_info.frame # correct
        print "Len Rects Frame: %d"%len(frame_info.rects)
        current_frame = frame.Frame_Info()
        # インスタンスコピー
        current_frame=frame_info.duplicate()
        # 現在のフレームrectsは初期化
        current_frame.rects=[]
        trackID=1

        # 2フレーム目以降
        if previous_frame is not None:
            deltas_frame=[]
            if frame_info.frame>1:
                print "Len Previous Rects Frame: %d"%len(previous_frame.rects)
                rect_idx=0
                for rect in previous_frame.rects:
                    print len(current_frame.rects)
                    # ????
                    rect.add_delta(deltas_video[frame_info.frame-2][rect_idx][0],deltas_video[frame_info.frame-2][rect_idx][1],deltas_video[frame_info.frame-2][rect_idx][2],deltas_video[frame_info.frame-2][rect_idx][3])
                    # max_iou rect
                    current_rect = multiclass_rectangle.pop_max_iou(frame_info.rects,rect)
                    if current_rect is not None:
                        # 以前のフレームのrectを代入
                        current_rect.load_trackID(rect.trackID)
                        # current rectとprevious rectの差が閾値以上
                        # current rectを描写し,current_rectの値を更新
                        ## ここも書き換えるひつようがある
                        ## 離れすぎてたらID消す
                        ## current_rectが枠外に出たら IDけす
                        current_rect.check_rects_motion(frame_info.filename, rect, deltas_video[frame_info.frame-2][rect_idx][0],deltas_video[frame_info.frame-2][rect_idx][1],deltas_video[frame_info.frame-2][rect_idx][2],deltas_video[frame_info.frame-2][rect_idx][3])
                        
                        # current_rectをcurrent_frameのrectsに追加
                        current_frame.append_labeled_rect(current_rect)
                        dx1=current_rect.x1-rect.x1
                        dx2=current_rect.x2-rect.x2
                        dy1=current_rect.y1-rect.y1
                        dy2=current_rect.y2-rect.y2
                        # 追加
                        # 3フレーム目から値を代入
                        deltas_frame.append((dx1,dx2,dy1,dy2))
                    else: break
                    #rect += 1
                    # if previous_frame.rectsをこえたら 新しいIDわりあて
                    

            # 2フレーム目        
            else:
                print "Len Previous Rects Frame: %d"%len(previous_frame.rects)
                for rect in previous_frame.rects:
                    print "A"+str(len(current_frame.rects))
                    current_rect = multiclass_rectangle.pop_max_iou(frame_info.rects,rect)
                    if current_rect is not None:
                        dx1=current_rect.x1-rect.x1
                        dx2=current_rect.x2-rect.x2
                        dy1=current_rect.y1-rect.y1
                        dy2=current_rect.y2-rect.y2
                        deltas_frame.append((dx1,dx2,dy1,dy2))
                        current_rect.load_trackID(rect.trackID)
                        # current_rectを追加 initはzero
                        current_frame.append_labeled_rect(current_rect)
                    else: break
            deltas_video.append(deltas_frame)
        # 1フレーム目    
        else:
            picked_rect=Utils_Tensorbox.NMS(frame_info.rects)
            # rect: Rectangle_Multiclass Object
            # bboxにIDを付加
            for rect in picked_rect:       
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

    return tracked_video

