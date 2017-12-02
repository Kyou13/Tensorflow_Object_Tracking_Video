# Memo
- MainProject: VID_tensorbox

- Utils_Video.extract_frames_incten: フレームごとに画像生成

- Utils_Image.resizeImage: tensorbox用に画像をリサイズ

- Urils_Tensorbox.bbox_det_TENSORBOX_multiclass
  - In: (frame_tensorbox, parh_video_folder, argx.hypes, args.weights, pred_idl)
  - Return: video_info

  
# 詳細
- bbox_det_TENSORBOX_multiclass
  return Video_info(class) frame_infoを含むlist
  1. tensorbox 設定ファイル読み込み(hypes)
  2. TensorBoxモデル作成 output: pred_box, pred_logits, pred_confidence
  3. 
  
  - multiclass_rectangle
      MainFarse
  - get_multiclass_rectangles
    - 閾値以上のaccもつ矩形を検出
    - silhouetteIDとは

- Utils_Video.recurrent_track_object
  検出データをトラッキング
  1. Video_infoからframe_infoに分割

  2. frameごとの検出矩形割り出す
  - Urils_Tensorbox.NMS

## 出力フォーマット
- (Movie_name)/(Movie_name).idl
- frame, label, trackID, (conf)信頼性, xmin, ymin, xmax, ymax


TODO
frame Obj とRectangle_Multiclass Objについてやる
