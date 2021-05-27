import glob
from pathlib import Path
import cv2
from utils import local_maxima, optimum, remove_outside_plot
import os


def cal_tp_fp_fn(ori, gt_img, pre_img):
    # save_error_path = "/home/hyeonwoo/research/20201203/Image-level/step1/error/"
    dist_threshold = 10
    dist_peak = 2
    peak_thresh = 100
    # 정답이미지에서 픽셀255인 좌표찾기
    # gt = target_peaks_gen((gt_img).astype(np.uint8))
    gt = local_maxima(gt_img, peak_thresh, dist_peak)
    # 예측 이미지에서 밝기가 가장 밝은 곳의 중심좌표찾기
    res = local_maxima(pre_img, peak_thresh, dist_peak)
    associate_id = optimum(gt, res, dist_threshold)

    gt_final, no_detected_id = remove_outside_plot(
        gt, associate_id, 0, pre_img.shape
    )
    res_final, overdetection_id = remove_outside_plot(
        res, associate_id, 1, pre_img.shape
    )

    # show_res(
    #     ori,
    #     gt,
    #     res,
    #     no_detected_id,
    #     overdetection_id,
    #     path=str(save_error_path + os.path.basename(ori_path)),
    # )

    # truepositive
    tp = associate_id.shape[0]
    # falsenegative
    fn = gt_final.shape[0] - associate_id.shape[0]
    # falsepositive
    fp = res_final.shape[0] - associate_id.shape[0]
    if fn > 0 or fp > 0:
        label = 0
    if fn == 0 and fp == 0:
        label = 1
    return label




for i in range(1):
    out_path = "/home/hyeonwoo/research/semi/con_to_BMP2/Experiment4/Result/Discriminator/step{}/save_selected_top".format(str(i))
    precision_txt = out_path / Path("precision_txt")
    ori_path = out_path / Path("ori")
    pgt_path = out_path / Path("pgt")
    gt_path = out_path / Path("gt")
    ori_paths = sorted(ori_path.glob("*.tif"))
    tps = 0
    fps = 0
    for path in ori_paths:
        ori = cv2.imread(str(ori_path / Path(os.path.basename(path))), 0)
        pgt = cv2.imread(str(pgt_path / Path(os.path.basename(path))), 0)
        gt = cv2.imread(str(gt_path / Path(os.path.basename(path))), 0)
        label = cal_tp_fp_fn(ori, gt, pgt)
        if label == 1:
            tps += 1
            # cv2.imwrite(
            #     "/home/hyeonwoo/research/20201203/Image-level/step1/TP/ori/" + os.path.basename(i), ori)
            # cv2.imwrite(
            #     "/home/hyeonwoo/research/20201203/Image-level/step1/TP/gt/" + os.path.basename(i), gt)
            # cv2.imwrite(
            #     "/home/hyeonwoo/research/20201203/Image-level/step1/TP/pgt/" + os.path.basename(i), pgt)
        if label == 0:
            fps += 1
            # cv2.imwrite(
            #     "/home/hyeonwoo/research/20201203/Image-level/step1/FP/ori/" + os.path.basename(i), ori)
            # cv2.imwrite(
            #     "/home/hyeonwoo/research/20201203/Image-level/step1/FP/gt/" + os.path.basename(i), gt)
            # cv2.imwrite(
            #     "/home/hyeonwoo/research/20201203/Image-level/step1/FP/pgt/" + os.path.basename(i), pgt)

    precision = tps / (tps + fps)
    with precision_txt.open(mode="a") as f:
        f.write("%.12f\n" % (precision))
    print(precision)

