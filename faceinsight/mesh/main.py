# vi: set ft=python sts=4 ts=4 sw=4 et:

"""
The pipeline of 3DDFA prediction: given one cropped face image, predict the 3d
face vertices, 68 landmarks and visualization.

[todo]
1. CPU optimization: https://pmchojnacki.wordpress.com/2018/10/07/slow-pytorch-cpu-performance
"""

import os
import argparse

import numpy as np
import scipy.io as sio
import cv2
import dlib
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

from faceevolve.models import mobilenet_v1
from faceevolve.mesh.ddfa import ToTensorGjz, NormalizeGjz
from faceevolve.mesh.inference import get_suffix
from faceevolve.mesh.inference import predict_68pts, predict_dense
from faceevolve.mesh.inference import parse_roi_box_from_bbox
from faceevolve.mesh.inference import parse_roi_box_from_landmark
from faceevolve.mesh.inference import crop_img, draw_landmarks
from faceevolve.mesh.inference import dump_to_ply, dump_vertex
from faceevolve.mesh.inference import get_colors, write_obj_with_colors
from faceevolve.mesh.estimate_pose import parse_pose
from faceevolve.mesh.cv_plot import plot_pose_box
from faceevolve.mesh.render import get_depths_image, cget_depths_image, cpncc
from faceevolve.mesh.paf import gen_img_paf

STD_SIZE = 120


def main(args):
    """Main function of 3DDFA."""
    # 1. load pre-tained model
    arch = 'mobilenet_1'
    # 62 dims = 12(pose) + 40(shape) +10(expression)
    model = getattr(mobilenet_v1, arch)(num_classes=62)
    # load model weights
    model_dir = os.path.dirname(mobilenet_v1.__file__)
    checkpoint_fp = os.path.join(model_dir, '3ddfa_phase1_wpdc_vdc.pth.tar')
    checkpoint = torch.load(checkpoint_fp,
                        map_location=lambda storage, loc: storage)['state_dict']

    model_dict = model.state_dict()
    # because the model is trained by multiple gpus
    # prefix module should be removed
    for k in checkpoint.keys():
        model_dict[k.replace('module.', '')] = checkpoint[k]
    model.load_state_dict(model_dict)
    if args.mode == 'gpu':
        cudnn.benchmark = True
        model = model.cuda()
    model.eval()

    # 2. load dlib model for face detection and landmark used for face cropping
    if args.dlib_landmark:
        dlib_landmark_model = os.path.join(model_dir,
                                'dlib_shape_predictor_68_face_landmarks.dat')
        face_regressor = dlib.shape_predictor(dlib_landmark_model)
    if args.dlib_bbox:
        face_detector = dlib.get_frontal_face_detector()

    # 3. forward
    tri = sio.loadmat('visualize/tri.mat')['tri']
    transform = transforms.Compose([ToTensorGjz(),
                                    NormalizeGjz(mean=127.5, std=128)])
    for img_fp in args.files:
        img_ori = cv2.imread(img_fp)
        if args.dlib_bbox:
            rects = face_detector(img_ori, 1)
        else:
            rects = []

        if len(rects) == 0:
            rects = dlib.rectangles()
            rect_fp = img_fp + '.bbox'
            lines = open(rect_fp).read().strip().split('\n')[1:]
            for l in lines:
                l, r, t, b = [int(_) for _ in l.split(' ')[1:]]
                rect = dlib.rectangle(l, r, t, b)
                rects.append(rect)

        pts_res = []
        # Camera matrix collection
        Ps = []
        # pose collection, [todo: validate it]
        poses = []
        # store multiple face vertices
        vertices_lst = []
        ind = 0
        suffix = get_suffix(img_fp)
        for rect in rects:
            # whether use dlib landmark to crop image, if not,
            # use only face bbox to calc roi bbox for cropping
            if args.dlib_landmark:
                # - use landmark for cropping
                pts = face_regressor(img_ori, rect).parts()
                pts = np.array([[pt.x, pt.y] for pt in pts]).T
                roi_box = parse_roi_box_from_landmark(pts)
            else:
                # - use detected face bbox
                bbox = [rect.left(), rect.top(), rect.right(), rect.bottom()]
                roi_box = parse_roi_box_from_bbox(bbox)

            img = crop_img(img_ori, roi_box)

            # forward: one step
            img = cv2.resize(img,
                             dsize=(STD_SIZE, STD_SIZE),
                             interpolation=cv2.INTER_LINEAR)
            input = transform(img).unsqueeze(0)
            with torch.no_grad():
                if args.mode == 'gpu':
                    input = input.cuda()
                param = model(input)
                param = param.squeeze().cpu().numpy().flatten().astype(np.float32)

            # 68 pts
            pts68 = predict_68pts(param, roi_box)

            # two-step for more accurate bbox to crop face
            if args.bbox_init == 'two':
                roi_box = parse_roi_box_from_landmark(pts68)
                img_step2 = crop_img(img_ori, roi_box)
                img_step2 = cv2.resize(img_step2,
                                       dsize=(STD_SIZE, STD_SIZE),
                                       interpolation=cv2.INTER_LINEAR)
                input = transform(img_step2).unsqueeze(0)
                with torch.no_grad():
                    if args.mode == 'gpu':
                        input = input.cuda()
                    param = model(input)
                    param = param.squeeze().cpu().numpy().flatten().astype(np.float32)

                pts68 = predict_68pts(param, roi_box)

            pts_res.append(pts68)
            P, pose = parse_pose(param)
            Ps.append(P)
            poses.append(pose)

            # dense face 3d vertices
            if args.dump_ply or args.dump_vertex or args.dump_depth or args.dump_pncc or args.dump_obj:
                vertices = predict_dense(param, roi_box)
                vertices_lst.append(vertices)
            if args.dump_ply:
                dump_to_ply(vertices, tri,
                            '{}_{}.ply'.format(img_fp.replace(suffix, ''), ind))
            if args.dump_vertex:
                dump_vertex(vertices,
                            '{}_{}.mat'.format(img_fp.replace(suffix, ''), ind))
            if args.dump_pts:
                wfp = '{}_{}.txt'.format(img_fp.replace(suffix, ''), ind)
                np.savetxt(wfp, pts68, fmt='%.3f')
                print('Save 68 3d landmarks to {}'.format(wfp))
            if args.dump_roi_box:
                wfp = '{}_{}.roibox'.format(img_fp.replace(suffix, ''), ind)
                np.savetxt(wfp, roi_box, fmt='%.3f')
                print('Save roi box to {}'.format(wfp))
            if args.dump_paf:
                wfp_paf = '{}_{}_paf.jpg'.format(img_fp.replace(suffix, ''), ind)
                wfp_crop = '{}_{}_crop.jpg'.format(img_fp.replace(suffix, ''), ind)
                paf_feature = gen_img_paf(img_crop=img, param=param,
                                          kernel_size=args.paf_size)

                cv2.imwrite(wfp_paf, paf_feature)
                cv2.imwrite(wfp_crop, img)
                print('Dump to {} and {}'.format(wfp_crop, wfp_paf))
            if args.dump_obj:
                wfp = '{}_{}.obj'.format(img_fp.replace(suffix, ''), ind)
                colors = get_colors(img_ori, vertices)
                write_obj_with_colors(wfp, vertices, tri, colors)
                print('Dump obj with sampled texture to {}'.format(wfp))
            ind += 1

        if args.dump_pose:
            # Camera matrix (without scale), and
            # pose (yaw, pitch, roll, to verify)
            # P, pose = parse_pose(param)  
            img_pose = plot_pose_box(img_ori, Ps, pts_res)
            wfp = img_fp.replace(suffix, '_pose.jpg')
            cv2.imwrite(wfp, img_pose)
            print('Dump to {}'.format(wfp))
        if args.dump_depth:
            wfp = img_fp.replace(suffix, '_depth.png')
            # python version
            # depths_img = get_depths_image(img_ori, vertices_lst, tri-1)
            # cython version
            depths_img = cget_depths_image(img_ori, vertices_lst, tri - 1)
            cv2.imwrite(wfp, depths_img)
            print('Dump to {}'.format(wfp))
        if args.dump_pncc:
            wfp = img_fp.replace(suffix, '_pncc.png')
            # cython version
            pncc_feature = cpncc(img_ori, vertices_lst, tri - 1)
            # cv2.imwrite will swap RGB -> BGR
            cv2.imwrite(wfp, pncc_feature[:, :, ::-1])
            print('Dump to {}'.format(wfp))
        if args.dump_res:
            draw_landmarks(img_ori, pts_res,
                           wfp=img_fp.replace(suffix, '_3DDFA.jpg'),
                           show_flg=args.show_flg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='3DDFA inference pipeline')
    parser.add_argument('-f', '--files', nargs='+',
                        help='single/multiple image files fed into network')
    parser.add_argument('-m', '--mode', default='cpu', type=str,
                        help='gpu or cpu mode')
    parser.add_argument('--bbox_init', default='one', type=str,
                        help='one|two: one- or two-step bbox initialization')
    parser.add_argument('--show_flg', default=False, action='store_true',
                        help='show the visualization result')
    parser.add_argument('--dump_res', default=False, action='store_true',
                        help='write out the visualization image and key points')
    parser.add_argument('--dump_vertex', default=False, action='store_true',
                        help='write out the dense face vertices to mat')
    parser.add_argument('--dump_ply', default=False, action='store_true')
    parser.add_argument('--dump_pts', default=False, action='store_true')
    parser.add_argument('--dump_roi_box', default=False, action='store_true')
    parser.add_argument('--dump_pose', default=False, action='store_true')
    parser.add_argument('--dump_depth', default=False, action='store_true')
    parser.add_argument('--dump_pncc', default=False, action='store_true')
    parser.add_argument('--dump_paf', default=False, action='store_true')
    parser.add_argument('--paf_size', default=3, type=int,
                        help='PAF feature kernel size')
    parser.add_argument('--dump_obj', default=False, action='store_true')
    parser.add_argument('--dlib_bbox', default=True, action='store_true',
                        help='use dlib to predict bbox')
    parser.add_argument('--dlib_landmark', default=True, action='store_true',
                        help='use dlib landmark to crop image')

    args = parser.parse_args()
    main(args)

