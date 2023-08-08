#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   deepface3d_render.py (theta->render)
@Time    :   2023/03/28 20:36:54
@Author  :   Weihao Xia 
@Version :   1.0
@Desc    :   3dmm parameters --> mesh -> rendered image
'''

import numpy as np
import argparse
import trimesh
from scipy.io import savemat, loadmat
import torch
from models.base_model import BaseModel
from models import networks
from models.bfm import ParametricFaceModel
from util import util 
from util.nvdiffrast import MeshRenderer
from util.preprocess import estimate_norm_torch

def merge_coeff_v1(coeffs_dict):
    """
    Merge the coeffs of the layers into a single tensor.
    Return:
        coeffs      -- torch.tensor, size (B, 256)
    Parameters:
        coeffs_dict -- a dict of torch.tensors
    """
    id_coeffs = coeffs_dict['id']
    exp_coeffs = coeffs_dict['exp']
    tex_coeffs = coeffs_dict['tex']
    angles = coeffs_dict['angle']
    gammas = coeffs_dict['gamma']
    translations = coeffs_dict['trans']  
    coeffs = torch.cat((id_coeffs, exp_coeffs, tex_coeffs, angles, gammas, translations), dim=1)
    return coeffs

def merge_coeff(coeffs_dict):
    """
    Merge the coeffs of the layers into a single tensor.
    """
    # The keys of `coeffs_dict` are the names of the layers.
    # The values of `coeffs_dict` are the coefficients of the layers.
    coeffs = []
    for key in coeffs_dict:
        coeffs.append(coeffs_dict[key])
    # Concatenate the coefficients of the layers along the second dimension
    # so that the shape of the coefficients is (B, 256)
    coeffs = torch.cat(coeffs, dim=1)
    return coeffs

def save_mesh(name, pred_vertex, pred_color, face_buf):
    """
    Save the mesh to a .obj file
    Parameters:
        name        -- str, the name of the output file
        pred_vertex -- torch.tensor, size (B, N, 3)
        pred_color  -- torch.tensor, size (B, N, 3)
        face_buf    -- torch.tensor, size (B, F, 3)
    """
    recon_shape = pred_vertex  # get reconstructed shape
    recon_shape[..., -1] = 10 - recon_shape[..., -1] # from camera space to world space
    recon_shape = recon_shape.cpu().numpy()[0]
    recon_color = pred_color
    recon_color = recon_color.cpu().numpy()[0]
    tri = face_buf.cpu().numpy()
    mesh = trimesh.Trimesh(vertices=recon_shape, faces=tri, vertex_colors=np.clip(255. * recon_color, 0, 255).astype(np.uint8), process=False)
    mesh.export(name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #
    parser.add_argument('--isTrain', type=bool, default='Falese', help='train or test')
    parser.add_argument('--rank', type=int, default=0, help='gpu rank')

    # net structure and parameters
    parser.add_argument('--net_recon', type=str, default='resnet50', choices=['resnet18', 'resnet34', 'resnet50'], help='network structure')
    parser.add_argument('--init_path', type=str, default='checkpoints/init_model/resnet50-0676ba61.pth')
    parser.add_argument('--use_last_fc', type=util.str2bool, nargs='?', const=True, default=False, help='zero initialize the last fc')
    parser.add_argument('--bfm_folder', type=str, default='BFM')
    parser.add_argument('--bfm_model', type=str, default='BFM_model_front.mat', help='bfm model')

    # renderer parameters
    parser.add_argument('--focal', type=float, default=1015.)
    parser.add_argument('--center', type=float, default=112.)
    parser.add_argument('--camera_d', type=float, default=10.)
    parser.add_argument('--z_near', type=float, default=5.)
    parser.add_argument('--z_far', type=float, default=15.)
    parser.add_argument('--use_opengl', type=util.str2bool, nargs='?', const=True, default=True, help='use opengl context or not')
    opt = parser.parse_args()

    net_recon = networks.define_net_recon(
        net_recon=opt.net_recon, use_last_fc=opt.use_last_fc, init_path=opt.init_path
    )

    facemodel = ParametricFaceModel(
        bfm_folder=opt.bfm_folder, camera_distance=opt.camera_d, focal=opt.focal, center=opt.center,
        is_train=opt.isTrain, default_name=opt.bfm_model
    )

    fov = 2 * np.arctan(opt.center / opt.focal) * 180 / np.pi
    renderer = MeshRenderer(
        rasterize_fov=fov, znear=opt.z_near, zfar=opt.z_far, rasterize_size=int(2 * opt.center), use_opengl=opt.use_opengl
    )

    device = torch.device(opt.rank)
    torch.cuda.set_device(device)
    facemodel.to(device)

    # dict_keys(['__header__', '__version__', '__globals__', 'id', 'exp', 'tex', 'angle', 'gamma', 'trans', 'lm68'])
    mat_fname = '000002.mat'
    pred_coeffs = loadmat(mat_fname)
    # pred_coeffs_dict = {key: torch.tensor(pred_coeffs[key]) for key in pred_coeffs}
    pred_coeffs_key = ['id', 'exp', 'tex', 'angle', 'gamma', 'trans'] # be the same order as in `split_coeff(coeffs_dict)`.
    pred_coeffs_dict = {key: torch.from_numpy(pred_coeffs[key]) for key in pred_coeffs_key} 
    coeffs = merge_coeff(pred_coeffs_dict)
    # check if the two methods are the same
    # print(torch.allclose(coeffs_v1, coeffs)) # True

    pred_vertex, pred_tex, pred_color, pred_lm = facemodel.compute_for_render(coeffs.to(device))
    pred_mask, _, pred_face = renderer(pred_vertex, facemodel.face_buf, feat=pred_color)

    # save the rendered face image
    pred_face_numpy_raw = 255. * pred_face.detach().cpu().permute(0, 2, 3, 1).numpy()
    pred_face_numpy = np.clip(pred_face_numpy_raw, 0, 255).astype(np.uint8)
    util.save_image(pred_face_numpy[0], mat_fname.replace('mat', 'png')) # (224, 224, 3)

    # save the mesh
    save_mesh(mat_fname.replace('mat', 'obj'), pred_vertex, pred_color, facemodel.face_buf)