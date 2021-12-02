
import os
import io
import PIL.Image, PIL.ImageDraw
import base64
import zipfile
import json
import requests
import numpy as np
import copy
import matplotlib.pylab as pl
import glob
from pathlib import Path
import tempfile

from IPython.display import Image, HTML, clear_output
from tqdm import tqdm_notebook, tnrange

os.environ['FFMPEG_BINARY'] = 'ffmpeg'


import torch

# torch.set_default_tensor_type('torch.cuda.FloatTensor')
# print("Torch version:", torch.__version__)

import pydiffvg
import skimage
import skimage.io
import random
import ttools.modules
import argparse
import math
import torchvision
import torchvision.transforms as transforms
import requests
from io import BytesIO


import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import PIL
from time import time

import clip
import torch.nn.functional as F
from torchvision.datasets import CIFAR100

def imread(url, max_size=None, mode=None):
  if url.startswith(('http:', 'https:')):
    r = requests.get(url)
    f = io.BytesIO(r.content)
  else:
    f = url
  img = PIL.Image.open(f)
  if max_size is not None:
    img = img.resize((max_size, max_size))
  if mode is not None:
    img = img.convert(mode)
  img = np.float32(img)/255.0
  return img

def checkin(img, out_path=None):
    save_img(img, str(out_path))
    return out_path

def save_img(img, file_name):
    img = np.transpose(img, (1, 2, 0))
    img = np.clip(img, 0, 1)
    img = np.uint8(img * 254)
    pimg = PIL.Image.fromarray(img, mode="RGB")
    pimg.save(file_name)

def np2pil(a):
  if a.dtype in [np.float32, np.float64]:
    a = np.uint8(np.clip(a, 0, 1)*255)
  return PIL.Image.fromarray(a)

def imwrite(f, a, fmt=None):
  a = np.asarray(a)
  if isinstance(f, str):
    fmt = f.rsplit('.', 1)[-1].lower()
    if fmt == 'jpg':
      fmt = 'jpeg'
    f = open(f, 'wb')
  np2pil(a).save(f, fmt, quality=95)

def imencode(a, fmt='jpeg'):
  a = np.asarray(a)
  if len(a.shape) == 3 and a.shape[-1] == 4:
    fmt = 'png'
  f = io.BytesIO()
  imwrite(f, a, fmt)
  return f.getvalue()

def im2url(a, fmt='jpeg'):
  encoded = imencode(a, fmt)
  base64_byte_string = base64.b64encode(encoded).decode('ascii')
  return 'data:image/' + fmt.upper() + ';base64,' + base64_byte_string

def imshow(a, fmt='jpeg'):
  display(Image(data=imencode(a, fmt)))


def tile2d(a, w=None):
  a = np.asarray(a)
  if w is None:
    w = int(np.ceil(np.sqrt(len(a))))
  th, tw = a.shape[1:3]
  pad = (w-len(a))%w
  a = np.pad(a, [(0, pad)]+[(0, 0)]*(a.ndim-1), 'constant')
  h = len(a)//w
  a = a.reshape([h, w]+list(a.shape[1:]))
  a = np.rollaxis(a, 2, 1).reshape([th*h, tw*w]+list(a.shape[4:]))
  return a

from torchvision import utils
def show_img(img):
    img = np.transpose(img, (1, 2, 0))
    img = np.clip(img, 0, 1)
    img = np.uint8(img * 254)
    # img = np.repeat(img, 4, axis=0)
    # img = np.repeat(img, 4, axis=1)
    pimg = PIL.Image.fromarray(img, mode="RGB")
    imshow(pimg)

def zoom(img, scale=4):
  img = np.repeat(img, scale, 0)
  img = np.repeat(img, scale, 1)
  return img






#@title Style Loss and Drawing Functions {vertical-output: true}
def pil_resize_long_edge_to(pil, trg_size):
  short_w = pil.width < pil.height
  ar_resized_long = (trg_size / pil.height) if short_w else (trg_size / pil.width)
  resized = pil.resize((int(pil.width * ar_resized_long), int(pil.height * ar_resized_long)), PIL.Image.BICUBIC)
  return resized


class Vgg16_Extractor(nn.Module):
    def __init__(self, space):
        super().__init__()
        self.vgg_layers = models.vgg16(pretrained=True).features

        for param in self.parameters():
            param.requires_grad = False
        self.capture_layers = [1,3,6,8,11,13,15,22,29]
        self.space = space

    def forward_base(self, x):
        feat = [x]
        for i in range(len(self.vgg_layers)):
            x = self.vgg_layers[i](x)
            if i in self.capture_layers: feat.append(x)
        return feat

    def forward(self, x):
        if self.space != 'vgg':
            x = (x + 1.) / 2.
            x = x - (torch.Tensor([0.485, 0.456, 0.406]).to(x.device).view(1, -1, 1, 1))
            x = x / (torch.Tensor([0.229, 0.224, 0.225]).to(x.device).view(1, -1, 1, 1))
        feat = self.forward_base(x)
        return feat

    def forward_samples_hypercolumn(self, X, samps=100):
        feat = self.forward(X)

        xx,xy = np.meshgrid(np.arange(X.shape[2]), np.arange(X.shape[3]))
        xx = np.expand_dims(xx.flatten(),1)
        xy = np.expand_dims(xy.flatten(),1)
        xc = np.concatenate([xx,xy],1)

        samples = min(samps,xc.shape[0])

        np.random.shuffle(xc)
        xx = xc[:samples,0]
        yy = xc[:samples,1]

        feat_samples = []
        for i in range(len(feat)):

            layer_feat = feat[i]

            # hack to detect lower resolution
            if i>0 and feat[i].size(2) < feat[i-1].size(2):
                xx = xx/2.0
                yy = yy/2.0

            xx = np.clip(xx, 0, layer_feat.shape[2]-1).astype(np.int32)
            yy = np.clip(yy, 0, layer_feat.shape[3]-1).astype(np.int32)

            features = layer_feat[:,:, xx[range(samples)], yy[range(samples)]]
            feat_samples.append(features.clone().detach())

        feat = torch.cat(feat_samples,1)
        return feat

# Tensor and PIL utils

def pil_loader(path):
    with open(path, 'rb') as f:
        img = PIL.Image.open(f)
        return img.convert('RGB')

def pil_loader_internet(url):
    response = requests.get(url)
    img = PIL.Image.open(BytesIO(response.content))
    return img.convert('RGB')

def tensor_resample(tensor, dst_size, mode='bilinear'):
    return F.interpolate(tensor, dst_size, mode=mode, align_corners=False)

def pil_resize_short_edge_to(pil, trg_size):
    short_w = pil.width < pil.height
    ar_resized_short = (trg_size / pil.width) if short_w else (trg_size / pil.height)
    resized = pil.resize((int(pil.width * ar_resized_short), int(pil.height * ar_resized_short)), PIL.Image.BICUBIC)
    return resized

def pil_resize_long_edge_to(pil, trg_size):
    short_w = pil.width < pil.height
    ar_resized_long = (trg_size / pil.height) if short_w else (trg_size / pil.width)
    resized = pil.resize((int(pil.width * ar_resized_long), int(pil.height * ar_resized_long)), PIL.Image.BICUBIC)
    return resized

def np_to_pil(npy):
    return PIL.Image.fromarray(npy.astype(np.uint8))

def pil_to_np(pil):
    return np.array(pil)

def tensor_to_np(tensor, cut_dim_to_3=True):
    if len(tensor.shape) == 4:
        if cut_dim_to_3:
            tensor = tensor[0]
        else:
            return tensor.data.cpu().numpy().transpose((0, 2, 3, 1))
    return tensor.data.cpu().numpy().transpose((1,2,0))

def np_to_tensor(npy, space):
    if space == 'vgg':
        return np_to_tensor_correct(npy)
    return (torch.Tensor(npy.astype(np.float) / 127.5) - 1.0).permute((2,0,1)).unsqueeze(0)

def np_to_tensor_correct(npy):
    pil = np_to_pil(npy)
    transform = transforms.Compose([transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(pil).unsqueeze(0)

# Laplacian Pyramid

def laplacian(x):
    # x - upsample(downsample(x))
    return x - tensor_resample(tensor_resample(x, [x.shape[2] // 2, x.shape[3] // 2]), [x.shape[2], x.shape[3]])

def make_laplace_pyramid(x, levels):
    pyramid = []
    current = x
    for i in range(levels):
        pyramid.append(laplacian(current))
        current = tensor_resample(current, (max(current.shape[2] // 2,1), max(current.shape[3] // 2,1)))
    pyramid.append(current)
    return pyramid

def fold_laplace_pyramid(pyramid):
    current = pyramid[-1]
    for i in range(len(pyramid)-2, -1, -1): # iterate from len-2 to 0
        up_h, up_w = pyramid[i].shape[2], pyramid[i].shape[3]
        current = pyramid[i] + tensor_resample(current, (up_h,up_w))
    return current

def sample_indices(feat_content, feat_style):
    indices = None
    const = 128**2 # 32k or so
    feat_dims = feat_style.shape[1]
    big_size = feat_content.shape[2] * feat_content.shape[3] # num feaxels

    stride_x = int(max(math.floor(math.sqrt(big_size//const)),1))
    offset_x = np.random.randint(stride_x)
    stride_y = int(max(math.ceil(math.sqrt(big_size//const)),1))
    offset_y = np.random.randint(stride_y)
    xx, xy = np.meshgrid(np.arange(feat_content.shape[2])[offset_x::stride_x], np.arange(feat_content.shape[3])[offset_y::stride_y] )

    xx = xx.flatten()
    xy = xy.flatten()
    return xx, xy

def spatial_feature_extract(feat_result, feat_content, xx, xy):
    l2, l3 = [], []
    device = feat_result[0].device

    # for each extracted layer
    for i in range(len(feat_result)):
        fr = feat_result[i]
        fc = feat_content[i]

        # hack to detect reduced scale
        if i>0 and feat_result[i-1].size(2) > feat_result[i].size(2):
            xx = xx/2.0
            xy = xy/2.0

        # go back to ints and get residual
        xxm = np.floor(xx).astype(np.float32)
        xxr = xx - xxm

        xym = np.floor(xy).astype(np.float32)
        xyr = xy - xym

        # do bilinear resample
        w00 = torch.from_numpy((1.-xxr)*(1.-xyr)).float().view(1, 1, -1, 1).to(device)
        w01 = torch.from_numpy((1.-xxr)*xyr).float().view(1, 1, -1, 1).to(device)
        w10 = torch.from_numpy(xxr*(1.-xyr)).float().view(1, 1, -1, 1).to(device)
        w11 = torch.from_numpy(xxr*xyr).float().view(1, 1, -1, 1).to(device)

        xxm = np.clip(xxm.astype(np.int32),0,fr.size(2)-1)
        xym = np.clip(xym.astype(np.int32),0,fr.size(3)-1)

        s00 = xxm*fr.size(3)+xym
        s01 = xxm*fr.size(3)+np.clip(xym+1,0,fr.size(3)-1)
        s10 = np.clip(xxm+1,0,fr.size(2)-1)*fr.size(3)+(xym)
        s11 = np.clip(xxm+1,0,fr.size(2)-1)*fr.size(3)+np.clip(xym+1,0,fr.size(3)-1)

        fr = fr.view(1,fr.size(1),fr.size(2)*fr.size(3),1)
        fr = fr[:,:,s00,:].mul_(w00).add_(fr[:,:,s01,:].mul_(w01)).add_(fr[:,:,s10,:].mul_(w10)).add_(fr[:,:,s11,:].mul_(w11))

        fc = fc.view(1,fc.size(1),fc.size(2)*fc.size(3),1)
        fc = fc[:,:,s00,:].mul_(w00).add_(fc[:,:,s01,:].mul_(w01)).add_(fc[:,:,s10,:].mul_(w10)).add_(fc[:,:,s11,:].mul_(w11))

        l2.append(fr)
        l3.append(fc)

    x_st = torch.cat([li.contiguous() for li in l2],1)
    c_st = torch.cat([li.contiguous() for li in l3],1)

    xx = torch.from_numpy(xx).view(1,1,x_st.size(2),1).float().to(device)
    yy = torch.from_numpy(xy).view(1,1,x_st.size(2),1).float().to(device)

    x_st = torch.cat([x_st,xx,yy],1)
    c_st = torch.cat([c_st,xx,yy],1)
    return x_st, c_st

def pairwise_distances_cos(x, y):
    x_norm = torch.sqrt((x**2).sum(1).view(-1, 1))
    y_t = torch.transpose(y, 0, 1)
    y_norm = torch.sqrt((y**2).sum(1).view(1, -1))
    dist = 1.-torch.mm(x, y_t)/x_norm/y_norm
    return dist

def pairwise_distances_sq_l2(x, y):
    x_norm = (x**2).sum(1).view(-1, 1)
    y_t = torch.transpose(y, 0, 1)
    y_norm = (y**2).sum(1).view(1, -1)
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    return torch.clamp(dist, 1e-5, 1e5)/x.size(1)

def distmat(x, y, cos_d=True):
    if cos_d:
        M = pairwise_distances_cos(x, y)
    else:
        M = torch.sqrt(pairwise_distances_sq_l2(x, y))
    return M

def content_loss(feat_result, feat_content):
    d = feat_result.size(1)

    X = feat_result.transpose(0,1).contiguous().view(d,-1).transpose(0,1)
    Y = feat_content.transpose(0,1).contiguous().view(d,-1).transpose(0,1)

    Y = Y[:,:-2]
    X = X[:,:-2]
    # X = X.t()
    # Y = Y.t()

    Mx = distmat(X, X)
    Mx = Mx#/Mx.sum(0, keepdim=True)

    My = distmat(Y, Y)
    My = My#/My.sum(0, keepdim=True)

    d = torch.abs(Mx-My).mean()# * X.shape[0]
    return d

def rgb_to_yuv(rgb):
    C = torch.Tensor([[0.577350,0.577350,0.577350],[-0.577350,0.788675,-0.211325],[-0.577350,-0.211325,0.788675]]).to(rgb.device)
    yuv = torch.mm(C,rgb)
    return yuv

def style_loss(X, Y, cos_d=True):
    d = X.shape[1]

    if d == 3:
        X = rgb_to_yuv(X.transpose(0,1).contiguous().view(d,-1)).transpose(0,1)
        Y = rgb_to_yuv(Y.transpose(0,1).contiguous().view(d,-1)).transpose(0,1)
    else:
        X = X.transpose(0,1).contiguous().view(d,-1).transpose(0,1)
        Y = Y.transpose(0,1).contiguous().view(d,-1).transpose(0,1)

    # Relaxed EMD
    CX_M = distmat(X, Y, cos_d=True)

    if d==3: CX_M = CX_M + distmat(X, Y, cos_d=False)

    m1, m1_inds = CX_M.min(1)
    m2, m2_inds = CX_M.min(0)

    remd = torch.max(m1.mean(), m2.mean())

    return remd

def moment_loss(X, Y, moments=[1,2]):
    loss = 0.
    X = X.squeeze().t()
    Y = Y.squeeze().t()

    mu_x = torch.mean(X, 0, keepdim=True)
    mu_y = torch.mean(Y, 0, keepdim=True)
    mu_d = torch.abs(mu_x - mu_y).mean()

    if 1 in moments:
        # print(mu_x.shape)
        loss = loss + mu_d

    if 2 in moments:
        X_c = X - mu_x
        Y_c = Y - mu_y
        X_cov = torch.mm(X_c.t(), X_c) / (X.shape[0] - 1)
        Y_cov = torch.mm(Y_c.t(), Y_c) / (Y.shape[0] - 1)

        # print(X_cov.shape)
        # exit(1)

        D_cov = torch.abs(X_cov - Y_cov).mean()
        loss = loss + D_cov

    return loss


def calculate_loss(feat_result, feat_content, feat_style, indices, content_weight, moment_weight=1.0):
  # spatial feature extract
  num_locations = 1024
  spatial_result, spatial_content = spatial_feature_extract(feat_result, feat_content, indices[0][:num_locations], indices[1][:num_locations])
  # loss_content = content_loss(spatial_result, spatial_content)

  d = feat_style.shape[1]
  spatial_style = feat_style.view(1, d, -1, 1)
  feat_max = 3+2*64+128*2+256*3+512*2 # (sum of all extracted channels)

  loss_remd = style_loss(spatial_result[:, :feat_max, :, :], spatial_style[:, :feat_max, :, :])

  loss_moment = moment_loss(spatial_result[:,:-2,:,:], spatial_style, moments=[1,2]) # -2 is so that it can fit?
  # palette matching
  content_weight_frac = 1./max(content_weight,1.)
  loss_moment += content_weight_frac * style_loss(spatial_result[:,:3,:,:], spatial_style[:,:3,:,:])

  loss_style = loss_remd + moment_weight * loss_moment
  # print(f'Style: {loss_style.item():.3f}, Content: {loss_content.item():.3f}')

  style_weight = 1.0 + moment_weight
  loss_total = (loss_style) / (content_weight + style_weight)
  return loss_total

def get_image_augmentation(use_normalized_clip):
    augment_trans = transforms.Compose([
        transforms.RandomPerspective(fill=1, p=1, distortion_scale=0.5),
        transforms.RandomResizedCrop(224, scale=(0.7,0.9)),
    ])

    if use_normalized_clip:
        augment_trans = transforms.Compose([
        transforms.RandomPerspective(fill=1, p=1, distortion_scale=0.5),
        transforms.RandomResizedCrop(224, scale=(0.7,0.9)),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])
    return augment_trans

def initialize_curves(num_paths, canvas_width, canvas_height):
    shapes = []
    shape_groups = []
    for i in range(num_paths):
        num_segments = random.randint(1, 3)
        num_control_points = torch.zeros(num_segments, dtype = torch.int32) + 2
        points = []
        p0 = (random.random(), random.random())
        points.append(p0)
        for j in range(num_segments):
            radius = 0.1
            p1 = (p0[0] + radius * (random.random() - 0.5), p0[1] + radius * (random.random() - 0.5))
            p2 = (p1[0] + radius * (random.random() - 0.5), p1[1] + radius * (random.random() - 0.5))
            p3 = (p2[0] + radius * (random.random() - 0.5), p2[1] + radius * (random.random() - 0.5))
            points.append(p1)
            points.append(p2)
            points.append(p3)
            p0 = p3
        points = torch.tensor(points)
        points[:, 0] *= canvas_width
        points[:, 1] *= canvas_height
        path = pydiffvg.Path(num_control_points = num_control_points, points = points, stroke_width = torch.tensor(1.0), is_closed = False)
        shapes.append(path)
        path_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([len(shapes) - 1]), fill_color = None, stroke_color = torch.tensor([random.random(), random.random(), random.random(), random.random()]))
        shape_groups.append(path_group)
    return shapes, shape_groups

def render_drawing(shapes, shape_groups,\
                   canvas_width, canvas_height, n_iter, save=False):
    scene_args = pydiffvg.RenderFunction.serialize_scene(\
        canvas_width, canvas_height, shapes, shape_groups)
    render = pydiffvg.RenderFunction.apply
    img = render(canvas_width, canvas_height, 2, 2, n_iter, None, *scene_args)
    img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3, device = pydiffvg.get_device()) * (1 - img[:, :, 3:4])
    if save:
        pydiffvg.imwrite(img.cpu(), '/content/res/iter_{}.png'.format(int(n_iter)), gamma=1.0)
    img = img[:, :, :3]
    img = img.unsqueeze(0)
    img = img.permute(0, 3, 1, 2) # NHWC -> NCHW
    return img


def style_clip_draw(prompt, style_path,
                    num_paths=256, num_iter=1000, max_width=50,
                    num_augs=4, style_weight=1.,
                    neg_prompt=None, neg_prompt_2=None,
                    use_normalized_clip=False,
                    debug=False):
    '''
    Perform StyleCLIPDraw using a given text prompt and style image
    args:
        prompt (str) : Text prompt to draw
        style_path(str) : Style image path or url
    kwargs:
        num_paths (int) : Number of brush strokes
        num_iter(int) : Number of optimization iterations
        max_width(float) : Maximum width of a brush stroke in pixels
        num_augs(int) : Number of image augmentations
        style_weight=(float) : What to multiply the style loss by
        neg_prompt(str) : Negative prompt. None if you don't want it
        neg_prompt_2(str) : Negative prompt. None if you don't want it
        use_normalized_clip(bool)
        debug(bool) : Print intermediate canvases and losses for debugging
    return
        np.ndarray(canvas_height, canvas_width, 3)
    '''
    out_path = Path(tempfile.mkdtemp()) / "out.png"

    text_input = clip.tokenize(prompt).to(device)

    if neg_prompt is not None: text_input_neg1 = clip.tokenize(neg_prompt).to(device)
    if neg_prompt_2 is not None: text_input_neg2 = clip.tokenize(neg_prompt_2).to(device)

    # Calculate features
    with torch.no_grad():
        text_features = model.encode_text(text_input)
        if neg_prompt is not None: text_features_neg1 = model.encode_text(text_input_neg1)
        if neg_prompt_2 is not None: text_features_neg2 = model.encode_text(text_input_neg2)

    canvas_width, canvas_height = 224, 224

    # Image Augmentation Transformation
    augment_trans = get_image_augmentation(use_normalized_clip)

    # Initialize Random Curves
    shapes, shape_groups = initialize_curves(num_paths, canvas_width, canvas_height)

    points_vars = []
    stroke_width_vars = []
    color_vars = []
    for path in shapes:
        path.points.requires_grad = True
        points_vars.append(path.points)
        path.stroke_width.requires_grad = True
        stroke_width_vars.append(path.stroke_width)
    for group in shape_groups:
        group.stroke_color.requires_grad = True
        color_vars.append(group.stroke_color)

    # Optimizers
    lr = 1
    points_optim = torch.optim.Adam(points_vars, lr=1.0*lr)
    width_optim = torch.optim.Adam(stroke_width_vars, lr=0.1*lr)
    color_optim = torch.optim.Adam(color_vars, lr=0.01*lr)

    style_pil = PIL.Image.open(str(style_path)).convert("RGB")
    style_pil = pil_resize_long_edge_to(style_pil, canvas_width)
    style_np = pil_to_np(style_pil)
    style = (np_to_tensor(style_np, "normal").to(device)+1)/2


    # Extract style features from style image
    feat_style = None
    for i in range(5):
        with torch.no_grad():
        # r is region of interest (mask)
            feat_e = extractor.forward_samples_hypercolumn(style, samps=1000)
            feat_style = feat_e if feat_style is None else torch.cat((feat_style, feat_e), dim=2)

    # Run the main optimization loop
    for t in range(num_iter):

        # Anneal learning rate (makes videos look cleaner)
        if t == int(num_iter * 0.5):
            for g in points_optim.param_groups:
                g['lr'] = 0.4
        if t == int(num_iter * 0.75):
            for g in points_optim.param_groups:
                g['lr'] = 0.1

        points_optim.zero_grad()
        width_optim.zero_grad()
        color_optim.zero_grad()

        img = render_drawing(shapes, shape_groups, canvas_width, canvas_height, t, save=(t % 5 == 0))

        loss = 0
        img_augs = []
        if t < .9*num_iter:
            for n in range(num_augs):
                img_augs.append(augment_trans(img))
            im_batch = torch.cat(img_augs)
            image_features = model.encode_image(im_batch)
            for n in range(num_augs):
                loss -= torch.cosine_similarity(text_features, image_features[n:n+1], dim=1)
                if neg_prompt is not None: loss += torch.cosine_similarity(text_features_neg1, image_features[n:n+1], dim=1) * 0.3
                if neg_prompt_2 is not None: loss += torch.cosine_similarity(text_features_neg2, image_features[n:n+1], dim=1) * 0.3


        # Do style optimization
        feat_content = extractor(img)

        xx, xy = sample_indices(feat_content[0], feat_style)

        np.random.shuffle(xx)
        np.random.shuffle(xy)

        styleloss = calculate_loss(feat_content, feat_content, feat_style, [xx, xy], 0)

        loss += styleloss * style_weight

        loss.backward()
        points_optim.step()
        width_optim.step()
        color_optim.step()

        for path in shapes:
            path.stroke_width.data.clamp_(1.0, max_width)
        for group in shape_groups:
            group.stroke_color.data.clamp_(0.0, 1.0)

        if t % 20 == 0:
            with torch.no_grad():
                shapes_resized = copy.deepcopy(shapes)
                for i in range(len(shapes)):
                    shapes_resized[i].stroke_width = shapes[i].stroke_width * 4
                    for j in range(len(shapes[i].points)):
                        shapes_resized[i].points[j] = shapes[i].points[j] * 4
                img = render_drawing(shapes_resized, shape_groups, canvas_width*4, canvas_height*4, t)
                yield checkin(img.detach().cpu().numpy()[0], out_path)
                print('Iteration:', t, '\tRender loss:', loss.item())

    with torch.no_grad():
        shapes_resized = copy.deepcopy(shapes)
        for i in range(len(shapes)):
            shapes_resized[i].stroke_width = shapes[i].stroke_width * 4
            for j in range(len(shapes[i].points)):
                shapes_resized[i].points[j] = shapes[i].points[j] * 4
        img = render_drawing(shapes_resized, shape_groups, canvas_width*4, canvas_height*4, t).detach().cpu().numpy()[0]
        save_img(img, str(out_path))
        yield out_path


import cog

device, model, preprocess, extractor = None, None, None, None

class Predictor(cog.Predictor):
    def setup(self):
        global device, model, preprocess, extractor
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        pydiffvg.set_print_timing(False)
        # Use GPU if available
        pydiffvg.set_use_gpu(torch.cuda.is_available())
        pydiffvg.set_device(device)

        # Load the model
        model, preprocess = clip.load('ViT-B/32', device, jit=False)

        extractor = Vgg16_Extractor(space="normal").to(device)



    @cog.input("prompt", type=str, default="A person watching TV.",
               help="Text description of the desired drawing")
    @cog.input("style_image", type=Path, help="Style Image")
    @cog.input("num_paths", type=int, default=256, help="Number of drawing strokes.")
    @cog.input("num_iterations", type=int, default=500, help="Number of optimization iterations")
    @cog.input("style_strength", type=int, default=50, help="How strong the style should be. 100 (max) is a lot. 0 (min) is no style.")
    def predict(self, prompt, style_image, num_paths, num_iterations,
                style_strength=50):
        """Run a single prediction on the model"""
        assert isinstance(num_paths, int) and num_paths > 0, 'num_paths should be an positive integer'
        assert isinstance(num_iterations, int) and num_iterations > 0, 'num_iterations should be an positive integer'
        # assert num_iterations < 350, 'num_iterations must be less than 350 or else the process will timeout'
        assert isinstance(style_strength, int) and style_strength >= 0 and style_strength <= 100, \
                'style_strength should be a positive integer less than 100'
        assert style_image is not None, 'style_image must be specified'
        assert prompt is not None and len(prompt) > 0, 'prompt must be specified'

        style_weight = 4 * (style_strength/100)

        for path in style_clip_draw(prompt, str(style_image), num_paths=num_paths,\
                          num_iter=num_iterations, style_weight=style_weight, num_augs=10):
            yield path

        return path
