import os
import sys
import numpy as np
import json
import random
import time
import jittor as jt
from jittor import nn
from tqdm import tqdm, trange
from natsort import natsorted
from run_nerf_helpers_deform import *
from load_audface_multiid import load_audface_data

jt.flags.use_cuda = 1   # use cuda
np.random.seed(0)
DEBUG = False

def batchify_cache(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches."""
    if chunk is None:
        return fn

    def ret(inputs, training = False):

        ret_list = [fn(inputs[i:i+chunk], training=training) for i in range(0, int(inputs.shape[0]), chunk)]

        return jt.concat((ret for ret in ret_list), 0)
    return ret

def batchify(fn, chunk, aud_para, world_fn = lambda x:x, gather_func = None, lip_rect=None):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn

    def ret(inputs, training = False, world_fn=world_fn):
        #add
        embedded = inputs[0]
        attention_poses = inputs[1]
        intrinsic = inputs[2]
        images_features = inputs[3]
        pts = inputs[4]
        input_paras, loss_translation = gather_func(world_fn(pts), attention_poses, intrinsic, images_features, aud_para, lip_rect)
        ret_list = fn([embedded, input_paras, pts], training=training)
        # ret_list[1] decoder_input
        if fn.coarse:
            return ret_list[0], ret_list[1], loss_translation
        else:
            return ret_list[0], None, loss_translation
    return ret


def run_network(inputs, viewdirs, aud_para, fn, embed_fn, embeddirs_fn, netchunk=1024*64, attention_poses=None, intrinsic=None, training=False,
                images_features=None, world_fn=None, gather_func=None, lip_rect = None):
    """Prepares inputs and applies network 'fn'.
    """
    inputs_flat = jt.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)
    aud = aud_para.unsqueeze(0).expand(inputs_flat.shape[0], -1)
    embedded = jt.concat((embedded, aud), -1)  # 将编码后的aud加入聚合
    if viewdirs is not None:
        # 视图不为 None，即输入了视图方向，那么我们就应该考虑对视图方向作出处理，用以生成颜色
        input_dirs = viewdirs[:, None].expand(inputs.shape)
        input_dirs_flat =jt.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)   # 对输入方向进行编码
        embedded = jt.concat([embedded, embedded_dirs], -1)    # 将编码后的方向加入聚合

    outputs_flat, attention_cache, loss_translation = batchify(fn, netchunk, aud_para, world_fn = world_fn, gather_func = gather_func, lip_rect=lip_rect)([embedded, attention_poses,
                                                                                           intrinsic, images_features, inputs_flat], training)

    outputs = jt.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs, attention_cache, loss_translation


def batchify_rays(lip_rect, rays_flat, bc_rgb, aud_para, chunk=1024*32, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}
    all_loss_translation = []

    # chunk 是并行处理的光束数量
    for i in range(0, rays_flat.shape[0], chunk):
        ret, loss_translation = render_rays(rays_flat[i:i+chunk], bc_rgb[i:i+chunk],
                          aud_para, lip_rect=lip_rect, **kwargs)
        all_loss_translation.append(loss_translation)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])
    loss_translation = jt.mean(jt.stack(all_loss_translation))

    all_ret = {k: jt.concat(all_ret[k], 0) for k in all_ret}
    return all_ret, loss_translation


def render_dynamic_face_new(H, W, focal, cx, cy, chunk=1024*32, rays=None, bc_rgb=None, aud_para=None,
                        c2w=None, ndc=True, near=0., far=1.,
                        use_viewdirs=False, c2w_staticcam=None, attention_images=None, attention_poses=None, intrinsic=None, render_pose=None,
                        attention_embed_fn=None, attention_embed_ln=None, feature_extractor=None, rotation_embed_fn = None, rotation_embed_ln = None, use_render_pose = True, lip_rect=None,
                        **kwargs):
    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, focal, c2w, cx, cy)
        bc_rgb = bc_rgb.reshape(-1, 3)
    else:
        rays_o, rays_d = rays

    # 如果使用视图方向，根据光线的 ray_d 计算单位方向作为 view_dirs
    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, focal, c2w_staticcam, cx, cy)
        viewdirs = viewdirs / jt.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = jt.reshape(viewdirs, [-1, 3]).float()

    sh = rays_d.shape  # [..., 3]
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, focal, 1., rays_o, rays_d)

    # Create ray batch
    rays_o = jt.reshape(rays_o, [-1, 3]).float()
    rays_d = jt.reshape(rays_d, [-1, 3]).float()

    # 生成光线的远近端，用于确定边界框，并将其聚合到 rays 中
    near, far = near * \
        jt.ones_like(rays_d[..., :1]), far * \
        jt.ones_like(rays_d[..., :1])
    rays = jt.concat([rays_o, rays_d, near, far], -1)

    # 视图方向聚合到光线中
    if use_viewdirs:
        rays = jt.concat([rays, viewdirs], -1)

    #module for image feature
    viewpoints = attention_poses[...,3]
    embedded_viewpoints = attention_embed_fn(viewpoints)
    bc_viewpoints = jt.broadcast(embedded_viewpoints[:,None,None], attention_images.shape[:-1] + (attention_embed_ln,))
    if use_render_pose:
        bc_render_transl = jt.broadcast(attention_embed_fn(render_pose[...,3])[None,None,None], attention_images.shape[:-1] + (attention_embed_ln,))
        bc_viewpoints = jt.concat((bc_viewpoints, bc_render_transl), -1)

    rgb_vp=jt.concat((attention_embed_fn(attention_images),bc_viewpoints),-1)
    rgb_vp = rgb_vp.permute(0, 3, 1, 2)
    images_features = feature_extractor(rgb_vp, attention_embed_ln)

    all_ret, loss_translation = batchify_rays(lip_rect, rays, bc_rgb, aud_para, chunk,attention_poses=attention_poses,intrinsic=intrinsic,images_features=images_features,**kwargs)

    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = jt.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map', 'last_weight']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k: all_ret[k] for k in all_ret if k not in k_extract}

    return ret_list + [ret_dict] + [loss_translation]


def render_path(args, torso_bcs, render_poses, aud_paras, bc_img, hwfcxy, attention_poses, attention_images,intrinsic,
                chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0, lip_rect=None):
    H, W, focal, cx, cy = hwfcxy

    if render_factor != 0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor

    rgbs = []
    disps = []
    last_weights = []
    for i, c2w in enumerate(tqdm(render_poses)):
        if i>=0:
            bc_img = jt.array(np.array(imageio.v2.imread(torso_bcs[i]))).float() / 255.0
            rgb, disp, acc, last_weight, _, _ = render_dynamic_face_new(
                H, W, focal, cx, cy, chunk=chunk, c2w=c2w[:3, :4], aud_para=aud_paras[i], bc_rgb=bc_img,
                attention_poses=attention_poses, attention_images=attention_images, intrinsic=intrinsic, render_pose=None, lip_rect = lip_rect, **render_kwargs)
            rgbs.append(rgb.numpy())
            disps.append(disp.numpy())
            last_weights.append(last_weight.numpy())
            if i == 0:
                print(rgb.shape, disp.shape)

            if savedir is not None:
                rgb8 = to8b(rgbs[-1])
                filename = os.path.join(savedir, '{:03d}.png'.format(i))
                imageio.v2.imwrite(filename, rgb8)
        else:
            continue

    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)
    last_weights = np.stack(last_weights, 0)

    return rgbs, disps, last_weights


def create_nerf(args):
    """Instantiate NeRF's MLP model.
    """
    # 位置编码，返回一个编码器embed_fn以及编码后的维数input_ch
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(
            args.multires_views, args.i_embed)
    output_ch = 5 if args.N_importance > 0 else 4
    skips = [4]

    attention_embed_fn, attention_embed_ln = get_embedder(5,0)
    attention_embed_fn_2, attention_embed_ln_2 = get_embedder(2,0,9)
    if args.dataset_type == 'llff' or args.use_quaternion or args.use_rotation_embed: #or args.dataset_type == 'shapenet':
        rotation_embed_fn, rotation_embed_ln = get_embedder(2,0,4)
    else:
        rotation_embed_fn, rotation_embed_ln = None, 0

    # 对应论文3.2节中对 N reference images 用 a two-layer convolutional network 提取 pixel aligned 特征
    model_obj = Feature_extractor()#shoule be: num_embed * embed_ln + num_rot * rotation_embed_ln
    grad_vars = list(model_obj.parameters())

    # 对应论文3.3节中 a three-layer MLP，args.num_reference_images 默认是 4
    position_warp = Position_warp(255, args.num_reference_images)
    grad_vars += list(position_warp.parameters())

    hidden_dim = 128
    iters = 2
    num_slots = 2
    num_features = num_slots * hidden_dim
    attention_module = SlotAttention(num_slots, hidden_dim, 130, iters=iters)
    grad_vars += list(attention_module.parameters())

    nerf_model = Face_Feature_NeRF(D=args.netdepth, W=args.netwidth,
                     input_ch=input_ch, dim_aud=args.dim_aud,
                     output_ch=output_ch, skips=skips,
                     input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs, dim_image_features = num_features)
    nerf_model_with_attention = nerf_attention_model(nerf_model, slot_att=attention_module, embed_fn=attention_embed_fn,
                                                     embed_ln=input_ch, embed_fn_2=attention_embed_fn_2, embed_ln_2=attention_embed_ln_2, coarse=True, num_samples=args.N_samples)
    grad_vars += list(nerf_model.parameters())

    models = {'model': nerf_model_with_attention, 'attention_model': attention_module, 'feature_extractor': model_obj}

    nerf_model_fine = None
    if args.N_importance > 0:
        nerf_model_fine = Face_Feature_NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                                       input_ch=input_ch, dim_aud=args.dim_aud,
                                       output_ch=output_ch, skips=skips,
                                       input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs, dim_image_features=num_features)
        nerf_model_fine_with_attention = nerf_attention_model(nerf_model_fine, attention_module, attention_embed_fn, input_ch,
                                               attention_embed_fn_2, attention_embed_ln_2, coarse=False, num_samples=args.N_importance+args.N_samples)
        models['model_fine'] = nerf_model_fine_with_attention
        grad_vars += list(nerf_model_fine.parameters())

    #feature fusion module
    world_fn = lambda x: x
    index_func = make_indices
    def gather_indices(pts, attention_poses, intrinsic, images_features, aud_para, lip_rect):
        H,W = images_features.shape[1:3]
        H=int(H)
        W=int(W)

        indices = jt.round(index_func(pts, attention_poses, intrinsic, H, W)).int()
        indices = indices.long()
        # default args.use_feature_map = False
        # pts的特征
        if not args.use_feature_map:
            features = [images_features[i][indices[i][:,0],indices[i][:,1]] for i in range(images_features.shape[0])]
            features = jt.concat([features[i].unsqueeze(0) for i in range(images_features.shape[0])], 0)
        else:
            features = [images_features[i][jt.meshgrid(indices[i][:, 0], indices[i][:, 1])[0].reshape(-1,2)] for i in range(images_features.shape[0])]
            features = jt.concat([features[i].unsqueeze(0) for i in range(images_features.shape[0])], 0)

        #3d positional encoding
        embed_fn_warp, input_ch_warp = get_embedder(10, 0)
        translation = jt.stack([position_warp(embed_fn_warp(pts), aud_para.detach(), features[i].detach()) for i in range(args.num_reference_images)])#[4,65536,3]
        #rect = lip_rect
        # 对应论文中 Lr 公式(5)
        loss_translation = jt.mean(translation**2,0)
        indices = indices + translation
        indices = jt.maximum(jt.minimum(indices, jt.array([H - 1., W - 1.])), jt.array([0, 0]))

        def grid_sampler_unnormalize(coord, size, align_corners):
            if align_corners:
                return 2*coord/(size-1)-1
            else:
                return (2*coord+1)/size-1

        indices_ = grid_sampler_unnormalize(indices, H, align_corners=False)

        if args.render_only:
            try:
                indices_ = indices_.reshape(indices_.shape[0], 1024, -1, indices_.shape[2])
            except:
                pdb.set_trace()
        else:
            indices_ = indices_.reshape(indices_.shape[0], args.N_rand, -1, indices_.shape[2])

        indices_ = jt.concat((indices_[:,:,:,1].unsqueeze(-1),indices_[:,:,:,0].unsqueeze(-1)),-1)
        # Since the hard index operation Fn(un′, vn′) is not differentiable
        # introduce a soft index function
        # bilinear sampling
        features = nn.grid_sample(images_features.permute(0,3,1,2), indices_, padding_mode='border', align_corners=False)
        features = features.reshape(args.num_reference_images, 128, -1).permute(0,2,1)
        return jt.concat((features, indices.int().float()), -1), loss_translation

    def network_query_fn(inputs, viewdirs, aud_para, network_fn, attention_poses, intrinsic, training, images_features, netchunk, lip_rect): \
        return run_network(inputs, viewdirs, aud_para, network_fn,
                           embed_fn=embed_fn, embeddirs_fn=embeddirs_fn, netchunk=netchunk, attention_poses = attention_poses,
                           intrinsic = intrinsic, training = training, images_features = images_features, world_fn = world_fn, gather_func = gather_indices, lip_rect=lip_rect)

    # Create optimizer
    optimizer = jt.optim.Adam(
        params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0
    basedir = args.basedir  # basedir = dataset/finetune_models
    expname = args.expname

    ##########################

    # Load checkpoints
    # train_base: ft_path = None
    # train: ft_path = dataset/base_model/300000_head.tar
    # test: ft_path = dataset/finetune_models/${names}/${iters}
    if args.ft_path is not None and args.ft_path != 'None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in natsorted(
            os.listdir(os.path.join(basedir, expname))) if 'tar' in f]

    print('Found ckpts', ckpts)
    learned_codes_dict = None
    AudNet_state = None
    AudAttNet_state = None
    optimizer_aud_state = None
    optimizer_audatt_state = None

    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = jt.load(ckpt_path)

        start = ckpt['global_step']
        if args.render_only:
            optimizer.load_state_dict(ckpt['optimizer_state_dict']) 
        AudNet_state = ckpt['network_audnet_state_dict']
        optimizer_aud_state = ckpt['optimizer_aud_state_dict']
        
        # Load model
        nerf_model_with_attention.load_state_dict(ckpt['network_fn_state_dict'])
        model_obj.load_state_dict(ckpt['unet_state_dict'])
        if args.render_only:
            position_warp.load_state_dict(ckpt['position_warp_state_dict'])
        attention_module.load_state_dict(ckpt['attention_state_dict'])

        if nerf_model_fine is not None:
            print('Have reload the fine model parameters. ')
            nerf_model_fine_with_attention.load_state_dict(ckpt['network_fine_state_dict'])
        if 'network_audattnet_state_dict' in ckpt:
            AudAttNet_state = ckpt['network_audattnet_state_dict']
        if 'optimize_audatt_state_dict' in ckpt:
            optimizer_audatt_state = ckpt['optimize_audatt_state_dict']
    models['optimizer'] = optimizer
    ##########################

    render_kwargs_train = {
        'network_query_fn': network_query_fn,
        'perturb': args.perturb,
        'N_importance': args.N_importance,  # 每条光线上细采样点的数量
        'network_fine': nerf_model_fine_with_attention,
        'N_samples': args.N_samples,    # 每条光线上粗采样点的数量
        'network_fn': nerf_model_with_attention, 
        'use_viewdirs': args.use_viewdirs,  # 是否使用视点方向
        'white_bkgd': args.white_bkgd,  # 是否将透明背景用白色填充
        'raw_noise_std': args.raw_noise_std,    # 归一化密度
        'training': True,
        'feature_extractor': model_obj,
        'position_warp_model':position_warp,
        'attention_embed_fn': attention_embed_fn,
        'attention_embed_ln': attention_embed_ln,
        'rotation_embed_fn': rotation_embed_fn,
        'rotation_embed_ln': rotation_embed_ln,
        'use_render_pose': not args.no_render_pose
    }

    # NDC only good for LLFF-style forward facing data
    # NDC 空间，只对前向场景有效，具体解释见 NeRF 论文
    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {
        k: render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.
    render_kwargs_test['training'] = False

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer, learned_codes_dict, \
        AudNet_state, optimizer_aud_state, AudAttNet_state, optimizer_audatt_state, models


def raw2outputs(raw, z_vals, rays_d, bc_rgb, raw_noise_std=0, white_bkgd=False, pytest=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """

    def raw2alpha(raw, dists, act_fn=nn.relu): return 1. - \
        jt.exp(-(act_fn(raw)+1e-6)*dists)

    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = jt.concat([dists, jt.array([1e10]).expand(
        dists[..., :1].shape)], -1)  # [N_rays, N_samples]

    dists = dists * jt.norm(rays_d[..., None, :], dim=-1)

    # 每个点的 RGB 值
    rgb = jt.sigmoid(raw[..., :3])  # [N_rays, N_samples, 3]
    
    # We treat the background, torso and neck parts together as the rendering 'background' 
    # and restore it frame by frame from the original videos.
    # We set the color of the last point of each ray as the corresponding background pixel 
    # to render a natural background including the torso part.
    rgb = jt.concat((rgb[:, :-1, :], bc_rgb.unsqueeze(1)), dim=1)
    noise = 0.
    if raw_noise_std > 0.:
        noise = jt.randn(raw[..., 3].shape) * raw_noise_std

        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[..., 3].shape)) * raw_noise_std
            noise = jt.array(noise)

    alpha = raw2alpha(raw[..., 3] + noise, dists)  # [N_rays, N_samples]
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    weights = alpha * \
        jt.cumprod(
            jt.concat([jt.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = jt.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]
    depth_map = jt.sum(weights * z_vals, -1)
    disp_map = 1./jt.maximum(1e-10 * jt.ones_like(depth_map),
                            depth_map / jt.sum(weights, -1))
    acc_map = jt.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map[..., None])

    return rgb_map, disp_map, acc_map, weights, depth_map, weights*z_vals


def render_rays(ray_batch,
                bc_rgb,
                aud_para,
                network_fn,
                network_query_fn,
                N_samples,
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                white_bkgd=False,
                raw_noise_std=0.,
                verbose=False,
                pytest=False,
                training=False,attention_poses=None,intrinsic=None,images_features=None,position_warp_model=None, lip_rect=None):
    """Volumetric rendering.
    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn: function. Model for predicting RGB and density at each point
        in space.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      white_bkgd: bool. If True, assume a white background.
      raw_noise_std: ...
      verbose: bool. If True, print more debugging info.
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]  # [N_rays, 3] each
    viewdirs = ray_batch[:, -3:] if ray_batch.shape[-1] > 8 else None
    bounds = jt.reshape(ray_batch[..., 6:8], [-1, 1, 2])
    near, far = bounds[..., 0], bounds[..., 1]  # [-1,1]

    # 每个光束上取 N_samples 个点，默认 64 个
    t_vals = jt.linspace(0., 1., steps=N_samples)
    if not lindisp:
        z_vals = near * (1.-t_vals) + far * (t_vals)
    else:
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

    z_vals = z_vals.expand([N_rays, N_samples])

    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = jt.concat([mids, z_vals[..., -1:]], -1)
        lower = jt.concat([z_vals[..., :1], mids], -1)
        # stratified samples in those intervals
        t_rand = jt.rand(z_vals.shape)

        # Pytest, overwrite u with numpy's fixed random numbers
        if pytest:
            np.random.seed(0)
            t_rand = np.random.rand(*list(z_vals.shape))
            t_rand = jt.array(t_rand)
        t_rand[..., -1] = 1.0
        z_vals = lower + (upper - lower) * t_rand

    # 生成光线上每个采样点的位置
    pts = rays_o[..., None, :] + rays_d[..., None, :] * \
        z_vals[..., :, None]  # [N_rays, N_samples, 3]

    # 将光线上的每个点投入到 MLP 网络 network_fn 中前向传播得到每个点对应的 （RGB，A）
    raw, attention_cache, loss_translation = network_query_fn(pts, viewdirs, aud_para, network_fn, attention_poses, intrinsic, training, images_features, 1024*64, lip_rect)
    # 这一步相当于是在做 volume render，将光束颜色合成图像上的点
    rgb_map, disp_map, acc_map, weights, depth_map, depth_grid = raw2outputs(raw, z_vals, rays_d, bc_rgb, raw_noise_std, white_bkgd, pytest=pytest)

    # 下面是有精细网络的情况，会再算一遍上述步骤，然后也封装到ret
    if N_importance > 0:

        rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map

        z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        z_samples = sample_pdf(
            z_vals_mid, weights[..., 1:-1], N_importance, det=(perturb == 0.), pytest=pytest)
        z_samples = z_samples.detach()

        # z_vals, _ = jt.sort(jt.concat([z_vals, z_samples], -1), -1)
        _, z_vals = jt.argsort(jt.concat([z_vals, z_samples], -1), -1, False)
        
        pts = rays_o[..., None, :] + rays_d[..., None, :] * \
            z_vals[..., :, None]  # [N_rays, N_samples + N_importance, 3]

        run_fn = network_fn if network_fine is None else network_fine

        raw, _, loss_translation = network_query_fn(pts, viewdirs, aud_para, run_fn, attention_poses, intrinsic, training, images_features, 1024*64*3, lip_rect)
        rgb_map, disp_map, acc_map, weights, depth_map, depth_grid = raw2outputs(
            raw, z_vals, rays_d, bc_rgb, raw_noise_std, white_bkgd, pytest=pytest)

    # According to (6) Lr' = (1-density) * Lr
    loss_translation = jt.mean(jt.mean(loss_translation, 1) * (1-jt.minimum(depth_grid.detach(), jt.array([1])).reshape(-1)))
    ret = {'rgb_map': rgb_map, 'disp_map': disp_map, 'acc_map': acc_map}
    if retraw:
        ret['raw'] = raw
    if N_importance > 0:
        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0
        ret['z_std'] = jt.var(z_samples, dim=-1, unbiased=False)  # [N_rays]
        ret['last_weight'] = weights[..., -1]

    # for k in ret:
    #     if (jt.isnan(ret[k]).any() or jt.isinf(ret[k]).any()) and DEBUG:
    #         print(f"! [Numerical Error] {k} contains nan or inf.")
    return ret, loss_translation


def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')
    parser.add_argument("--expname", type=str,
                        help='experiment name')
    parser.add_argument("--expname_finetune", type=str,
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/',
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern',
                        help='input data directory')

    # training options
    parser.add_argument("--netdepth", type=int, default=8,
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256,
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8,
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256,
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=850,
                        help='batch size (number of random rays per gradient step)')#1024
    parser.add_argument("--lrate", type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=500,
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024,
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*64,
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_false',
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None,
                        help='specific weights npy file to reload for coarse network')
    parser.add_argument("--N_iters", type=int, default=600000,
                        help='number of iterations')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64,
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=128,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_false',
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0,
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10,
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4,
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0.,
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true',
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # training options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops')

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='audface',
                        help='options: llff / blender / deepvoxels')
    parser.add_argument("--testskip", type=int, default=1,
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    # deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek',
                        help='options : armchair / cube / greek / vase')

    # blender flags
    parser.add_argument("--white_bkgd", action='store_false',
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true',
                        help='load blender synthetic data at 400x400 instead of 800x800')

    # face flags
    parser.add_argument("--with_test", type=int, default=0,
                        help='whether to use test set')
    parser.add_argument("--dim_aud", type=int, default=64,
                        help='dimension of audio features for NeRF')
    parser.add_argument("--sample_rate", type=float, default=0.95,
                        help="sample rate in a bounding box")
    parser.add_argument("--near", type=float, default=0.3,
                        help="near sampling plane")
    parser.add_argument("--far", type=float, default=0.9,
                        help="far sampling plane")
    parser.add_argument("--test_file", type=str, default='transforms_test.json',
                        help='test file')
    parser.add_argument("--aud_file", type=str, default='aud.npy',
                        help='test audio deepspeech file')
    parser.add_argument("--win_size", type=int, default=16,
                        help="windows size of audio feature")
    parser.add_argument("--smo_size", type=int, default=8,
                        help="window size for smoothing audio features")
    parser.add_argument('--nosmo_iters', type=int, default=100000,
                        help='number of iterations befor applying smoothing on audio features')#300000

    # llff flags
    parser.add_argument("--factor", type=int, default=8,
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true',
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true',
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true',
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8,
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=100,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img",     type=int, default=500,
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=5000,
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=150000,
                        help='frequency of testset saving')#10000
    parser.add_argument("--i_video",   type=int, default=50000,
                        help='frequency of render_poses video saving')
    #add some paras
    parser.add_argument("--num_reference_images",   type=int, default=4,
                        help='the number of reference input images k')
    parser.add_argument("--select_nearest",   type=int, default=0,
                        help='whether to select the k-nearest images as the reference')
    parser.add_argument("--use_quaternion",   type=bool, default=False)
    parser.add_argument("--use_rotation_embed",   type=bool, default=False)
    parser.add_argument("--no_render_pose", type=bool, default=True)
    parser.add_argument("--use_warp", type=bool, default=True)
    parser.add_argument("--indices_before_iter", type=int, default=0)
    parser.add_argument("--translation_iter", type=int, default=0)
    parser.add_argument("--L2loss_weight", type=float, default=5e-9)
    parser.add_argument("--use_feature_map", type=bool, default=False)
    parser.add_argument("--selectimg_for_heatmap", type=int, default=0)
    parser.add_argument("--train_length", type=int, default=15)
    parser.add_argument("--need_torso", type=bool, default=True)
    parser.add_argument("--bc_type", type=str, default='torso_imgs')
    parser.add_argument("--refer_from_train", type=int, default=1)
    return parser


def train():

    parser = config_parser()
    args = parser.parse_args()
    print(args.near, args.far)
    # Load data

    if args.dataset_type == 'audface':
        if args.with_test == 1:
            images, poses, auds, bc_img, hwfcxy, lip_rects, torso_bcs, _ = \
                load_audface_data(args.datadir, 
                                    args.testskip,  # testskip=1
                                    args.test_file,  # transforms_val.json
                                    args.aud_file, 
                                    need_lip=True, 
                                    need_torso = args.need_torso,
                                    bc_type=args.bc_type)
            #images = np.zeros(1)
        else:
            images, poses, auds, bc_img, hwfcxy, sample_rects, i_split, id_num, lip_rects, torso_bcs = \
                load_audface_data(args.datadir,
                                    args.testskip, 
                                    train_length = args.train_length, 
                                    need_lip=True, 
                                    need_torso = args.need_torso, 
                                    bc_type=args.bc_type)

        #print('Loaded audface', images['0'].shape, hwfcxy, args.datadir)
        if args.with_test == 0:
            print('Loaded audface', images['0'].shape, hwfcxy, args.datadir)
            #all id has the same split, so this part can be shared
            i_train, i_val = i_split['0']
        else:
            print('Loaded audface', len(images), hwfcxy, args.datadir)
        near = args.near
        far = args.far
    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    # Cast intrinsics to right types
    H, W, focal, cx, cy = hwfcxy
    H, W = int(H), int(W)
    hwf = [H, W, focal]
    hwfcxy = [H, W, focal, cx, cy]

    intrinsic = np.array([[focal, 0., W / 2],
                          [0, focal, H / 2],
                          [0, 0, 1.]])
    intrinsic = jt.array(intrinsic).float()

    # if args.render_test:
    #     render_poses = np.array(poses[i_test])

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    expname_finetune = args.expname_finetune
    os.makedirs(os.path.join(basedir, expname_finetune), exist_ok=True)
    f = os.path.join(basedir, expname_finetune, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname_finetune, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer, learned_codes, \
    AudNet_state, optimizer_aud_state, AudAttNet_state, optimizer_audatt_state, models = create_nerf(args)

    global_step = start

    AudNet = AudioNet(args.dim_aud, args.win_size)
    AudAttNet = AudioAttNet()
    optimizer_Aud = jt.optim.Adam(
        params=list(AudNet.parameters()), lr=args.lrate, betas=(0.9, 0.999))
    optimizer_AudAtt = jt.optim.Adam(
        params=list(AudAttNet.parameters()), lr=args.lrate, betas=(0.9, 0.999))

    if AudNet_state is not None:
        AudNet.load_state_dict(AudNet_state)
    if optimizer_aud_state is not None:
        optimizer_Aud.load_state_dict(optimizer_aud_state)
    if AudAttNet_state is not None:
        AudAttNet.load_state_dict(AudAttNet_state)
    if optimizer_audatt_state is not None:
        optimizer_AudAtt.load_state_dict(optimizer_audatt_state)
    bds_dict = {
        'near': near,
        'far': far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    if args.render_only:
        print('RENDER ONLY')
        with jt.no_grad():
            images_refer, poses_refer, auds_refer, bc_img_refer, _ , lip_rects_refer, _, _ = \
                load_audface_data(args.datadir, args.testskip, 'transforms_train.json', args.aud_file, need_lip=True, need_torso=False, bc_type=args.bc_type)

            images_refer = jt.concat([jt.array(np.array(imageio.v2.imread(images_refer[i]))).unsqueeze(0) for i in
                                range(len(images_refer))], 0).float()/255.0
            poses_refer = jt.array(poses_refer).float()
            # Default is smoother render_poses path
            #the data loader return these: images, poses, auds, bc_img, hwfcxy
            bc_img = jt.array(np.array(bc_img)).float() / 255.0
            poses = jt.array(poses).float()
            auds = jt.array(auds).float()
            testsavedir = os.path.join(basedir, expname_finetune, 'renderonly_{}_{:06d}'.format(
                'test' if args.render_test else 'path', start))
            os.makedirs(testsavedir, exist_ok=True)

            print('test poses shape', poses.shape)
            #select reference images for the test set
            if args.refer_from_train:
                perm = [50,100,150,200]
                perm = perm[0:args.num_reference_images]
                attention_images = images_refer[perm]
                attention_poses = poses_refer[perm, :3, :4]
            else:
                perm = np.random.randint(images_refer.shape[0]-1, size=4).tolist()
                attention_images_ = np.array(images)[perm]
                attention_images = jt.concat([jt.array(np.array(imageio.v2.imread(i))).unsqueeze(0) for i in
                                          attention_images_], 0).float() / 255.0
                attention_poses = poses[perm, :3, :4]

            auds_val = []
            if start < args.nosmo_iters:
                auds_val = AudNet(auds)
            else:
                print('Load the smooth audio for rendering!')
                for i in range(poses.shape[0]):
                    smo_half_win = int(args.smo_size / 2)
                    left_i = i - smo_half_win
                    right_i = i + smo_half_win
                    pad_left, pad_right = 0, 0
                    if left_i < 0:
                        pad_left = -left_i
                        left_i = 0
                    if right_i > poses.shape[0]:
                        pad_right = right_i - poses.shape[0]
                        right_i = poses.shape[0]
                    auds_win = auds[left_i:right_i]
                    if pad_left > 0:
                        auds_win = jt.concat(
                            (jt.zeros_like(auds_win)[:pad_left], auds_win), dim=0)
                    if pad_right > 0:
                        auds_win = jt.concat(
                            (auds_win, jt.zeros_like(auds_win)[:pad_right]), dim=0)
                    auds_win = AudNet(auds_win)
                    #aud = auds_win[smo_half_win]
                    aud_smo = AudAttNet(auds_win)
                    auds_val.append(aud_smo)
                auds_val = jt.stack(auds_val, 0)

            with jt.no_grad():
                rgbs, disp, last_weight = render_path(args, torso_bcs, poses, auds_val, bc_img, hwfcxy, attention_poses,attention_images,
                            intrinsic, args.chunk, render_kwargs_test, gt_imgs=None, savedir=testsavedir, lip_rect=lip_rects_refer[perm])

            np.save(os.path.join(testsavedir, 'last_weight.npy'), last_weight)
            print('Done rendering', testsavedir)
            # imageio.v2.mimwrite(os.path.join(
            #     testsavedir, 'video.mp4'), to8b(rgbs), fps=25, quality=8)
            return


    # Prepare raybatch tensor if batching random rays
    N_rand = args.N_rand
    print('N_rand', N_rand, 'no_batching',
          args.no_batching, 'sample_rate', args.sample_rate)
    use_batching = not args.no_batching

    if use_batching:
        # For random ray batching
        print('get rays')
        rays = np.stack([get_rays_np(H, W, focal, p, cx, cy)
                         for p in poses[:, :3, :4]], 0)  # [N, ro+rd, H, W, 3]
        print('done, concats')
        # [N, ro+rd+rgb, H, W, 3]
        rays_rgb = np.concatenate([rays, images[:, None]], 1)
        # [N, H, W, ro+rd+rgb, 3]
        rays_rgb = np.transpose(rays_rgb, [0, 2, 3, 1, 4])
        rays_rgb = np.stack([rays_rgb[i]
                             for i in i_train], 0)  # train images only
        # [(N-1)*H*W, ro+rd+rgb, 3]
        rays_rgb = np.reshape(rays_rgb, [-1, 3, 3])
        rays_rgb = rays_rgb.astype(np.float32)
        print('shuffle rays')
        np.random.shuffle(rays_rgb)

        print('done')
        i_batch = 0

    if use_batching:
        rays_rgb = jt.array(rays_rgb)

    N_iters = args.N_iters + 1
    print('Begin')
    print('TRAIN views are', i_train)
    print('VAL views are', i_val)

    start = start + 1
    for i in trange(start, N_iters):
        time0 = time.time()
        # Sample random ray batch
        # use_batching = False (defalut)
        if use_batching:
            # Random over all images
            batch = rays_rgb[i_batch:i_batch+N_rand]  # [B, 2+1, 3*?]
            batch = jt.transpose(batch, 0, 1)
            batch_rays, target_s = batch[:2], batch[2]

            i_batch += N_rand
            if i_batch >= rays_rgb.shape[0]:
                print("Shuffle data after an epoch!")
                rand_idx = jt.randperm(rays_rgb.shape[0])
                rays_rgb = rays_rgb[rand_idx]
                i_batch = 0

        else:
            # Random from one image
            #1.Select a id for training
            # 随机选取一个id来训练
            select_id = np.random.choice(id_num)
            #bc_img_ = torch.Tensor(bc_img[str(select_id)]).to(device).float()/255.0
            poses_ = jt.array(poses[str(select_id)]).float()
            auds_ = jt.array(auds[str(select_id)]).float()
            i_train, i_val = i_split[str(select_id)]
            img_i = np.random.choice(i_train)
            image_path = images[str(select_id)][img_i]

            # In 3.4 We treat the background, torso and neck parts together as the rendering 'background'
            bc_img_ = jt.array(np.array(imageio.v2.imread(torso_bcs[str(select_id)][img_i]))).float()/255.0

            # 随机选取该id的一张图像和对应aud帧来训练
            target = jt.array(np.array(imageio.v2.imread(image_path))).float()/255.0
            pose = poses_[img_i, :3, :4]
            rect = sample_rects[str(select_id)][img_i]  # face_rect
            aud = auds_[img_i]

            #select the attention pose and image
            if args.select_nearest: # whether to select the k-nearest images as the reference
                current_poses = poses[str(select_id)][:, :3, :4]
                current_images = images[str(select_id)]  # top size was set at 4 for reflective ones
                current_images = jt.concat([jt.array(np.array(imageio.v2.imread(current_images[i]))).unsqueeze(0) for i in range(current_images.shape[0])], 0)
                current_images = current_images.float() / 255.0
                attention_poses, attention_images = get_similar_k(pose, current_poses, current_images, top_size=None, k = 20)
            else:
                # 删掉 img_i 后剩下的图像
                i_train_left = np.delete(i_train, np.where(np.array(i_train) == img_i))
                # np.random.permutation() 随机排列序列
                perm = np.random.permutation(i_train_left)[:args.num_reference_images]#selete num_reference_images images from the training set as reference
                attention_images = images[str(select_id)][perm]
                attention_images = jt.concat([jt.array(np.array(imageio.v2.imread(attention_images[i]))).unsqueeze(0) for i in range(args.num_reference_images)],0)
                attention_images = attention_images.float()/255.0
                attention_poses = poses[str(select_id)][perm, :3, :4]
                lip_rect = jt.array(lip_rects[str(select_id)][perm])

            attention_poses = jt.array(attention_poses).float()
            if global_step >= args.nosmo_iters:
                smo_half_win = int(args.smo_size / 2)   # args.smo_size default 8
                left_i = img_i - smo_half_win
                right_i = img_i + smo_half_win
                pad_left, pad_right = 0, 0
                if left_i < 0:
                    pad_left = -left_i
                    left_i = 0
                if right_i > i_train.shape[0]:
                    pad_right = right_i-i_train.shape[0]
                    right_i = i_train.shape[0]
                auds_win = auds_[left_i:right_i]
                if pad_left > 0:
                    auds_win = jt.concat(
                        (jt.zeros_like(auds_win)[:pad_left], auds_win), dim=0)
                if pad_right > 0:
                    auds_win = jt.concat(
                        (auds_win, jt.zeros_like(auds_win)[:pad_right]), dim=0)
                auds_win = AudNet(auds_win)     # 窗口内 auds 提取出来的特征 auds_window
                aud = auds_win[smo_half_win]
                aud_smo = AudAttNet(auds_win)   # 获取smooth后的aud特征
            else:
                aud = AudNet(aud.unsqueeze(0))
            if N_rand is not None:
                # 生成这张图像中每个像素点对应的光线的原点和方向
                rays_o, rays_d = get_rays(
                    H, W, focal, jt.array(pose), cx, cy)  # (H, W, 3), (H, W, 3)
                
                # 生成每个像素点的笛卡尔坐标，前 precrop_iters 生成图像中心的像素坐标坐标
                if i < args.precrop_iters:
                    dH = int(H//2 * args.precrop_frac)
                    dW = int(W//2 * args.precrop_frac)
                    coords = jt.stack(
                        jt.meshgrid(
                            jt.linspace(H//2 - dH, H//2 + dH - 1, 2*dH),
                            jt.linspace(W//2 - dW, W//2 + dW - 1, 2*dW)
                        ), -1)
                    if i == start:
                        print(
                            f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {args.precrop_iters}")
                else:
                    # 生成图像中每个像素的坐标
                    coords = jt.stack(jt.meshgrid(jt.linspace(
                        0, H-1, H), jt.linspace(0, W-1, W)), -1)  # (H, W, 2)

                coords = jt.reshape(coords, [-1, 2])  # (H * W, 2)

                # 在训练的时候并不是给图像中每个像素都打光线，而是加载一批光线，批大小为 N_rand
                if args.sample_rate > 0:
                    rect_inds = (coords[:, 0] >= rect[0]) & (
                        coords[:, 0] <= rect[0] + rect[2]) & (
                            coords[:, 1] >= rect[1]) & (
                                coords[:, 1] <= rect[1] + rect[3])
                    # face
                    coords_rect = coords[rect_inds]
                    # not face
                    coords_norect = coords[jt.array(~(rect_inds.numpy()))]
                    # 根据采样率计算 N_rand 在脸部的分配的光线
                    rect_num = int(N_rand*args.sample_rate)
                    norect_num = N_rand - rect_num
                    select_inds_rect = np.random.choice(
                        coords_rect.shape[0], size=[rect_num], replace=False)  # (N_rand,)
                    # (N_rand, 2)
                    select_coords_rect = coords_rect[select_inds_rect].long()
                    select_inds_norect = np.random.choice(
                        coords_norect.shape[0], size=[norect_num], replace=False)  # (N_rand,)
                    # (N_rand, 2)
                    select_coords_norect = coords_norect[select_inds_norect].long(
                    )
                    select_coords = jt.concat(
                        (select_coords_rect, select_coords_norect), dim=0)
                else:
                    select_inds = np.random.choice(
                        coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
                    select_coords = coords[select_inds].long()

                rays_o = rays_o[select_coords[:, 0],
                                select_coords[:, 1]]  # (N_rand, 3)
                rays_d = rays_d[select_coords[:, 0],
                                select_coords[:, 1]]  # (N_rand, 3)
                batch_rays = jt.stack([rays_o, rays_d], 0)
                target_s = target[select_coords[:, 0],
                                  select_coords[:, 1]]  # (N_rand, 3)
                bc_rgb = bc_img_[select_coords[:, 0],
                                select_coords[:, 1]]


        #####  Core optimization loop  #####
        if global_step >= args.nosmo_iters:
            rgb, disp, acc, _, extras, loss_translation = render_dynamic_face_new(H, W, focal, cx, cy, chunk=args.chunk, rays=batch_rays,
                                                            aud_para=aud_smo, bc_rgb=bc_rgb,
                                                            verbose=i < 10, retraw=True, attention_images = attention_images,
                                                            attention_poses = attention_poses, intrinsic = intrinsic, render_pose=pose,lip_rect = lip_rect, **render_kwargs_train)
        else:
            rgb, disp, acc, _, extras, loss_translation = render_dynamic_face_new(H, W, focal, cx, cy, chunk=args.chunk, rays=batch_rays,
                                                            aud_para=aud, bc_rgb=bc_rgb,
                                                            verbose=i < 10, retraw=True, attention_images = attention_images,
                                                            attention_poses = attention_poses, intrinsic = intrinsic, render_pose=pose,lip_rect = lip_rect, **render_kwargs_train)

        
        img_loss = img2mse(rgb, target_s)
        trans = extras['raw'][..., -1]
        loss = img_loss + args.L2loss_weight * loss_translation
        psnr = mse2psnr(img_loss)

        if 'rgb0' in extras:
            img_loss0 = img2mse(extras['rgb0'], target_s)
            loss = loss + img_loss0
            psnr0 = mse2psnr(img_loss0)


        # loss.backward()
        # optimizer.step()
        # optimizer_Aud.step()
        optimizer.step(loss,retain_graph=True)
        if global_step >= args.nosmo_iters:
            optimizer_Aud.step(loss,retain_graph=True)
            optimizer_AudAtt.step(loss)
        else:
            optimizer_Aud.step(loss)
            
        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1500
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate

        for param_group in optimizer_Aud.param_groups:
            param_group['lr'] = new_lrate

        for param_group in optimizer_AudAtt.param_groups:
            param_group['lr'] = new_lrate*5
        ################################
        dt = time.time()-time0

        # Rest is logging
        if i % args.i_weights == 0:
            path = os.path.join(basedir, expname_finetune, '{:06d}_head.tar'.format(i))
            jt.save({
                'global_step': global_step,
                'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
                'network_audnet_state_dict': AudNet.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'optimizer_aud_state_dict': optimizer_Aud.state_dict(),
                'network_audattnet_state_dict': AudAttNet.state_dict(),
                'optimizer_audatt_state_dict': optimizer_AudAtt.state_dict(),
                'unet_state_dict': render_kwargs_train['feature_extractor'].state_dict(),
                'attention_state_dict': models['attention_model'].state_dict(),
                'position_warp_state_dict':render_kwargs_train['position_warp_model'].state_dict(),
            }, path)
            print('Saved checkpoints at', path)

        
        if i % args.i_print == 0:
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}")

        global_step += 1


if __name__ == '__main__':
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')

    train()
