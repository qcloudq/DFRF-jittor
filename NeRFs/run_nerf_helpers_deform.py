import numpy as np
from jittor import nn
import pdb
import imageio
import jittor as jt
import os
jt.flags.use_cuda = 1   # use cuda

# Misc
def img2mse(x, y): return jt.mean((x - y) ** 2)


def mse2psnr(x): return -10. * jt.log(x) / jt.log(jt.array([10.]))


def to8b(x): return (255*np.clip(x, 0, 1)).astype(np.uint8)


# Positional encoding (section 5.1)
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        # 如果包含原始位置
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)   # 把一个不对数据做出改变的匿名函数添加到列表中
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2.**jt.linspace(0., max_freq, steps=N_freqs)   # 得到 [2^0, 2^1, ... ,2^(L-1)] 参考NeRF论文 5.1 中的公式
        else:
            freq_bands = jt.linspace(2.**0., 2.**max_freq, steps=N_freqs)   # 得到 [2^0, 2^(L-1)] 的等差数列，列表中有 L 个元素

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn,
                                 freq=freq: p_fn(x * freq)) # sin(x * 2^n)  参考位置编码公式
                out_dim += d    # 每使用子编码公式一次就要把输出维度加 3，因为每个待编码的位置维度是 3

        self.embed_fns = embed_fns  # 相当于是一个编码公式列表
        self.out_dim = out_dim

    def embed(self, inputs):
        # 对各个输入进行编码，给定一个输入，使用编码列表中的公式分别对他编码
        return jt.concat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0, input_dims = 3):
    if i == -1:
        return nn.Identity(), 3

    embed_kwargs = {
        'include_input': True,  # 如果为真，最终的编码结果包含原始坐标
        'input_dims': input_dims,   # 输入给编码器的数据的维度
        'max_freq_log2': multires-1,
        'num_freqs': multires,  # 即NeRF论文中 5.1 节位置编码公式中的 L 
        'log_sampling': True,
        'periodic_fns': [jt.sin, jt.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    def embed(x, eo=embedder_obj): return eo.embed(x)   # embed 现在相当于一个编码器，具体的编码公式与NeRF论文中的一致。
    return embed, embedder_obj.out_dim


# Audio feature extractor
# Audio attention module
class AudioAttNet(nn.Module):
    def __init__(self, dim_aud=32, seq_len=8):
        super(AudioAttNet, self).__init__()
        self.seq_len = seq_len
        self.dim_aud = dim_aud
        self.attentionConvNet = nn.Sequential(  # b x subspace_dim x seq_len
            nn.Conv1d(self.dim_aud, 16, kernel_size=3,
                      stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02),
            nn.Conv1d(16, 8, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02),
            nn.Conv1d(8, 4, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02),
            nn.Conv1d(4, 2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02),
            nn.Conv1d(2, 1, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02)
        )
        self.attentionNet = nn.Sequential(
            nn.Linear(in_features=self.seq_len,
                      out_features=self.seq_len, bias=True),
            nn.Softmax(dim=1)
        )

    def execute(self, x):
        y = x[..., :self.dim_aud].permute(1, 0).unsqueeze(
            0)  # 2 x subspace_dim x seq_len
        y = self.attentionConvNet(y)
        y = self.attentionNet(y.view(1, self.seq_len)).view(self.seq_len, 1)
        # print(y.view(-1).data)
        return jt.sum(y*x, dim=0)
# Model


# Audio feature extractor
class AudioNet(nn.Module):
    def __init__(self, dim_aud=76, win_size=16):
        super(AudioNet, self).__init__()
        self.win_size = win_size
        self.dim_aud = dim_aud
        self.encoder_conv = nn.Sequential(  # n x 29 x 16
            nn.Conv1d(29, 32, kernel_size=3, stride=2,
                      padding=1, bias=True),  # n x 32 x 8
            nn.LeakyReLU(0.02),
            nn.Conv1d(32, 32, kernel_size=3, stride=2,
                      padding=1, bias=True),  # n x 32 x 4
            nn.LeakyReLU(0.02),
            nn.Conv1d(32, 64, kernel_size=3, stride=2,
                      padding=1, bias=True),  # n x 64 x 2
            nn.LeakyReLU(0.02),
            nn.Conv1d(64, 64, kernel_size=3, stride=2,
                      padding=1, bias=True),  # n x 64 x 1
            nn.LeakyReLU(0.02),
        )
        self.encoder_fc1 = nn.Sequential(
            nn.Linear(64, 64),
            nn.LeakyReLU(0.02),
            nn.Linear(64, dim_aud),
        )

    def execute(self, x):
        half_w = int(self.win_size/2)
        x = x[:, 8-half_w:8+half_w, :].permute(0, 2, 1)
        x = self.encoder_conv(x).squeeze(-1)
        # x = self.encoder_fc1(x).squeeze()
        x = self.encoder_fc1(x)
        shape = list(x.shape)
        new_shape = [s for s in shape if s > 1]
        x = x.reshape(new_shape)
        return x


# Ray helpers
def get_rays(H, W, focal, c2w, cx=None, cy=None):
    # pytorch's meshgrid has indexing='ij'
    i, j = jt.meshgrid(jt.linspace(0, W-1, W), jt.linspace(0, H-1, H))
    i = i.t()
    j = j.t()
    if cx is None:
        cx = W*.5
    if cy is None:
        cy = H*.5
    dirs = jt.stack(
        [(i-cx)/focal, -(j-cy)/focal, -jt.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    rays_d = jt.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d


def get_rays_np(H, W, focal, c2w, cx=None, cy=None):
    if cx is None:
        cx = W*.5
    if cy is None:
        cy = H*.5
    i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-cx)/focal, -(j-cy)/focal, -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))
    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[..., 0] / rays_o[..., 2]
    o1 = -1./(H/(2.*focal)) * rays_o[..., 1] / rays_o[..., 2]
    o2 = 1. + 2. * near / rays_o[..., 2]

    d0 = -1./(W/(2.*focal)) * \
        (rays_d[..., 0]/rays_d[..., 2] - rays_o[..., 0]/rays_o[..., 2])
    d1 = -1./(H/(2.*focal)) * \
        (rays_d[..., 1]/rays_d[..., 2] - rays_o[..., 1]/rays_o[..., 2])
    d2 = -2. * near / rays_o[..., 2]

    rays_o = jt.stack([o0, o1, o2], -1)
    rays_d = jt.stack([d0, d1, d2], -1)

    return rays_o, rays_d


# Hierarchical sampling (section 5.2)


def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / jt.sum(weights, -1, keepdims=True)
    cdf = jt.cumsum(pdf, -1)
    # (batch, len(bins))
    cdf = jt.concat([jt.zeros_like(cdf[..., :1]), cdf], -1)

    # Take uniform samples
    if det:
        u = jt.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = jt.rand(list(cdf.shape[:-1]) + [N_samples])

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = jt.array(u)

    # Invert CDF
    u = u.contiguous()
    inds = jt.searchsorted(cdf, u, right=True)
    below = jt.maximum(jt.zeros_like(inds-1), inds-1)
    above = jt.minimum((cdf.shape[-1]-1) * jt.ones_like(inds), inds)
    inds_g = jt.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = jt.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = jt.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1]-cdf_g[..., 0])
    denom = jt.ternary(denom < 1e-5, jt.ones_like(denom), denom)
    t = (u-cdf_g[..., 0])/denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1]-bins_g[..., 0])

    return samples


class Feature_extractor(nn.Module):
    def __init__(self, input_dim=66):
        super(Feature_extractor, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(input_dim, 64, kernel_size=3, stride=1, padding=1, bias=True), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True), nn.ReLU())
        self.fc = nn.Linear(64, 128)


    def execute(self, input, embed_ln):
        local = self.conv1(input)
        local = self.conv2(local)
        local = self.fc(local.permute(0,2,3,1))
        return local


#the attention module for feature fusion
class SlotAttention(nn.Module):
    def __init__(self, num_slots, dim, input_dim, iters=3, eps=1e-8, hidden_dim=128):
        super(SlotAttention, self).__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5
        self.dim = dim
        self.input_dim = input_dim

        self.slots_mu = nn.Parameter(jt.randn(1, 1, dim))

        self.slots_logsigma = nn.Parameter(jt.zeros(1, 1, dim))

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        self.gru = nn.GRUCell(dim, dim)

        hidden_dim = max(dim, hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim)
        )

        self.norm_input = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

        self.fc1 = nn.Sequential( nn.Linear(self.input_dim, 32), nn.ReLU())
        self.fc6 = nn.Linear(32, 32)
        self.fc7 = nn.Sequential(nn.Linear(225, dim), nn.LeakyReLU())

    def execute(self, inputs, embedded_pts, num_slots=None):
        max_rel = self.fc1(inputs)
        max_rel = self.fc6(max_rel)
        inputs = jt.concat((inputs, embedded_pts, max_rel), -1)
        inputs = self.fc7(inputs)

        b, n, d = inputs.shape
        n_s = num_slots if num_slots is not None else self.num_slots

        mu = self.slots_mu.expand(b, n_s, -1)
        sigma = self.slots_logsigma.exp().expand(b, n_s, -1)

        slots = mu + sigma * jt.randn(mu.shape)

        inputs = self.norm_input(inputs)
        k, v = self.to_k(inputs), self.to_v(inputs)

        for _ in range(self.iters):
            slots_prev = slots

            slots = self.norm_slots(slots)
            q = self.to_q(slots)

            dots = jt.linalg.einsum('bid,bjd->bij', q, k) * self.scale
            # attn = dots.softmax(dim=1) + self.eps
            attn = nn.softmax(dots, dim=1) + self.eps
            attn = attn / attn.sum(dim=-1, keepdims=True)

            updates = jt.linalg.einsum('bjd,bij->bid', v, attn)

            slots = self.gru(
                updates.reshape(-1, d),
                slots_prev.reshape(-1, d)
            )

            slots = slots.reshape(b, -1, d)
            slots = slots + self.mlp(self.norm_pre_ff(slots))

        ### This isn't part of the original slot attention model, but downsamples result if necessary to correct length
        attention_outputs = jt.reshape(slots, (-1, n_s * self.dim))

        return attention_outputs


def invert(mat):
    rot = mat[...,:3,:3]
    trans = mat[...,:3,3,None]
    rot_t = rot.permute(0,2,1)
    trans_t = -1 * rot.permute(0,2,1) @ trans
    return jt.concat((rot_t, trans_t), -1)
#reproject the corresponding 3d position
#pts [n_pts,3]: points in 3d space
#attention_poses[n_views,3,4]: matrices corresponding to the different viewpoints of a given scene
#intrinsic[3,4]: intrinsic matrix
#returns image plane pixel locations of rays originating at all of the
#attention_poses and going through one of the given points. The output tensor has shape [n_views,n_pts,2]

# 即根据论文3.2节中 For a 3D query point p = (x, y, z) ∈ P
# we project it back to the 2D image spaces of these references using intrinsics {Kn}
# and camera poses {Rn, Tn} and get the corresponding 2D coordinate.
def make_indices(pts, attention_poses, intrinsic, H, W):
    hom_points = jt.concat((pts, jt.broadcast(jt.array([1.0]), pts.shape[:-1] + (1,))), -1)
    extrinsic = invert(attention_poses)[:, :3]
    focal = intrinsic[0, 0]

    pt_camera = jt.broadcast(hom_points[None, ...],
                                (extrinsic.shape[0], hom_points.shape[0], hom_points.shape[1])) @ extrinsic.permute(0, 2, 1)
    pt_camera = focal / pt_camera[:, :, 2][..., None] * pt_camera
    final = 1.0 / focal * (pt_camera @ jt.transpose(intrinsic,0,1))
    final = jt.flip(final, dim=[-1])[..., 1:]
    final = (jt.array([0., W]) - final) * jt.array([-1., 1.])
    #final = torch.round(final)
    final = jt.maximum(jt.minimum(final, jt.array([H-1.,W-1.])), jt.array([0,0]))
    #final = final.int()
    return final


class Face_Feature_NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, dim_aud=76,
                 output_ch=4, skips=[4], use_viewdirs=False, dim_image_features = 0):
        """
        """
        super(Face_Feature_NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.dim_aud = dim_aud
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        self.dim_image_features = dim_image_features
        input_ch_all = input_ch + dim_aud + dim_image_features
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch_all, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch_all, W) for i in range(D-1)])

        # Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)])

        # Implementation according to the paper
        self.views_linears = nn.ModuleList(
            [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//4)])

        if use_viewdirs:
            # self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W//2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def execute(self, x):
        input_pts, input_views = jt.split(
            x, [self.input_ch+self.dim_aud+self.dim_image_features, self.input_ch_views], dim=-1)

        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = nn.relu(h)
            if i in self.skips:
                h = jt.concat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)#the density output, 1-D
            feature = h  # self.feature_linear(h)
            h = jt.concat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = nn.relu(h)

            rgb = self.rgb_linear(h)#the rgb output, 3-D
            outputs = jt.concat([rgb, alpha], -1)#all output, 4-D
        else:
            outputs = self.output_linear(h)

        return outputs

    def load_weights_from_keras(self, weights):
        assert self.use_viewdirs, "Not implemented if use_viewdirs=False"

        # Load pts_linears
        for i in range(self.D):
            idx_pts_linears = 2 * i
            self.pts_linears[i].weight.data = jt.array(
                np.transpose(weights[idx_pts_linears]))
            self.pts_linears[i].bias.data = jt.array(
                np.transpose(weights[idx_pts_linears+1]))

        # Load feature_linear
        # idx_feature_linear = 2 * self.D
        # self.feature_linear.weight.data = jt.array(
        #     np.transpose(weights[idx_feature_linear]))
        # self.feature_linear.bias.data = jt.array(
        #     np.transpose(weights[idx_feature_linear+1]))

        # Load views_linears
        idx_views_linears = 2 * self.D + 2
        self.views_linears[0].weight.data = jt.array(
            np.transpose(weights[idx_views_linears]))
        self.views_linears[0].bias.data = jt.array(
            np.transpose(weights[idx_views_linears+1]))

        # Load rgb_linear
        idx_rbg_linear = 2 * self.D + 4
        self.rgb_linear.weight.data = jt.array(
            np.transpose(weights[idx_rbg_linear]))
        self.rgb_linear.bias.data = jt.array(
            np.transpose(weights[idx_rbg_linear+1]))

        # Load alpha_linear
        idx_alpha_linear = 2 * self.D + 6
        self.alpha_linear.weight.data = jt.array(
            np.transpose(weights[idx_alpha_linear]))
        self.alpha_linear.bias.data = jt.array(
            np.transpose(weights[idx_alpha_linear+1]))


#nerf attention model
class nerf_attention_model(nn.Module):
    def __init__(self, nerf_model, slot_att, embed_fn, embed_ln, embed_fn_2, embed_ln_2, coarse, num_samples):
        super(nerf_attention_model, self).__init__()
        self.nerf_model = nerf_model    # Face_Feature_NeRF
        self.embed_fn, self.embed_ln = embed_fn, embed_ln
        self.embed_fn_2, self.embed_ln_2 = embed_fn_2, embed_ln_2
        self.slot_att = slot_att
        self.coarse = coarse
        self.num_samples = num_samples

    def execute(self, inputs, training=None):
        nerf_inputs = inputs[0]
        local = inputs[1]
        pts = inputs[2]

        embedded_pts = nerf_inputs[...,:self.embed_ln]
        embedded_pts = jt.broadcast(embedded_pts[None], (local.shape[0], local.shape[1], embedded_pts.shape[-1]) )
        attention_outputs = self.slot_att(local.permute(1,0,2), embedded_pts.permute(1,0,2))
        decoder_input = jt.concat((attention_outputs, nerf_inputs), -1)
        return self.nerf_model(decoder_input), decoder_input


def get_similar_k(pose, pose_set, img_set, top_size = None, num_from_top = 1, k = 2):
    vp = pose[:,3]
    vp_set = pose_set[:,:,3]
    vp_set_norm = jt.norm(vp_set, axis = -1)[...,None]
    vp_norm = jt.norm(vp, axis = -1)
    simil = jt.sum( (vp / vp_norm) * (vp_set / vp_set_norm) , -1)
    sorted_inds = jt.argsort(simil, descending=True)
    if top_size is None:
        return pose_set[sorted_inds[:k]], np.take(img_set, sorted_inds[:k], axis = 0)
    else:
        rand_idxs = np.random.choice(np.arange(1,top_size), num_from_top, replace = False)
        sorted_inds = np.take(sorted_inds, rand_idxs, axis = 0)
        #use np take so that img_set is not all copied to GPU
        return pose_set[sorted_inds], np.take(img_set, sorted_inds, axis = 0)


def read_img(image_path_list):
    imgs = []
    for fname in image_path_list:
        imgs.append(imageio.imread(fname))
    imgs = jt.array(imgs).float()/255.0
    return imgs


class Position_warp(nn.Module):
    def __init__(self, input_dim, reference_num):
        super(Position_warp, self).__init__()
        self.input_dim = input_dim
        self.reference_num = reference_num
        self.mlp = nn.Sequential(nn.Linear(self.input_dim, 128), nn.ReLU(), nn.Linear(128, 32), nn.ReLU(), nn.Linear(32,2))
        self.sigmoid = nn.Sigmoid()
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                # m.weight.data.normal_(0, 1e-3)
                m.weight.gauss_(0, 1e-3)
                m.bias.zero_()

    def execute(self, query_pts, aud, reference_feature):
        input = jt.concat((query_pts, aud.broadcast((query_pts.shape[0],64)), reference_feature),1)
        output = self.mlp(input)
        output = jt.maximum(jt.minimum(output, jt.array([10,10])), jt.array([-10,-10]))
        return output



