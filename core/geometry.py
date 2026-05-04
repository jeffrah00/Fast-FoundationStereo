import torch,os,sys
import torch.nn.functional as F
from core.utils.utils import bilinear_sampler, bilinear_sampler1d
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')

class Combined_Geo_Encoding_Volume:
    def __init__(self, init_fmap1, init_fmap2, geo_volume, num_levels=2):
        self.num_levels = num_levels
        self.geo_volume_pyramid = []
        self.init_corr_pyramid = []

        # all pairs correlation
        init_corr = Combined_Geo_Encoding_Volume.corr(init_fmap1, init_fmap2)

        b, h, w, _, w2 = init_corr.shape
        b, c, d, h, w = geo_volume.shape
        geo_volume = geo_volume.permute(0, 3, 4, 1, 2).reshape(b*h*w, c, 1, d)

        init_corr = init_corr.view(b*h*w, 1, 1, w2)
        self.geo_volume_pyramid.append(geo_volume)
        self.init_corr_pyramid.append(init_corr)
        for _ in range(self.num_levels-1):
            geo_volume = F.avg_pool2d(geo_volume, [1,2], stride=[1,2])
            self.geo_volume_pyramid.append(geo_volume)

        for _ in range(self.num_levels-1):
            init_corr = F.avg_pool2d(init_corr, [1,2], stride=[1,2])
            self.init_corr_pyramid.append(init_corr)



    def __call__(self, disp, coords, dx, low_memory=True):
        b, _, h, w = disp.shape
        out_pyramid = []
        for i in range(self.num_levels):
            geo_volume = self.geo_volume_pyramid[i]
            x0 = dx + disp.view(b*h*w, 1, 1, 1) / 2**i
            if low_memory:
                geo_volume = bilinear_sampler1d(geo_volume, x0, mode='bilinear', align_corners=True)
            else:
                y0 = torch.zeros_like(x0)
                disp_lvl = torch.cat([x0,y0], dim=-1)
                geo_volume = bilinear_sampler(geo_volume, disp_lvl, low_memory=low_memory)
            geo_volume = geo_volume.view(b, h, w, -1)

            init_corr = self.init_corr_pyramid[i]   # (B*H*W, 1, 1, W2)
            init_x0 = coords.view(b*h*w, 1, 1, 1)/2**i - disp.view(b*h*w, 1, 1, 1) / 2**i + dx
            if low_memory:
                init_corr = bilinear_sampler1d(init_corr, init_x0, mode='bilinear', align_corners=True)
            else:
                init_coords_lvl = torch.cat([init_x0,y0], dim=-1)
                init_corr = bilinear_sampler(init_corr, init_coords_lvl, low_memory=low_memory)
            init_corr = init_corr.view(b, h, w, -1)

            out_pyramid.append(geo_volume)
            out_pyramid.append(init_corr)

        out_pyramid = torch.cat(out_pyramid, dim=-1)
        return out_pyramid.permute(0, 3, 1, 2)   #(B,C,H,W)


    @staticmethod
    def corr(fmap1, fmap2, normalize=True):
        B, D, H, W1 = fmap1.shape
        _, _, _, W2 = fmap2.shape
        if normalize:
            f1 = F.normalize(fmap1.float(), dim=1)  # (B, D, H, W1)
            f2 = F.normalize(fmap2.float(), dim=1)  # (B, D, H, W2)
        else:
            f1 = fmap1.float()
            f2 = fmap2.float()
        # bmm: (B*H, W1, D) x (B*H, D, W2) -> (B*H, W1, W2)
        f1_bm = f1.permute(0, 2, 3, 1).reshape(B * H, W1, D)
        f2_bm = f2.permute(0, 2, 1, 3).reshape(B * H, D, W2)
        corr = torch.bmm(f1_bm, f2_bm).reshape(B, H, W1, W2)
        return corr.unsqueeze(3).to(fmap1.dtype)  # (B, H, W1, 1, W2)