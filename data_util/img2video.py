import imageio as ig
import os

if __name__ == '__main__':
    basedir='/root/DFRF/dataset/finetune_models/cnn_pytorch_val/renderonly_path_339999_pytorch'
    savedir='/root/DFRF/dataset/output_videos'
    name='cnn_340000_pytorch.mp4'
    frames = []
    source = [os.path.join(basedir, f) for f in sorted(os.listdir(os.path.join(basedir))) if 'png' in f]
    length = len(source)
    for img in source:
        frames.append(ig.imread(img))
    ig.mimwrite(os.path.join(savedir,name), frames, fps=25, quality=8)
    print('已完成')