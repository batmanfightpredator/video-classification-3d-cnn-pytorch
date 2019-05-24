import os
import sys
import json
import subprocess
import numpy as np
import torch
from torch import nn

from opts import parse_opts
from model import generate_model
from mean import get_mean
from classify import classify_video

if __name__=="__main__":
    opt = parse_opts()
    #归一化处理
    opt.mean = get_mean()
    #模型名称与深度
    opt.arch = '{}-{}'.format(opt.model_name, opt.model_depth)
    opt.sample_size = 112
    opt.sample_duration = 16
    opt.n_classes = 400
    
    #模型生成
    model = generate_model(opt)
    print('loading model {}'.format(opt.model))
    #导入模型
    model_data = torch.load(opt.model)
    assert opt.arch == model_data['arch']
    model.load_state_dict(model_data['state_dict'])
    model.eval()
    #verbose作用？
    if opt.verbose:
        print(model)
    
    #导入文件（需要测试的文件）
    input_files = []
    with open(opt.input, 'r') as f:
        for row in f:
            input_files.append(row[:-1])#row[:-1]去掉row最后一个
    
    #类型名
    class_names = []
    with open('class_names_list') as f:
        for row in f:
            class_names.append(row[:-1])
    
    
    ffmpeg_loglevel = 'quiet'
    if opt.verbose:
        ffmpeg_loglevel = 'info'
    
    #删除tmp文件夹
    if os.path.exists('tmp'):
        subprocess.call('rm -rf tmp', shell=True)
    
    #测试
    outputs = []
    for input_file in input_files:
        video_path = os.path.join(opt.video_root, input_file)#视频路径
        if os.path.exists(video_path):
            print(video_path)
            subprocess.call('mkdir tmp', shell=True)
            subprocess.call('ffmpeg -i {} tmp/image_%05d.jpg'.format(video_path),
                            shell=True)

            result = classify_video('tmp', input_file, class_names, model, opt)
            outputs.append(result)#测试结果

            subprocess.call('rm -rf tmp', shell=True)
        else:
            print('{} does not exist'.format(input_file))

    if os.path.exists('tmp'):
        subprocess.call('rm -rf tmp', shell=True)
    
    #保存结果
    with open(opt.output, 'w') as f:
        json.dump(outputs, f)
