#给train_list,test_list分别添加Violence的内容
import os
#训练，验证，训练：6:2:2，这里因为Execute数据量少，可能要多点训练集
#Execute:28，4,2
#Fire:138,46,46
#WithGun:187,62,63
paths='E:/pythonProject/PaddlePeddle/violenceRecognization2/TSN/data2/ucf101/videos/WithGun'
out_path='E:/pythonProject/PaddlePeddle/violenceRecognization2/TSN/temp.txt'
f = open(out_path,'r+')
filenames = os.listdir(paths)

for filename in filenames:
    if os.path.splitext(filename)[1] == '.avi':
        f.write(paths.split('/')[-1]+'/'+filename+'\n')