import os
# 图片文件夹路径
pic_paths= "/mnt/7c29025e-8959-4011-9c6c-607048c15cfc/ZS/job_works/calib/MonocularCalibration/calib_imgs/abstra_image/"
f=open('names.txt', 'w')
filenames=os.listdir(pic_paths)
filenames.sort()
for filename in filenames:
     # 图片绝对路径
    out_path=pic_paths + filename
    print(out_path)
    f.write(out_path+'\n')
f.close()


