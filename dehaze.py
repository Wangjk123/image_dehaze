import cv2;
import math;
import numpy as np;
from skimage.metrics import mean_squared_error as MSE
from skimage.metrics import structural_similarity as SSIM
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.color import deltaE_ciede2000 as CIEDE2000

def DarkChannel(im,sz):
    b,g,r = cv2.split(im)
    dc = cv2.min(cv2.min(r,g),b);
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(sz,sz))
    dark = cv2.erode(dc,kernel)
    return dark

def AtmLight(im,dark):
    [h,w] = im.shape[:2]
    imsz = h*w
    numpx = int(max(math.floor(imsz/1000),1))
    darkvec = dark.reshape(imsz);
    imvec = im.reshape(imsz,3);

    indices = darkvec.argsort();
    indices = indices[imsz-numpx::]

    atmsum = np.zeros([1,3])
    for ind in range(1,numpx):
       atmsum = atmsum + imvec[indices[ind]]

    A = atmsum / numpx;
    return A

def TransmissionEstimate(im,A,sz):
    omega = 0.95;
    im3 = np.empty(im.shape,im.dtype);

    for ind in range(0,3):
        im3[:,:,ind] = im[:,:,ind]/A[0,ind]

    transmission = 1 - omega*DarkChannel(im3,sz);
    return transmission

def Guidedfilter(im,p,r,eps):
    mean_I = cv2.boxFilter(im,cv2.CV_64F,(r,r));
    mean_p = cv2.boxFilter(p, cv2.CV_64F,(r,r));
    mean_Ip = cv2.boxFilter(im*p,cv2.CV_64F,(r,r));
    cov_Ip = mean_Ip - mean_I*mean_p;

    mean_II = cv2.boxFilter(im*im,cv2.CV_64F,(r,r));
    var_I   = mean_II - mean_I*mean_I;

    a = cov_Ip/(var_I + eps);
    b = mean_p - a*mean_I;

    mean_a = cv2.boxFilter(a,cv2.CV_64F,(r,r));
    mean_b = cv2.boxFilter(b,cv2.CV_64F,(r,r));

    q = mean_a*im + mean_b;
    return q;

def TransmissionRefine(im,et):
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY);
    gray = np.float64(gray)/255;
    r = 60;
    eps = 0.0001;
    t = Guidedfilter(gray,et,r,eps);

    return t;

def Recover(im,t,A,tx = 0.1):
    res = np.empty(im.shape,im.dtype);
    t = cv2.max(t,tx);

    for ind in range(0,3):
        res[:,:,ind] = (im[:,:,ind]-A[0,ind])/t + A[0,ind]

    return res

# ----------------------------------------------------------------------
# 局部直方图均衡

def comp_hist(data):
    hist = np.zeros((3,256,))
    for d in range(0,3):
        for i in range(0,data.shape[0]):
            for j in range(0,data.shape[1]):
                if data[i,j,d]<0 or data[i,j,d]>255:
                    print("error: value out of bounds\n")
                hist[d, data[i,j]] += 1
    return hist

def equalization(data):
    # get pmf
    pmf = comp_hist(data) / (data.shape[0] * data.shape[1])
    # get pdf
    pdf = np.empty_like(pmf)
    base = np.array([0,0,0])
    for i in range(0,256):
        for d in range(0,3):
            base[d] += pmf[d,i]
            pdf[d,i] = base[d]
    # compute 对应关系
    match = np.round(255*pdf)
    return match

def local_adjust(data,radius):
    height = data.shape[0]
    width = data.shape[1]
    len = 2*radius+1
    # 邻域半径过大
    if len > height or len > width:
        print("error: the radius is too big\n")
    # 初始化邻域
    local_data = data[0:len,0:len,:]
    check_point0 = np.array([len-1,0])            #左下检查点
    check_point1 = np.array([len-1,len-1])        #右下检查点
    mid_index = np.array([radius,radius]) #中心指示器
    direction = 0                       #移动方向(左右)
    row_index = 0                       #行替换指示器
    col_index = 0                       #列替换指示器
    local_adj_data = np.empty_like(data)
    # 将边缘上的原始数据直接填入(含少量重复操作)
    for i in range(0,radius):
        for j in range(0,width):
            local_adj_data[i,j,:] = data[i,j,:]
            local_adj_data[-i,j,:] = data[-i,j,:]
    for i in range(0,height):
        for j in range(0,radius):
            local_adj_data[i,j,:] = data[i,j,:]
            local_adj_data[i,-j,:] = data[i,-j,:]
    while(1):
        # 计算当前中心点对应灰度
        match = equalization(local_data)
        mid_value = data[mid_index[0],mid_index[1],:]
        for d in range(0,3):
            local_adj_data[mid_index[0],mid_index[1],d] = match[d, mid_value[d]]
        ## 开始移动
        # 若当前向右
        if(direction==0):
            # 若可以向右
            if(check_point1[1]+1<width):
                # 数据替换
                local_data[:,col_index] = data[(check_point1[0]-2*radius):check_point1[0]+1,check_point1[1]+1]
                # 检查点更新
                check_point0[1] += 1
                check_point1[1] += 1
                mid_index[1] += 1
                # 列替换指示器更新
                col_index = (col_index + 1) % len
            # 若不能向右
            else:
                # 若可以向下
                if(check_point1[0]+1<height):
                    # 数据替换
                    local_data[row_index,:] = data[check_point1[0]+1,check_point0[1]:check_point1[1]+1]
                    # 检查点更新
                    check_point0[0] += 1
                    check_point1[0] += 1
                    mid_index[0] += 1
                    # 行替换指示器更新
                    row_index = (row_index + 1) % len
                    # 方向指示器更新
                    direction = 1
                # 若不能向下
                else:
                    # 遍历结束
                    break
        # 若当前向左
        else:
            # 若可以向左
            if(check_point0[1]-1>=0):
                # 数据替换
                local_data[:,col_index] = data[(check_point0[0]-2*radius):check_point0[0]+1,check_point0[1]-1]
                # 检查点更新
                check_point0[1] -= 1
                check_point1[1] -= 1
                mid_index[1] -= 1
                # 列替换指示器更新
                col_index = (col_index + 1) % len
            # 若不能向左
            else:
                # 若可以向下
                if(check_point0[0]+1<height):
                    # 数据替换
                    local_data[row_index,:] = data[check_point0[0]+1,check_point0[1]:check_point1[1]+1]
                    # 检查点更新
                    check_point0[0] += 1
                    check_point1[0] += 1
                    mid_index[0] += 1
                    # 行替换指示器更新
                    row_index = (row_index + 1) % len
                    # 方向指示器更新
                    direction = 0
                # 若不能向下
                else:
                    # 遍历结束
                    break
    return local_adj_data

if __name__ == '__main__':
    import sys
    try:
        fn = sys.argv[1]
    except:
        fn = './image/15.png'

    def nothing(*argv):
        pass

    src = cv2.imread(fn);

    I = src.astype('float64')/255;
 
    dark = DarkChannel(I,15);
    A = AtmLight(I,dark);
    te = TransmissionEstimate(I,A,15);
    t = TransmissionRefine(src,te);
    J = Recover(I,t,A,0.1);

    local_hist = local_adjust(src, 7)

    cv2.imshow("local hist", local_hist)
    cv2.imwrite("./image/local_hist",local_hist)

    cv2.imshow("dark",dark);
    cv2.imshow("t",t);
    cv2.imshow('I',src);
    cv2.imshow('J',J);
    cv2.imwrite("./image/J.png",J*255);
   

    print("MSE of dark channel is ", MSE(src,J*255))
    print("SSIM of dark channel is", SSIM(src,J*255))
    print("PSNR of dark channel is", PSNR(src,J*255))
    print("CIEDE2000 of dark channel is", CIEDE2000(src,J*255))

    print("MSE of local hist is ", MSE(src,local_hist))
    print("SSIM of local hist is", SSIM(src,local_hist))
    print("PSNR of local hist is", PSNR(src,local_hist))
    print("CIEDE2000 of local hist is", CIEDE2000(src,local_hist))

    cv2.waitKey();
