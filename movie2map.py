# coding: UTF-8
from PIL import Image

import numpy as np
import os

# movie2map : 動画からマップを作成するツール ver0.1 by pensil 2019.02.26
# ソースコードを参考にしていただくことはかまいませんが、著作権は放棄していません

class SlideAnalyzer:
    """
    画像スライド解析に必要なnarray行列を保持し、解析を行うクラス
    """
    def __init__(self, h, w, rc, d, k):
        self.w  = w    # w  : 画像の幅
        self.h  = h    # h  : 画像の高さ
        self.rc = rc   # rc : スライド解析に使用する、ランダム座標数
        self.d  = d    # d  : 画像全体のうちdの1をスライド可能とする
        self.k  = k    # k  : 解析の結果、想定されたスライド位置の上位いくつまで再検証するか
        self.sw = int(w / d)             # sw : 横軸スライド可能サイズ
        self.sh = int(h / d)             # sh : 縦軸スライド可能サイズ
        self.bw = int(self.sw / 2)       # bw : 行列のインデックスと横軸の差異
        self.bh = int(self.sh / 2)       # bh : 行列のインデックスと縦軸の差異
        self.rw = int(w - self.bw)       # rw : 2つの画像比較における、重なり評価エリアの幅
        self.rh = int(h - self.bh)       # rh : 2つの画像比較における、重なり評価エリアの高さ
        self.sx = np.zeros(self.sw, dtype='int')   # sx : スライド比較におけるソース画像の左座標
        self.dx = np.zeros(self.sw, dtype='int')   # dx : スライド比較における比較画像の左座標
        self.sy = np.zeros(self.sh, dtype='int')   # sy : スライド比較におけるソース画像の上座標
        self.dy = np.zeros(self.sh, dtype='int')   # dy : スライド比較における比較画像の上座標
        self.rx = np.random.randint(int(w/16), self.rw-int(w/16), rc)  # rx : 比較評価に使うランダムな座標 x軸
        self.ry = np.random.randint(int(h/16), self.rh-int(h/16), rc)  # ry : 比較評価に使うランダムな座標 y軸
        self.sx[self.bw : self.sw] = np.arange(0, self.sw - self.bw, 1, dtype = 'int')
        self.dx[0 : self.bw] = np.arange(self.bw, 0, -1, dtype = 'int')
        self.sy[self.bh : self.sh] = np.arange(0, self.sh - self.bh, 1, dtype = 'int')
        self.dy[0 : self.bh] = np.arange(self.bh, 0, -1, dtype = 'int')
        tx = self.rx.repeat(self.sh * self.sw).reshape(rc, self.sw, self.sh).transpose(2, 1, 0)
        ty = self.ry.repeat(self.sh * self.sw).reshape(rc, self.sw, self.sh).transpose(2, 1, 0)
        asize = self.sh * self.sw * rc
        self.srxi = ((self.sx.repeat(self.sh * rc).reshape(self.sw, self.sh, rc).transpose(1, 0, 2)) + tx).reshape(asize)
        self.drxi = ((self.dx.repeat(self.sh * rc).reshape(self.sw, self.sh, rc).transpose(1, 0, 2)) + tx).reshape(asize)
        self.sryi = ((self.sy.repeat(self.sw * rc).reshape(self.sh, self.sw, rc)) + ty).reshape(asize)
        self.dryi = ((self.dy.repeat(self.sw * rc).reshape(self.sh, self.sw, rc)) + ty).reshape(asize)
        print ('search indexes : {:>15}'.format(self.srxi.shape[0]))

    def search(self, src, dst):
        ddif = np.where(np.abs(src[self.sryi, self.srxi, :] - dst[self.dryi, self.drxi, :]) <20, 0, 1)
        res = np.sum(ddif, axis=1).reshape(self.sh * self.sw, self.rc).sum(axis=1)
        mindif = 0.0
        minidx = -1
        for idx in np.argpartition(res, self.k)[:self.k]:
            [y, x] = np.unravel_index(idx, (self.sh, self.sw))
            [csx, csy, cdx, cdy] = [self.sx[x], self.sy[y], self.dx[x], self.dy[y]]
            crw = self.w - csx if csx > cdx else self.w - cdx
            crh = self.h - csy if csy > cdy else self.h - cdy
            dif = diffimage(src[csy:csy+crh,csx:csx+crw,:],dst[cdy:cdy+crh,cdx:cdx+crw,:])
            if dif < mindif or minidx == -1:
                mindif = dif
                minidx = idx
        [y, x] = np.unravel_index(minidx, (self.sh, self.sw))
        return [self.bw - x, self.bh - y, mindif]

def diffimage(src, dst): return np.sum((src - dst)**2)/np.size(src)

def getImage(c): return np.array(Image.open(str.format('work/{:06}.png', c)), dtype=np.float)

def putImage(img, filename): Image.fromarray(np.array(img, dtype=np.uint8)).save(filename)

def bokashi(img):
    img2 = img.copy()
    img2[0:h-1,:] += img[1:h,:]*2
    img2[1:h,:] += img[0:h-1,:]*2
    img2[:,0:w-1] += img[:,1:w]*2
    img2[:,1:w] += img[:,0:w-1]*2
    return img2

print ('read images --- ')
startIndex = 1
endIndex = 0

images = []
mask_images = []
posx = []
posy = []

c = startIndex
im1 = getImage(c)
[x, y, mx, my] = [0, 0, 0, 0]
[h, w] = im1[:,:,0].shape
print ('size of movie (width, height) : {:>6},{:>6}'.format(w, h))

an1 = SlideAnalyzer(h, w, 40, 8, 40)
an2 = SlideAnalyzer(h, w, 50, 4, 50)
an3 = SlideAnalyzer(h, w, 60, 2, 50)
an4 = SlideAnalyzer(h, w, 60, 1, 80)

prev = im1
posx.append(x)
posy.append(y)
images.append(c)
c+= 1

maskbase_test = np.zeros((h, w), dtype=np.float)
maskvar_test = np.zeros((h, w), dtype=np.float)

while os.path.exists(str.format('work/{:06}.png', c)):
    print (str.format('{:06}.png ----------- ', c))
    img = getImage(c)
    diff = diffimage(img, prev)
    if diff < 200:
        print ('  skip     : {:>6},{:>6},{:>6},{:>6},{:>10.4f}'.format(x, y, 0, 0, diff))
    else:
        [mx, my, difs] = an1.search(img, prev)
        if difs > 1000: # 35
            print ('  up to sa2: {:>6},{:>6},{:>6},{:>6},{:>10.4f}'.format(x, y, mx,my,difs))
            [mx, my, difs] = an2.search(img, prev)
            if difs > 1200: # 33
                print ('  up to sa3: {:>6},{:>6},{:>6},{:>6},{:>10.4f}'.format(x, y, mx,my,difs))
                [mx, my, difs] = an3.search(img, prev)
                if difs > 1400: # 30
                    print ('  up to sa4: {:>6},{:>6},{:>6},{:>6},{:>10.4f}'.format(x, y, mx,my,difs))
                    [mx, my, difs] = an4.search(img, prev)
                    if difs > 1500: # 30
                        # シーン前後の差が大きすぎるため解析を終了します
                        print ('  not found!: ',mx,my,difs)
                        #break
        x += mx
        y += my
        print ('  pos      : {:>6},{:>6},{:>6},{:>6},{:>10.4f}'.format(x, y, mx,my,difs))

        if mx < 0:
            [sx, dx, rw] = [-mx, 0, w + mx]
        else:
            [sx, dx, rw] = [0, mx, w - mx]
        if my < 0:
            [sy, dy, rh] = [-my, 0, h + my]
        else:
            [sy, dy, rh] = [0, my, h - my]

        if mx != 0 or my != 0:
            maskbase_test[sy:sy+rh, sx:sx+rw] += np.sum((prev[0:rh, 0:rw] - img[dy:dy+rh, dx:dx+rw])**2, axis=2)
            maskbase_test[dy:dy+rh, dx:dx+rw] += np.sum((prev[sy:sy+rh, sx:sx+rw] - img[0:rh, 0:rw])**2, axis=2)
            maskvar_test += np.sum((img - prev)**2, axis=2)

        posx.append(x)
        posy.append(y)
        images.append(c)
        prev = img
    c+=1
    if c > endIndex and endIndex > 0:
        break

posxl=np.array(posx)
posyl=np.array(posy)

# マスクの生成
maskf = np.zeros((h, w, 3), dtype=im1.dtype)
masktest = np.zeros((h, w, 3), dtype=im1.dtype)

maskbase_testb = bokashi(maskbase_test)
maskvar_testb = bokashi(maskvar_test)

#if os.path.exists('mask.png'):
if 0: # 既存のマスクを使用する
    print ('use mask file : mask.png')
    mask = np.array(Image.open('mask.png'))
    maskf = np.where(mask < 128, 0, 1)
else:
    # 分散をとって、マスクを作る
    mask = np.zeros((h, w, 3), dtype=im1.dtype)

    mask[:,:,0] = np.where(maskbase_testb > np.std(maskbase_testb) * (-0.7) + np.mean(maskbase_testb), 255, 0) * np.where(maskvar_testb > np.std(maskvar_testb) * (-0.5) + np.mean(maskvar_testb), 1, 0)
    mask[:,:,1] = mask[:,:,0]
    mask[:,:,2] = mask[:,:,0]

    maskf[:,:,0] = np.where(maskbase_testb > np.std(maskbase_testb) * (-0.7) + np.mean(maskbase_testb), 1, 0) * np.where(maskvar_testb > np.std(maskvar_testb) * (-0.5) + np.mean(maskvar_testb), 1, 0)
    maskf[:,:,1] = maskf[:,:,0]
    maskf[:,:,2] = maskf[:,:,0]

    print ('write mask file : mask.png')
    putImage(mask,'mask.png')

[xmax, xmin, ymax, ymin] = [int(np.max(posxl)), int(np.min(posxl)), int(np.max(posyl)), int(np.min(posyl))]
[xsize, ysize] = [xmax - xmin + w, ymax - ymin + h]
print ('xsize, ysize : {:>6},{:>6}'.format(xsize, ysize))

mapdata = np.zeros((ysize, xsize, 3), dtype=im1.dtype)
mapmask = np.zeros((ysize, xsize, 3), dtype=im1.dtype)

[lx, ly] = [posx[0]-xmin+32, posy[0]-ymin+32]

for i in range(len(posx)):
    [x, y] = [int(posx[i]-xmin), int(posy[i]-ymin)]
    if ((lx - x)**2 + (ly - y)**2) > 1024:
        print ('{:06}.png ----------- marge'.format(images[i]))
        data = getImage(images[i]) * maskf
        data[np.where(mapmask[y:y+h,x:x+w]>0)] = 0
        mapdata[y:y+h,x:x+w] += data
        mapmask[y:y+h,x:x+w] += maskf
        [lx, ly] = [x, y]
    else:
        print ('{:06}.png ----------- skip '.format(images[i]))

putImage(mapdata,'map.png')
print ('Complete!!')
