# coding: UTF-8
from PIL import Image

import numpy as np
import os
import time

# movie2map : 動画からマップを作成するツール ver0.1 by pensil 2019.02.26
# ソースコードを参考にしていただくことはかまいませんが、著作権は放棄していません

start = time.time()

def diffimage(src, dst): return np.sum((src - dst)**2)/np.size(src)

def getImage(c): return np.array(Image.open(str.format('work/{:06}.png', c)), dtype=np.float)

def putImage(img, filename): Image.fromarray(np.array(img, dtype=np.uint8)).save(filename)

def imageToGray(img): return 0.298912*img[:,:,0] + 0.586611 *img[:,:,1] + 0.114478**img[:,:,2]

def imageToBoolean(img, border): return np.where(imageToGray(img) < border, 1, 0)

class SquareIndex:
    def __init__(self, h, w, d, deps, count):
        self.w  = w    # w  : 画像の幅
        self.h  = h    # h  : 画像の高さ
        self.rw = 4
        self.rh = 4
        self.sw = w - self.rw + 1
        self.sh = h - self.rh + 1
        self.rx = np.arange(0, self.rw, 1, dtype = np.int).repeat(self.rh).reshape(self.rw, self.rh).T.reshape(-1)
        self.ry = np.arange(0, self.rh, 1, dtype = np.int).repeat(self.rw)
        self.rc = self.rw * self.rh
        self.sx = np.arange(0, self.sw, 1, dtype = np.int).repeat(self.sh).reshape(self.sw, self.sh).T.reshape(-1)
        self.sy = np.arange(0, self.sh, 1, dtype = np.int).repeat(self.sw)
        self.sc = self.sw * self.sh
        self.tx = self.rx.repeat(self.sc).reshape(self.rc, self.sc).T.reshape(-1)
        self.ty = self.ry.repeat(self.sc).reshape(self.rc, self.sc).T.reshape(-1)
        self.srx = self.sx.repeat(self.rc) + self.tx
        self.sry = self.sy.repeat(self.rc) + self.ty
        self.deps = deps
        self.count = count

    def searchXY(self, sxy, dxy):
        srcs = np.shape(sxy)[0]
        dsts = np.shape(dxy)[0]
        vxy = sxy.repeat(dsts).reshape(srcs, dsts).T.reshape(-1) - dxy.repeat(srcs)
        unique, counts = np.unique(vxy, return_counts=True)
        midx = np.argmax(counts)
        idx = unique[midx]
        mx, my = idx % 10000, idx // 10000
        if (mx > 5000):
            mx -= 10000
            my += 1
        return mx, my

    def limitSet(self, idx, deps):
        np.random.shuffle(idx)
        imax = np.shape(idx)[0]
        if (imax > deps):
            return idx[0:deps]
        return idx

    def searchRandom(self, sxy, dxy, deps):
        sxy1 = self.limitSet(sxy, deps)
        dxy1 = self.limitSet(dxy, deps)
        return self.searchXY(sxy1, dxy1)

    def convert(self, img):
        data = imageToBoolean(img, 40)
        srcv = data[self.sry, self.srx].reshape(self.sc, self.rc).sum(axis=1)
        si = np.array(np.where(srcv == 8)[0], dtype=np.int)
        [sy, sx] = np.unravel_index(si, (self.sh, self.sw))
        sxy = sy * 10000 + sx
        return sxy

    def search(self, srci, dsti):
        dup = np.intersect1d(srci, dsti)
        sxy = np.setdiff1d(srci, dup)
        dxy = np.setdiff1d(dsti, dup)
        srcs = np.shape(sxy)[0]
        dsts = np.shape(dxy)[0]
        dups = np.shape(dup)[0]
        if (dups > srcs / 2 or dups > dsts / 2):
            print(' no move many dot match! {:}, {:}, {:}'.format(srcs, dsts, dups))
            return 0, 0

        #print(' check! : {:}, {:})'.format(srcs, dsts))
        for c in range(1, self.count):
            deps = self.deps * c
            if (srcs <= deps and dsts <= deps):
                return self.searchXY(sxy, dxy)
            # ダブルチェックで一致しないとOKとしない
            mx1, my1 = self.searchRandom(sxy, dxy, deps)
            mx2, my2 = self.searchRandom(sxy, dxy, deps)
            if (mx1 == mx2 and my1 == my2): return mx1, my1
            mx3, my3 = self.searchRandom(sxy, dxy, deps)
            if (mx1 == mx3 and my1 == my3): return mx1, my1
            if (mx2 == mx3 and my2 == my3): return mx2, my2
            print(' up to pass{:>2} : {:}, {:}  vs  {:}, {:}   ({:} < {:}, {:})'.format(c, mx1, my1, mx2, my2, deps, srcs, dsts))

        print(' up to passALL: {:}, {:}  vs  {:}, {:} ({:}, {:})'.format(mx1, my1, mx2, my2, srcs, dsts))
        return self.searchXY(sxy, dxy)

def bokashi(img):
    img2 = img.copy()
    [h, w] = [img.shape[0], img.shape[1]]
    img2[0:h-1,:] += img[1:h,:]*2
    img2[1:h,:] += img[0:h-1,:]*2
    img2[:,0:w-1] += img[:,1:w]*2
    img2[:,1:w] += img[:,0:w-1]*2
    return img2

print ('read images --- ')
startIndex = 1
endIndex = 9999

images = []
posx = []
posy = []

c = startIndex
im1 = getImage(c)
[x, y, mx, my] = [0, 0, 0, 0]
[h, w] = im1[:,:,0].shape
print ('size of movie (width, height) : {:>6},{:>6}'.format(w, h))

sq = SquareIndex(h, w, 16, 800, 10)

prev = im1
previ = sq.convert(im1)
posx.append(x)
posy.append(y)
images.append(c)
c+= 1

test = np.zeros((h, w, 3), dtype=np.uint8)

maskvar_test = np.zeros((h, w), dtype=np.float)

while 1:
    if (os.path.exists(str.format('work/{:06}.png', c))):
        print (str.format('{:06}.png ----------- ', c))
        img = getImage(c)

        diff = diffimage(img, prev)
        if diff < 1000:
            print ('  skip      : {:>6},{:>6},{:>6},{:>6},{:>10.4f}'.format(x, y, 0, 0, diff))
        else:
            imgi = sq.convert(img)
            if (np.shape(imgi)[0] == 0):
                print ('  skip      : no data!!!')
            else:
                mx, my = sq.search(previ, imgi)

                if mx < 0:
                    [sx, dx, rw] = [-mx, 0, w + mx]
                else:
                    [sx, dx, rw] = [0, mx, w - mx]
                if my < 0:
                    [sy, dy, rh] = [-my, 0, h + my]
                else:
                    [sy, dy, rh] = [0, my, h - my]

                x += mx
                y += my
                print ('  pos       : {:>6},{:>6},{:>6},{:>6} {:>10.4f}'.format(x, y, mx, my, diff))

                if mx != 0 or my != 0:
                    maskvar_test += np.sum((img - prev)**2, axis=2)

                posx.append(x)
                posy.append(y)
                images.append(c)
                prev = img
                previ = imgi

    c+=1
    if c > endIndex:
        break

posxl=np.array(posx)
posyl=np.array(posy)

# マスクの生成
maskf = np.zeros((h, w, 3), dtype=im1.dtype)
maskvar_testb = bokashi(maskvar_test)

#if os.path.exists('mask.png'):
if 0: # 既存のマスクを使用する
    print ('use mask file : mask.png')
    mask = np.array(Image.open('mask.png'))
    maskf = np.where(mask < 128, 0, 1)
else:
    # 分散をとって、マスクを作る
    mask = np.zeros((h, w, 3), dtype=im1.dtype)

    maskf[:,:,0] = np.where(maskvar_testb > np.std(maskvar_testb) * (-0.5) + np.mean(maskvar_testb), 1, 0)
    maskf[:,:,1] = maskf[:,:,0]
    maskf[:,:,2] = maskf[:,:,0]

    mask = maskf * 255

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
process_time = time.time() - start

print ('Complete!! - ({:8.2f}sec)'.format(process_time))
