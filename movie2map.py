# coding: UTF-8
import numpy as np
import cv2
import sys
import argparse
import time

# movie2map : 動画からマップを作成するツール ver0.1 by pensil 2019.02.26
# ソースコードを参考にしていただくことはかまいませんが、著作権は放棄していません

#useffmpeg = False
#cap = cv2.VideoCapture('c:/workspace/test/2019_02_28_11_53_18.mp4')

def diffimage(src, dst): return np.sum((src - dst)**2)/np.size(src)

def getImage(cap, c, x, y, width, height, compress):
    #if (useffmpeg):
    #    return np.array(cv2.imread(str.format('work/{:06}.png', c)), dtype=np.float)
    cap.set(cv2.CAP_PROP_POS_FRAMES, c)
    ret, frame = cap.read()
    if not ret:
        return False
    frame = frame[y:y+height, x:x+width]
    if (compress != 1.0):
        frame = cv2.resize(frame, None, fx = compress, fy = compress)
    #print(ret)
    #print(cap.get(cv2.CAP_PROP_POS_FRAMES))
    #putImage(img, "work/test_out{:06}.png".format(c))
    return frame.astype(np.float)

def putImage(img, filename): cv2.imwrite(filename, np.array(img, dtype=np.uint8))

def imageToGray(img): return 0.298912*img[:,:,0] + 0.586611 *img[:,:,1] + 0.114478*img[:,:,2]

def imageToBoolean(img, border): return np.where(imageToGray(img) < border, 1, 0)

def diffmoveimage(src, dst, mx, my):
    [h, w] = [np.shape(src)[0], np.shape(src)[1]]
    if mx < 0:
        [sx, dx, rw] = [-mx,  0, w + mx]
    else:
        [sx, dx, rw] = [  0, mx, w - mx]
    if my < 0:
        [sy, dy, rh] = [-my,  0, h + my]
    else:
        [sy, dy, rh] = [  0, my, h - my]
    return np.sum((src[dy:dy+rh,dx:dx+rw]-dst[sy:sy+rh,sx:sx+rw])**2)/(rh * rw)

def testImage(c, prev, img, mx, my):
    [h, w] = [np.shape(prev)[0], np.shape(prev)[1]]
    if mx < 0:
        [sx, dx, cw] = [-mx,  0, w - mx]
    else:
        [sx, dx, cw] = [  0, mx, w + mx]
    if my < 0:
        [sy, dy, ch] = [-my,  0, h - my]
    else:
        [sy, dy, ch] = [  0, my, h + my]
    test = np.zeros((ch, cw, 3), dtype=np.uint8)
    test[sy:sy+h, sx:sx+w, 0] = imageToGray(prev)
    test[dy:dy+h, dx:dx+w, 1] = imageToGray(img)
    putImage(test, "work/test{:06}.png".format(c))

class SquareIndex:
    def __init__(self, h, w, d, deps, count):
        self.w  = w    # w  : 画像の幅
        self.h  = h    # h  : 画像の高さ
        self.rw = d
        self.rh = d
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
        self.d = d
        self.c = 2 ** (d - 1)

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
        return mx, my, counts[midx]/srcs

    def searchNXY(self, src, dst, sxy, dxy, deps):
        srcs = np.shape(sxy)[0]
        dsts = np.shape(dxy)[0]
        vxy = sxy.repeat(dsts).reshape(srcs, dsts).T.reshape(-1) - dxy.repeat(srcs)
        unique, counts = np.unique(vxy, return_counts=True)

        maxp = len(counts)
        if maxp > deps:
            maxp = deps
        cvals = np.zeros(maxp, dtype=np.float)
        idxes = np.argsort(-counts)[0:maxp]
        #unsorted_max_indices = np.argpartition(-counts, maxp)[:maxp]
        for i in range(maxp):
            idx = unique[i]
            mx, my = idx % 10000, idx // 10000
            if (mx > 5000):
                mx -= 10000
                my += 1
            cvals[i] = diffmoveimage(src, dst, mx, my)

        midx = np.argmin(cvals)
        idx = unique[idxes[midx]]
        mx, my = idx % 10000, idx // 10000
        if (mx > 5000):
            mx -= 10000
            my += 1
        return mx, my, counts[idxes[midx]]/srcs

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

    def searchNRandom(self, src, dst, sxy, dxy, deps, count):
        sxy1 = self.limitSet(sxy, deps)
        dxy1 = self.limitSet(dxy, deps)
        return self.searchNXY(src, dst, sxy1, dxy1, count)

    def convert(self, img, pbright):
        data = imageToBoolean(img, pbright)
        srcv = data[self.sry, self.srx].reshape(self.sc, self.rc).sum(axis=1)
        si = np.array(np.where(srcv == self.c)[0], dtype=np.int)
        [sy, sx] = np.unravel_index(si, (self.sh, self.sw))
        sxy = sy * 10000 + sx
        return sxy

    def search(self, src, dst, srci, dsti):
        dup = np.intersect1d(srci, dsti)
        sxy = np.setdiff1d(srci, dup)
        dxy = np.setdiff1d(dsti, dup)
        srcs = np.shape(sxy)[0]
        dsts = np.shape(dxy)[0]
        dups = np.shape(dup)[0]
        if (dups > srcs / 2 or dups > dsts / 2):
            print(' no move many dot match! {:}, {:}, {:}'.format(srcs, dsts, dups))
            return 0, 0, 0

        c = 1
        #print(' check! : {:}, {:})'.format(srcs, dsts))
        deps = self.deps * c
        if (srcs <= deps and dsts <= deps):
            #print(' self.searchNXY')
            return self.searchNXY(src, dst, sxy, dxy, self.count)

        # ダブルチェックで一致しないとOKとしない
        mx1, my1, c1 = self.searchRandom(sxy, dxy, deps)
        mx2, my2, c2 = self.searchRandom(sxy, dxy, deps)
        if (mx1 == mx2 and my1 == my2): return mx1, my1, c1
        mx3, my3, c3 = self.searchRandom(sxy, dxy, deps)
        if (mx1 == mx3 and my1 == my3): return mx1, my1, c1
        if (mx2 == mx3 and my2 == my3): return mx2, my2, c2

        # ここで決着がつかない場合は、画像比較で結論を出す
        #return self.searchRandom(src, dst, sxy, dxy, deps*10, self.count)
        return self.searchNRandom(src, dst, sxy, dxy, deps, self.count)
        #return self.searchRandom(sxy, dxy, deps*3)

def bokashi(img):
    img2 = img.copy()
    [h, w] = [img.shape[0], img.shape[1]]
    img2[0:h-1,:] += img[1:h,:]*2
    img2[1:h,:] += img[0:h-1,:]*2
    img2[:,0:w-1] += img[:,1:w]*2
    img2[:,1:w] += img[:,0:w-1]*2
    return img2

def movie2map(cap, outfile, startIndex, endIndex, rate, pcount, pbright, pdeps, testmode, iposx, iposy, width, height, compress, maskfile):
    start = time.time()

    images = []
    posx = []
    posy = []

    c = startIndex
    im1 = getImage(cap, c, iposx, iposy, width, height, compress)
    [x, y, mx, my] = [0, 0, 0, 0]
    [h, w] = im1[:,:,0].shape
    print ('size of image    : {:>6},{:>6}'.format(w, h))
    print ('')

    sq = SquareIndex(h, w, pdeps, pcount, 4)

    prev = im1
    previ = sq.convert(im1, pbright)
    posx.append(x)
    posy.append(y)
    images.append(c)
    c+= rate

    maskvar_test = np.zeros((h, w), dtype=np.float)

    while c < endIndex:
        print (str.format('frame {:}/{:} ({:>5.1f}%) ----------- ', int(c), int(endIndex), (c/endIndex)*100))
        img = getImage(cap, c, iposx, iposy, width, height, compress)

        diff = diffimage(img, prev)
        if diff < 200:
            print ('  skip      : {:>6},{:>6},{:>6},{:>6},{:>10.2f}% {:>10.4f}'.format(x, y, 0, 0, 0, diff))
        else:
            imgi = sq.convert(img, pbright)
            if (np.shape(imgi)[0] == 0):
                print ('  skip      : no data!!!')
            else:
                mx, my, cp = sq.search(prev, img, previ, imgi)
                if (testmode == True):
                    testImage(c, prev, img, mx, my)

                x += mx
                y += my
                print ('  pos       : {:>6},{:>6},{:>6},{:>6} {:>10.2f}% {:>10.4f}'.format(x, y, mx, my, cp*100, diff))

                if mx != 0 or my != 0:
                    if maskfile == None:
                        maskvar_test += np.sum((img - prev)**2, axis=2)

                posx.append(x)
                posy.append(y)
                images.append(c)
                prev = img
                previ = imgi

        c+=rate

    posxl=np.array(posx)
    posyl=np.array(posy)

    # マスクの生成
    maskf = np.zeros((int(height*compress), int(width*compress), 3), dtype=im1.dtype)
    maskvar_testb = bokashi(maskvar_test)

    if (maskfile != None): # 既存のマスクを使用する
        print ('use mask file : {:}'.format(maskfile))
        mask = cv2.imread(maskfile)
        maskf[:,:,0] = np.where(mask[:,:,0] < 128, 0, 1)
        maskf[:,:,1] = maskf[:,:,0]
        maskf[:,:,2] = maskf[:,:,0]
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
            print ('frame {:} ----------- marge'.format(int(images[i])))
            data = getImage(cap, images[i], iposx, iposy, width, height, compress) * maskf
            data[np.where(mapmask[y:y+h,x:x+w]>0)] = 0

            mapdata[y:y+h,x:x+w] += data
            mapmask[y:y+h,x:x+w] += maskf
            [lx, ly] = [x, y]
        else:
            print ('frame {:} ----------- skip '.format(int(images[i])))

    putImage(mapdata, outfile)
    process_time = time.time() - start

    print ('Complete!! -> {:} ({:8.2f}sec)'.format(outfile, process_time))

def main():
    print ('movie2map - a simple tool for generating 2D maps from 2D movies  v2.1-beta by pensil 2019.03.11')
    print ('')
    parser = argparse.ArgumentParser()
    parser.add_argument('input_filename')
    parser.add_argument('-o', metavar='OUTPUT', help='出力ファイル名')
    parser.add_argument('-m', metavar='MASK', help='マスクファイル 別に用意する場合のみ指定する')
    parser.add_argument('-s', metavar='START(sec)', type=int, default=0, help='開始位置(秒)')
    parser.add_argument('-e', metavar='END(sec)', type=int, default=180, help='終了位置(秒) デフォルト180秒 0にすると全て解析します')
    parser.add_argument('-r', metavar='RATE', type=int, default=20, help='解析フレーム間隔 小さくすると精度が 上がりますが時間がかかります')
    parser.add_argument('-c', metavar='COUNT', type=int, default=1500, help='サンプリング数(1000～5000) 多いほど 精度が上がりますが時間がかかります')
    parser.add_argument('-b', metavar='BRITENESS', type=int, default=45, help='サンプリング閾値(20～128) 指定数より暗いポイントを識別します')
    parser.add_argument('-d', metavar='DEPS', type=int, default=4, help='畳み込み範囲(2～5) 変更する必要はありません')
    parser.add_argument('-test', action="store_true", help='比較テスト画像をworkフォルダに出力します 動作チェック用')
    parser.add_argument('-x', metavar='XPOS', type=int, default=0, help='開始左座標')
    parser.add_argument('-y', metavar='YPOS', type=int, default=0, help='開始上座標')
    parser.add_argument('-width', default=0, type=int, help='幅')
    parser.add_argument('-height', default=0, type=int, help='高さ')
    parser.add_argument('-p', metavar='COMPRESSION', type=float, default=0.5, help='画像リサイズ率 1にすると原寸で処理 デフォルト0.5(半分)')

    if len(sys.argv)==1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    #print(args)

    cap = cv2.VideoCapture(args.input_filename)

    if (not cap.isOpened()):
        print('cannot open file:{:}'.format(args.input_filename))
        sys.exit(1)

    output = args.input_filename + '.png'
    if(args.o != None):
        output = args.o

    xpos = args.x
    ypos = args.y

    width = args.width
    if (width == 0):
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    if (width + xpos > cap.get(cv2.CAP_PROP_FRAME_WIDTH)):
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH) - xpos

    height = args.height
    if (height == 0):
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    if (height + ypos > cap.get(cv2.CAP_PROP_FRAME_HEIGHT)):
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) - ypos

    fps = cap.get(cv2.CAP_PROP_FPS)
    cof = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    mvs = cof / fps

    stf = args.s * fps
    if (stf > cof):
        print('invalid start point: {:} > {:}'.format(args.s, mvs))
        sys.exit(1)

    enf = args.e * fps
    if (enf > cof):
        enf = cof

    print('frames per second: {:>10}'.format(fps))
    print('count of frames: {:>10}'.format(cof))
    print('movie size(sec) :{:>10.2f}'.format(mvs))

    movie2map(cap, output, stf, enf, args.r, args.c, args.b, args.d, args.test, int(xpos), int(ypos), int(width), int(height), args.p, args.m)

main()