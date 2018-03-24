import numpy as np
import glob
import os
import sys
import multiprocessing
import ctypes
from scipy import stats,linalg
from sklearn import linear_model
import math
import time

#calculate array size given array.shape
def arrtol(shape):
    res = 1
    for x in shape:
        res = res*x
    return res

# Parallel processing
def regress(x,y):
    regr = linear_model.LinearRegression()
    regr.fit(x, y)
    r2 = regr.score(x,y)
    params = np.append(regr.intercept_,regr.coef_)
    predictions = regr.predict(x)
    try:
        newX = np.concatenate((np.ones((x.shape[0],1)),x),axis=1)
        if (len(newX)-newX.shape[1]) ==0:
            raise RuntimeError
        MSE = (sum((y-predictions)**2))/(len(newX)-newX.shape[1])
        var_b = MSE*(np.linalg.inv(np.dot(newX.T,newX)).diagonal())
        sd_b = np.sqrt(np.abs(var_b))
        if 0 in sd_b:
            raise RuntimeError
        ts_b = params / sd_b
        p_values =[2*(1-stats.t.cdf(np.abs(i),(len(newX)-1))) for i in ts_b]
        sd_b = np.round(sd_b,3)
        ts_b = np.round(ts_b,3)
        p_values = np.round(p_values,3)
        params = np.round(params,4)
        if 0 in (np.sqrt(ts_b**2+len(y)-3)):
            raise RuntimeError
        pr = ts_b/np.sqrt(ts_b**2+len(y)-3) #Partial correlation
        if np.Inf in pr:
            raise RuntimeError
        pr = pr[1:]
        tpr = pr*math.sqrt(len(y)-3)/np.sqrt(1-pr**2)
    except:
        p_values = np.array([-99,-99,-99])
        pr = np.array([-99,-99])
        tpr = np.array([-99,-99])

    return (regr.coef_,regr.intercept_,r2,p_values,pr,tpr)

def gridoperation(idx):
    gridr = idx[0]
    gridc = idx[1]
    gridndvi = np.memmap('regress.dat', dtype='float32', mode='r', shape=(3, 2159, 4320, 387))[0][gridr][gridc][:]
    gridpre = np.memmap('regress.dat', dtype='float32', mode='r', shape=(3, 2159, 4320, 387))[1][gridr][gridc][:]
    gridtemp = np.memmap('regress.dat', dtype='float32', mode='r', shape=(3, 2159, 4320, 387))[2][gridr][gridc][:]
    coes = np.array([0,0])
    inter = 0
    r2 = 0
    lag = [-1,-1]
    p_values = np.array([0,0,0])
    pr = np.array([-99,-99])
    tpr = np.array([-99,-99])
    for i in range(4):
        for j in range(4):
            arrlen = len(gridndvi)-max(i,j)
            y = gridndvi[max(i,j):]
            x1 = gridpre[i:arrlen+i]
            x2 = gridtemp[j:arrlen+j]
            v = np.stack((x1,x2,y))
            delidx = np.unique(np.concatenate((np.where(y<0)[0],np.where(x1<0)[0],np.where(x2<0)[0])))
            v = np.delete(v,delidx,axis=1)
            if v.shape[1] > 15:
                (coes0,inter0,r20,p_values0,pr0,tpr0) = regress(v[0:2].transpose(),v[2])
                if r20 > r2:
                    coes = coes0
                    inter = inter0
                    lag = [i,j]
                    r2 = r20
                    p_values = p_values0
                    pr = pr0
                    tpr = tpr0

    result_array[gridr][gridc] = np.concatenate((coes,np.array([inter]),np.array(lag),np.array([r2]),p_values,pr,tpr))
    sys.stdout.write('grid done %d,%d \r' %(idx[0],idx[1]))

def getindexarray(shape):
    row,col = np.indices(shape)
    x = np.array([row.flatten(),col.flatten()])
    return x.T

start_time = time.time()
#Make result_array for multiprocessing
result_array = np.empty((2159,4320,13))
res_shape = result_array.shape
result_array_base = multiprocessing.Array(ctypes.c_double, arrtol(res_shape))
result_array = np.ctypeslib.as_array(result_array_base.get_obj())
result_array = result_array.reshape(res_shape)

if __name__ == '__main__':
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    cores = int(20)
    print('Cores number: %d' %(cores))
    p = multiprocessing.Pool(processes=cores)
    idxarray = getindexarray((2159,4320))
    np.save('idx.npy',idxarray)
    idxarray = idxarray.tolist()
    for idx in idxarray:
        p.apply_async(gridoperation, args=(idx,))
    p.close()
    p.join()
    print('all subprocesses done...')
    np.save('res.npy',result_array)
    print("--- %s seconds ---" % (time.time() - start_time))

#Process result
result = np.load('res0.npy')

from osgeo import gdal
def P2tif(parray,cpytif,name,isint = False):
    if isint:
        tiftype = gdal.GDT_Int16
    else:
        tiftype = gdal.GDT_Float32
    cpyds = gdal.Open(cpytif)
    cpyband = cpyds.GetRasterBand(1)
    cpyarr = cpyband.ReadAsArray()
    gt = cpyds.GetGeoTransform()
    pj = cpyds.GetProjection()
    ds = band = None
    drv = gdal.GetDriverByName('GTiff')
    ds = drv.Create(name+'.tif', cpyarr.shape[1], cpyarr.shape[0],1,tiftype)
    ds.SetGeoTransform(gt)
    ds.SetProjection(pj)
    band = ds.GetRasterBand(1)
    band.WriteArray(parray)
    band.SetNoDataValue(-99)
    ds = band = None

namelist = ['coesPre','coesTemp','intercept','lagPre','lagTemp','R-square','p-inter','p-Pre','p-Temp','PC_Pre','PC_Temp','tPC_pre','tPC_temp']

cpytif = 'tem.tif'
for i in range(13):
    name = namelist[i]
    P2tif(result[:,:,i],cpytif,name)
