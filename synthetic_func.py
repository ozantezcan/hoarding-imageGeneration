import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import numpy as np
import re
from sklearn import linear_model
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.svm import SVC
import datetime
import os
import skimage.io
import shutil

def text2coor(textpath):
    f=open(textpath,'r')
    log=f.read()
    coor=np.asarray(re.split('\(|, |\)|\n',log))
    coor=np.delete(coor,np.where(coor=='')[0])
    coor=np.reshape(coor,(-1,3)).astype(np.float)
    return coor

def normalize(inarray,x,y,m=0,M=0):
    if(m==0 and M==0):
        m=np.min(inarray)
        M=np.max(inarray)
    print('Normalization started with m=%.2f and M=%.2f'%(m,M))
    A=np.array([[m,1],[M,1]])
    b=np.array([[x],[y]])
    b2=np.linalg.solve(A,b)
    return inarray*b2[0]+b2[1];
	
def prune_data(coor,corners):
    if((corners==[0,0,0,0,0,0]).all):
        return coor
    
    [x1,x2,y1,y2,z1,z2]=corners
    if x1>x2:
        temp=x1
        x1=x2
        x2=temp
    if y1>y2:
        temp=y1
        y1=y2
        y2=tmp
    if z1>z2:
        temp=z1
        z1=z2
        z2=temp
        
    idx=np.zeros(coor.shape[0])
    count=0
    for k in range(len(idx)):
        [x,y,z]=coor[k,:]
        if(x<x1 or x>x2 or y<y1 or y>y2 or z<z1 or z>z2):
            idx[count]=k
            count=count+1
    idx=idx[:count]
    coor=np.delete(coor,idx,axis=0)
    
    return coor
	
def featExtract(rootdir,logdir,rateddir,subdirslog,subdirsrated,nrmalize=True,\
                m=[[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]],):
    label=np.zeros((2000,1))
    room=np.zeros((2000,1))
    fvec = np.zeros((2000,4000,4))
    imcount=np.zeros((2000,1))

    for k in range(len(rateddir)):
        for l in range(len(subdirsrated)):
            for cir in range(1,10):
                fnames=os.listdir("".join((rootdir,rateddir[k],subdirsrated[l],'\\',str(cir))))
                for fname in fnames:
                    if fname.endswith('.png'):
                        imNumber=int(fname.split('_')[0])+k*1000
                        logpath="".join((rootdir,logdir[k],subdirslog[l],'\\',str(imNumber-k*1000),'.txt'))
                        coor = text2coor(logpath);
                        coor=prune_data(coor,m[l])
                        fvec[imNumber,:coor.shape[0],0]=1
                        fvec[imNumber,:coor.shape[0],1:4]=coor
                        del coor
                        imcount[imNumber] += 1
                        label[imNumber] += cir
                        room[imNumber]=l+1;



    label=np.delete(label,np.where(imcount==0)[0])
    room=np.delete(room,np.where(imcount==0)[0])
    fvec=np.delete(fvec,np.where(imcount==0)[0],axis=0)

    imcount=np.delete(imcount,np.where(imcount==0)[0])

    print(''.join(('Number of images is ',str(sum(imcount)))))
    print(''.join(('Number of frames is ',str(len(imcount)))))
    label=label/imcount
    print('Mean CIR values for scenes')
    print(label)

    print('Rooms of scenes')
    print(room)

    print('Shape of the feature vector')
    print(fvec.shape)

    #print('Feature vector for the 58th image')
    #print(fvec[58,:,:])

    fvec_n=fvec.copy()
    
    if(nrmalize):
        fvec_n[room==1,:,1]=normalize(fvec[room==1,:,1],-1,1,m=m[0][0],M=m[0][1])
        fvec_n[room==1,:,2]=normalize(fvec[room==1,:,2],-1,1,m=m[0][2],M=m[0][3])
        fvec_n[room==1,:,3]=normalize(fvec[room==1,:,3],-1,1,m=m[0][4],M=m[0][5])

        if(len(subdirslog)>1):
            fvec_n[room==2,:,1]=normalize(fvec[room==2,:,1],-1,1,m=m[1][0],M=m[1][1])
            fvec_n[room==2,:,2]=normalize(fvec[room==2,:,2],-1,1,m=m[1][2],M=m[1][3])
            fvec_n[room==2,:,3]=normalize(fvec[room==2,:,3],-1,1,m=m[1][4],M=m[1][5])

        if(len(subdirslog)>2):
            fvec_n[room==3,:,1]=normalize(fvec[room==3,:,1],-1,1,m=m[2][0],M=m[2][1])
            fvec_n[room==3,:,2]=normalize(fvec[room==3,:,2],-1,1,m=m[2][2],M=m[2][3])
            fvec_n[room==3,:,3]=normalize(fvec[room==3,:,3],-1,1,m=m[2][4],M=m[2][5])
        
        

    #print('Normalized feature vector for the 58th image')
    #print(fvec_n[58,:,:])
    return fvec_n,label,room
	
def heatmap(x,y,z,plot_b=False):

    clutter_img=np.zeros((555,555))

    for k in range(len(x)):
        clutter_img[x[k]-10:x[k]+10,z[k]-10:z[k]+10]= \
            np.maximum(clutter_img[x[k]-10:x[k]+10,z[k]-10:z[k]+10],y[k])
    if(plot_b==True):
        imgplot=plt.imshow(clutter_img)
        plt.colorbar(imgplot)
    return clutter_img
	
def feat_transform(fvec):
    color_levels=np.array([0,.3,.6,.7,.8,.9,.95,1])
    numScenes = fvec.shape[0]
    fvec_stat=np.zeros((numScenes,14))
    fvec_c = np.zeros(14)
    for k in range(numScenes):
        coors=fvec[k,:,:]
        coors=coors[coors[:,0]==1,:]
        coors=coors[:,1:4]

        x=coors[:,0]
        y=coors[:,1]
        z=coors[:,2]

        x=(11+np.round(256*(x+1))).astype(np.int)
        z=(11+np.round(256*(z+1))).astype(np.int)
        y=(y+1)/2.

        top_height=np.zeros(6)
        if(len(y)>0):
            top_height[0]=y[0]
            top_height[1]=np.mean(y[:10])
            top_height[2]=np.mean(y[:20])
            top_height[3]=np.mean(y[:50])
            top_height[4]=np.mean(y[:100])
            top_height[5]=np.mean(y[:1000])


        clutter_img=heatmap(x,y,z)
        clutter_levels=np.histogram(clutter_img,color_levels)[0]

        numObj=len(y)

        fvec_c[0]=numObj/1500
        fvec_c[1:8]=clutter_levels/(555*555/10)
        fvec_c[8:]=top_height

        fvec_stat[k,:]=fvec_c

    fvec_stat_01=fvec_stat/(np.ones((fvec_stat.shape[0],1))*np.max(fvec_stat,axis=0))
    return fvec_stat
	
def cross_val(clf,X,y,cv=4):
    n=len(y)
    idx=np.random.permutation(n)
    X=X[idx,:]
    y=y[idx]
    cv_id=np.round(np.linspace(0,n,cv+1))

    y_pred=np.zeros(y.shape)
    for k in range(len(cv_id)-1):
        id1=int(cv_id[k])
        id2=int(cv_id[k+1])
        X_test=X[id1:id2,:]
        X_tr=X.copy()
        X_tr=np.delete(X_tr,range(id1,id2),axis=0)
        y_tr=y.copy()
        y_tr=np.delete(y_tr,range(id1,id2))

        clf.fit(X_tr,y_tr)
        y_pred[id1:id2] = clf.predict(X_test)
    return y,y_pred
       
def cross_val2(clf,X,y,cv=4):
    n=len(y)
    idx=np.random.permutation(n)
    X=X[idx,:]
    y=y[idx]
    cv_id=np.round(np.linspace(0,n,cv+1))
    
    cir1=np.zeros(cv)
    cir2=np.zeros(cv)
    for k in range(len(cv_id)-1):
        id1=int(cv_id[k])
        id2=int(cv_id[k+1])
        X_tr=X[id1:id2,:]
        X_test=X.copy()
        X_test=np.delete(X_test,range(id1,id2),axis=0)
        y_tr=y[id1:id2]
        y_test=y.copy()
        y_test=np.delete(y_test,range(id1,id2))

        clf.fit(X_tr,y_tr)
        y_pred = clf.predict(X_test)
        cir1[k]=np.mean((np.abs(np.round(y_pred)-y_test)<=1).astype(np.float))
        cir2[k]=np.mean((np.abs(np.round(y_pred)-y_test)<=2).astype(np.float))
    return cir1,cir2
	
def swap_elements(array,i,j):
    tmp=array[i]
    array[i]=array[j]
    array[j]=tmp
    return array

def text2feat(textpath):
    f=open(textpath,'r')
    log=f.read()
    feat=np.asarray(re.split('\(|, |\)|\n',log))
    feat=np.delete(feat,np.where(feat=='')[0])
    if(feat[2]=='Corner Cubes for Room Dimensions'):
        corners=np.asarray([feat[3],feat[6],feat[4],feat[10],feat[5],feat[8]]).astype(np.float)
    else:
        corners=np.asarray([feat[2],feat[5],feat[3],feat[9],feat[4],feat[7]]).astype(np.float)
    idx=np.char.equal(feat,'Rotation').astype(np.int)
    idx = np.argwhere(idx)[0][0]

    feat=feat[idx+1:]
    feat=np.reshape(feat,(-1,3)).astype(np.float)
    coor=feat[::4,:]
    scle=feat[1::4,:]
    sze=feat[2::4,:]
    rot=feat[3::4,:]

    if(corners[1]<corners[0]):
        corners=swap_elements(corners,0,1)
    if(corners[3]<corners[2]):
        corners=swap_elements(corners,2,3)
    if(corners[5]<corners[4]):
        corners=swap_elements(corners,4,5)
        
    return coor,scle,sze,rot,corners

def prune_data2(coor,scle,sze,rot,corners):
    [x1,x2,y1,y2,z1,z2]=corners
    if x1>x2:
        temp=x1
        x1=x2
        x2=temp
    if y1>y2:
        temp=y1
        y1=y2
        y2=temp
    if z1>z2:
        temp=z1
        z1=z2
        z2=temp
        
    idx=np.zeros(coor.shape[0])
    count=0
    for k in range(len(idx)):
        [x,y,z]=coor[k,:]
        if(x<x1 or x>x2 or y<y1 or y>y2 or z<z1 or z>z2):
            idx[count]=k
            count=count+1
    idx=idx[:count]
    coor=np.delete(coor,idx,axis=0)
    scle=np.delete(scle,idx,axis=0)
    sze=np.delete(sze,idx,axis=0)
    rot=np.delete(rot,idx,axis=0)
    
    return coor,scle,sze,rot
	
def featExtract2(rootdir,subdirs,roomdirs,nrmalize=True,m=[[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]]):
    label=np.zeros((50000,1))
    room=np.zeros((50000,1))
    fvec = np.zeros((50000,15000,4))
    fvolume=np.zeros((50000,1))
    imcount=np.zeros((50000,1))
    

    room_dims=[[m[0][1]-m[0][0],m[0][3]-m[0][2],m[0][5]-m[0][4]],
               [m[1][1]-m[1][0],m[1][3]-m[1][2],m[1][5]-m[1][4]],
               [m[2][1]-m[2][0],m[2][3]-m[2][2],m[2][5]-m[2][4]]]
    vol_room=[room_dims[0][0]*room_dims[0][1]*room_dims[0][2],
              room_dims[1][0]*room_dims[1][1]*room_dims[1][2],
              room_dims[2][0]*room_dims[2][1]*room_dims[2][2]]
    print(vol_room)
    
    for k in range(len(subdirs)):
            for l in range(len(roomdirs)):
                for cir in range(1,10):
                    fnames=os.listdir("".join((rootdir,subdirs[k],roomdirs[l],'\\',str(cir))))
                    for fname in fnames:
                        if fname.endswith('.png'):
                            imNumber=int(fname.split('.')[0][2:])+k*1000
                            logpath="".join((rootdir,subdirs[k],'\\logs\\fr',str(imNumber-k*1000),'_log.txt'))
                            [coor,scle,sze,rot,corners] = text2feat(logpath);
                            [coor,scle,sze,rot] = prune_data2(coor,scle,sze,rot,corners)
                            fvec[imNumber,:coor.shape[0],0]=1
                            fvec[imNumber,:coor.shape[0],1:4]=coor
                            sze_obj=scle*sze
                            fvolume[imNumber]=np.sum(np.prod(sze_obj,axis=1))
                            del coor
                            imcount[imNumber] += 1
                            label[imNumber] += cir
                            room[imNumber]=l+1;
                        



    label=np.delete(label,np.where(imcount==0)[0])
    room=np.delete(room,np.where(imcount==0)[0])
    fvec=np.delete(fvec,np.where(imcount==0)[0],axis=0)
    fvolume=np.delete(fvolume,np.where(imcount==0)[0],axis=0)

    imcount=np.delete(imcount,np.where(imcount==0)[0])
    
    fvolume[room==1]=fvolume[room==1]/vol_room[0]
    fvolume[room==2]=fvolume[room==2]/vol_room[1]
    fvolume[room==3]=fvolume[room==3]/vol_room[2]
    print(''.join(('Number of images is ',str(sum(imcount)))))
    print(''.join(('Number of frames is ',str(len(imcount)))))
    label=label/imcount
    print('Mean CIR values for scenes')
    print(label)

    print('Rooms of scenes')
    print(room)

    print('Shape of the feature vector')
    print(fvec.shape)

    #print('Feature vector for the 58th image')
    #print(fvec[58,:,:])

    fvec_n=fvec.copy()

    if(nrmalize):
            fvec_n[room==1,:,1]=normalize(fvec[room==1,:,1],-1,1,m=m[0][0],M=m[0][1])
            fvec_n[room==1,:,2]=normalize(fvec[room==1,:,2],-1,1,m=m[0][2],M=m[0][3])
            fvec_n[room==1,:,3]=normalize(fvec[room==1,:,3],-1,1,m=m[0][4],M=m[0][5])

            if(len(roomdirs)>1):
                    fvec_n[room==2,:,1]=normalize(fvec[room==2,:,1],-1,1,m=m[1][0],M=m[1][1])
                    fvec_n[room==2,:,2]=normalize(fvec[room==2,:,2],-1,1,m=m[1][2],M=m[1][3])
                    fvec_n[room==2,:,3]=normalize(fvec[room==2,:,3],-1,1,m=m[1][4],M=m[1][5])

            if(len(roomdirs)>2):
                    fvec_n[room==3,:,1]=normalize(fvec[room==3,:,1],-1,1,m=m[2][0],M=m[2][1])
                    fvec_n[room==3,:,2]=normalize(fvec[room==3,:,2],-1,1,m=m[2][2],M=m[2][3])
                    fvec_n[room==3,:,3]=normalize(fvec[room==3,:,3],-1,1,m=m[2][4],M=m[2][5])
					
    return fvec_n,label,room
					
					
def feat_transform2(fvec,fsize):
    color_levels=np.array([0,.3,.6,.7,.8,.9,.95,1])
    numScenes = fvec.shape[0]
    fvec_stat=np.zeros((numScenes,14))
    fvec_c = np.zeros(14)
    for k in range(numScenes):
        coors=fvec[k,:,:]
        coors=coors[coors[:,0]==1,:]
        coors=coors[:,1:4]

        x=coors[:,0]
        y=coors[:,1]
        z=coors[:,2]

        x=(11+np.round(256*(x+1))).astype(np.int)
        z=(11+np.round(256*(z+1))).astype(np.int)
        y=(y+1)/2.

        top_height=np.zeros(6)
        if(len(y)>0):
            top_height[0]=y[0]
            top_height[1]=np.mean(y[:10])
            top_height[2]=np.mean(y[:20])
            top_height[3]=np.mean(y[:50])
            top_height[4]=np.mean(y[:100])
            top_height[5]=np.mean(y[:1000])


        clutter_img=heatmap(x,y,z)
        clutter_levels=np.histogram(clutter_img,color_levels)[0]

        numObj=len(y)

        fvec_c[0]=numObj/1500
        fvec_c[1:8]=clutter_levels/(555*555/10)
        fvec_c[8:]=top_height

        fvec_stat[k,:]=fvec_c

    fvec_stat_01=fvec_stat/(np.ones((fvec_stat.shape[0],1))*np.max(fvec_stat,axis=0))
    return fvec_stat
	
def predict_cir(clf,X):
    decisions_=clf.decision_function(X);
    decisions=np.zeros((decisions_.shape[0],19))
    for k in range(len(clf.classes_)):
        decisions[:,clf.classes_[k]]=decisions_[:,k]
    y_pred=np.argmax(decisions,axis=1)
    for i in range(1,19,2):
        idxs=np.argmax(decisions[:,[i-1,i+1]],axis=1)[y_pred==i]
        y_pred[y_pred==i]=(idxs==0)*(i-1)+(idxs==1)*(i+1)
    return y_pred/2
	
	
	
def best_param(fvec,label,c=np.power(2.,np.arange(2,8)),g=np.power(2.,np.arange(-3,5)),cv=3):
	cir1=np.zeros((len(g),len(c)))
	for k in range(len(c)):
		for l in range(len(g)):
			regr=OneVsOneClassifier(SVC(C=c[k],random_state=0,kernel='rbf',gamma=g[l],class_weight='balanced'))
			[y_cv, y_pred]=cross_val(regr,fvec,label,cv=cv) 
			#cir_1=np.mean((np.abs(np.ceil(y_pred/2)-np.ceil(y_cv/2))<=1).astype(np.float))
			cir_1=np.mean(y_cv==y_pred)
			cir1[l,k]=cir_1
	idx=np.unravel_index(cir1.argmax(), cir1.shape)
	
	regr=OneVsOneClassifier(SVC(C=c[idx[1]],random_state=0,kernel='rbf',gamma=g[idx[0]],class_weight='balanced'))
	[y_cv, y_pred]=cross_val(regr,fvec,label,cv=cv) 
	cir_1=np.mean((np.abs(np.ceil(y_pred/2)-np.ceil(y_cv/2))<=1).astype(np.float))
	print('MAX CIR-0='+str(cir1[idx])+', MAX CIR-1='+str(cir_1)+' achieved at C='+str(c[idx[1]])+', gamma='+str(g[idx[0]]))
	regr.fit(fvec,label)
	return regr
	
	
def featExtract3(rootdir,subdirs,roomdirs,nrmalize=True,m=[[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]],wrong_feat=False):
    room=np.zeros((50000,1))
    fvec = np.zeros((50000,15000,4))
    frame=np.arange(0,50000)

    for k in range(len(subdirs)):
            for l in range(len(roomdirs)):
                fnames=os.listdir("".join((rootdir,subdirs[k],roomdirs[l])))
                for fname in fnames:
                    if fname.endswith('.txt'):
                        imNumber=int(fname.split('_')[0][2:])+k*1000
                        logpath="".join((rootdir,subdirs[k],roomdirs[l],'\\',fname))
                        if(wrong_feat):
                            [coor,scle,sze,rot,corners] = text2feat2(logpath);
                        else:
                            [coor,scle,sze,rot,corners] = text2feat(logpath);
                        [coor,scle,sze,rot] = prune_data2(coor,scle,sze,rot,corners)
                        fvec[imNumber,:coor.shape[0],0]=1
                        fvec[imNumber,:coor.shape[0],1:4]=coor
                        del coor
                        room[imNumber]=l+1;

    fvec=np.delete(fvec,np.where(room==0)[0],axis=0)
    frame=np.delete(frame,np.where(room==0)[0])
    room=np.delete(room,np.where(room==0)[0])

    print(''.join(('Number of frames is ',str(len(room)))))

    print('Frame numbers')
    print(frame)
    
    print('Rooms of scenes')
    print(room)

    print('Shape of the feature vector')
    print(fvec.shape)

    #print('Feature vector for the 58th image')
    #print(fvec[58,:,:])

    fvec_n=fvec.copy()

    if(nrmalize):
            fvec_n[room==1,:,1]=normalize(fvec[room==1,:,1],-1,1,m=m[0][0],M=m[0][1])
            fvec_n[room==1,:,2]=normalize(fvec[room==1,:,2],-1,1,m=m[0][2],M=m[0][3])
            fvec_n[room==1,:,3]=normalize(fvec[room==1,:,3],-1,1,m=m[0][4],M=m[0][5])

            if(len(roomdirs)>1):
                    fvec_n[room==2,:,1]=normalize(fvec[room==2,:,1],-1,1,m=m[1][0],M=m[1][1])
                    fvec_n[room==2,:,2]=normalize(fvec[room==2,:,2],-1,1,m=m[1][2],M=m[1][3])
                    fvec_n[room==2,:,3]=normalize(fvec[room==2,:,3],-1,1,m=m[1][4],M=m[1][5])

            if(len(roomdirs)>2):
                    fvec_n[room==3,:,1]=normalize(fvec[room==3,:,1],-1,1,m=m[2][0],M=m[2][1])
                    fvec_n[room==3,:,2]=normalize(fvec[room==3,:,2],-1,1,m=m[2][2],M=m[2][3])
                    fvec_n[room==3,:,3]=normalize(fvec[room==3,:,3],-1,1,m=m[2][4],M=m[2][5])
    return fvec_n, room, frame
	
def selectImages(regr,fvec_tr,label_tr,fvec_test,room_test,im_count,min_cir=1,max_cir=9):
    regr.fit(fvec_tr,label_tr)
    label_new=regr.predict(fvec_test)
    rooms=np.unique(room_test)
    idx=-1*np.ones((len(rooms),im_count*(max_cir-min_cir+1)))
    
    for rm in rooms:
        cur_pos=0
        lbl=label_new.copy()
        lbl[room_test!=rm]=-1
        for k in range(min_cir,max_cir+1):
            idx_c=np.where(np.logical_and(lbl>=2*k,lbl<2*k+2).astype(np.float))[0]
            len_c=len(idx_c)
            if(len_c<im_count*k-cur_pos):
                idx[rm-1,cur_pos:cur_pos+len_c]=idx_c
                cur_pos=cur_pos+len_c
            else:
                print(np.floor(np.linspace(0,len_c-1,im_count*k-cur_pos)).astype(np.int))
                idx[rm-1,cur_pos:im_count*k]=idx_c[np.floor(np.linspace(0,len_c-1,im_count*k-cur_pos)).astype(np.int)]
                cur_pos=im_count*k
    return idx.astype(np.int),label_new
	
def saveImages(rootdir,subdirs,fr_num,im_num=6):
    destdir=rootdir+'-to-be-rated'
    if not os.path.exists(destdir+'\\logs'):
            os.makedirs(destdir+'\\logs')
    for k in range(len(subdirs)):
        if not os.path.exists(destdir+subdirs[k]):
            for cir in range(1,10):
                os.makedirs(destdir+subdirs[k]+'\\'+str(cir))
            os.makedirs(destdir+subdirs[k]+'\\type1')
        for fr in fr_num[k,:]:
            if(fr!=-1):
                if(im_num==8):
                    im_subs=[None]*8
                    for l in range(8):
                        im_subs[l]=skimage.io.imread("".join((rootdir,subdirs[k],'\\cam',str(l+1),'_fr',str(fr),'.png')))
                    im=np.zeros((1034,2600,3))
                    im[:512,522:1034,:]=im_subs[4];
                    im[:512,1044:1556,:]=im_subs[3];
                    im[:512,1566:2078,:]=im_subs[5];
                    im[522:1034,:512,:]=im_subs[1];
                    im[522:1034,522:1034,:]=im_subs[6];
                    im[522:1034,1044:1556,:]=im_subs[0];
                    im[522:1034,1566:2078,:]=im_subs[7];
                    im[522:1034,2088:,:]=im_subs[2];
                else:
                    im_subs=[None]*6
                    for l in range(6):
                        im_subs[l]=skimage.io.imread("".join((rootdir,subdirs[k],'\\cam',str(l+1),'_fr',str(fr),'.png')))
                    im=np.zeros((1034,1556,3))
                    im[:512,:512,:]=im_subs[4];
                    im[:512,522:1034,:]=im_subs[3];
                    im[:512,1044:,:]=im_subs[5];
                    im[522:1034,:512,:]=im_subs[1];
                    im[522:1034,522:1034,:]=im_subs[0];
                    im[522:1034,1044:,:]=im_subs[2];
                skimage.io.imsave("".join((destdir,subdirs[k],'\\type1\\fr',str(fr),'.png')),im.astype(np.uint8))
                logTarget="".join((rootdir,subdirs[k],'\\fr',str(fr),'_log.txt'))
                logDest="".join((destdir,'\\logs\\fr',str(fr),'_log.txt'))
                shutil.copy2(logTarget,logDest)	
def plot_hist(label,room,ttle='Room Histograms',cir_bins=range(1,11)):
    label=np.ceil(label)
    plt.figure()
    plt.subplot(1,3,1)
    plt.hist(label[room==1],cir_bins)
    plt.subplot(1,3,2)
    plt.hist(label[room==2],cir_bins)
    plt.title(ttle)
    plt.subplot(1,3,3)
    plt.hist(label[room==3],cir_bins)
    
	
def save_images_with_label(rootdir,subdirs,frame_test,room_test,labels,im_num=6):
    frames=[frame_test[room_test==1],frame_test[room_test==2],frame_test[room_test==3]]
    labels=labels.astype(np.int)
    labels=[labels[room_test==1],labels[room_test==2],labels[room_test==3]]
    destdir=rootdir+'-rated'
    if not os.path.exists(destdir+'\\logs'):
            os.makedirs(destdir+'\\logs')

    for k in range(2,len(subdirs)):
        if not os.path.exists(destdir+subdirs[k]):
            for cir in range(1,10):
                os.makedirs(destdir+subdirs[k]+'\\'+str(cir))
        for fr in range(len(frames[k])):
            fr_num=frames[k][fr]
            fr_label=labels[k][fr]
            if(fr!=0):
                for l in range(im_num):
                    targetdir="".join((rootdir,subdirs[k],'\\cam',str(l+1),'_fr',str(fr_num),'.png'))
                    destdirr="".join((destdir,subdirs[k],'\\',str(fr_label),'\\cam',str(l+1),'_fr',str(fr_num),'.png'))
                    shutil.copy2(targetdir,destdirr)
                logTarget="".join((rootdir,subdirs[k],'\\fr',str(fr_num),'_log.txt'))
                logDest="".join((destdir,'\\logs\\fr',str(fr_num),'_log.txt'))
                shutil.copy2(logTarget,logDest)
