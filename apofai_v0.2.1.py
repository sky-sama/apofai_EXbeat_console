from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
#import matplotlib.font_manager
import numpy as np
import scipy.io.wavfile as wav
from scipy.fft import rfft, rfftfreq,fft,fftfreq,ifft
import os.path as pth
#from scipy.ndimage import convolve1d as convolve
#import multiprocessing as process
#import threading
#import time
import scipy.signal
#SAMPLE_RATE = 44100
#DURATION = 5
SIGMA=1e-2
MU=0.5
rate=None;all_samples=None;path=None;name=None;val=0;x=None;y0=None;y1=None;bpm=None

def convfft(a, b):#使用快速傅里叶变换的快速卷积
    N = len(a)
    M = len(b)
    YN = N + M - 1
    FFT_N = 2 ** (int(np.log2(YN)) + 1)
    afft = fft(a, FFT_N)
    bfft = fft(b, FFT_N)
    abfft = afft * bfft
    y = ifft(abfft)[int(np.floor(M/2)):int(np.floor(N+M/2))]
    return np.real(y)

plt.rcParams['font.family']=['Microsoft YaHei']#给plot库导入中文字体

#指令
#怀疑开头的卡顿都用在def上了（解释性语言）

def cinfo(args):
  print("自动采音器v0.2 Console.ver\n\
作者：@bilibili自己想柠檬\n\
关注柠檬喵~关注柠檬谢谢喵~\n\
--------------------------------------------------\n\
当前可用指令：\n\
info\n\
help (demo)\n\
list (demo)\n\
open/openfile\n\
show\n\
std/savetimedata (useless)\n\
bpm\n\
val\n\
make/generate\n\
exit\n\
要求路径时可以使用相对路径（应该可以）\n\
--------------------------------------------------")

def chelp(args):
  print("待施工。至少先像cmd那样用下试试。\n除部分指令外，空格会分隔参数。\n需要参数的命令，将参数留空可查看该命令的帮助")

def clist(args):
  print("命令列表：\n"+"\n".join(commandlist.keys())+"\n待施工。以后会加命令描述")

def copenfile(args):
  global rate,all_samples,path,name,val,x,y0,y1
  if len(args)==0 or args[0].lower=="help":
    print("命令格式：\n\
    openfile path\n\
单参数命令。空格在该命令中不分隔参数。\n\
打开一个WAV音频文件，然后查看其波形。\n\
参数1：path：待打开的WAV文件路径。")
    return
  tmp=" ".join(args)
  if not pth.isfile(tmp):
    print("不是一个有效的路径，请重试")
    return
  path=tmp
  try:
    rate, all_samples = wav.read(path)
  except:
    print("文件读取失败：不是可用的文件格式或其他意外。当前仅支持读取WAV格式的音频。")
    return
  print("采样率",rate,"采样总数",len(all_samples),"数据类型（无用）",all_samples.dtype)
  name=path.split("\\")[-1:][0]
  x = np.linspace(0, len(all_samples)/rate, len(all_samples), endpoint=False)
  y0 = np.array(all_samples)[:,0]
  y1 = np.int32(y0)**2
  y1 = np.int16((y1/y1.max())*32767)
  cshow([])

def cshow(args):
  global rate,all_samples,path,name,val,x,y0,y1
  if path==None:
    print("你还没有选中文件。使用open指令打开一个WAV文件先~")
    return
  fig, ax = plt.subplots()
  plt.subplots_adjust(bottom=0.25)
  plt.title(f"{name}的采音预览")
  plt.plot(x, y0)
  plt.plot(x, y1)
  plt.xlabel("时间(s)")
  plt.axhline(y=0, color='k')
  sig=SIGMA*np.exp(val*0.5)
  norm=1/(np.sqrt(2*np.pi)*sig)*np.exp(-((x-MU*len(x)/rate)**2)/(2*sig**2))
  try:
    y2=convfft(np.int32(y1),norm)
  except:
    print("在处理音频时发生了错误。可能是内存不足。")
    return
  y2 = np.int16((y2/y2.max())*32767)
  peak=scipy.signal.find_peaks(y2)[0]
  line=ax.plot(x, y2)[0]
  dots=ax.plot(peak/rate,y2[peak],'o',markersize=1)[0]
  
  axfreq = plt.axes([0.1, 0.1, 0.8, 0.05])
  slider = Slider(
      ax=axfreq,
      label="平滑度",
      valmin=-5,
      valmax=5,
      valinit=val,
      valstep=0.1
  )
  
  def update(val):
    sig=SIGMA*np.exp(val*0.5)
    norm=1/(np.sqrt(2*np.pi)*sig)*np.exp(-((x-MU*len(x)/rate)**2)/(2*sig**2))
    try:
      y2=convfft(np.int32(y1),norm)
    except:
      print("在处理音频时发生了错误。可能是内存不足。")
      return
    y2 = np.int16((y2/y2.max())*32767)
    peak=scipy.signal.find_peaks(y2)[0]
    line.set_ydata(y2)
    dots.set_data(peak/rate,y2[peak])
  
  slider.on_changed(update)
  ini=slider.val

  plt.show(block=True)

  if not ini==slider.val:
    val=slider.val

def cstd(args):
  global rate,all_samples,path,name,val,x,y0,y1
  if path==None:
    print("你还没有选中文件。使用open指令打开一个WAV文件先~")
    return
  sig=SIGMA*np.exp(val*0.5)
  norm=1/(np.sqrt(2*np.pi)*sig)*np.exp(-((x-MU*len(x)/rate)**2)/(2*sig**2))
  try:
    y2=convfft(np.int32(y1),norm)
  except:
    print("在处理音频时发生了错误。可能是内存不足。")
    return
  y2 = np.int16((y2/y2.max())*32767)
  peak=scipy.signal.find_peaks(y2)[0]
  offset=peak/rate
  np.savetxt(f"{path}_{val}_timedata.txt",offset)
  print(f"文件已保存至{path}_{val}_timedata.txt")

def cbpm(args):
  global bpm
  if len(args)==0 or args[0].lower=="help":
    print(f"命令格式：\n\
    bpm bpm\n\
\n\
重设输出谱面BPM，或者改为自动获取BPM。\n\
参数1：bpm：将要设为的bpm。输入auto来使用自动获取（默认）\n\
当前bpm：{bpm}")
    return
  try:
    bpm=float(args[0])
    print(f"已将BPM设为{bpm}")
  except ValueError:
    if args[0].lower()=="auto":
      bpm=None
      print("BPM将自动测算")
    else:
      if input("无效输入。是否将BPM设为自动？(Y/N)").lower()=="y":
        bpm=None
        print("BPM将自动测算")
      else:
        print("已取消")

def cval(args):
  global val
  if len(args)==0 or args[0].lower=="help":
    print(f"命令格式：\n\
    val value\n\
重设采音平滑度。可以突破滑动条的范围限制和分度值限制。\n\
值越小采音强度越高，按键越密集。\n\
参数1：value：将要设为的值。\n\
当前平滑度：{val}")
    return
  try:
    val=float(args[0])
    print(f"已将平滑度设为{val}")
  except ValueError:
    print(f"无效输入。当前采音平滑度为{val}")

def cgenerate(args):
  global rate,all_samples,path,name,val,x,y0,y1,bpm
  if path==None:
    print("你还没有选中文件。使用open指令打开一个WAV文件先~")
    return
  sig=SIGMA*np.exp(val*0.5)
  norm=1/(np.sqrt(2*np.pi)*sig)*np.exp(-((x-MU*len(x)/rate)**2)/(2*sig**2))
  
  y2=convfft(np.int32(y1),norm)
  y2 = np.int16((y2/y2.max())*32767)
  peak=scipy.signal.find_peaks(y2)[0]
  
  if len(peak)==0:
    print("你干嘛，怎么一个音也没采到，老子不干了！")
    return
  if len(peak)==1:
    print("你干嘛，怎么只采到一个音，这让我怎么算BPM")
    return
  offset=peak/rate
  #np.savetxt(f"{name}_{val}_data.txt",offset)
  if bpm==None:
    offset0=offset[1:]-offset[:-1]
    bpmdt=np.median(offset0)
    bpm=np.int16(60/bpmdt)
  #bpm=300
  bpmdt=60/bpm
  offseti=offset[0]
  offset=offset-offseti
  if len(offset)%2==0:
    all=np.reshape(offset,(2,-1),"F")
    fire=np.concatenate((all[0][1:],[0]))
    ice=all[1]
    even=False
  else:
    all=np.reshape(offset[1:],(2,-1),"F")
    ice=all[0]
    fire=all[1]
    even=True
  combine=np.array([ice,fire])
  combineangle=combine/bpmdt*180
  overangle=combineangle.reshape(-1,order="F")
  if not even:
    overangle=overangle[:-1]
  deltaangle=overangle-np.concatenate(([0],overangle[:-1]))
  extraround=np.floor(deltaangle/360-1e-126)*2
  for i in range(len(combineangle[0])-1):
    if (deltaangle[2*i+1])%360==0 and extraround[2*i+1]!=0:
      extraround[2*i+1]+=1
  combineangle=np.array([(180-combineangle[0])%360,
                         (360-combineangle[1])%360
                         ])
  tmp=combineangle.reshape(-1,order="F")
  if not even:
    tmp=tmp[:-1]
  angledata=", ".join(str(i) for i in tmp)
  
  tmpstr="{\n\
  	\"angleData\": [0, "+f"{angledata}"+"], \n\
  	\"settings\":\n\
  	{\n\
  		\"version\": 13 ,\n\
  		\"artist\": \"\", \n\
  		\"specialArtistType\": \"None\", \n\
  		\"artistPermission\": \"\", \n\
  		\"song\": \"\", \n\
  		\"author\": \"\", \n\
  		\"separateCountdownTime\": true, \n\
  		\"previewImage\": \"\", \n\
  		\"previewIcon\": \"\", \n\
  		\"previewIconColor\": \"003f52\", \n\
  		\"previewSongStart\": 0, \n\
  		\"previewSongDuration\": 10, \n\
  		\"seizureWarning\": false, \n\
  		\"levelDesc\": \"\", \n\
  		\"levelTags\": \"\", \n\
  		\"artistLinks\": \"\", \n\
  		\"speedTrialAim\": 0, \n\
  		\"difficulty\": 1, \n\
  		\"requiredMods\": [],\n\
  		\"songFilename\": \""+f"{name}"+"\", \n\
  		\"bpm\": "+f"{bpm}"+", \n\
  		\"volume\": 100, \n\
  		\"offset\": "+f"{np.int32(offseti*1000)}"+", \n\
  		\"pitch\": 100, \n\
  		\"hitsound\": \"Kick\", \n\
  		\"hitsoundVolume\": 100, \n\
  		\"countdownTicks\": 4,\n\
  		\"trackColorType\": \"Rainbow\", \n\
  		\"trackColor\": \"debb7b\", \n\
  		\"secondaryTrackColor\": \"ffffff\", \n\
  		\"trackColorAnimDuration\": 2, \n\
  		\"trackColorPulse\": \"Forward\", \n\
  		\"trackPulseLength\": 100, \n\
  		\"trackStyle\": \"NeonLight\", \n\
  		\"trackTexture\": \"\", \n\
  		\"trackTextureScale\": 1, \n\
  		\"trackGlowIntensity\": 100, \n\
  		\"trackAnimation\": \"None\", \n\
  		\"beatsAhead\": 3, \n\
  		\"trackDisappearAnimation\": \"Shrink\", \n\
  		\"beatsBehind\": 0,\n\
  		\"backgroundColor\": \"000000\", \n\
  		\"showDefaultBGIfNoImage\": true, \n\
  		\"showDefaultBGTile\": true, \n\
  		\"defaultBGTileColor\": \"101121\", \n\
  		\"defaultBGShapeType\": \"Default\", \n\
  		\"defaultBGShapeColor\": \"ffffff\", \n\
  		\"bgImage\": \"\", \n\
  		\"bgImageColor\": \"ffffff\", \n\
  		\"parallax\": [100, 100], \n\
  		\"bgDisplayMode\": \"FitToScreen\", \n\
  		\"imageSmoothing\": true, \n\
  		\"lockRot\": false, \n\
  		\"loopBG\": false, \n\
  		\"scalingRatio\": 100,\n\
  		\"relativeTo\": \"Player\", \n\
  		\"position\": [0, 0], \n\
  		\"rotation\": 0, \n\
  		\"zoom\": 150, \n\
  		\"pulseOnFloor\": true,\n\
  		\"bgVideo\": \"\", \n\
  		\"loopVideo\": false, \n\
  		\"vidOffset\": 0, \n\
  		\"floorIconOutlines\": false, \n\
  		\"stickToFloors\": true, \n\
  		\"planetEase\": \"Linear\", \n\
  		\"planetEaseParts\": 1, \n\
  		\"planetEasePartBehavior\": \"Mirror\", \n\
  		\"defaultTextColor\": \"ffffff\", \n\
  		\"defaultTextShadowColor\": \"00000050\", \n\
  		\"congratsText\": \"\", \n\
  		\"perfectText\": \"\",\n\
  		\"legacyFlash\": false ,\n\
  		\"legacyCamRelativeTo\": false ,\n\
  		\"legacySpriteTiles\": false \n\
  	},\n\
  	\"actions\":\n\
  	["
  for i in range(len(extraround)):
    if extraround[i]!=0:
      tmpstr+="\n		{ \"floor\": "+f"{i+1}"+", \"eventType\": \"Pause\", \"duration\": "+f"{extraround[i]}"+", \"countdownTicks\": 0, \"angleCorrectionDir\": -1 },"
  tmpstr+="\n\
  	],\n\
  	\"decorations\":\n\
  	[\n\
    ]\n\
  }"
  with open(f"{path}_{val}.adofai","wt",1,"utf-8") as file:
    file.write(tmpstr)
    print(f"谱面文件已保存至{path}_{val}.adofai")

def cexit(args):
  exit()

def celse(args):
  print("不是可用的命令。输入list获取当前支持的命令列表")

commandlist={"info":cinfo,"help":chelp,"list":clist,"open":copenfile,"openfile":copenfile,"show":cshow,
             "std":cstd,"savetimedata":cstd,"bpm":cbpm,"val":cval,"make":cgenerate,"generate":cgenerate,
       		 "exit":cexit}
#Main:
cinfo([])
while True:
  command=input("=> ").strip().split(" ")
  if command[0]=="":
    continue
  commandlist.get(command[0].lower(),celse)(command[1:])