VER="v0.2.6"
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
#import matplotlib.font_manager
import numpy as np
import scipy.io.wavfile as wav
from scipy.fft import fft,ifft
import os.path as pth
#from scipy.ndimage import convolve1d as convolve
#import multiprocessing as process
#import threading
#import time
import scipy.signal
import subprocess
import re
import sys
#SAMPLE_RATE = 44100
#DURATION = 5
SIGMA=1e-2
MU=0.5
rate=None;all_samples=None;path=None;name=None;val=0;x=None;y0=None;y1=None;bpm=None
height=[0,32767];threshold=[0,32767];distance=0

#escape_character_list = {"\\":"x5c","a":"x07","b":"x08","t":"x09","n":"x0a","v":"x0b","f":"x0c","r":"x0d",'"':"x22","'":"x27"}

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

def cinfo(args):
  print(f"自动采音器{VER} Console.ver\n\
作者：@bilibili自己想柠檬\n\
关注柠檬喵~关注柠檬谢谢喵~\n\
--------------------------------------------------\n\
输入help获取基本的使用帮助\n\
输入list获取支持的所有命令\n\
--------------------------------------------------")

def chelp(args):
  print("基本格式：\n\tcommand [arg(s)]\n需要参数的命令，将参数留空可查看该命令的帮助。除部分指令外，空格会分隔不同参数。\n无需参数的命令，输入任何值不影响其执行。可在list命令中查看该命令的描述\n使用list命令获取当前版本支持的所有命令\n\
         (v0.2.6)现已支持引号\"'和转义字符\\。在命令末尾输入\" \\\"以启用转义，如使用\\x20表示空格等。")

def clist(args):
  print("当前可用的所有命令：\n"+"\n".join(pair[0]+"\t"+pair[1][1] for pair in commandlist.items())+"\n")

def copenfile(args,show=True):
  global rate,all_samples,path,name,val,x,y0,y1
  if len(args)==0 or args[0].lower=="help" or args[0].lower=="/?":
    print("命令格式：\n\
    {openfile|openonly} path\n\
单参数命令。空格在该命令中不分隔参数。\n\
打开一个媒体文件。如果不是.wav格式将自动调用ffmpeg转码，并将覆盖同名wav文件，请注意。\n\
参数1：path：待打开的媒体文件路径。\n\
示例：\n\
open C:\\Users\\\\Desktop\\_hitsound.wav\n\
打开文件并预览采音\n\
openonly F:\\A Dance of Fire and Ice\\背景视频.mp4\n\
抽取音频转码为.wav文件，但不打开采音预览界面。")
    return
  if args[0][0] == "'" or args[0][0] == "\"":
    tmp=args[0][1:-1]
  else:
    tmp=" ".join(args)
  if not pth.isfile(tmp):
    print("不是一个有效的路径，请重试")
    return
  if tmp[-4:].lower()!=".wav":
    tmp2=".".join(tmp.split(".")[:-1])+".wav"
    try:
      print("ffmpeg输出，看不懂没关系，反正我也不懂（总之没看到error就行")
      subprocess.Popen('ffmpeg -i '+f'\"{tmp}\" \"{tmp2}\"').wait()
    except:
      print("不是一个可识别的音频格式。请确保路径指向音频文件或包含音频的视频文件")
      return
    if not pth.isfile(tmp2):
      print("不是一个可识别的音频格式。请确保路径指向音频文件或包含音频的视频文件")
      return
    path=tmp2
  else:
    path=tmp
  try:
    rate, all_samples = wav.read(path)
  except:
    print("文件读取失败：不是可用的文件格式或其他意外。")
    return
  print("采样率",rate,"采样总数",len(all_samples),"数据类型（无用）",all_samples.dtype)
  name=path.split("\\")[-1:][0]
  x = np.linspace(0, len(all_samples)/rate, len(all_samples), endpoint=False)
  y0 = np.array(all_samples)
  if y0.ndim == 2 :
    y0 = (y0[:,0]+y0[:,1])/2
  y1 = np.int32(y0)**2
  y1 = np.int16((y1/y1.max())*32767)
  if show:
    cshow([])

def copenonly(args):
  copenfile(args,show=False)

def cshow(args):
  global rate,all_samples,path,name,val,x,y0,y1,height
  if path==None:
    print("你还没有选中文件。使用open指令打开一个WAV文件先~")
    return
  try:
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)
    plt.title(f"{name}的采音预览")
    plt.plot(x, y0)
    plt.plot(x, y1)
    plt.xlabel("时间(s)")
    plt.axhline(y=0, color='k')
  except Exception as err:
    print(f"绘制图像时发生错误。可能是内存不足。\n{err}")
    return
  sig=SIGMA*np.exp(val*0.5)
  norm=1/(np.sqrt(2*np.pi)*sig)*np.exp(-((x-MU*len(x)/rate)**2)/(2*sig**2))
  try:
    y2=convfft(np.int32(y1),norm)
  except Exception as err:
    print(f"在处理音频时发生了错误。可能是内存不足。\n{err}")
    return
  y2 = np.int16((y2/y2.max())*32767)
  peak=scipy.signal.find_peaks(y2,height)[0]
  try:
    minline=ax.axhline(y=height[0], color='k')
    maxline=ax.axhline(y=height[1], color='k')
    line=ax.plot(x, y2)[0]
    dots=ax.plot(peak/rate,y2[peak],'o',markersize=1)[0]
  except Exception as err:
    print(f"绘制图像时发生错误。可能是内存不足。\n{err}")
    return
  
  axval = plt.axes([0.1, 0.14, 0.8, 0.02])
  sliderval = Slider(
      ax=axval,
      label="平滑度",
      valmin=-5,
      valmax=5,
      valinit=val,
      valstep=0.1
  )
  axmin = plt.axes([0.1, 0.12, 0.8, 0.02])
  slidermin = Slider(
      ax=axmin,
      label="阈值（小）",
      valmin=0,
      valmax=32767,
      valinit=height[0],
      valstep=1
  )
  axmax = plt.axes([0.1, 0.1, 0.8, 0.02])
  slidermax = Slider(
      ax=axmax,
      label="阈值（大）",
      valmin=0,
      valmax=32767,
      valinit=height[1],
      valstep=1
  )
  
  def valupdate(val):
    sig=SIGMA*np.exp(val*0.5)
    norm=1/(np.sqrt(2*np.pi)*sig)*np.exp(-((x-MU*len(x)/rate)**2)/(2*sig**2))
    try:
      y2=convfft(np.int32(y1),norm)
    except Exception as err:
      print(f"在处理音频时发生了错误。可能是内存不足。\n{err}")
      return
    y2 = np.int16((y2/y2.max())*32767)
    peak=scipy.signal.find_peaks(y2,[slidermin.val,slidermax.val])[0]
    line.set_ydata(y2)
    dots.set_data(peak/rate,y2[peak])
  
  def minupdate(min):
    peak=scipy.signal.find_peaks(y2,[min,slidermax.val])[0]
    minline.set_ydata([min,min])
    dots.set_data(peak/rate,y2[peak])
  def maxupdate(max):
    peak=scipy.signal.find_peaks(y2,[slidermin.val,max])[0]
    maxline.set_ydata([max,max])
    dots.set_data(peak/rate,y2[peak])
  
  sliderval.on_changed(valupdate)
  slidermin.on_changed(minupdate)
  slidermax.on_changed(maxupdate)
  ini=sliderval.val

  plt.show(block=True)

  height=[slidermin.val,slidermax.val]
  if not ini==sliderval.val:
    val=sliderval.val

def cstd(args):
  global rate,all_samples,path,name,val,x,y0,y1,height
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
  peak=scipy.signal.find_peaks(y2,height)[0]
  offset=peak/rate
  curbpm=60/(offset-np.concatenate(([0],offset[:-1])))
  np.savetxt(f"{path}_{val}_timedata.txt",curbpm)
  print(f"文件已保存至{path}_{val}_timedata.txt")

def cbpm(args):
  global bpm
  if len(args)==0 or args[0].lower=="help" or args[0].lower=="/?":
    print(f"命令格式：\n\
    bpm bpm\n\
\n\
重设输出谱面BPM，或者改为自动获取BPM。\n\
参数1：bpm：将要设为的bpm。输入auto来使用自动获取（默认）\n\
当前bpm：{bpm}\n\
示例：\n\
bpm 150\n\
已将BPM设为150\n\
bpm auto\n\
BPM将自动测算")
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
  if len(args)==0 or args[0].lower=="help" or args[0].lower=="/?":
    print(f"命令格式：\n\
    val [value] [min] [max]\n\
重设采音相关参数。\n\
参数1：value：（可选的）采音平滑度。值越小采音强度越高，按键越密集。可以突破滑动条的范围限制和分度值限制。\n\
参数2：min：（可选的）采音阈值（最小值）。低于此值则不采此音。只接受0~32767之间的整数值。\n\
参数2：max：（可选的）采音阈值（最大值）。高于此值则不采此音。只接受0~32767之间的整数值。\n\
当前平滑度：{val}\n\
当前采音阈值：{height}\n\
示例：\n\
val -21 0 32767\n\
已将平滑度设为-21\t已将采音阈值（最小值）设为0\t已将采音阈值（最大值）设为32767\n\
val 0\n\
已将平滑度设为0\n\
val  100\n\
已将采音阈值（最小值）设为100\n\
val   32000\n\
已将采音阈值（最大值）设为32000")
    return
  if not args[0]=="":
    try:
      val=float(args[0])
      print(f"已将平滑度设为{val}")
    except ValueError:
      print(f"参数1无效。当前采音平滑度为{val}")
  if not (len(args)<2 or args[1]==""):
    try:
      height[0]=np.int16(args[1])
      print(f"已将采音阈值（最小值）设为{height[0]}")
    except ValueError:
      print(f"参数2无效。当前采音阈值（最小值）为{height[0]}")
  if not (len(args)<3 or args[2]==""):
    try:
      height[1]=np.int16(args[2])
      print(f"已将采音阈值（最大值）设为{height[1]}")
    except ValueError:
      print(f"参数3无效。当前采音阈值（最大值）为{height[1]}")

def chz(args):
  global y0,path
  if path==None:
    print("你还没有选中文件。使用open指令打开一个WAV文件先~")
    return
  peak=scipy.signal.find_peaks(-y0)[0]
  make(peak,"hz")

def cgenerate(args):
  global rate,all_samples,path,name,val,x,y0,y1,height
  if path==None:
    print("你还没有选中文件。使用open指令打开一个WAV文件先~")
    return
  sig=SIGMA*np.exp(val*0.5)
  norm=1/(np.sqrt(2*np.pi)*sig)*np.exp(-((x-MU*len(x)/rate)**2)/(2*sig**2))
  
  y2=convfft(np.int32(y1),norm)
  y2 = np.int16((y2/y2.max())*32767)
  peak=scipy.signal.find_peaks(y2,height)[0]
  make(peak,val)

def make(peak,exdescription):
  global rate,path,bpm
  if len(peak)==0:
    print("你干嘛，怎么一个音也没采到，老子不干了！")
    return
  offset=peak/rate
  #np.savetxt(f"{name}_{val}_data.txt",offset)
  if bpm==None:
    if len(peak)==1:
      print("你干嘛，怎么只采到一个音，这让我怎么算BPM")
      return
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
  with open(f"{path}_{exdescription}.adofai","wt",1,"utf-8") as file:
    file.write(tmpstr)
    print(f"谱面文件已保存至{path}_{exdescription}.adofai")

def cexit(args):
  print("")
  sys.exit()

def crun(args):
  if len(args)==0 or args[0].lower=="/?":
    print("命令格式：\n\
    & args\n\
运行外部程序或指令。\n\
第一个参数指定接受参数的程序或指令，其余的参数作为参数输入程序或指令。\n\
可以运行path路径内包含的程序或指令。\n\
示例：\n\
& help\n\
非常离谱的是，cmd的部分指令（如echo，cd，tree）是封装在其内部的指令，这些指令不能直接调用\n\
而另一部分（如help（离谱），xcopy，cmd）是放在System32下的程序，这些可以直接调用。\n\
所以如果你闲着没事看了一下help，里面很多指令都是不可用的。\n\
& ffmpeg -i ...\n\
该命令也可以运行同目录下的其他程序，比如ffmpeg（还有自己）。")
    return
  try:
    subprocess.Popen(" ".join(args)).wait()
  except Exception as err:
    print(f"{err}")

def celse(args):
  if args == ["bad_quotation_marks",]:
    print("发生错误：错误的引号数量或结构。请检查输入。")
  else:
    print("不是可用的命令。输入list获取当前支持的命令列表")

pre_escape_list = {'\\"':'\\x22',"\\'":"\\x27","\\\\":"\\x5c"}
escape_character_list = {"\\a":"\a","\\b":"\b","\\t":"\t","\\n":"\n","\\v":"\v","\\f":"\f","\\r":"\r"}

def precode(matchitem):
  return pre_escape_list.get(matchitem.group(),"\?")
#re.sub("\\\\[\"\'\\\\]",precode,{#})

def quode(strin:str,unlimited=True):
  inquote = False
  dblquote = True
  strout = ""
  for char in strin:
    if inquote:
      if (char == "\"" and dblquote)or(char == "'" and(not dblquote)):
        inquote = False
      if char == " ":
        if unlimited:
          strout += "\\x20"
        else:
          strout += "\v"
      else:
        strout += char
    else:
      if char == "\"":
        inquote = True
        dblquote = True
      elif char == "'":
        inquote = True
        dblquote = False
      strout += char
  if inquote:
    return "error bad_quotation_marks"
  else:
    return strout

def decode(matchitem):
  escaped_char = matchitem.group()
  if escaped_char == "\v":
    return " "
  elif escaped_char.startswith("\\x"):
    # 处理十六进制转义序列
    return chr(int(escaped_char[2:], 16))
    # 处理八进制转义序列
  elif 48 <= ord(escaped_char[1]) and ord(escaped_char[1]) <= 55:
    return chr(int(escaped_char[1:], 8))
  else:
    return escape_character_list.get(escaped_char,escaped_char)
#re.sub("\\\\[abtnvfr]|\\\\x[0-9a-fA-F]{2}|\\\\[0-7]{1,3}",decode,{#})

commandlist={"info":(cinfo,"\t显示初始信息"),
             "help":(chelp,"\t显示基本帮助"),
             "list":(clist,"\t显示所有可用的命令列表（本指令）"),
             "open":(copenfile,"\t=openfile"),
             "openfile":(copenfile,"打开要处理的媒体文件并显示预览"),
             "openonly":(copenonly,"仅打开文件而不显示预览（适合目的性强时）"),
             "show":(cshow,"\t显示预览"),
             "std":(cstd,"\t=savetimedata"),
             "savetimedata":(cstd,"以curbpm形式存储采音信息（v0.2.2新）"),
             "bpm":(cbpm,"\t查看或更改输出谱面BPM"),
             "val":(cval,"\t查看或更改采音相关参数"),
             "hz":(chz,"\t针对赫兹级采音优化的采音。跳过音高计算，直接生成.adofai谱面文件"),
             "make":(cgenerate,"\t=generate"),
             "generate":(cgenerate,"生成.adofai谱面文件"),
       		   "exit":(cexit,"\t退出"),
             "&":(crun,"\t运行外部命令()")}

def inputandprocess():
  tmp=input("=> ").strip()
  if tmp[-2:] == " \\":
    return [re.sub("\\\\[abtnvfr]|\\\\x[0-9a-fA-F]{2}|\\\\[0-7]{1,3}",decode,arg) for arg in quode(re.sub("\\\\[\"\'\\\\]",precode,tmp[:-2])).split(" ")]
  else:
    return [re.sub("\\v",decode,arg) for arg in quode(tmp,False).split(" ")]

#Main:
cinfo([])
while True:
  command=inputandprocess()
  if command[0]=="":
    continue
  commandlist.get(command[0].lower(),(celse,))[0](command[1:])
#乐。开心.jpg
