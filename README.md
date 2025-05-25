# useWhisperX
```
(useWhisperX) yuabe@g16:/mnt/kiso-qnap3/yuabe/m1/useWhisperX$ python3 whisperX.py
```
で動く．

(05/25)
```
(base) yuabe@g23:/mnt/kiso-qnap3/yuabe/m1/useWhisperX$ conda create -p ~/envs/whisperx-v3 python=3.10 -y
(base) yuabe@g23:/mnt/kiso-qnap3/yuabe/m1/useWhisperX$ conda info -e
# conda environments:
#
                         /home/yuabe/envs/whisperx-v3
base                  *  /mnt/kiso-qnap3/yuabe/m1/anaconda3
jmoshi312                /mnt/kiso-qnap3/yuabe/m1/anaconda3/envs/jmoshi312
moshiFT312               /mnt/kiso-qnap3/yuabe/m1/anaconda3/envs/moshiFT312
remdis                   /mnt/kiso-qnap3/yuabe/m1/anaconda3/envs/remdis
useWhisperX              /mnt/kiso-qnap3/yuabe/m1/anaconda3/envs/useWhisperX
whisper310               /mnt/kiso-qnap3/yuabe/m1/anaconda3/envs/whisper310

(base) yuabe@g23:/mnt/kiso-qnap3/yuabe/m1/useWhisperX$ conda activate /home/yuabe/envs/whisperx-v3
(whisperx-v3) yuabe@g23:/mnt/kiso-qnap3/yuabe/m1/useWhisperX$ pip install whisperx
```
NAS上にはpip install whisperxができなかったので，~/envs/whisperx-v3 に作った．
💡
g23 でしか作業できない！！

(追記)
この後失敗したので，使わなくなりました．．
原因は多分nvccコマンドが使えないことかなと．．