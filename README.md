# KsponSpeech.preprocess

### Repository for pre-processing KsponSpeech corpus provided by AI Hub.

<img src="https://img.shields.io/badge/License-MIT-yellow" height=20> [<img src="https://img.shields.io/badge/chat-on%20gitter-4fb99a" height=20>](https://gitter.im/Korean-Speech-Recognition/community)  
  
## Intro

`KsponSpeech.preprocess` is repository for pre-processing `KsponSpeech corpus` provided by AI Hub.  
I pre-process for a `Speech Recognition` task.   
This repository provides **filtering transcript**, **create character labels**, **create id transcript**.   
For detail, I explain in the `Pre-process` part below.         
  
`KsponSpeech corpus` is a **1000h** Korean speech data corpus provided by [AI Hub](http://www.aihub.or.kr/) in Korea.   
Anyone can download this dataset just by applying. The transcription rules can see [here](http://www.aihub.or.kr/sites/default/files/2019-12/%ED%95%9C%EA%B5%AD%EC%96%B4%20%EC%9D%8C%EC%84%B1%20%EC%A0%84%EC%82%AC%EA%B7%9C%EC%B9%99%20v1.0.pdf).  
  
## Pre-process
  
The text for provided by the AI Hub is as follows.  
(**Spell transcription**) / (**Pronunciation transcription**), noise, groping, etc.. are labeled in detail.   
I pre-process for a `Speech Recognition` task.   
So, I left only the labels that i thought were necessary for the automatic speech recognition system (ASR).   
   
### Filtering transcript

* Raw data
```
"b/ 아/ 모+ 몬 소리야 (70%)/(칠 십 퍼센트) 확률이라니 n/" 
``` 
  
* Delete noise labels, such as b/, n/, / ..
```
"아/ 모+ 몬 소리야 (70%)/(칠 십 퍼센트) 확률이라니"
```
  
* Use pronunciation transcription
```
"아/ 모+ 몬 소리야 칠 십 퍼센트 확률이라니"
```
  
* Delete labels such as '/', '*', '+', etc. (used for gantour representation)
```
"아 모 몬 소리야 칠 십 퍼센트 확률이라니"
```
  
### Create character labels
  
Create character label file as csv format.  
  
|id|char|freq|  
|:--:|:----:|:----:|   
|0|\<sos\>|0|   
|1|<eos\>|0|   
|2|\<pad\>|0|  
|3| |5774462|   
|4|.|640924|   
|5|그|556373|     
|.|.|.|  
|.|.|.|     
|2336|뷁|1|      
|2337|꺟|1|  
  
### Create id transcript
  
Create id transcript according to character label csv file.  
  
* Before
```
근데 칠십 퍼센트가 커 보이긴 하는데 이백 벌다 백 사십 벌면 빡셀걸?
```
  
* After
```
35 11 0 318 119 0 489 551 156 6 0 379 0 42 3 144 0 14 4 11 0 3 248 0 355 15 0 248 0 34 119 0 355 24 0 588 785 104 12
```
  
## Get Started
  
### Method 1: Run `main.py`  
```
$ python main.py --datasetpath your_path --script_prefix KsponScript_
```  

### Method 2: Run `run.sh`  

  
* run.sh
  
Set the dataset path in run.sh
```
DATASET_PATH=''                   # SET YOUR KsponSpeech corpus PATH
SCRIPT_PREFIX='KsponScript_'      # IF YOU WANT, CHANGE AS CUSTOM.


python main.py -dataset_path $DATASET_PATH -script_prefix $SCRIPT_PREFIX
```  
  
Run run.sh
```
$ ./run.sh
```
  
## Troubleshoots and Contributing
  
If you have any questions, bug reports, and feature requests, please [open an issue](https://github.com/sooftware/KsponSpeech.preprocess/issues) on Github.   
For live discussions, please go to our [gitter](https://gitter.im/Korean-Speech-Recognition/community) or Contacts sh951011@gmail.com please.  
  
I appreciate any kind of feedback or contribution.  Feel free to proceed with small issues like bug fixes, documentation improvement.  For major contributions and new features, please discuss with the collaborators in corresponding issues.  
  
### Citing
```
@github{
  title = {KsponSpeech.preprocess},
  author = {Soohwan Kim},
  publisher = {GitHub},
  url = {https://github.com/sooftware/KsponSpeech.preprocess},
  year = {2020}
}
```
