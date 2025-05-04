# coding = utf-8

import random
import time
from datetime import datetime, timedelta
import pickle
import logging

from urllib.request import urlopen
from urllib.parse import quote
import bs4

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import platform
import os

BASE_DIR = os.path.expanduser('~/ProjectAuto')
temp_dir = os.path.expanduser('~/ProjectAuto/temp')
log_dir = os.path.expanduser('~/ProjectAuto/log')
data_dir = os.path.expanduser('~/ProjectAuto/data')
font_dir = os.path.expanduser('~/ProjectAuto/fonts')
   
folder_lst = [BASE_DIR, temp_dir, log_dir, data_dir, font_dir]

for f in folder_lst:
    os.makedirs(f, exist_ok=True)

def get_fonts():
    """
    """
    import os
    from pathlib import Path

    # 폰트 저장 디렉토리 설정
    font_dir = BASE_DIR + '/fonts/NanumGothic'
    font_dir = Path(font_dir)

    # 디렉토리가 없으면 생성
    font_dir.mkdir(parents=True, exist_ok=True)

    # 폰트 파일 찾기 (Regular 및 Bold)
    nanum_gothic_regular = list(font_dir.glob('*NanumGothic.ttf'))
    nanum_gothic_bold = list(font_dir.glob('*NanumGothicBold.ttf'))

    if nanum_gothic_regular:
        pass
    else:        
        raise ValueError(f"""NanumGothic Regular 폰트 파일을 찾을 수 없습니다. \nhttps://fonts.google.com/download?family=Nanum+Gothic 에서 폰트를 다운로드하고\n{BASE_DIR}/fonts 폴더 밑에 ttf파일을 옮겨주세요.""")
    
    if nanum_gothic_bold:
        pass
    else:        
        raise ValueError(f"""NanumGothic Bold 폰트 파일을 찾을 수 없습니다.\nhttps://fonts.google.com/download?family=Nanum+Gothic 에서 폰트를 다운로드하고\n{BASE_DIR}/fonts 폴더 밑에 ttf파일을 옮겨주세요.""")

    return {"regular": f"{BASE_DIR}/fonts/NanumGothic/NanumGothic.ttf",
            "bold": f"{BASE_DIR}/fonts/NanumGothic/NanumGothicBold.ttf"}



if __name__ == '__main__':
    print('main')
    font_path = get_fonts()
    print(font_path)
    