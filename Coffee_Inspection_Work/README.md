# 1. 라즈베리파이
## (1) 라즈베리파이 카메라 모듈
### 가. Product Name : Pi NoIR - Raspberry Pi Infrared Camera Module
### 나. Resolution : 8-megapixel
### 다. Still picture resolution : 3280 x 2464
### 라. Max image transfer rate
    - 1080p : 30fps(encode and decode)
    - 720p : 60fps

## (2) 라즈베리파이 OS 설치
### 가. SD카드 포멧
    - 컴퓨터 관리 -> 저장소 -> 디스크 관리 -> 디스크 1 우클릭 -> 새단순볼륨 생성 -> 완전 포멧 
### 나.https://www.raspberrypi.org/downloads/ 접속
### 다. Raspberry Pi Imager for Windows 실행
### 라. 가이드 대로 수행.

## (3) 라즈베리파이 samba 설치
    - sudo apt-get update
    - sudo apt-get install samba samba-common-bin
    - sudo smbpasswd -a pi
    - sudo nano /etc/samba/smb.conf
            [pi]
            comment = pi sharing folder 
            path = /home/pi(공유폴더 경로)
            valid user = pi
            writable = yes
            read only = no
            browseable = yes
    - sudo /etc/init.d/samba restart
    - 윈도우에서 \\192.168.0.14\pi 라고 입력
    
## (4) 라즈베리파이 외부접속 허용
    - 윈도우 cmd 창 
    - mongod --bind_ip 0.0.0.0
    


