import numpy as np
import matplotlib.pyplot as plt
import scipy
import pandas as pd
def readfile(filename):
  df = pd.read_csv(filename,header = None)
  df.drop(df.columns[[0, 1, 2,3,4,5]], axis=1, inplace=True)
  df = df.transpose()
  df.columns = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8']
  b1 = (10**(df['B1']/10))*100
  b2 = (10**(df['B2']/10))*100
  b3 = (10**(df['B3']/10))*100
  b4 = (10**(df['B4']/10))*100
  b5 = (10**(df['B5']/10))*100
  b6 = (10**(df['B6']/10))*100
  b7 = (10**(df['B7']/10))*100
  b8 = (10**(df['B8']/10))*100
  b = np.concatenate((b1,b2,b3,b4,b5,b6,b7,b8))
  return b 

def groundfile(gf):
  dfg = pd.read_csv(gf,header = None)
  dfg.drop(dfg.columns[[0, 1, 2,3,4,5]], axis=1, inplace=True)
  dfg = dfg.transpose()
  dfg.columns = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8']
  b1 = (10**(dfg['B1']/10))*100
  b2 = (10**(dfg['B2']/10))*100
  b3 = (10**(dfg['B3']/10))*100
  b4 = (10**(dfg['B4']/10))*100
  b5 = (10**(dfg['B5']/10))*100
  b6 = (10**(dfg['B6']/10))*100
  b7 = (10**(dfg['B7']/10))*100
  b8 = (10**(dfg['B8']/10))*100
  bg = np.concatenate((b1,b2,b3,b4,b5,b6,b7,b8))
  return bg  

def skyfile(sf):
  dfs = pd.read_csv(sf,header = None)
  dfs.drop(dfs.columns[[0, 1, 2,3,4,5]], axis=1, inplace=True)
  dfs = dfs.transpose()
  dfs.columns = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8']
  b1 = (10**(dfs['B1']/10))*100
  b2 = (10**(dfs['B2']/10))*100
  b3 = (10**(dfs['B3']/10))*100
  b4 = (10**(dfs['B4']/10))*100
  b5 = (10**(dfs['B5']/10))*100
  b6 = (10**(dfs['B6']/10))*100
  b7 = (10**(dfs['B7']/10))*100
  b8 = (10**(dfs['B8']/10))*100
  bs = np.concatenate((b1,b2,b3,b4,b5,b6,b7,b8))
  return bs  

b = readfile('zero1.csv')
bg = groundfile('ground1.csv')
bs = skyfile('sky1.csv')
x = np.arange(1418, 1422, 4/1656)
print(x)

finalb1 = b - 0.5*bg
#finalb2 = b - 0.4*bg
#plt.plot(x,b)
plt.plot(x,finalb1)
#plt.plot(x,finalb2)


plt.plot(x,bs)
plt.plot(x,b)

def readfile(filename):
  df = pd.read_csv(filename,header = None)
  df.drop(df.columns[[0, 1, 2,3,4,5]], axis=1, inplace=True)
  df = df.transpose()
  df.columns = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8']
  b1 = (10**(df['B1']/10))*100
  b2 = (10**(df['B2']/10))*100
  b3 = (10**(df['B3']/10))*100
  b4 = (10**(df['B4']/10))*100
  b5 = (10**(df['B5']/10))*100
  b6 = (10**(df['B6']/10))*100
  b7 = (10**(df['B7']/10))*100
  b8 = (10**(df['B8']/10))*100
  b = np.concatenate((b1,b2,b3,b4,b5,b6,b7,b8))
  return b 


def skyfile(sf):
  dfs = pd.read_csv(sf,header = None)
  dfs.drop(dfs.columns[[0, 1, 2,3,4,5]], axis=1, inplace=True)
  dfs = dfs.transpose()
  dfs.columns = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8']
  b1 = (10**(dfs['B1']/10))*100
  b2 = (10**(dfs['B2']/10))*100
  b3 = (10**(dfs['B3']/10))*100
  b4 = (10**(dfs['B4']/10))*100
  b5 = (10**(dfs['B5']/10))*100
  b6 = (10**(dfs['B6']/10))*100
  b7 = (10**(dfs['B7']/10))*100
  b8 = (10**(dfs['B8']/10))*100
  bs = np.concatenate((b1,b2,b3,b4,b5,b6,b7,b8))
  return bs  

b = readfile('zero1.csv')
bg = groundfile('ground1.csv')
bs = skyfile('sky1.csv')
x = np.arange(1418, 1422, 4/1656)
print(x)

finalb = b - bs
plt.plot(x,b)



Experiment 2

def readfilea(filename):
  df = pd.read_csv(filename,header = None)
  df.drop(df.columns[[0, 1, 2,3,4,5]], axis=1, inplace=True)
  df = df.transpose()
  df.columns = ['B1']
  b1 = (10**(df['B1']/10))
  ba = b1
  return ba

def readfileb(filename):
  df = pd.read_csv(filename,header = None)
  df.drop(df.columns[[0, 1, 2,3,4,5]], axis=1, inplace=True)
  df = df.transpose()
  df.columns = ['B1']
  b1 = (10**(df['B1']/10))
  bb = b1
  return bb

ba = readfilea('ten2a.csv')
bb = readfileb('ten2b.csv')

bc = ba - bb
x = np.linspace(1419.5, 1421.5, 513)

plt.plot(x,ba)
plt.plot(x,bb)
plt.plot(x,bc)

plt.plot(x,bc)

print(bc.argmax())
print(x[245])

'''files = ['zero2a.csv', 'zero2b.csv','ten2a.csv', 'ten2b.csv','twenty2a.csv', 'twenty2b.csv','thirty2a.csv', 'thirty2b.csv','forty2a.csv', 'forty2b.csv','fifty2a.csv', 'fifty2b.csv','sixty2a.csv', 'sixty2b.csv','seventy2a.csv', 'seventy2b.csv']
files = np.array(files)
for i in files:
  ba = readfilea(i)
  bb = readfileb(i+1)
  bc = ba - bb 
  i = i+2

plt.plot(x,bc)'''

files = ['zero2a.csv', 'ten2a.csv','twenty2a.csv','thirty2a.csv','forty2a.csv','fifty2a.csv','sixty2a.csv','seventy2a.csv', 'threefivezero2a.csv']
files2 = ['zero2b.csv','ten2b.csv','twenty2b.csv','thirty2b.csv', 'forty2b.csv','fifty2b.csv','sixty2b.csv','seventy2b.csv', 'threefivezero2b.csv']
files3 = ['zero2a.csv', 'ten2a.csv','twenty2a.csv','thirty2a.csv','forty2a.csv','fifty2a.csv','sixty2a.csv','seventy2a.csv','threefivezero2a.csv', 'zero2b.csv','ten2b.csv','twenty2b.csv','thirty2b.csv', 'forty2b.csv','fifty2b.csv','sixty2b.csv','seventy2b.csv', 'threefivezero2b.csv']
files = np.array(files)
files2 = np.array(files2)
files3 = np.array(files3)
rows, cols = 3, 3
fig, ax = plt.subplots(rows, cols, figsize=(15, 15))
i=[0,0,0,1,1,1,2,2,2]
j=[0,1,2,0,1,2,0,1,2]
for k in range(0, 9):
              AA = readfilea(files3[k])
              BB = readfileb(files3[k+9])
              CC = AA - BB
              CF=CC.to_numpy()  
              #plt.plot(x, CC, c=np.random.rand(3,))
              ax[i[k],j[k]].plot(x, CF, c=np.random.rand(3,))
              
              



def grounda(filename):
  df = pd.read_csv(filename,header = None)
  df.drop(df.columns[[0, 1, 2,3,4,5]], axis=1, inplace=True)
  df = df.transpose()
  df.columns = ['B1']
  b1 = (10**(df['B1']/10))
  bga = b1
  return bga

def groundb(filename):
  df = pd.read_csv(filename,header = None)
  df.drop(df.columns[[0, 1, 2,3,4,5]], axis=1, inplace=True)
  df = df.transpose()
  df.columns = ['B1']
  b1 = (10**(df['B1']/10))
  bgb = b1
  return bgb

bga = grounda('ground2a.csv')
bgb = groundb('ground2b.csv')

bg3 = bga - bgb

def skya(filename):
  df = pd.read_csv(filename,header = None)
  df.drop(df.columns[[0, 1, 2,3,4,5]], axis=1, inplace=True)
  df = df.transpose()
  df.columns = ['B1']
  b1 = (10**(df['B1']/10))
  bsa = b1
  return bsa

def skyb(filename):
  df = pd.read_csv(filename,header = None)
  df.drop(df.columns[[0, 1, 2,3,4,5]], axis=1, inplace=True)
  df = df.transpose()
  df.columns = ['B1']
  b1 = (10**(df['B1']/10))
  bsb = b1
  return bsb

bsa = skya('sky2a.csv')
bsb = skyb('sky2b.csv')

bs3 = bsa - bsb

bd = bc - bg3
plt.plot(bga)
plt.plot(bgb)

den = bga - bsa
num = bsa*300 - bga*5
Tr = num/den
Tr
plt.plot(x,Tr)
plt.show()
x = np.linspace(1419.5, 1421.5, 513)

numm = Tr*(ba - bga) + ba*300
denn = bga
Tsource = numm/denn
Tsource
plt.plot(x,Tsource)
plt.show()
kf = pd.DataFrame(Tsource)
kf.to_csv(r'C:\Users\Somani\Downloads\ts.csv')
rf = pd.DataFrame(Tr)
rf.to_csv(r'tr1.csv')
xf = pd.DataFrame(x)
xf.to_csv(r'x1.csv')
#plt.plot(x,bs3)
plt.plot(x,bc)
plt.show()

'''da = pd.read_csv('tr.csv')
print(da)

trr = np.asarray(da['B1'])
print(trr)'''

def readfilea(filename):
  df = pd.read_csv(filename,header = None)
  df.drop(df.columns[[0, 1, 2,3,4,5]], axis=1, inplace=True)
  df = df.transpose()
  df.columns = ['B1']
  b1 = (10**(df['B1']/10))
  ba = b1
  return ba

def readfileb(filename):
  df = pd.read_csv(filename,header = None)
  df.drop(df.columns[[0, 1, 2,3,4,5]], axis=1, inplace=True)
  df = df.transpose()
  df.columns = ['B1']
  b1 = (10**(df['B1']/10))
  bb = b1
  return bb

ba = readfilea('seventy2a.csv')
bb = readfileb('seventy2b.csv')

bc = ba - bb
x = np.linspace(1419.5, 1421.5, 513)

plt.plot(x,ba)
plt.plot(x,bb)
plt.plot(x,bc)

den = bga - bsa
num = bsa*300 - bga*5
Tr = num/den
Tr
plt.plot(x,trr)
plt.show()
x = np.linspace(1419.5, 1421.5, 513)

numm = trr*(ba - bga) + ba*300
denn = bga
Tsource = numm/denn
Tsource
plt.plot(x,Tsource)
plt.show()
kf = pd.DataFrame(Tsource)
kf.to_csv(r'ts70.csv')
rf = pd.DataFrame(Tr)
rf.to_csv(r'tr1.csv')
xf = pd.DataFrame(x)
xf.to_csv(r'x1.csv')
#plt.plot(x,bs3)
plt.plot(x,bc)
plt.show()

from google.colab import drive
drive.mount('/content/drive')

Corrected Velocity

!pip install PyAstronomy

from __future__ import print_function, division
from PyAstronomy import pyasl
import math
import datetime

# for seventy
# Coordinates of telescope
longitude = 73.8253
latitude = 18.5593
altitude = 554

# Coordinates of source (RA2000, DEC2000) RA_hr RA_min RA_sec DEC_deg DEC_min DEC_sec. Note DEC must be signed + or -.
hd1 = "20 07 57.31 +32 20 24"
obs_ra_2000, obs_dec_2000 = pyasl.coordsSexaToDeg(hd1)

# Time of observation converted to Julian Date
dt = datetime.datetime(2022, 12, 17, 17, 35, 54)
jd = pyasl.jdcnv(dt)

# Calculate barycentric correction (debug=True show
# various intermediate results)
corr, hjd = pyasl.helcorr(longitude, latitude, altitude, obs_ra_2000, obs_dec_2000, jd, debug=True)

#print("Barycentric correction [km/s]: ", corr)
#print("Heliocentric Julian day: ", hjd)

# Calculate LSR correction
v_sun = 20.5 # peculiar velocity (km/s) of sun w.r.t. LSR (The Solar Apex. Nature 162, 920 (1948). https://doi.org/10.1038/162920a0)
# solar apex
sun_ra = math.radians(270.2)
sun_dec = math.radians(28.7)

obs_dec = math.radians(obs_dec_2000)
obs_ra = math.radians(obs_ra_2000)

a = math.cos(sun_dec) * math.cos(obs_dec)
b = (math.cos(sun_ra) * math.cos(obs_ra)) + (math.sin(sun_ra) * math.sin(obs_ra))
c = math.sin(sun_dec) * math.sin(obs_dec)
v_rs = v_sun * ((a * b) + c)

v_lsr = corr + v_rs
print("LSR correction [km/s]: ", -v_lsr)
print("Positive value means receding (redshift) source, negative value means approaching (blueshift) source")
