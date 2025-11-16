from obspy import read
import os
import pandas as pd


lst = []

path = '/media/ekarkooti/ext4/Kahbasi-PhD/Kaki/raw-gcf-data'
#
path = r'X:\Local Network Data - Raw\Ahar_Varzaghan\All_Data'
path_csv = r'X:\Local Network Data - Raw\Ahar_Varzaghan\AharVarzaghan_station-sensor-region-coords.csv'
#
path = r'F:\Local_Network_Data-Raw\Qeshm\qeshm1_gcf'
path_csv = r'F:\Local_Network_Data-Raw\Qeshm\station-sensor-coords.csv'
#


df = pd.read_csv(path_csv)
#df = df[['sensor', 'region', 'station']]
#
for root, dirs, files in os.walk(path):
    for fname in files:
        if fname.endswith('.gcf'):
            try:
                st = read(f'{root}/{fname}')
                for tr in st:
                    gcf = tr.stats.gcf
                    el = (gcf['system_id'], gcf['stream_id'])
                    lst.append(el)
                break
            except Exception as error:
                print(root, fname, error)

with open('output.ini', 'w') as output:
    print('[EXPORTINFO]', file=output)
    ii = 0
    for el in lst:
        system_id, stream_id = el
        sta = ''
        chan = 'HH' + stream_id[-2].upper()
        net = 'QM'
        loc = ''
        print(f'{system_id}-{stream_id}=sta:{sta} chan:{chan} net:{net} loc:{loc}', file=output)
        ii += 1
        if ii == 3:
            print('', file=output)
            ii = 0
