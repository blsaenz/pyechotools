# blsaenz

import math,sys,os,csv
import numpy as np
import datetime as dt
import matplotlib.pylab as plt
import matplotlib as mpl
import pandas as pd
import xarray as xr
from numba import jit
# mpl.use('Qt5Agg')  # interactive mode works with this, pick one
mpl.use('TkAgg')  # interactive mode works with this, pick one
from scipy.interpolate import griddata
from scipy.optimize import minimize
import scipy.io as spio


# Below are some routines for pulling data from gilder matlab data,
# but they are not generalized.
# =================================================================


def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], spio.matlab.mat_struct):
            dict[key] = _todict(dict[key])
    return dict

def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict

def dive_matlab_to_df(do_613=False,data_path=r"X:\data\ross_sea_gliders\Glider_Data_2022-23__"):

    hyd_keys = ['pressure', 'depth', 'time', 'dive_num', 'direction', 'profile_num', 'temp', 'cons_temp', 'salinity', 'abs_salinity', 'sigma0', 'salinity_nocorr', 'conductivity', 'rho', 'w_H2O', 'lat', 'lon', 'oxygen', 'PAR', 'Scatter_470', 'Scatter_700', 'Chlorophyll']

    if do_613:
        matdata = loadmat(os.path.join(data_path,"sg613_c.mat"))
        root_key = 'data_613'
    else:
        matdata = loadmat(os.path.join(data_path,"sg676.mat"))
        root_key = 'data_613'

    hyd = matdata[root_key]['hydrography']
    hyd_ravel = {}
    for i in range(len(hyd)):
        h = _todict(hyd[i])
        for field in hyd_keys:
            if i==0:
                hyd_ravel[field] = h[field]
            else:
                hyd_ravel[field] = np.hstack([hyd_ravel[field],h[field]])

    eng = matdata[root_key]['eng']
    for i in range(len(eng)):
        e = _todict(eng[i])
        if i==0:
            glide_angle = e['pitchAng']
            roll = e['rollAng']
            #heading = e['head']
        else:
            glide_angle = np.hstack([glide_angle,e['pitchAng']])
            roll = np.hstack([roll, e['rollAng']])
            #heading = np.hstack([glide_angle, e['head']])
    glide_angle[glide_angle==0] = np.nan
    roll[roll==0] = np.nan
    #heading[heading==0] = np.nan

    hyd_ravel['pitch'] = glide_angle
    hyd_ravel['roll'] = roll
    #hyd_ravel['heading'] = heading

    # convert matlab datenum to pandas timestamp
    hyd_ravel['timestamp'] = pd.to_datetime(hyd_ravel['time'] - 719529, unit='D')
    out_fname = "sg613_c_hyd_20230717.csv" if do_613 else "sg676_hyd_20230426.csv"
    pd.DataFrame(hyd_ravel).to_csv(os.path.join(data_path,out_fname))


def dive_matlab_raw_to_df(do_613=False,data_path=r"X:\data\ross_sea_gliders\Glider_Data_non-acoustic"):

    raw_keys = ['lat','lon','time', 'dive','cond','temp','poc','sal','ox','osat','chlorophyll','scatter700','scatter470','ps','insitu_t']

    if do_613:
        matdata = loadmat(os.path.join(data_path,"raw_data_sg613_c.mat"))
        root_key = 'sg613_raw'
    else:
        matdata = loadmat(os.path.join(data_path,"raw_data_sg676.mat"))
        root_key = 'sg676_raw'

    hyd = matdata[root_key]
    hyd_ravel = {}
    #for i in range(len(hyd)):
    #    h = _todict(hyd[i])
    for field in raw_keys:
        hyd_ravel[field] = hyd[field]

    hyd_ravel['timestamp'] = pd.to_datetime(hyd_ravel['time'] - 719529, unit='D')
    out_fname = "sg613_raw_20240320.csv" if do_613 else "sg676_raw_20240320.csv"
    pd.DataFrame(hyd_ravel).to_csv(os.path.join(data_path,out_fname))


# Helper routines
# =================================================================


def unpack2byte(twobytes):
    return (twobytes[0] & 0x7F)<<7 | twobytes[1] & 0x7F

def unpack_dive(threebytes):
    dive_type = (threebytes[0] & 0x40)>>6
    dive_num = (threebytes[0] & 0x03)<<14 | (threebytes[1] & 0x7F)<<7 | threebytes[2] & 0x7F
    return dive_type,dive_num

def unpack_ping_num(fourbytes):
    return (fourbytes[0] & 0x7F)<<21 | (fourbytes[1] & 0x7F)<<14 | (fourbytes[2] & 0x7F)<<7 | fourbytes[3] & 0x7F


def SSMackenzie1981(T, S, D):
    ''' Returns the speed of sound (m/s) according to Mackenzie 1981. T =
    temperature in degC; S = salinity in PPT; D = depth in m,
    where -2 < T < 30; 25 < S < 40; 0 < D < 8000
    '''

    if np.any(T < -2)or np.any(T > 30.):
        raise ValueError('Temperature not in range (-2:30) for Mackenzie Sound Speed Calc')
    if np.any(S < 25) or np.any(S > 40):
        raise ValueError('Salinity not in range (25:40) for Mackenzie Sound Speed Calc')
    if np.any(D < 0) or np.any(D > 8000):
        raise ValueError('Depth not in range  (0:8000) for Mackenzie Sound Speed Calc')

    return 1448.96 + 4.591 * T - 0.05304 * T ** 2 + 2.374e-4 * T ** 3 + 1.34 * (S - 35) + \
           0.0163 * D + 1.675e-7 * D ** 2 - 0.01025 * T * (S - 35) - 7.139e-13 * T * D ** 3

#@jit(nopython=True)
def ABSORPFrancisGarrison1982(T, S, D, C, f, pH=8.0):
    '''

    :param T: temperature (C)
    :param S: salinity (PSU)
    :param D: depth (m)
    :param C:
    :param f: sound frequency (Hz)
    :param pH: pH (pH scale)
    :return: sounds absorption in dB km-1
    '''
    f_kHz = f * 0.001

    if f < 200 or f > 1e6:
        return None
        #raise ValueError('Frequency not in range (200:1e6 Hz) for FrancisGarrison Absorption Calc:(f = %f)' % f)

    if np.any(T <= 20):
        A3 = 4.937e-4 - 2.59e-5 * T + 9.11e-7 * T ** 2 - 1.5e-8 * T ** 3
    else:
        A3 = 3.964e-4 - 1.146e-5 * T + 1.45e-7 * T ** 2 - 6.5e-10 * T ** 3
    P3 = 1.0 - 3.83e-5 * D + 4.9e-10 * D ** 2
    if np.any(S > 0):
        A1 = 8.86 * 10 ** (0.78 * pH - 5) / C
        A2 = 21.44 * S * (1.0 + 0.025 * T) / C
        f1 = 2.8 * np.sqrt(S / 35) * 10 ** (4 - (1245 / (T + 273)))
        f2 = 8.17 * 10 ** (8 - (1990 / (T + 273))) / (1.0 + 0.0018 * (S - 35))
        P2 = 1.0 - 1.37e-4 * D + 6.2e-9 * D ** 2
        return f_kHz ** 2 * (A1 * f1 / (f1 ** 2 + f_kHz ** 2) + A2 * P2 * f2 / (f2 ** 2 + f_kHz ** 2) + A3 * P3)
    else:
        return f_kHz ** 2 * A3 * P3



#@jit(nopython=True)
def db_from_volts(volts,range,alpha,gain,cal_coef):
    Sv = 20.0*math.log10(volts) + 20.0*math.log10(range) + 2.*alpha*range - gain + cal_coef
    return Sv


@jit(nopython=True)
def Sv_from_volts_old(volts,r,alpha,gain,cal_coef):
    ni,nj = np.shape(volts)
    dB = np.full((ni,nj),np.nan)
    for i in range(ni):
        for j in range(nj):
            dB[i,j] = 20.0*math.log10(volts[i,j]) + 20.0*math.log10(r[i,j]) + 2.*alpha[i,j]*r[i,j] - gain + cal_coef
    return dB

def Sv_from_volts(volts,r,alpha,gain,cal_coef):
    return 20.0*np.log10(volts) + 20.0*np.log10(r) + 2.*alpha*r - gain + cal_coef

@jit(nopython=True)
def haversine(lat1,lon1,lat2,lon2):
    '''Return distance in km between two points on earth surface'''
    R = 6372.8 # km radius of earth
    dLat = np.deg2rad(lat2 - lat1)
    dLon = np.deg2rad(lon2 - lon1)
    lat1 = np.deg2rad(lat1)
    lat2 = np.deg2rad(lat2)
    a = np.sin(dLat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dLon/2)**2
    c = 2*np.arcsin(np.sqrt(a))
    return R * c

@jit(nopython=True)
def dist_process(lat,lon):
    dist = np.full(len(lat),np.nan)
    dist[0] = 0.0
    for i in range(1,len(lat)):
        dist[i] = haversine(lat[i-1],lon[i-1],lat[i],lon[i])
    return dist

def _sv_to_linear(sv):
    return 10 ** (sv / 10.0)

def sv_to_linear(sv_arr):
    '''sv to linear where zeros are stored as -999.0'''
    sv_arr[sv_arr<=-999.0] = np.nan
    svlin = _sv_to_linear(sv_arr)
    svlin[np.isnan(svlin)] = 0.0
    return svlin

def linear_to_sv(linear):
    sv = 10.0 * np.log10(linear)
    sv[~np.isfinite(sv)] = -999.0
    return sv

def linear_to_sv1(linear):
    sv = 10.0 * np.log10(linear)
    if np.isfinite(sv):
        return sv
    else:
        return -999.0
    return sv

def average_sv(sv_arr):
    sv_arr[sv_arr<=-999.0] = np.nan
    svlin = 10 ** (sv_arr / 10.0)
    svlin[np.isnan(svlin)] = 0.0
    svlin_mean = np.nanmean(svlin)
    sv_mean = 10.0 * np.log10(svlin_mean)
    if np.isfinite(sv_mean):
        return sv_mean
    else:
        return -999.0

def count_sv_gr_zero(sv_arr):
    sv_arr[sv_arr<=-999.0] = np.nan
    return np.sum(np.isfinite(sv_arr))

def weighted_average_w_nan(data,weights):
    if np.sum(weights) > 0:
        return np.average(data,weights=weights)
    else:
        return np.nan

def weighted_average_w_nan_sv(data,weights):
    if np.sum(weights) > 0:
        data_linear = sv_to_linear(data)
        mean = np.average(data_linear,weights=weights)
        return linear_to_sv1(mean)
    else:
        return np.nan


def calc_add_depth_correct(df,mount_angle,name_space=' '):
    df['Depth_corr'] = (df[name_space+'Depth_mean'] - df['depth']) * np.cos(np.pi * (df['pitch'] + mount_angle) / 180.)
    df['Depth_corr'] *= np.cos(np.pi * df['roll'] / 180.)
    df['Depth_corr'] += df['depth']
    return df



# Main class that can process a .853 file
# =================================================================

class Eight53(object):
    return_header_len = 54
    return_header = [
        ('IKX', (np.uint8, 3)),
        ('ID', (np.uint8, 1)),
        ('Status', (np.uint8, 1)),
        ('PingNo', (np.uint8, 4)),
        ('range', (np.uint8, 1)),  # 0=25 m; 1=50m; 2=100m
        ('gain', (np.uint8, 1)),   # 1=20db; 2=40dB
        ('freq', (np.uint8, 1)),  # 0=120 kHz; 1=300 kHz
        ('pulse_length', (np.uint8, 1)),  # 0=100 microsec
        ('mode', (np.uint8, 1)),  # 0=normal; 1=Glider; 2=stand alone
        ('PingRate', (np.uint8, 2)),  # 1=20db; 2=40dB
        ('Blank1', (np.uint8, 3)),
        ('DiveNo', (np.uint8, 3)),
        ('Day', (np.uint8, 1)),
        ('Mo', (np.uint8, 1)),
        ('Year', (np.uint8, 2)),
        ('Hour', (np.uint8, 1)),
        ('Min', (np.uint8, 1)),
        ('Sec', (np.uint8, 1)),
        ('MS', (np.uint8, 2)),
        ('Blank2', (np.uint8, 24)),
    ]
    return_echo_data_lon = 213
    return_echo = [
        ('bins', (np.uint8, 200)),
        ('End', (np.uint8, 1)),
    ]

    hdr_fields = ['dtt','ping_num','range','gain','freq','pulse_length','mode','ping_rate','dive_type','dive_num']
    byte_hdr_fields = ['range','gain','freq','pulse_length','mode',]

    _depth_corrected = None
    _range = None
    Sv = None

    def __init__(self,filepath,mount_angle=0):
        # mount angle is positive tilting toward the bow/front of the glider, where zero if pointing exactly downward.
        # this is the opposite of how the mount ange is reported in Guihen et al. 2014
        # glide_slope in the metadata is expected to use the same convention
        self.mount_angle = mount_angle
        self.filepath = filepath
        self.read_raw_853(filepath)

    def append_header(self,hdrs,hdr_bytes):
        hdrs['ping_num'].append(unpack_ping_num(hdr_bytes['PingNo'][0]))
        print('Ping Num: ',hdrs['ping_num'][-1])
        hdrs['dtt'].append(dt.datetime(unpack2byte(hdr_bytes['Year'][0]),hdr_bytes['Mo'][0],hdr_bytes['Day'][0],
                                 hdr_bytes['Hour'][0],hdr_bytes['Min'][0],hdr_bytes['Sec'][0],unpack2byte(hdr_bytes['MS'][0])*1000))
        hdrs['ping_rate'].append(unpack2byte(hdr_bytes['PingRate'][0]))
        d_t,d_n = unpack_dive(hdr_bytes['DiveNo'][0])
        hdrs['dive_type'].append(d_t)
        hdrs['dive_num'].append(d_n)
        for f in self.byte_hdr_fields:
            hdrs[f].append(hdr_bytes[f][0])
       #return hdrs

    def eof(self,hdr):
        test = hdr['IKX'].tobytes()
        if test == '':
            return True
        return False

    def read_raw_853(self,filepath):

        hdrs = {f:[] for f in self.hdr_fields}
        pings = []
        with open(filepath) as fp:

            for i in range(30000):
                hdr_bytes = np.fromfile(fp,self.return_header,1)
                if len(hdr_bytes['PingNo']) == 0:
                    break
                self.append_header(hdrs,hdr_bytes)
                ping_bytes = np.fromfile(fp,self.return_echo,1)
                pings.append(10.0**(-ping_bytes['bins'][0].astype(np.float32)/20.0))

        for f in self.hdr_fields:
            setattr(self,f,np.array(hdrs[f]))
        self.pings = np.transpose(np.array(pings))
        self.npings = len(self.dtt)


    def get_meta_from_nc(self,nc_file):

        # find closest index in netcdf dimension
        ds = xr.open_dataset(nc_file)
        date64 = self.dtt.astype(np.datetime64)
        dtime = ds.time.values

        # perhaps need to find only single closest value, then interpolate? Or just plain interpolate?
        self.metai = np.array([np.argmin(np.abs(dtime-d1)) for d1 in date64])

        self.depth = ds['depth'].values[self.metai]
        self.lat = ds['latitude'].values[self.metai]
        self.lon = ds['longitude'].values[self.metai]

    def get_meta_from_df(self,df):

        df_dive = pd.DataFrame({'timestamp':pd.to_datetime(self.dtt),'dummy':0}).set_index('timestamp')
        df_dive = pd.merge_asof(df_dive,df,left_index=True,right_index=True,direction='nearest')

        self.dive_meta = df_dive

        self.pitch = df_dive['pitch'].values
        self.roll = df_dive['roll'].values
        #self.heading = df_dive['heading'].values
        self.depth = df_dive['depth'].values
        self.lat = df_dive['lat'].values
        self.lon = df_dive['lon'].values


    def calc_Sv(self,T=None,S=None,freq=120000.,depth=50.):

        # no single value, attempt to use dive data
        if T is None or S is None:
            depth = self.depth_corrected()
            (ndepths,_) = np.shape(depth)
            S = np.tile(self.dive_meta['salinity'], (ndepths,1))
            T = np.tile(self.dive_meta['temp'], (ndepths,1))

        C = SSMackenzie1981(T, S, depth) # sound speed
        alpha = ABSORPFrancisGarrison1982(T, S, depth, C, freq)/1000. # absorption
        self.Sv = Sv_from_volts(self.pings,self.trange(),alpha,40.0,0)

    def get_depth_raster(self):
        pass

    def bin_range(self,future_range_identifier=None):
        return np.arange(0.25, 100, 0.5)

    def trange(self):
        if self._range is None:
            self._range = np.tile(np.asarray([self.bin_range()]).transpose(), (1, self.npings))
        return self._range

    def depth_corrected(self):
        if self._depth_corrected is None:
            self._depth_corrected = np.copy(self.trange())
            self.swath_depth = np.full(len(self.depth),np.nan)
            self.pitch_roll_factor = np.full(len(self.depth),np.nan)
            self.pitch_w_mount_angle = np.full(len(self.depth),np.nan)
            for i,d_glider in enumerate(self.depth):
                self._depth_corrected[:,i] *= np.cos(np.pi*(self.pitch[i]+self.mount_angle)/180.)
                self._depth_corrected[:,i] *= np.cos(np.pi*self.roll[i]/180.)
                self._depth_corrected[:,i] += d_glider
                self.swath_depth[i] = np.max(self._depth_corrected[:,i]) - np.min(self._depth_corrected[:,i])
                self.pitch_roll_factor = np.pi*(self.pitch[i]+self.mount_angle)/180. * np.pi*self.roll[i]/180.
                self.pitch_w_mount_angle[i] = self.pitch[i]+self.mount_angle

        return self._depth_corrected

    def plot_Sv_depth_corrected(self,savefigname=None,show=True):
        self.depth_corr = np.copy(self.trange())
        for i,d_glider in enumerate(self.depth):
            self.depth_corr[:,i] += d_glider
        depth_corr = self.depth_corrected()
        dmin = np.min(depth_corr)
        dmax = np.max(depth_corr)
        dspace = np.linspace(dmin,dmax,int((dmax-dmin)*4))
        pspace = np.arange(self.npings)
        dgrid,pgrid = np.meshgrid(dspace,pspace)
        Sv_map = griddata((depth_corr.ravel(), np.tile(np.asarray([np.arange(self.npings)]),(200,1)).ravel()),
                          self.Sv.ravel(),
                          (dgrid, pgrid),
                          method='linear')
        for i in range(self.npings):
            Sv_map[i,dgrid[i,:] > self.depth[i]+100] = np.nan
            Sv_map[i,dgrid[i,:] < self.depth[i]] = np.nan
        #im = plt.pcolor(np.flipud(Sv_map.transpose()), vmin=-80, vmax=-35)

        fig,ax = plt.subplots(1,1,figsize=(12,6))

        im = ax.pcolor(pgrid,-1*dgrid,Sv_map, vmin=-80, vmax=-35)
        plt.colorbar(im)
        ax.set_xlabel('Ping Number')
        ax.set_ylabel('Depth')
        _,fname = os.path.split(self.filepath)
        ax.set_title(fname)
        if savefigname is not None:
            plt.savefig(savefigname,dpi=400,bbox_inches='tight')
        if show:
            plt.show()

    def to_echoview_csv(self,ev_filename):

        ev_headers = ['Ping_index','Distance_GPS','Distance_vl','Ping_date','Ping_time','Ping_milliseconds',
                      'Latitude','Longitude','Depth_start','Depth_stop','Range_start','Range_stop','Sample_count']

        depths = self.depth_corrected()
        rbins = self.bin_range()
        rmin = str(rbins.min())
        rmax = str(rbins.max())
        nbins = str(len(rbins))

        with open(ev_filename,'w') as ev:
            ev.write(','.join(ev_headers)+'\n')
            for i in range(self.npings):
                if i==0:
                    d = 0.0
                else:
                    d = d + haversine(self.lat[i-1],self.lon[i-1],self.lat[i],self.lon[i])
                pingline = ','.join([str(i),str(d),str(d),self.dtt[i].strftime('%Y-%m-%d'),
                                     self.dtt[i].strftime('%H:%M:%S'),str(int(self.dtt[i].microsecond/1000)),
                                     str(self.lat[i]),str(self.lon[i]),str(depths[:,i].min()),
                                    str(depths[:,i].max()),rmin,rmax,nbins])
                pingline += ',' + ",".join(np.char.mod('%f', self.Sv[:,i])) + '\n'
                ev.write(pingline)


# Example processing of multiple .853 files, assuming that all needed
# extra data is in a pandas-readable CSV file, with labels.
# =================================================================


def process_dives_2022_23(mount_angle=26.0,raw_file_pattern='A.853',outpath=None):

    # get glider data
    #dive_df = pd.read_csv(r"X:\data\ross_sea_gliders\Glider_Data_2022-23\sg676_hyd.csv",parse_dates=['timestamp'])
    dive_df = pd.read_csv(r"X:\data\ross_sea_gliders\Glider_Data_2022-23__\sg676_hyd_newll.csv",parse_dates=['timestamp'])
    default_outpath = r"X:\data\ross_sea_gliders\Glider_Data_2022-23__\ES853"

    op = default_outpath if outpath is None else outpath

    # crude fixing of bad CT values
    dive_df['salinity'][dive_df['salinity'] <= 25] = np.nan
    dive_df['temp'][dive_df['temp'] <= -2.0] = np.nan

    dive_df = dive_df.set_index('timestamp').interpolate('linear')


    for i in range(1,345):  # 344 dived
        for a_b in ['A','B']:
            if not (i == 238 and a_b == 'A') and not (i == 245 and a_b == 'A') and not (i == 293 and a_b == 'A'):
                fname = "ES%05i"%i + a_b + ".853"
                if raw_file_pattern in fname:
                    filepath = os.path.join(default_outpath,fname)
                    p = Eight53(filepath,mount_angle=mount_angle)
                    p.get_meta_from_df(dive_df)
                    p.calc_Sv()
                    fig = p.plot_Sv_depth_corrected(savefigname=os.path.join(op,fname[:-4]+'_Sv_raw.png'),show=False)
                    plt.close()
                    p.to_echoview_csv(os.path.join(op,fname[:-4] + '.sv.csv'))

# Some examples of creating echoview-compatible data files (in case
# you want to filter by excessive ptich/roll, etc.)
# =================================================================

def create_heave_pitch_roll():
    dive_df = pd.read_csv(r"X:\data\ross_sea_gliders\Glider_Data_2022-23__\sg676_hyd_newll.csv",parse_dates=['timestamp'])
    #dive_df = dive_df.set_index('timestamp').interpolate('linear')
    #dtt = dive_df.index.to_pydatetime().tolist()
    n_meta = len(dive_df)

    # with open(r"X:\data\ross_sea_gliders\Glider_Data_2022-23\sg676_depth.EVL",'w') as f:
    #     f.write("EVBD 3 6.1.73.30395\n")
    #     f.write("%i\n"%n_meta)
    #     for index, row in dive_df.iterrows():
    #         dt1 = row['timestamp'].to_pydatetime()
    #         f.write("%s %s%04i  %s 3 \n"%(dt1.strftime("%Y%m%d"),
    #                                       dt1.strftime("%H%M%S"),
    #                                       int(dt1.microsecond/100),
    #                                       str(row['depth'])[0:9]))

    # with open(r"X:\data\ross_sea_gliders\Glider_Data_2022-23\sg676_corrected.pitch.csv",'w') as f:
    #     f.write("Pitch_date,Pitch_time,Pitch_milliseconds,Pitch_angle\n")
    #     for index, row in dive_df.iterrows():
    #         dt1 = row['timestamp'].to_pydatetime()
    #         f.write(",".join((dt1.strftime("%Y-%m-%d"),
    #                                       dt1.strftime("%H:%M:%S"),
    #                                       str(int(dt1.microsecond/1000)),
    #                                       str(row['pitch']+22.5))) + "\n")

    # with open(r"X:\data\ross_sea_gliders\Glider_Data_2022-23\sg676.roll.csv",'w') as f:
    #     f.write("Roll_date,Roll_time,Roll_milliseconds,Roll_angle\n")
    #     for index, row in dive_df.iterrows():
    #         dt1 = row['timestamp'].to_pydatetime()
    #         f.write(",".join((dt1.strftime("%Y-%m-%d"),
    #                                       dt1.strftime("%H:%M:%S"),
    #                                       str(int(dt1.microsecond/1000)),
    #                                       str(row['roll']))) + "\n" )


    with open(r"X:\data\ross_sea_gliders\Glider_Data_2022-23__\sg676_interp_time.gps.csv",'w') as f:
        f.write("GPS_date,GPS_time,GPS_milliseconds,Latitude,Longitude\n")
        for index, row in dive_df.iterrows():
            dt1 = row['timestamp'].to_pydatetime()
            f.write(",".join((dt1.strftime("%Y-%m-%d"),
                                          dt1.strftime("%H:%M:%S"),
                                          str(int(dt1.microsecond/1000)),
                                          str(row['lat']),
                                          str(row['lon']))) + "\n" )

