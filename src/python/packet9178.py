"""Integrity checks for the observed datatype-13, two-channel 9178 packet layout.

Header clock decoding is a labelled candidate, never an automatic sample epoch.
"""
from datetime import datetime
from pathlib import Path
import numpy as np

PACKET_BYTES=262400
HEADER_BYTES=256
SAMPLES_PER_PACKET=65536

def _nonnegative(value,name):
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)) or value<0:
        raise ValueError(f'{name} must be a nonnegative integer')
    return int(value)

def _clock_candidate(h):
    try:
        stamps=[]
        for row in (h[0],h[-1]):
            date=f'{int(row[60]):06x}';time=f'{int(row[61]):06x}';sub=int(row[62])
            if len(date)!=6 or len(time)!=6 or not(date+time).isdigit() or sub>=100000000:
                return None
            dt=datetime.strptime('20'+date+time,'%Y%m%d%H%M%S')
            stamps.append(dt.strftime('%Y-%m-%dT%H:%M:%S')+f'.{sub*10:09d}')
        return {'encoding_hypothesis':'word60 YYMMDD BCD; word61 HHMMSS BCD; word62 10ns ticks',
                'first_local':stamps[0],'last_local':stamps[1],
                'timezone':None,'sample_epoch_confirmed':False}
    except (ValueError,OverflowError):
        return None

def audit_file(path,start_packet=0,packet_count=None):
    path=Path(path);start_packet=_nonnegative(start_packet,'start_packet')
    total,tail=divmod(path.stat().st_size,PACKET_BYTES)
    if tail or total==0:raise ValueError(f'{path}: incomplete or empty packet file')
    if packet_count is None:packet_count=total-start_packet
    packet_count=_nonnegative(packet_count,'packet_count')
    if packet_count==0 or start_packet+packet_count>total:raise ValueError('packet range outside file')
    mm=np.memmap(path,mode='r',dtype=np.uint8,offset=start_packet*PACKET_BYTES,shape=(packet_count,PACKET_BYTES))
    h=mm[:,:HEADER_BYTES].copy().view('<u4').reshape(packet_count,64)
    del mm
    valid=((h[:,0]==0x18efdc01)&(h[:,2]==PACKET_BYTES)&((h[:,3]&0xff)==0x11)
           &((h[:,3]>>16)==32)&((h[:,4]&0xffff)==2)&((h[:,4]>>16)==0)&((h[:,6]&0xff)==13))
    if not valid.all():raise ValueError(f'{path}: invalid packet header at index {start_packet+int(np.flatnonzero(~valid)[0])}')
    counters=h[:,1].astype(np.int64);deltas=(np.diff(counters))%(1<<32)
    if np.any(deltas!=1):
        i=int(np.flatnonzero(deltas!=1)[0]);raise ValueError(f'{path}: non-contiguous packet counter at index {start_packet+i}: {counters[i]} -> {counters[i+1]}')
    return {'path':str(path.resolve()),'total_packets':total,'start_packet':start_packet,'packets':packet_count,
            'first_counter':int(counters[0]),'last_counter':int(counters[-1]),'counter_contiguous':True,
            'samples_per_packet_per_input':SAMPLES_PER_PACKET,'clock_candidate':_clock_candidate(h)}

def audit_group(paths):
    reports=[audit_file(path) for path in paths]
    if not reports:raise ValueError('at least one terminal file is required')
    key=lambda r:(r['first_counter'],r['last_counter'],r['packets'])
    if any(key(r)!=key(reports[0]) for r in reports[1:]):
        raise ValueError('terminal packet counter origins/ranges differ; align samples before channelization')
    return {'files':reports,'packet_counter_ranges_match':True,'sample_clock_synchronization_proven':False,
            'note':'Equal counters do not establish clock/LO phase synchronization or analog cable delay.'}

def chunk_epoch(unix_start,start_packet,start_sequence,sample_rate_hz=250e6):
    start_packet=_nonnegative(start_packet,'start_packet');start_sequence=_nonnegative(start_sequence,'start_sequence')
    if not np.isfinite(unix_start) or unix_start<0 or not np.isfinite(sample_rate_hz) or sample_rate_hz<=0:
        raise ValueError('start epoch must be nonnegative and sample rate positive, both finite')
    if unix_start==0:return 0.0
    return float(unix_start+start_packet*SAMPLES_PER_PACKET/sample_rate_hz+start_sequence*40e-6)
