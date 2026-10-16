#!/usr/bin/env python3
"""Export authoritative ITRF-derived ENU geometry for the explicit-frequency FX runner."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src/python'))
from array_geometry import load_itrf_array, ordered_antenna_names, geometry_metadata


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--coordinates',type=Path,required=True)
    parser.add_argument('--mapping',type=Path,required=True)
    parser.add_argument('--frequency-start-hz',type=float,required=True,help='RF center of the first input channel, explicitly supplied')
    parser.add_argument('--channel-width-hz',type=float,required=True,help='signed RF channel increment in input order')
    parser.add_argument('--channels',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.channels<1 or not np.isfinite([args.frequency_start_hz,args.channel_width_hz]).all() or args.channel_width_hz==0:
        raise ValueError('positive channel count, finite first frequency and nonzero increment required')
    freq=args.frequency_start_hz+np.arange(args.channels)*args.channel_width_hz
    if not np.isfinite(freq).all() or (freq<=0).any():
        raise ValueError('all RF channel frequencies must be positive and finite')
    mapping=yaml.safe_load(args.mapping.read_text())
    geometry=load_itrf_array(args.coordinates,ordered_antenna_names(mapping['antenna_to_epic_index']))
    metadata=geometry_metadata(geometry)
    metadata.update(frequency_authority='caller_supplied_RF_centers',mapping_file=str(args.mapping.resolve()))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('xb') as stream:
        np.savez(stream,positions_m=geometry['positions_m'],frequencies_hz=freq,
                 antenna_names=np.asarray(geometry['antenna_names']),geometry_metadata_json=json.dumps(metadata,ensure_ascii=False))
    print(json.dumps(dict(output=str(args.output),**metadata),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
