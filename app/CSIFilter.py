# CSIFilter.py
# import necessary libraries
import numpy as np
import pandas as pd
from CSIKit.reader import get_reader
from CSIKit.util.csitools import get_CSI
import os


# class : CSIFilter
# description: Filter CSI data from a pcap file and return it as a csv file
"""
class : CSIFilter
description: Filter CSI data from a pcap file and return it as a csv file

Metadata Fields:
    timestamp       (float) : Time of arrival in seconds (s), with microsecond precision
    rssi            (int)   : Received Signal Strength Indicator in dBm
    frame_control   (int)   : 802.11 Frame Control bitfield
    source_mac      (str)   : Transmitter’s MAC address (hex string)
    core            (int)   : Broadcom chipset core number
    seq_num         (int)   : 802.11 sequence number
    channel_spec    (str)   : Channel + bandwidth specifier (hex bitfield)
    spatial_stream  (int)   : MIMO spatial stream index
    chip            (str)   : Chipset identifier

Metric Fields:
    amplitude       (float) : CSI amplitude (linear scale, arbitrary units)
    phase           (float) : CSI phase in radians
    real            (float) : Real part of the CSI complex response
    imag            (float) : Imaginary part of the CSI complex response
"""
class CSIFilter:
    # constructor
    # data_dir: directory where the pcap or csv file is located
    # save_dir: directory where the csv file will be saved
    # mode: "raw" for raw pcap data, "processed" for csv data
    def __init__(self, data_dir, save_dir, mode="raw"):
        self.data_dir = data_dir
        self.save_dir = save_dir
        self.loaded = None
        self.loaded_metadata = None

        if mode == "processed":
            self.loaded = self._load_processed(data_dir)
        elif mode == "raw":
            self.loaded = self._load_and_process_raw(data_dir)
            self.loaded_metadata = self.load_metadata_pcap(data_dir)

        else:
            raise ValueError("mode must be 'raw' or 'processed'")

    # method to load and process raw pcap data
    def _load_processed(self, csv_dir=None):
        # if csv_dir is not provided, use the self.data_dir
        if csv_dir is None:
            csv_dir = self.data_dir
        # if csv_dir is not a csv file, raise an error
        if not csv_dir.endswith(".csv"):
            raise ValueError("Processed mode requires a CSV file")
        elif csv_dir.endswith(".csv"):
            return pd.read_csv(csv_dir)
        return pd.read_csv(csv_dir)

    # method to load and process raw pcap data
    def _load_and_process_raw(self, pcap_dir=None):
        # if pcap_dir is not provided, use the self.data_dir
        if pcap_dir is None:
            pcap_dir = self.data_dir
        # if pcap_dir is not a pcap file, raise an error
        if not pcap_dir.endswith(".pcap"):
            raise ValueError("Raw mode requires a PCAP file")
        # create a reader for the pcap file
        reader = get_reader(pcap_dir)
        # read the pcap file with scaled=True to apply amplitude/phase scaling
        csi_data = reader.read_file(pcap_dir, scaled=True)

        # extract the CSI matrix with complex metric
        csi_matrix, n_frames, n_subcarr = get_CSI(csi_data, metric="complex")

        # create records for each frame, subcarrier, rx, tx
        records = []
        for f in range(n_frames):
            for sc in range(n_subcarr):
                for rx in range(csi_matrix.shape[2]):
                    for tx in range(csi_matrix.shape[3]):
                        val = csi_matrix[f, sc, rx, tx]
                        records.append(
                            {
                                "frame": f,
                                "subcarrier": sc,
                                "rx": rx,
                                "tx": tx,
                                "amplitude": np.abs(val),
                                "phase": np.angle(val),
                                "real": val.real,
                                "imag": val.imag,
                            }
                        )
        # create a DataFrame from the records and save it as a csv file
        df = pd.DataFrame.from_records(records)
        # return the DataFrame
        return df

    # method to load metadata from a pcap file
    def load_metadata_pcap(self, pcap_path=None):

        if pcap_path is None:
            pcap_path = self.data_dir
        if not pcap_path.lower().endswith(".pcap"):
            raise ValueError("load_metadata_pcap requires a PCAP file")

        reader = get_reader(pcap_path)
        csi_data = reader.read_file(pcap_path, scaled=True)

        # Optional top‐level timestamps list
        timestamps = getattr(csi_data, "timestamps", None)

        records = []
        for i, frame in enumerate(csi_data.frames):
            rec = {"frame": i}
            # timestamp if available
            if timestamps is not None:
                rec["timestamp"] = int(timestamps[i] * 1e6)
            # common CSIFrame attributes (use getattr to avoid AttributeError)
            rec["rssi"] = getattr(frame, "rssi", None)
            rec["frame_control"] = getattr(frame, "frame_control", None)
            rec["source_mac"] = getattr(frame, "source_mac", None)
            rec["core"] = getattr(frame, "core", None)
            # CSIKit might call these seq_num or sequence_no
            rec["seq_num"] = getattr(
                frame, "seq_num", getattr(frame, "sequence_no", None)
            )
            rec["channel_spec"] = getattr(frame, "channel_spec", None)
            rec["spatial_stream"] = getattr(frame, "spatial_stream", None)
            rec["chip"] = getattr(frame, "chip", None)
            records.append(rec)

        df_meta = pd.DataFrame.from_records(records).set_index("frame")
        return df_meta

    # method to save data to a csv file
    def save_csv(
        self, data_fname: str = "csi_data.csv", meta_fname: str = "csi_metadata.csv"
    ):
        os.makedirs(self.save_dir, exist_ok=True)
        data_path = os.path.join(self.save_dir, data_fname)
        meta_path = os.path.join(self.save_dir, meta_fname)
        # save the loaded data to the csv file
        self.loaded.to_csv(data_path, index=True)
        self.loaded_metadata.to_csv(meta_path, index=True)

    
