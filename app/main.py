
import numpy as np
import CSIFilter as cf

cfo = cf.CSIFilter(data_dir="pcaps/CSI.pcap", save_dir="pcaps", mode="raw")
cfo.rssi_filter()
cfo.save_csv(data_fname="csi_data.csv", meta_fname="csi_metadata.csv")
cfo.csi_unwrap()