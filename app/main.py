
import numpy as np
import CSIFilter as cf
import gc

cfo = cf.CSIFilter(data_dir="pcaps/CSI.pcap", save_dir="pcaps", mode="raw")
cfo.rssi_filter()
cfo.save_csv(data_fname="csi_data.csv", meta_fname="csi_metadata.csv")
cfo.calculate_doppler_shift_per_frame(cfo.calculate_doppler_shift())
cfo.calculate_doppler_shift_per_frame_pca(cfo.calculate_doppler_shift())

del cfo
gc.collect()

cfo = cf.CSIFilter(data_dir="pcaps/CSI_nowall_moving2.pcap", save_dir="pcaps", mode="raw")
cfo.rssi_filter()
cfo.save_csv(data_fname="csi_datawalk.csv", meta_fname="csi_metadatawalk.csv")
cfo.calculate_doppler_shift_per_frame(cfo.calculate_doppler_shift(), "cdfpswalk.csv")
cfo.calculate_doppler_shift_per_frame_pca(cfo.calculate_doppler_shift(), "cdspfspwalkpca.csv")


del cfo
gc.collect()

cfo = cf.CSIFilter(data_dir="pcaps/CSI_nowall_static1.pcap", save_dir="pcaps", mode="raw")
cfo.rssi_filter()
cfo.save_csv(data_fname="csi_datastop.csv", meta_fname="csi_metadatastop.csv")
cfo.calculate_doppler_shift_per_frame(cfo.calculate_doppler_shift(), "cdfpsstop.csv")
cfo.calculate_doppler_shift_per_frame_pca(cfo.calculate_doppler_shift(), "cdspfsstop.csv")

