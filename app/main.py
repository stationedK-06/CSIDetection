
import numpy as np
import CSIFilter as cf
import gc

cfo = cf.CSIFilter(data_dir="pcaps/CSI.pcap", save_dir="pcaps", mode="raw")
cfo.rssi_filter()
cfo.save_csv(data_fname="csi_data.csv", meta_fname="csi_metadata.csv")
cfo.calculate_doppler_shift_per_frame(cfo.calculate_doppler_shift_savitzky_golay_filter())
cfo.calculate_doppler_shift_per_frame_pca(cfo.calculate_doppler_shift_savitzky_golay_filter())

del cfo
gc.collect()

cfo = cf.CSIFilter(data_dir="pcaps/CSI_nowall_moving2.pcap", save_dir="pcaps", mode="raw")
cfo.rssi_filter()
cfo.save_csv(data_fname="csi_datawalk.csv", meta_fname="csi_metadatawalk.csv")
cfo.calculate_doppler_shift_per_frame(cfo.calculate_doppler_shift_savitzky_golay_filter(), "cdfpswalk.csv")
cfo.calculate_doppler_shift_per_frame_pca(cfo.calculate_doppler_shift_savitzky_golay_filter(), "cdspfspwalkpca.csv")


del cfo
gc.collect()

cfo = cf.CSIFilter(data_dir="pcaps/CSI_nowall_static1.pcap", save_dir="pcaps", mode="raw")
cfo.rssi_filter()
cfo.save_csv(data_fname="csi_datastop.csv", meta_fname="csi_metadatastop.csv")
cfo.calculate_doppler_shift_per_frame(cfo.calculate_doppler_shift_savitzky_golay_filter(), "cdfpsstop.csv")
cfo.calculate_doppler_shift_per_frame_pca(cfo.calculate_doppler_shift_savitzky_golay_filter(), "cdspfsstoppca.csv")

del cfo
gc.collect()

cfo = cf.CSIFilter(data_dir="pcaps/CSI_nowall_static2.pcap", save_dir="pcaps", mode="raw")
cfo.rssi_filter()
cfo.save_csv(data_fname="csi_datastop2.csv", meta_fname="csi_metadatastop2.csv")
cfo.calculate_doppler_shift_per_frame(cfo.calculate_doppler_shift_savitzky_golay_filter(), "cdfpsstop2.csv")
cfo.calculate_doppler_shift_per_frame_pca(cfo.calculate_doppler_shift_savitzky_golay_filter(), "cdspfsstop2pca.csv")


del cfo
gc.collect()

cfo = cf.CSIFilter(data_dir="pcaps/CSI_nowall_moving1.pcap", save_dir="pcaps", mode="raw")
cfo.rssi_filter()
cfo.save_csv(data_fname="csi_datastop1.csv", meta_fname="csi_metadatastop1.csv")
cfo.calculate_doppler_shift_per_frame(cfo.calculate_doppler_shift_savitzky_golay_filter(), "cdfpsmoving1.csv")
cfo.calculate_doppler_shift_per_frame_pca(cfo.calculate_doppler_shift_savitzky_golay_filter(), "cdspfsmoving1pca.csv")



