# csipass_dump.py
import numpy as np
import pandas as pd
from CSIKit.reader import get_reader
from CSIKit.util.csitools import get_CSI

# 1) Reader 생성 및 파일 읽기 (scaled=True → amplitude/phase 스케일 적용)
reader   = get_reader("pcaps/CSI.pcap")
csi_data = reader.read_file("pcaps/CSI.pcap", scaled=True)

# 2) get_CSI 로 (frames, subcarriers, rx, tx) 구조의 complex 행렬 추출
#    metric="complex" 로 하면 real+imag 두 값 모두 얻을 수 있고,
#    amplitude, phase 는 .abs(), .angle() 로 추출합니다.
csi_matrix, n_frames, n_subcarr = get_CSI(csi_data, metric="complex")

# 3) CSV용 레코드 생성
records = []
for f in range(n_frames):
    for sc in range(n_subcarr):
        for rx in range(csi_matrix.shape[2]):
            for tx in range(csi_matrix.shape[3]):
                val = csi_matrix[f, sc, rx, tx]
                records.append({
                    "frame":      f,
                    "subcarrier": sc,
                    "rx":         rx,
                    "tx":         tx,
                    "amplitude":  np.abs(val),
                    "phase":      np.angle(val),
                    "real":       val.real,
                    "imag":       val.imag
                })

# 4) DataFrame 생성 및 저장
df = pd.DataFrame.from_records(records)
df.to_csv("csi_full_dump.csv", index=False)
print(f"Saved {len(df)} rows to csi_full_dump.csv")
