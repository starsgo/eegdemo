import matplotlib.pyplot as plt
import mne
from dataPrepare import raw,picks
from scipy.io import savemat
import numpy as np

raw.plot(start=1040, n_channels=8, scalings=0.0001, duration=60)


events, _ = mne.events_from_annotations(raw)    #events （n,3） 那一秒，持续多久，发生的事件类型id
event_id = dict({'769': 7, '770': 8, '771': 9, '772': 10})  #evnet_id:78910对应了左右脚舌头运动意图提示

epochs = mne.Epochs(raw, events, event_id, tmin=-1., tmax=4., proj=True, picks=picks, baseline=(-1.0, 0), preload=True)
data = epochs.get_data()

labels = epochs.events[:, -1] - 7

bciciv = {
    'data': data,
    'label': labels
}
savemat('bciciv_2a01T.mat', bciciv)


all_evoked = epochs.average() # (22, 1251)
plt.plot(all_evoked.times, all_evoked.data.T)
plt.show()
