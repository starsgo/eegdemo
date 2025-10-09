import mne
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import savemat

mne.viz.set_browser_backend('matplotlib')   # 关键一步

datas = []  #(9*288, 1, 22, 1251)
labels = [] #(9*288, 1)
# savefilename = f"bciciv_all5_exclude8.mat"
for i in [0]:
# for i in [7]:
    # 原始数据读取
    filename = f"../BCICIV_2a_gdf/A0{i+1}T.gdf"
    print(filename)
    savefilename = f"bciciv_2a_{i+1}T.mat"
    raw = mne.io.read_raw_gdf(filename)
    # Pre-load the data
    raw.load_data()
    fs = raw.info['sfreq']  # 250Hz
    mapping = {
        'EEG-Fz': 'eeg',
        'EEG-0': 'eeg',
        'EEG-1': 'eeg',
        'EEG-2': 'eeg',
        'EEG-3': 'eeg',
        'EEG-4': 'eeg',
        'EEG-5': 'eeg',
        'EEG-C3': 'eeg',
        'EEG-6': 'eeg',
        'EEG-Cz': 'eeg',
        'EEG-7': 'eeg',
        'EEG-C4': 'eeg',
        'EEG-8': 'eeg',
        'EEG-9': 'eeg',
        'EEG-10': 'eeg',
        'EEG-11': 'eeg',
        'EEG-12': 'eeg',
        'EEG-13': 'eeg',
        'EEG-14': 'eeg',
        'EEG-Pz': 'eeg',
        'EEG-15': 'eeg',
        'EEG-16': 'eeg',
        'EOG-left': 'eog',
        'EOG-central': 'eog',
        'EOG-right': 'eog'
    }
    raw.set_channel_types(mapping)
    electrode_names = [
        'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz',
        'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz',
        'P2', 'POz'
    ]
    raw.info['bads'] += ['EOG-left','EOG-central','EOG-right']
    raw.rename_channels({raw.ch_names[i]: electrode_names[i] for i in range(22)})
    raw.set_montage('standard_1005')
    picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=False, exclude='bads')
    # raw.plot(start=1040, n_channels=8, scalings=0.0001, duration=60)
    # raw.plot_psd(average=True)

    # 陷波滤波
    raw = raw.notch_filter(freqs=(50))      # 中国大陆陷波60Hz
    raw.filter(1, 38, picks=picks, fir_design='firwin')
    # raw.plot(start=1040, n_channels=8, scalings=0.0001, duration=60)
    # raw.plot_psd(average=True)

    #主成分分析
    # ica = mne.preprocessing.ICA(n_components=15, random_state=97, max_iter=800)
    # ica.fit(raw)
    # ica.plot_sources(raw, show_scrollbars=False, start=1040, stop = 1100)

    # 主成分1对应眨眼噪声，去除
    # ica.exclude = [1]
    # ica.apply(raw)
    # raw.plot(start=1040, n_channels=8, scalings=0.0001, duration=60)

    events, _ = mne.events_from_annotations(raw)    #events （n,3） 那一秒，持续多久，发生的事件类型id
    event_id = dict({'769': 7, '770': 8, '771': 9, '772': 10})  #evnet_id:78910对应了左右脚舌头运动意图提示
    ids = np.unique(events[:,2])
    event_ids = {k:v for k,v in event_id.items() if v in ids}
    print(events.shape)

    # epochs = mne.Epochs(raw, events, event_id, tmin=-1., tmax=4., proj=True, picks=picks, baseline=(-1.0, 0), preload=True)
    epochs = mne.Epochs(raw, events, event_ids, tmin=0, tmax=5, proj=True, picks=picks, baseline=None, preload=True)
    data = epochs.get_data()
    data = data[:,:,:-1]
    label = epochs.events[:, -1] - 7
    print(data.shape)
    print(label.shape)

    all_evoked = epochs.average()  # (22, 1251)
    plt.plot(all_evoked.times, all_evoked.data.T)
    plt.show()

    datas.extend(data)
    labels.extend(label)

bciciv = {
    'data': datas,
    'label': labels
}
savemat(savefilename, bciciv)


