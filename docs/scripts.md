# Scripts

[初始化](#設置初始環境)

[讀取資料](#建立資料庫與訓練資料)

[產生訓練資料](#02_generate_datasetpy)

[檢視做出來的檔案](#03_plot_instancepy)

[開始訓練模型](#訓練模型)

[預測TFRecord](#預測與評估結果)

[評估結果](#09_model_evaluationpy)

[畫出結果](#10_plot_predict_instancepy)

[預測連續資料](#預測連續資料)

[評估連續資料](#評估連續資料結果)

[定位](#定位)

## 設置初始環境

### 00_initialize.py

初始化 將所需的資料夾與檔案創好

```python
import seisblue

config = seisblue.utils.Config(initialize=True)
```

## 建立資料庫與訓練資料

### 01_load_sql_database.py

開啟資料庫,若是第一次建立資料庫, build = True

```python 
db = seisblue.sql.Client(database="HL2019.db",build=True)
inspector = seisblue.sql.DatabaseInspector(db)
```

把測站參數讀進database,並且給予network名稱

此測站檔案應存放在家目錄底下的Geom資料夾中(此資料夾會在00_initialize)

```python
db.read_hyp(hyp="STATION0.2019HL.HYP", network='HP')
inspector.inventory_summery()
```

sfile應存於家目錄底下的Catalog中，之後便可透過以下code將事件與pick讀進database中， 並且標註為 'manual'

```python
events = seisblue.io.read_event_list('HL2019')
db.add_sfile_events(events=events, tag="manual")
```

顯示目前 database 的基本資料

```python
inspector.event_summery()
inspector.pick_summery()
```

劃出事件分布，範圍與格式可微調

```python
inspector.plot_map()
```

### 02_generate_dataset.py

```python
import seisblue
```

給予要使用的 database 與對應的 pick tag

```python
database = 'HL2019.db'
tag = 'manual'
```

讀取 database

```python
db = seisblue.sql.Client(database=database)
inspector = seisblue.sql.DatabaseInspector(db)
```

從 database 取出要使用的pick

```python
pick_list = db.get_picks(tag=tag)
```

將 pick 做成對應的 TFRecord

連續資料的存放位置要在 config.yml 中更改 SDS_ROOT 的路徑

TFRecord 存放的位置可以在 config.yml 中更改

```python
tfr_converter = seisblue.components.TFRecordConverter(trace_length=33)
tfr_converter.convert_training_from_picks(pick_list, tag, database)
```

將做好的 TFRecord 的絕對路徑及相關資訊寫進 database 中

```python
config = seisblue.utils.Config()
tfr_list = seisblue.utils.get_dir_list(config.train, suffix='.tfrecord')
db.clear_table(table='waveform')  # 清空欄位
db.clear_table(table='tfrecord')  # 清空欄位
db.read_tfrecord_header(tfr_list)
```

寫出 waveform 相關資訊

```python
inspector.waveform_summery()
```

### 03_plot_instance.py

```python
import seisblue
```

讀取 database

```python
database = 'HL2019.db'
db = seisblue.sql.Client(database=database)
```

畫 TFrecord

```python
waveforms = db.get_waveform()
for waveform in waveforms:
    instance = seisblue.core.Instance(waveform)
    instance.plot()
```

## 訓練模型

### 05_training.py

```python
import seisblue
```

讀取 database

```python
database = 'HL2019.db'
db = seisblue.sql.Client(database)
```

取得要訓練的 list

```python
tfr_list = db.get_tfrecord(from_date='2019-01-01', to_date='2019-05-09', column='path')
tfr_list = seisblue.utils.flatten_list(tfr_list)
```

設定訓練參數並且開始訓練

```python
model_instance = 'test_model'  # model名稱
trainer = seisblue.model.trainer.GeneratorTrainer(database)
trainer.train_loop(tfr_list, model_instance,
                   batch_size=64, epochs=10,
                   plot=True)
```

## 預測與評估結果

### 07_predict.py

```python
import seisblue
```

選取 Database 與要使用的模型(模型會放在家目錄中的Models)

```python
model = 'HP2017_0104_EQ_noise.h5'
database = 'HL2019.db'
db = seisblue.sql.Client(database)
```

選取要進行預測的TFRecord

```python
tfr_list = db.get_tfrecord(column='path')
tfr_list = seisblue.utils.flatten_list(tfr_list)
```

進行預測

```python
evaluator = seisblue.model.evaluator.GeneratorEvaluator(database, model)
evaluator.predict(tfr_list)
```

結果會存放於家目錄底下的 TFRecord/model_name/ 底下

### 09_model_evaluation.py

```python
import seisblue
```

建立要評估的 tfrecord_list

```python
tfr_path = '/home/andy/TFRecord/Eval/model_name.h5/2019/HL'
tfr_list = seisblue.utils.get_dir_list(tfr_path, suffix='.tfrecord')
```

開始評估predict 後的 TFRecord 並且 output precision, recall, F1 score

```python
evaluator = seisblue.model.evaluator.GeneratorEvaluator()
evaluator.score(tfr_list, height=0.4, delta=0.2, error_distribution=True)
```

### 10_plot_predict_instance.py

```python
import seisblue
```

建立要檢視的 tfrecord_list

```python
tfr_path = '/home/andy/TFRecord/Eval/model_name.h5/2019/HL'
tfr_list = seisblue.utils.get_dir_list(tfr_path, suffix='.tfrecord')
dataset = seisblue.io.read_dataset(tfr_list)
```

畫圖

```python
for item in dataset:
    instance = seisblue.core.Instance(item)
    instance.plot(threshold=0.4)
```

畫出模型預測的 pick 的噪訊比

```python
psnr, ssnr = seisblue.qc.get_snr_list(dataset)
seisblue.plot.plot_snr_distribution(psnr)
seisblue.plot.plot_snr_distribution(ssnr)
```

## 預測連續資料

### 11_continues_data_parallel.py

```python
import seisblue
import os
import multiprocessing as mp
```

設定 GPU 可平行運算，需使用 def main()執行此程式

```python
def main():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    mp.set_start_method('spawn')
```

指定 database 與 SDS_ROOT 中 network 的路徑

```python
    database = 'SF2021_04.db'
network_path_in_SDS_ROOT = '/home/andy/SDS_ROOT/2021/SF/'
```

取得 station_list

```python
    station_list = os.listdir(network_path_in_SDS_ROOT).sort()
```

開始預測連續資料

```python
    seisblue.utils.parallel(station_list,
                            func=seisblue.model.evaluator.predict_parallel,
                            database=database,
                            model_instance='/home/andy/Models/HP2017_0104_EQ_noise.h5',  # 指定使用模型
                            start_time='2021-04-01',
                            end_time='2021-06-01',
                            batch_size=1,
                            cpu_count=3)  # 同時進行的執行緒，此部分需考量 GPU 的記憶體大小
```

刪除相同的 picks

```python
    db = seisblue.sql.Client(database)
db.remove_duplicates('pick', ['time', 'phase', 'station', 'tag'])
```

```python
if __name__ == "__main__":
    main()
```

## 評估連續資料結果

### 12_compare_dataset.py

```python
import seisblue
```

比較 dataset 中 manual 和 predict 的 recall

```python
seisblue.compare.compare_dataset_pick('HP2017_EQ.db', from_time='2017-05-01 00:00:00',
                                      to_time='2017-06-30 23:59:59')
```

## 定位

### 13_associate

database 資料位置可從 config.yml 中修改

執行此程式前，需先將 HYP 檔案與 SEISAN 檔案放入家目錄底下的 Associator 資料夾中

```python
import seisblue
```

從 database 列出需要定位的 picks

```python
db = seisblue.sql.Client(database='HL2019.db')
picks = db.get_picks(tag='manual',
                     station='HL*',
                     from_time='2019-04-01',
                     to_time='2019-05-30')
```

將 picks 讀入 assoicate database

```python
db_assoc = seisblue.associator.sql.Client('assoc.db', remove=True)
db_assoc.read_picks(picks)
```

產生一個 LocalAssociator

```python
associator = seisblue.associator.core.LocalAssociator('assoc.db',
                                                      assoc_ot_uncert=3,
                                                      nsta_declare=3)

associator = seisblue.associator.sql.get_associator(picks)
```

做出可能的事件

```
associator.id_candidate_events()
```

定位

```python
associator.associate_candidates()
```

### 14_associate_event_to_sfile.py

```python
import seisblue
```

從 associate database 取得資料

```python
config = seisblue.utils.Config()
database = '2017may_june.db'
associates = seisblue.sql.get_associates(database=database)
```

寫成 sfile

```python
for associate in associates:
    picks = seisblue.sql.pick_assoc_id(database=database, assoc_id=associate.id)
    seisblue.io.output_sfile(associate.origin_time,
                             associate.latitude,
                             associate.longitude,
                             associate.depth,
                             picks,
                             out_dir='/home/user/Catalog/associate_sfile_output')  # 指定資料夾
```