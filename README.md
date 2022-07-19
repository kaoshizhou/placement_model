# placement_model

Arguments are in `src/config.py`, you can set `epochs`, `batch size`, `output dir` and so on.

## data
+ 2m_data_new: [2m_data_new] (https://microsoftapc-my.sharepoint.com/:u:/g/personal/shizsu_microsoft_com/Ee0QlZVmx2tAov9YMGbrf8cBP-5R1GclPuIsYxyoDR4ZyA)
+ 2m_data_with_type: [2m_data_with_type] (https://microsoftapc-my.sharepoint.com/:u:/g/personal/shizsu_microsoft_com/EXXWoVNdgT9JiW7Qs1kljIoB3wN5OvT_fOc1vUzawxCjNw?email=v-zhajiang%40microsoft.com&e=ImS6JH)

## training 
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 train.py
```
### 4 task setting
| | 1 | 2 | 3 | 4 |
| :----: | :----: | :----: | :----: | :----: |
|dataset| 2m_data_new |2m_data_new|2m_data_with_type|2m_data_with_type|
|model| t5-base | t5-large | t5-base | t5-large |
