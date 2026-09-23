# 读 csv , 查看 base_id, app_id, location_id 的最大最小值
import pandas as pd
import os

def check(csv_file):
    df = pd.read_csv(csv_file)
    print(f"File: {csv_file}")
    print(f"Number of records: {len(df)}")
    print(f"Base ID - Min: {df['location'].min()}, Max: {df['location'].max()}")
    print(f"App ID - Min: {df['app_id'].min()}, Max: {df['app_id'].max()}")
    print(f"app cate ID - Min: {df['app_cate_id'].min()}, Max: {df['app_cate_id'].max()}")

if __name__ == "__main__":
    # files = [
    #     "data/processed_data/shanghai/app_usage_records/app_usage_record_20160420.csv",
    #     "data/processed_data/shanghai/app_usage_records/app_usage_record_20160421.csv",
    #     "data/processed_data/shanghai/app_usage_records/app_usage_record_20160422.csv",
    #     "data/processed_data/shanghai/app_usage_records/app_usage_record_20160423.csv",
    #     "data/processed_data/shanghai/app_usage_records/app_usage_record_20160424.csv",
    #     "data/processed_data/shanghai/app_usage_records/app_usage_record_20160425.csv",
    # ]
    files = [
        "data/nanchang/app_usage_records/app_usage_record_20220519.csv",
        "data/nanchang/app_usage_records/app_usage_record_20220520.csv",
        "data/nanchang/app_usage_records/app_usage_record_20220521.csv",
        "data/nanchang/app_usage_records/app_usage_record_20220522.csv",
        "data/nanchang/app_usage_records/app_usage_record_20220523.csv",
        "data/nanchang/app_usage_records/app_usage_record_20220524.csv",
        "data/nanchang/app_usage_records/app_usage_record_20220525.csv",
    ]

    for file in files:
        if os.path.exists(file):
            check(file)
        else:
            print(f"File not found: {file}")