import pandas as pd

# 加载CSV文件
file_path = './Figures/average_max_metrics.csv'  # 根据需要更改路径
data = pd.read_csv(file_path)


def format_metric(value, std):
    return f"${value:.3f}_{{\\pm{std:.3f}}}$"


data['Acc'] = data.apply(lambda row: format_metric(row['Average_Max_Test_Acc'], row['Acc_std']), axis=1)
data['F1 Score'] = data.apply(lambda row: format_metric(row['Average_Max_F1_Score'], row['F1_std']), axis=1)
data['AUC'] = data.apply(lambda row: format_metric(row['Average_Max_AUC'], row['AUC_std']), axis=1)

formatted_data = data[['Dataset', 'Framework', 'Acc', 'F1 Score', 'AUC']]

print(formatted_data)

output_file_path = './Figures/sformatted_metrics_output.csv'  
formatted_data.to_csv(output_file_path, index=False)