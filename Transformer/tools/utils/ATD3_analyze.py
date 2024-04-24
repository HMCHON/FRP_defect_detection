import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib


class TemperatureAnalysis:
    def __init__(self, base_path, result_path, num_files, mode):
        self.base_path = base_path
        self.result_path = result_path
        self.num_files = num_files
        self.mode = mode

    def save_data(self, data, file_name):
        """
        Saves the provided DataFrame `data` to a CSV file named `file_name` in the `result_path`.
        """
        file_path = os.path.join(self.result_path, file_name)
        data.to_csv(file_path, index=False)
        print(f"Data saved to {file_path}")

    def load_and_combine_temperature_data(self):
        temperature_data = pd.DataFrame()
        for i in range(1, self.num_files + 1):
            file_path = os.path.join(self.base_path, f"temperature_Temperature {i}_sorted.csv")
            temp_df = pd.read_csv(file_path)
            temp_df['temperature'] = pd.to_numeric(temp_df['temperature'], errors='coerce')
            temp_df.rename(columns={'temperature': f'temperature_t{i}'}, inplace=True)
            if temperature_data.empty:
                temperature_data = temp_df
            else:
                temperature_data = pd.merge(temperature_data, temp_df, on='node', how='outer')
        return temperature_data

    def calculate_differences(self, df):
        delta_dict = {}
        for i in range(1, self.num_files):
            delta_key = f'ΔT{i}{i + 1}'
            delta_dict[delta_key] = df[f'temperature_t{i + 1}'] - df[f'temperature_t{i}']
        for i in range(1, self.num_files - 1):
            delta_delta_key = f'ΔT{i}{i + 1}{i + 2}'
            delta_dict[delta_delta_key] = delta_dict[f'ΔT{i + 1}{i + 2}'] - delta_dict[f'ΔT{i}{i + 1}']
        df = pd.concat([df, pd.DataFrame(delta_dict)], axis=1)
        return df

    def plot_and_save(self, df):
        matplotlib.use('TkAgg')  # 환경에 맞는 백엔드 설정
        plt.figure(figsize=(20, 10), dpi=300)
        unique_nodes = df['node'].unique()
        line_styles = ['-']

        for idx, node in enumerate(unique_nodes):
            node_data = df[df['node'] == node]
            x_vals = []
            y_vals = []
            for i in range(1, self.num_files):
                if self.mode == 1:
                    delta_col = f'ΔT{i}{i + 1}'
                elif self.mode == 2:
                    delta_col = f'ΔT{i}{i + 1}{i + 2}'
                else:
                    delta_col = f'temperature_t{i}'

                if delta_col in node_data:
                    value = node_data[delta_col].values[0]
                    x_vals.append(i)
                    y_vals.append(value)
            line_style = line_styles[idx % len(line_styles)]
            plt.plot(x_vals, y_vals, label=f'Node {node}', linewidth=0.5, color='black', linestyle=line_style)

        plt.xlabel('Time (s)')
        plt.ylabel('Rate of Change of Temperature Change')
        plt.title('Rate of Change of Temperature Change Over Time for All Nodes')
        plt.grid(True)
        folder_name = os.path.basename(self.result_path)
        file_name = f"{self.mode}_{folder_name}_Rate_of_Change_of_Temperature_Over_Time_for_All_Nodes.png"
        save_path = os.path.join(self.result_path, file_name)
        plt.savefig(save_path, dpi=300)

    def run(self):
        results_file = os.path.join(self.result_path, 'result.csv')
        temperature_data_file = os.path.join(self.result_path, 'temperature_data.csv')

        if os.path.exists(results_file):
            results = pd.read_csv(results_file)
        else:
            data = self.load_and_combine_temperature_data()
            results = self.calculate_differences(data)
            self.save_data(data, 'temperature_data.csv')
            self.save_data(results, 'result.csv')

        self.plot_and_save(results)


def subdirectories(directory):
    # directory 아래의 모든 디렉토리를 순회
    for root, dirs, files in os.walk(directory):
        for name in dirs:
            print(os.path.join(root, name))

# 사용 예시:
TopDirectory = '/home/lams/Desktop/PycharmProjects/FRP_defect_detection/Transformer/dataset/ATD3_2'
for root, dirs, files in os.walk(TopDirectory):
    for name in dirs:
        BasePath = os.path.join(root, name)
        print(BasePath)

        if BasePath != '/home/lams/Desktop/PycharmProjects/FRP_defect_detection/Transformer/dataset/ATD3_2/Air_Defect_50_result' and\
                BasePath != '/home/lams/Desktop/PycharmProjects/FRP_defect_detection/Transformer/dataset/ATD3_2/Vacuum_Defect_40_result' and \
                BasePath != '/home/lams/Desktop/PycharmProjects/FRP_defect_detection/Transformer/dataset/ATD3_2/Vacuum_Defect_50_result':

            num_files = 900  # 파일 수
            for i in [0, 1,2]:
                mode = i  # 선택한 모드

                analysis = TemperatureAnalysis(BasePath, BasePath, num_files, mode)
                analysis.run()
