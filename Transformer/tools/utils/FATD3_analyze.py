from PIL import Image
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib
import re

class TemperatureAnalysis:
    def __init__(self, target, data_path, number_list, mode, plot=False):
        self.target = target
        self.data_path = data_path
        self.number_list = number_list
        self.mode = mode
        self.plot = plot


    def save_data(self, data, file_name):
        """
        Saves the provided DataFrame `data` to a CSV file named `file_name` in the `result_path`.
        """
        file_path = os.path.join(self.data_path, file_name)
        data.to_csv(file_path, index=False)
        print(f"Data saved to {file_path}")

    def load_and_combine_temperature_data(self):
        temperature_data = pd.DataFrame()
        for i in self.number_list:
            num = str(re.findall(r'\d+', i)[0]).zfill(4)
            file_path = os.path.join(self.data_path, i)
            temp_df = pd.read_csv(file_path, header=None, low_memory=False)
            rows, columns = temp_df.shape
            temp_df = temp_df.stack().reset_index()
            temp_df.columns = ['row', 'column', 'temperature']
            temp_df['node'] = temp_df['row'] * columns + temp_df['column']  # 동적으로 크기에 맞춰 node 번호 계산
            temp_df['temperature'] = pd.to_numeric(temp_df['temperature'], errors='coerce')
            temp_df.rename(columns={'temperature': f'temperature_t{num}'}, inplace=True)
            temp_df = temp_df[['node', f'temperature_t{num}']]

            if temperature_data.empty:
                temperature_data = temp_df
            else:
                temperature_data = pd.merge(temperature_data, temp_df, on='node', how='outer')
        return temperature_data

    def calculate_differences(self, df):
        delta_dict = {}
        for i in range(len(self.number_list)-1):
            check = re.findall(r'\d+', self.number_list[i])[0]
            num = str(re.findall(r'\d+', self.number_list[i])[0]).zfill(4)
            num1 = str(re.findall(r'\d+', self.number_list[i+1])[0]).zfill(4)
            delta_key = f'ΔT{num},{num1}'
            delta_dict[delta_key] = df[f'temperature_t{num1}'] - df[f'temperature_t{num}']
        for i in range(len(self.number_list)-2):
            num = str(re.findall(r'\d+', self.number_list[i])[0]).zfill(4)
            num1 = str(re.findall(r'\d+', self.number_list[i+1])[0]).zfill(4)
            num2 = str(re.findall(r'\d+', self.number_list[i+2])[0]).zfill(4)
            delta_delta_key = f'ΔT{num},{num1},{num2}'
            delta_dict[delta_delta_key] = delta_dict[f'ΔT{num1},{num2}'] - delta_dict[f'ΔT{num},{num1}']
        df = pd.concat([df, pd.DataFrame(delta_dict)], axis=1)
        return df

    def plot_and_save(self, df, mode):
        matplotlib.use('TkAgg')  # 환경에 맞는 백엔드 설정
        plt.figure(figsize=(20, 10), dpi=300)
        unique_nodes = df['node'].unique()
        line_styles = ['-']

        unique_nodes = np.array([25000])

        for idx, node in enumerate(unique_nodes):
            node_data = df[df['node'] == node]
            x_vals = []
            y_vals = []
            if mode == 1:
                for i in range(len(self.number_list)-1):
                    delta_col = f'ΔT{self.number_list[i]},{self.number_list[i+1]}'
                    if delta_col in node_data:
                        value = node_data[delta_col].values[0]
                        x_vals.append(i)
                        y_vals.append(value)
            elif mode == 2:
                for i in range(len(self.number_list)-2):
                    delta_col = f'ΔT{self.number_list[i]},{self.number_list[i+1]},{self.number_list[i+2]}'
                    if delta_col in node_data:
                        value = node_data[delta_col].values[0]
                        x_vals.append(i)
                        y_vals.append(value)
            else:
                for i in range(len(self.number_list)):
                    delta_col = f'temperature_t{self.number_list[i]}'
                    if delta_col in node_data:
                        value = node_data[delta_col].values[0]
                        x_vals.append(i)
                        y_vals.append(value)

            # Apply smooth line
            smoothed_y_vals = np.convolve(y_vals, np.ones(5) / 5, mode='valid')
            smoothed_x_vals = x_vals[:len(smoothed_y_vals)]
            line_style = line_styles[idx % len(line_styles)]
            plt.plot(smoothed_x_vals, smoothed_y_vals, label=f'Node {node}', linewidth=0.5, color='black', linestyle=line_style)

            # # Not apply smooth line
            # line_style = line_styles[idx % len(line_styles)]
            # plt.plot(x_vals, y_vals, label=f'Node {node}', linewidth=0.5, color='black', linestyle=line_style)

        plt.xlabel('Time (s)')
        plt.ylabel('Rate of Change of Temperature Change')
        plt.title('Rate of Change of Temperature Change Over Time for All Nodes')
        plt.grid(True)
        file_name = f"test_{mode}_{self.target}_Rate_of_Change_of_Temperature_Over_Time_for_All_Nodes.png"
        save_path = os.path.join(self.data_path, file_name)
        plt.savefig(save_path, dpi=300)


    def run(self):
        results_file = os.path.join(self.data_path, 'result.csv')
        temperature_data_file = os.path.join(self.data_path, 'temperature_data.csv')

        if os.path.exists(results_file):
            results = pd.read_csv(results_file)
        else:
            data = self.load_and_combine_temperature_data()
            results = self.calculate_differences(data)
            self.save_data(data, 'temperature_data.csv')
            self.save_data(results, 'result.csv')

        if self.plot == True:
            self.plot_and_save(results, self.mode)


def subdirectories(directory):
    # directory 아래의 모든 디렉토리를 순회
    for root, dirs, files in os.walk(directory):
        for name in dirs:
            print(os.path.join(root, name))

