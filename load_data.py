import numpy as np
import pandas as pd

def read_range_var(excel):
    df = pd.read_excel(excel)

    x_plus = 7
    y_plus = 16

    T1_min = df[f'Unnamed: {5+x_plus}'][14 + y_plus * 2]
    T1_max = df[f'Unnamed: {5+x_plus}'][13 + y_plus * 2]
    T2_min = df[f'Unnamed: {6+x_plus}'][14 + y_plus * 2]
    T2_max = df[f'Unnamed: {6+x_plus}'][13 + y_plus * 2]
    B_min = df[f'Unnamed: {3+x_plus}'][14 + y_plus * 2]
    B_max = df[f'Unnamed: {3+x_plus}'][13 + y_plus * 2]
    G_min = df[f'Unnamed: {2+x_plus}'][14 + y_plus * 2]
    G_max = df[f'Unnamed: {2+x_plus}'][13 + y_plus * 2]
    R_min = df[f'Unnamed: {1+x_plus}'][14 + y_plus * 2]
    R_max = df[f'Unnamed: {1+x_plus}'][13 + y_plus * 2]



    # print(type(df['Unnamed: 5'][14]))
    # if type(df['Unnamed: 5'][15]) is not int:
    #     print("###")
    mask_list = []
    range_list = []
    for j in range(19) :
        for i in range(4) :

            if type(df[f'Unnamed: {5+i * x_plus}'][14 + j * y_plus]) is int:
                T1_min = df[f'Unnamed: {5+i * x_plus}'][14 + j *y_plus]
                T1_max = df[f'Unnamed: {5+i * x_plus}'][13 + j *y_plus]
                T2_min = df[f'Unnamed: {6+i * x_plus}'][14 + j *y_plus]
                T2_max = df[f'Unnamed: {6+i * x_plus}'][13 +j * y_plus]
                B_min = df[f'Unnamed: {3+i * x_plus}'][14 + j *y_plus]
                B_max = df[f'Unnamed: {3+i * x_plus}'][13 + j *y_plus]
                G_min = df[f'Unnamed: {2+i * x_plus}'][14 + j *y_plus]
                G_max = df[f'Unnamed: {2+i * x_plus}'][13 + j *y_plus]
                R_min = df[f'Unnamed: {1+i * x_plus}'][14 + j *y_plus]
                R_max = df[f'Unnamed: {1+i * x_plus}'][13 + j *y_plus]
                
                mask_list.append([[T1_min, T1_max], [T2_min, T2_max]])
                range_list.append([[B_min, G_min, R_min], [B_max, G_max, R_max]])
                
    return mask_list, range_list
    print(mask_list)
    print(range_list)

if __name__ == "__main__":
    mask_list, range_list = read_range_var("Book1.xlsx")
    