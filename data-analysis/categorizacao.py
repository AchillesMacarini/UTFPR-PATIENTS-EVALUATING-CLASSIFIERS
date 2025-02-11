import import_data
import matplotlib
import pandas as pd

data = import_data.raw_data

dataType = data.dtypes

quantitavos = dataType[dataType != 'object'].index.tolist()
qualitativos = dataType[dataType == 'object'].index.tolist()

def sortDiscrete(caminho,coluna ,mapeamento):
    df = pd.read_csv(caminho)
    mapping = {
        'Excellent': 4,
        'Very good': 3,
        'Good': 2,
        'Fair': 1,
        'Poor': 0
    }
    df['ID'] = df[coluna].map(mapeamento)
    df = df.sort_values(by='ID')

    df.to_csv(caminho,index=False)



# print(f"Dados quantitativos: \n {quantitavos}")

# print(f"\nDados qualitativos: \n {qualitativos}")

# estatisticas = import_data.data[quantitavos].describe()
# print(f"\n Estatisticas Descritivas: \n {estatisticas}")

# Tentativa de discretizar os dados
for column in qualitativos:

    df = pd.DataFrame(data[column].drop_duplicates().reset_index(drop = True))
    df['ID'] = df.index
    df.to_csv(f'data-set\{column}_discretizacao.csv', index=False)

# Alguns precisam de correção, não estão ordinais



sortDiscrete('data-set\GeneralHealth_discretizacao.csv','GeneralHealth',{
    'Excellent': 4,
    'Very good': 3,
    'Good': 2,
    'Fair': 1,
    'Poor': 0
})
sortDiscrete('data-set\HadDiabetes_discretizacao.csv','HadDiabetes',{
    'Yes': 3,
    'Yes, but only during pregnancy (female)': 2,
    'No, pre-diabetes or borderline diabetes': 1,
    'No': 0
})
sortDiscrete('data-set\SmokerStatus_discretizacao.csv','SmokerStatus',{
    'Current smoker - now smokes every day': 3,
    'Current smoker - now smokes some days': 2,
    'Former smoker': 1,
    'Never smoked': 0
})

sortDiscrete('data-set\TetanusLast10Tdap_discretizacao.csv','TetanusLast10Tdap',{
    'Yes, received Tdap': 3,
    'Yes, received tetanus shot but not sure what type': 2,
    'Yes, received tetanus shot, but not Tdap': 1,
    'No, did not receive any tetanus shot in the past 10 years': 0
})

df = pd.read_csv('data-set\AgeCategory_discretizacao.csv')
df = df.drop(columns=['ID'])
df = df.sort_values(by='AgeCategory')
df = df.reset_index(drop=True)
df['ID'] = df.index

df.to_csv('data-set\AgeCategory_discretizacao.csv', index=False)


