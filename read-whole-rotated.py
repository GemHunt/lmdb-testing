import pandas as pd
df = pd.read_csv('out.dat')
df.columns = ['filename','class','pred','result']
print df

scale = df['result'].sum()/360
print scale
all_classes = df.groupby('pred')['result'].sum()
print all_classes
scaled = all_classes.multiply(1/scale, fill_value=0)
print scaled
print scaled.sum()

al = all_classes.to_frame()
print al
al.columns =  ['pred','scale']
df = pd.merge(df,al,on='pred', how='inner')
print df


