import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv('train1.csv')
## analysing gender
approved=df[df['Loan_Status']=='Y']['Gender'].value_counts()
declined=df[df['Loan_Status']=='N']['Gender'].value_counts()
sf=pd.DataFrame([approved,declined])
sf.index=['approved','declined']
sf.plot(kind='bar',stacked=True,figsize=(15,8))
#plt.show()

##analyse married status
approved=df[df['Loan_Status']=='Y']['Married'].value_counts()
declined=df[df['Loan_Status']=='N']['Married'].value_counts()
sf=pd.DataFrame([approved,declined])
sf.index=['approved','declined']
sf.plot(kind='bar',stacked=True,figsize=(15,8))
#plt.show()

#analysing dependents
approved=df[df['Loan_Status']=='Y']['Dependents'].value_counts()
declined=df[df['Loan_Status']=='N']['Dependents'].value_counts()
sf=pd.DataFrame([approved,declined])
sf.index=['approved','declined']
sf.plot(kind='bar',stacked=True,figsize=(15,8))
#plt.show()

#analysing educate
approved=df[df['Loan_Status']=='Y']['Education'].value_counts()
declined=df[df['Loan_Status']=='N']['Education'].value_counts()
sf=pd.DataFrame([approved,declined])
sf.index=['approved','declined']
sf.plot(kind='bar',stacked=True,figsize=(15,8))
#plt.show()
#employed
approved=df[df['Loan_Status']=='Y']['Self_Employed'].value_counts()
declined=df[df['Loan_Status']=='N']['Self_Employed'].value_counts()
sf=pd.DataFrame([approved,declined])
sf.index=['approved','declined']
sf.plot(kind='bar',stacked=True,figsize=(15,8))
#plt.show()


#analyzing income
ax=df[df['Loan_Status']=='Y']['ApplicantIncome']
az=df[df['Loan_Status']=='N']['ApplicantIncome']
plt.hist([ax,az],stacked=True,bins=5,label=['approved','declined'])
#plt.show()


xc=df.groupby('Gender')['Married']

def get_combined_data():
    train=pd.read_csv('train1.csv')
    test=pd.read_csv('test1.csv')
    target=train.Loan_Status
    train.drop('Loan_Status',1,inplace=True)
    combined=train.append(test)
    combined.reset_index(inplace=True)
    combined.drop('index',inplace=True,axis=1)
    return combined

combined=get_combined_data()
#filling gender
combined['Gender'].fillna('Male',inplace=True)
#filling married feature
s=combined[combined['Married'].isnull()].index
for i in s:
  if (combined.loc[i,'Gender']=='Male'):
     combined.loc[i,'Married']='Yes'
  else:
     combined.loc[i,'Married']='No'


#print combined['Married'].isnull().value_counts()
#combined.info()
#asd=combined.groupby(['Gender','Married'])['Dependents'].value_counts()
aq=combined[combined['Dependents'].isnull()].index
for i in aq:
  if (combined.loc[i,'Married']=='No'):
     combined.loc[i,'Dependents']='0'
  else:
     if(combined.loc[i,'Gender']=='Male'):
       combined.loc[i,'Dependents']='2'
     else:
        combined.loc[i,'Dependents']='1'

asx=combined.groupby(['Gender','Married','Education'])['Self_Employed'].value_counts()
#print asx
combined['Self_Employed'].fillna('No',inplace=True)


azx=combined.groupby(['Gender'])['LoanAmount'].value_counts()
#print azx

s=combined[combined['Gender']=='Male']['LoanAmount'].median()
t=combined[combined['Gender']=='Female']['LoanAmount'].median()

d=combined[combined['LoanAmount'].isnull()].index
for i in d:
  if (combined.loc[i,'Gender']=='Male'):
     combined.loc[i,'LoanAmount']=s
  else:
     combined.loc[i,'LoanAmount']=t

azx=combined.groupby(['Gender'])['Loan_Amount_Term'].value_counts()
combined['Loan_Amount_Term'].fillna(360,inplace=True)


azx=combined.groupby(['Gender'])['Credit_History'].value_counts()
combined['Credit_History'].fillna(1,inplace=True)


##dummy encoding

gender_dummies=pd.get_dummies(combined['Gender'],prefix='Gender')
combined=pd.concat([combined,gender_dummies],axis=1)
combined.drop('Gender',axis=1,inplace=True)


married_dummies=pd.get_dummies(combined['Married'],prefix='Married')
combined=pd.concat([combined,married_dummies],axis=1)
combined.drop('Married',axis=1,inplace=True)

dependent_dummies=pd.get_dummies(combined['Dependents'],prefix='Dependents')
combined=pd.concat([combined,dependent_dummies],axis=1)
combined.drop('Dependents',axis=1,inplace=True)

Education_dummies=pd.get_dummies(combined['Education'],prefix='Education')
combined=pd.concat([combined,Education_dummies],axis=1)
combined.drop('Education',axis=1,inplace=True)

employed_dummies=pd.get_dummies(combined['Self_Employed'],prefix='Self_Employed')
combined=pd.concat([combined,employed_dummies],axis=1)
combined.drop('Self_Employed',axis=1,inplace=True)
s=pd.Series([1,2,3,4,5,6,7,8,9,10])
for i in s :
  combined[i*1000]=np.NaN
for i in combined.index:
  if (combined.loc[i,'ApplicantIncome']<=1000) :
   combined.loc[i,1000]=1
for i in combined.index:
  if (combined.loc[i,'ApplicantIncome']<=2000 and combined.loc[i,'ApplicantIncome']>1000) :
   combined.loc[i,2000]=1
for i in combined.index:
  if (combined.loc[i,'ApplicantIncome']<=3000 and combined.loc[i,'ApplicantIncome']>2000) :
   combined.loc[i,3000]=1
for i in combined.index:
  if (combined.loc[i,'ApplicantIncome']<=4000 and combined.loc[i,'ApplicantIncome']>3000) :
   combined.loc[i,4000]=1
for i in combined.index:
  if (combined.loc[i,'ApplicantIncome']<=5000 and combined.loc[i,'ApplicantIncome']>4000) :
   combined.loc[i,5000]=1
for i in combined.index:
  if (combined.loc[i,'ApplicantIncome']<=6000 and combined.loc[i,'ApplicantIncome']>5000) :
   combined.loc[i,6000]=1
for i in combined.index:
  if (combined.loc[i,'ApplicantIncome']<=7000 and combined.loc[i,'ApplicantIncome']>6000) :
   combined.loc[i,7000]=1
for i in combined.index:
  if (combined.loc[i,'ApplicantIncome']<=8000 and combined.loc[i,'ApplicantIncome']>7000 ):
   combined.loc[i,8000]=1

for i in combined.index:
  if (combined.loc[i,'ApplicantIncome']<=9000 and combined.loc[i,'ApplicantIncome']>8000) :
   combined.loc[i,8000]=1


for i in combined.index:
  if (combined.loc[i,'ApplicantIncome']<=10000 and combined.loc[i,'ApplicantIncome']>9000) :
   combined.loc[i,9000]=1

for i in combined.index:
  if (combined.loc[i,'ApplicantIncome']<=100000 and combined.loc[i,'ApplicantIncome']>10000) :
   combined.loc[i,10000]=1


for i in s:
  combined[i*1000].fillna(0,inplace=True)



combined.drop('ApplicantIncome',axis=1,inplace=True)


t=pd.Series(['0c','1000c','2000c','3000c','4000c','5000c','6000c','7000c','8000c','9000c','10000c'])
for i in t :
  combined[i]=np.NaN
for i in combined.index:
  if (combined.loc[i,'CoapplicantIncome']==0) :
   combined.loc[i,'0c']=1
for i in combined.index:
  if (combined.loc[i,'CoapplicantIncome']<=1000 and combined.loc[i,'CoapplicantIncome']>0) :
   combined.loc[i,'1000c']=1
for i in combined.index:
  if (combined.loc[i,'CoapplicantIncome']<=2000 and combined.loc[i,'CoapplicantIncome']>1000) :
   combined.loc[i,'2000c']=1
for i in combined.index:
  if (combined.loc[i,'CoapplicantIncome']<=3000 and combined.loc[i,'CoapplicantIncome']>2000) :
   combined.loc[i,'3000c']=1
for i in combined.index:
  if (combined.loc[i,'CoapplicantIncome']<=4000 and combined.loc[i,'CoapplicantIncome']>3000) :
   combined.loc[i,'4000c']=1
for i in combined.index:
  if (combined.loc[i,'CoapplicantIncome']<=5000 and combined.loc[i,'CoapplicantIncome']>4000) :
   combined.loc[i,'5000c']=1
for i in combined.index:

  if (combined.loc[i,'CoapplicantIncome']<=6000 and combined.loc[i,'CoapplicantIncome']>5000) :
   combined.loc[i,'6000c']=1
for i in combined.index:
  if (combined.loc[i,'CoapplicantIncome']<=7000 and combined.loc[i,'CoapplicantIncome']>6000) :
   combined.loc[i,'7000c']=1
for i in combined.index:
  if (combined.loc[i,'CoapplicantIncome']<=8000 and combined.loc[i,'CoapplicantIncome']>7000 ):
   combined.loc[i,'8000c']=1

for i in combined.index:
  if (combined.loc[i,'CoapplicantIncome']<=9000 and combined.loc[i,'CoapplicantIncome']>8000) :
   combined.loc[i,'8000c']=1


for i in combined.index:
  if (combined.loc[i,'CoapplicantIncome']<=10000 and combined.loc[i,'CoapplicantIncome']>9000) :
   combined.loc[i,'9000c']=1

for i in combined.index:
  if (combined.loc[i,'CoapplicantIncome']<=100000 and combined.loc[i,'CoapplicantIncome']>10000) :
   combined.loc[i,'10000c']=1


for i in t:
  combined[i].fillna(0,inplace=True)


#combined.info()
#print combined['1000c'].value_counts()
combined.drop('CoapplicantIncome',axis=1,inplace=True)



x=pd.Series(['100s','200s','300s','400s','500s','600s','700s'])
for i in x:
  combined[i]=np.NaN
for i in combined.index:
  if (combined.loc[i,'LoanAmount']<=100):
   combined.loc[i,'100s']=1
for i in combined.index:
  if (combined.loc[i,'LoanAmount']<=200 and combined.loc[i,'LoanAmount']>100):
   combined.loc[i,'200s']=1
for i in combined.index:
  if (combined.loc[i,'LoanAmount']<=300 and combined.loc[i,'LoanAmount']>200):
   combined.loc[i,'300s']=1
for i in combined.index:
  if (combined.loc[i,'LoanAmount']<=400 and combined.loc[i,'LoanAmount']>300):
   combined.loc[i,'400s']=1
for i in combined.index:
  if (combined.loc[i,'LoanAmount']<=500 and combined.loc[i,'LoanAmount']>400):
   combined.loc[i,'500s']=1
for i in combined.index:
  if (combined.loc[i,'LoanAmount']<=600 and combined.loc[i,'LoanAmount']>500):
   combined.loc[i,'600s']=1
for i in combined.index:
  if (combined.loc[i,'LoanAmount']<=700 and combined.loc[i,'LoanAmount']>600):
   combined.loc[i,'700s']=1

combined.drop('LoanAmount',axis=1,inplace=True)
for i in x:
  combined[i].fillna(0,inplace=True)

term_dummies=pd.get_dummies(combined['Loan_Amount_Term'],prefix='Loan_Amount_Term')
combined=pd.concat([combined,term_dummies],axis=1)
combined.drop('Loan_Amount_Term',axis=1,inplace=True)

credit_dummies=pd.get_dummies(combined['Credit_History'],prefix='Credit_History')
combined=pd.concat([combined,credit_dummies],axis=1)
combined.drop('Credit_History',axis=1,inplace=True)
area_dummies=pd.get_dummies(combined['Property_Area'],prefix='Property_Area')
combined=pd.concat([combined,area_dummies],axis=1)
combined.drop('Property_Area',axis=1,inplace=True)
combined.info()
combined.drop('Loan_ID',axis=1,inplace=True)

for i in df.index:
  if (df.loc[i,'Loan_Status']=='Y'):
   df.loc[i,'Loan_Status']=1.0
  else:
     df.loc[i,'Loan_Status']=0.0

df['Loan_Status']=df['Loan_Status'].astype(float)
print df['Loan_Status'].dtypes
train=combined.head(614)

test=combined.iloc[614:]
targets=df.Loan_Status

import sklearn
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(train,targets,test_size=0.2)
X_train=np.array(X_train)
y_train=np.array(y_train)
X_test=np.array(X_test)
y_test=np.array(y_test)
import keras
from keras.models import Sequential
from keras.layers import Dense
classifier=Sequential()
classifier.add(Dense(kernel_initializer="uniform", activation="relu", input_dim=57, units=29))
classifier.add(Dense(kernel_initializer="uniform", activation="relu",units=10))
classifier.add(Dense(units=1, activation="sigmoid", kernel_initializer="uniform"))
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
classifier.fit(X_train,y_train,batch_size=10,epochs=100,initial_epoch=0)
test=np.array(test)
ax=classifier.predict(test)
output=(ax>0.5).astype(int)
final=pd.DataFrame()
xc=pd.read_csv('test1.csv')
final['Loan_ID']=xc['Loan_ID']
final['Loan_Status']=output
final[['Loan_ID','Loan_Status']].to_csv('loanoutput.csv',index=False)

sd=pd.read_csv('loanoutput.csv')
for i in sd.index:
  if (sd.loc[i,'Loan_Status']==1):
   sd.loc[i,'Loan_Status']='Y'
  else:
     sd.loc[i,'Loan_Status']='N'

sd.to_csv('loanout.csv',index=False)

