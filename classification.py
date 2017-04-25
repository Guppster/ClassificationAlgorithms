import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
%matplotlib inline

# Load Data from CSV file
df=pd.read_csv('LoanPaymentsDataClean.csv')
df.head()

# Convert to date time object
df['due_date']=pd.to_datetime(df['due_date'])
df['effective_date']=pd.to_datetime(df['effective_date'])
df.head()

# See how many of each class is in our data set
df['loan_status'].value_counts()

# Functions for Plotting

def NiceHist(Name,df,H=False):
    Paid=df.ix[df['loan_status']=='PAIDOFF',Name].values
    notPaid=df.ix[df['loan_status']=='COLLECTION',Name].values
    Max=np.array([Paid.max(),notPaid.max()]).max()
    Min=np.array([Paid.min(),notPaid.min()]).min()
    bins = np.linspace(Min, Max,10)

    plt.hist(Paid, bins, alpha=0.5, label='PAIDOFF',color='g')
    plt.hist(notPaid, bins, alpha=0.5, label='COLLECTION',color='r')
    plt.legend(loc='upper left')
    plt.xlabel(Name)
    plt.title('Histogram of '+Name+ ' for Different Classes' )
    plt.ylabel('Number of people')
    plt.show()

def CoolPlot(df,Name_x,Name_y):       # no labels
    nullfmt = NullFormatter()   
    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    bottom_h = left_h = left + width + 0.02
    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.2]
    rect_histy = [left_h, bottom, 0.2, height]

   # nullfmt = NullFormatter()         # no labels

    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    bottom_h = left_h = left + width + 0.02
    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.2]
    rect_histy = [left_h, bottom, 0.2, height]

    # start with a rectangular Figure
    plt.figure(1, figsize=(8, 8))

    axScatter = plt.axes(rect_scatter)
    axHistx = plt.axes(rect_histx)
    axHisty = plt.axes(rect_histy)

    # no labels
    axHistx.xaxis.set_major_formatter(nullfmt)
    axHisty.yaxis.set_major_formatter(nullfmt)

    # the scatter plot:
    whatcolor=df['loan_status'].apply(lambda x: 'g' if x=='PAIDOFF' else 'r' )

    axScatter.scatter(df[Name_x], df[Name_y],c=whatcolor,marker=(5, 0)) 
    axScatter.set_xlabel(Name_x)
    axScatter.set_ylabel(Name_y)
    
    Paid_x=df.ix[df['loan_status']=='PAIDOFF',Name_x].values
    notPaid_x=df.ix[df['loan_status']=='COLLECTION',Name_x].values
    Max_x=np.array([Paid_x.max(),notPaid_x.max()]).max()
    Min_x=np.array([Paid_x.min(),notPaid_x.min()]).min()
    bins_x = np.linspace(Min_x, Max_x,10)

    axHistx.hist(Paid_x, bins_x, alpha=0.5, label='PAIDOFF',color='g')
    axHistx.hist(notPaid_x, bins_x, alpha=0.5, label='COLLECTION',color='r')
    axHistx.legend(loc='upper left')

    Paid_y=df.ix[df['loan_status']=='PAIDOFF',Name_y].values
    notPaid_y=df.ix[df['loan_status']=='COLLECTION',Name_y].values
    Max_y=np.array([Paid_y.max(),notPaid_y.max()]).max()
    Min_y=np.array([Paid_y.min(),notPaid_y.min()]).min()
    bins_y = np.linspace(Min_y, Max_y,10)
    axHisty.hist(Paid_y, bins_y, alpha=0.5, label='PAIDOFF',color='g', orientation='horizontal')
    axHisty.hist(notPaid_y, bins_y, alpha=0.5, label='COLLECTION',color='r', orientation='horizontal')

    def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


#Plot some stuff with our new functions
NiceHist('Principal',df)

NiceHist('terms',df)

NiceHist('age',df)

CoolPlot(df,'Principal','age')

#PreProcess some data (Feature selection / extraction)
df['dayofweek']=df['effective_date'].dt.dayofweek
NiceHist('dayofweek',df)

CoolPlot(df,'dayofweek','terms')

#These graphs reveal that people who get the loan at the end of the week, dont pay it off

#Use feature binarization to set a threshold value of less than day 4 of the week
df['weekend']=df['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)
df.head()

#Checking gender trends
df.groupby(['Gender'])['loan_status'].value_counts(normalize=True)

#Checkin education trends
df.groupby(['education'])['loan_status'].value_counts(normalize=True)

#PreProcess gender values (0 for male, 1 for female)
df['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)
df.head()

df[['Principal','terms','age','Gender','education']].head()

#Convert categorical variables to binary variables and append to Data Frame
Feature=df[['Principal','terms','age','Gender','weekend']]
Feature=pd.concat([Feature,pd.get_dummies(df['education'])], axis=1)
Feature.drop(['Master or Above'], axis = 1,inplace=True)
Feature.head()

#Define a feature set X
X=Feature
X[0:5]

#Define lables
y=df['loan_status'].values
y[0:5]

#Normalize Data
X= preprocessing.StandardScaler().fit(X).transform(X)
X[0:5]

#Train Test Split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print 'Train set:', X_train.shape,  y_train.shape
print 'Test set:', X_test.shape,  y_test.shape

#Start with the actual Classifcation

#K Nearest Neighbor (K-NN)

#Training

k = 3

#Train Model and Predict  
neigh = KNeighborsClassifier(n_neighbors=k).fit(X_train,y_train)

#Predicting
yhat=neigh.predict(X_test)
yhat[0:5]

#Evalute how accurate the prediction was
np.mean(yhat==y_test)

#Testing other values of K
Ks=7
mean_acc=np.zeros((Ks-1))
std_acc=np.zeros((Ks-1))
ConfustionMx=[];

for n in range(1,Ks):
    
    neigh = KNeighborsClassifier(n_neighbors=n).fit(X_train,y_train)
    yhat=neigh.predict(X_test)
    
    mean_acc[n-1]=np.mean(yhat==y_test);
    
    std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])
    ConfustionMx.append(confusion_matrix(yhat,y_test,labels=['PAIDOFF','COLLECTION' ]))

mean_acc

#Plot model accuracy for different K
plt.plot(range(1,Ks),mean_acc,'g')
plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
plt.legend(('Accuracy ', '+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Nabors (K)')
plt.tight_layout()
plt.show()

#Display Best
print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1) 

plt.figure()
plot_confusion_matrix(ConfustionMx[mean_acc.argmax()], classes=['PAIDOFF','COLLECTION' ],title='Confusion matrix, with normalization',normalize=True)
