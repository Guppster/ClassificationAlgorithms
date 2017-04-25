import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
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


NiceHist('Principal',df)

NiceHist('terms',df)

NiceHist('age',df)

