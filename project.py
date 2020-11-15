import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pylab import rcParams
import seaborn as sns ; sns.set(style="ticks",color_codes=True)
rcParams['figure.figsize']=15,10

d.read_csv('C:/Users/Spardha/Desktop/online labs/conce.csv')

df.head(n=5)

df.describe()

df.columns=['Cement','slag','ash','water','super','coarse','fine',
           'age','strength']
           
 features=[
    'Cement','slag','ash','water','super','coarse','fine','age'
]
target='strength'

x=df[features]
y=df[target]

sns.pairplot(df);

conda install -c districtdatalabs yellowbrick

from yellowbrick.model_selection import FeatureImportances

from sklearn.linear_model import Lasso

fig=plt.figure()
ax=fig.add_subplot()
labels=list(map(lambda s: s.title(),features))
viz=FeatureImportances(Lasso(),ax=ax,label=labels,relative=False)

viz.fit(x,y)
viz.poof()


from yellowbrick.target import BalancedBinningReference
visual=BalancedBinningReference()
visual.fit(y)
visual.poof()

from yellowbrick.regressor import PredictionError
from sklearn.model_selection import train_test_split

x_train, x_test, y_train,y_test=train_test_split(x,y,test_size=0.3)

visual1=PredictionError(Lasso(),size=(600,400))
visual1.fit(x_train,y_train)
visual1.score(x_test,y_test)

visual1.finalize()
visual1.ax.set_xlabel('measured concerete strength')
visual1.ax.set_ylabel('predicted concerete strength')

from yellowbrick.regressor import ResidualsPlot
visual2=ResidualsPlot(Lasso(),size=(800,600))

visual2.fit(x_train,y_train)
visual2.score(x_test,y_test)
visual2.poof

from sklearn.model_selection import KFold
from yellowbrick.model_selection import CVScores

_ , ax=plt.subplots()
cv=KFold(12)
viz=CVScores(Lasso(),ax=ax,cv=cv,scoring='r2')
viz.fit(x_train,y_train).poof()

from yellowbrick.model_selection import LearningCurve
from sklearn.linear_model import LassoCV

sizes=np.linspace(0.3,1.0,10)
viz=LearningCurve(LassoCV(),train_sizes=sizes,scoring='r2')
viz.fit(x,y)
viz.poof()

from yellowbrick.regressor import AlphaSelection

alpha=np.logspace(-10,1,400)
model=LassoCV(alphas=alpha)
viz=AlphaSelection(model,size=(800,600))
viz.fit(x,y)
viz.poof()
