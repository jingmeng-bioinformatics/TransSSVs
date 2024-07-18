from sklearn.metrics import precision_recall_curve,average_precision_score
import numpy as np
import pandas as pd

import plotly.offline as py
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots


save_fig = go.Figure()


dash=None



EXPERIMENTS = [
		['7_trans','TransSSVs-07', '#E71F19'],
		['varnet','VarNet', '#008B8B'],
		['strelka2','Strelka2', '#0000ff'],
		['neusomatic','NeuSomatic', '#C9A77C'],
		['mutect2','Mutect2', '#F6BBC6'],['deepssv','DeepSSVs','#7e2065'],['varscan2','VarScan2','#41ae3c']]




for type in 'all','snp','indel':
	for name in 'AML','MB','CLL','M1','M2','M3':
		save_fig = go.Figure()
		for i, e in enumerate(EXPERIMENTS):
			path='TransSSVs/prroc/data/'+name+'/'+str(name)+'_'+e[0]+'_'+str(type)+'.txt'
			a=pd.read_table(path,header=None)
			all_number=a[a[1]==1].shape[0]
			aa=a[a[0]>0]
			max_number=aa[aa[1]==1].shape[0]
			max_recall=max_number/all_number
			b=list(a.loc[:,1])
			c=list(a.loc[:,0])
			pp,rr,tt=precision_recall_curve(b,c)
			a_data=pd.DataFrame({'precision':pp,'recall':rr})
			a_data_ok=a_data[a_data['recall']<=max_recall]
			rr=list(a_data_ok['recall'])
			pp=list(a_data_ok['precision'])
			ROC=0
			for k in range(len(rr)-1):
				ROC=ROC+(rr[k+1]-rr[k])*(pp[k+1]+pp[k])
			roc=-0.5*ROC
			trace = go.Scatter(x = rr,y = pp,mode = 'lines',line = dict(color = (e[2]),width = 2,dash = dash),name = e[1]+'_'+str(name)+'_'+str(type)+': '+str(round(roc,3)))
			save_fig.add_trace(trace)
			margin=go.layout.Margin(
			l=0, #left margin
			r=0, #right margin
			b=0, #bottom margin
			t=40  #top margin
			)
			layout = dict(
						  margin = margin,
						  title = '',
						  title_x=0.5, # position title
						  plot_bgcolor='#fff',
	#					   paper_bgcolor='#fff',
						  font=dict(
								family="Helvetica",
								size=14,
							),
						  xaxis = dict(
								tickmode = 'linear',
								linecolor='black',
								tick0 = 0,
								# showline=True,
								dtick = 0.2,
								title = 'Recall',mirror=True,
								range=[0, 1.], autorange=False, rangemode='nonnegative'
							),
						  yaxis = dict(
								tickmode = 'linear',
								linecolor='black',
								tick0 = 0,
								# showline=True,
								dtick = 0.2,
								title = 'Precision',mirror=True,
								range=[0, 1.], autorange=False, rangemode='nonnegative'
							)
						  )
			save_fig.layout = layout
			save_fig.update_xaxes(range=[0, 1.], showline=True) # , gridwidth=1
			save_fig.update_yaxes(range=[0, 1.],showline=True)#, linewidth=0.5, gridcolor='#EFECF6')
			save_fig.update_layout(width=300, height=300,showlegend=False)
		save_fig.show()
		save_path='tu/'+str(name)+'_'+str(type)+'.pdf'
		save_fig.write_image(save_path)
