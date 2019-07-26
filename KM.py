#Read in data (Input)
dataset = pd.read_csv('H:/Juan Lu/AF/Outcome/af_bleed.csv')

#Processing Column
dataset = dataset.drop(["Column","You","Don't","Need"
                         ], axis=1)
#Group Data
groups = dataset['desired_group_column']
ix = (groups == 1)  # can be anything here number/string
                    # this value will be used to classify the data into two groups. 

#Time
time = dataset['your_time_col']

#Event
event = dataset['your_event_col']

#Fit the model
kmf.fit(time[~ix], event[~ix], label = 'Event without Group')
#Output plot
ax = kmf.plot()  
#ax = kmf.plot(ax=ax)

#Fit and plot
kmf.fit(time[ix], event[ix], label='Event with Group')
ax = kmf.plot(ax=ax)
