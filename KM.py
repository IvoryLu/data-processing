dataset = pd.read_csv('H:/Juan Lu/AF/Outcome/af_bleed.csv')

dataset = dataset.drop(["Column","You","Don't","Need"
                         ], axis=1)

groups = dataset['desired_group_column']
ix = (groups == 1) # can be anything here number/string

time = dataset['your_time_col']

event = dataset['your_event_col']

kmf.fit(time[~ix], event[~ix], label = 'Event without Group')
ax = kmf.plot() 
#ax = kmf.plot(ax=ax)
kmf.fit(time[ix], event[ix], label='Event with Group')
ax = kmf.plot(ax=ax)
