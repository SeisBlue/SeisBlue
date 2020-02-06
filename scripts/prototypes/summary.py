from seisnn.io import read_hyp, read_event_list
from seisnn.pick import get_pick_dict

geom = read_hyp('HL2017.HYP')
events = read_event_list('HL2017')

pick_dict = get_pick_dict(events)
