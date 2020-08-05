from seisnn.utils import get_config
from seisnn.data.io import read_kml_placemark, write_hyp_station

config = get_config()

kml = 'Deployed2018HL.kml'

geom = read_kml_placemark(kml)
write_hyp_station(geom, 'HL2018_Zland.HYP')
