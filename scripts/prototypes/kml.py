from seisnn.utils import get_config
from seisnn.io import read_kml_placemark, write_hyp_station

config = get_config()

kml = 'HL2019_M_deployed.kml'

geom = read_kml_placemark(kml)
write_hyp_station(geom, 'HL2019_M.HYP')
