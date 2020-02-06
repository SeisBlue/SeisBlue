from seisnn.utils import get_config
from seisnn.io import read_kml

config = get_config()

kml = 'HL2019_M_deployed.kml'

geom = read_kml(kml)

