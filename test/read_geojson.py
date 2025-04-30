import geopandas as gpd
import matplotlib.pyplot as plt

gdf = gpd.read_file("./data/geojson/target_00000.geojson")

gdf = gdf.set_crs(epsg=26917, allow_override=True)

gdf = gdf.to_crs(epsg=4326)

gdf.plot()
plt.show()
