from pyrocko.tectonics import PeterBird2003

poi_lat = 12.4
poi_lon = 133.5

PB = PeterBird2003()
plates = PB.get_plates()

for plate in plates:
    if plate.contains_point((poi_lat, poi_lon)):
        print plate.name
        print PB.full_name(plate.name)

'''
>>> PS
>>> Philippine Sea Plate
'''
