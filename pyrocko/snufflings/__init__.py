import minmax, rms, iris_data, stalta, geofon
modules = [minmax, rms, iris_data, stalta, geofon]

def __snufflings__():
    snufflings = []
    for mod in modules:
        snufflings.extend(mod.__snufflings__())
        
    for snuffling in snufflings:
        snuffling.setup()
        snuffling.set_name( snuffling.get_name() + ' (builtin)' )

    return snufflings
    
    
