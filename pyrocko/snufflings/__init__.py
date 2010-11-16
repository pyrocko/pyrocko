import minmax, rms
modules = [minmax, rms]

def __snufflings__():
    snufflings = []
    for mod in modules:
        snufflings.extend(mod.__snufflings__())
        
    return snufflings
    
    