import minmax
modules = [minmax]

def __snufflings__():
    snufflings = []
    for mod in modules:
        snufflings.extend(mod.__snufflings__())
        
    return snufflings
    
    