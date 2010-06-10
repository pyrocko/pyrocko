


class DatalessSeedReader:
    
    def __init__(self):
        
    
    
    def 
        
    def read(self, filename):
        f = open(filename, 'r')
        data = f.read(1024)
        
        self.parse_control_header(data)
        
    def parse_control_header(self, data)
    
        sequence_number = int(data[0:6])
        type_code = data[6]
        continuation_code = data[7]
        blockette_type = data[8:11]
        blockette_length = int(data[11:15])
        
        print sequence_number
        print type_code
        print continuation_code
        print blockette_type
        print blockette_length
        
        
        
        
        