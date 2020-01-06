from pyrocko.plot.gmtpy import GMT

gmt = GMT(config={'PS_PAGE_COLOR': '247/247/240'})
gmt.psbasemap(
    R=(0, 5, 0, 5),
    J='X%gi/%gi' % (5, 3),
    B='%g:Time:/%g:Amplitude:SWne' % (1, 1))

# Make four different datasets

# (1) a nested list, with the first dim corresponding to columns
data_as_columns = [[0, 1, 2, 3, 4, 5],  [0, 1, 0.5, 1, 0.5, 1]]

# (2) a nested list, with the first dim corresponding to rows
data_as_rows = [[0, 1],  [1, 2],  [2, 3],  [3, 3.5],  [4, 3],  [5, 2]]

# (3) a string containing an ascii table
data_as_string = b'''0 5
1 4
2 3.5
3 4
4 4.5
5 5'''

# (4) write ascii table in a temporary file...

# Get a filename in the private tempdir of the GMT instance.
# Files in that directory get deleted automatically.
filename = gmt.tempfilename('table.txt')

f = open(filename, 'w')
f.write('0 3\n1 3\n5 1.2\n')
f.close()

# Plot the four datasets
#
# The kind of input is selected with the keyword arguments beginning
# with 'in_'.
#
# Specifying R=True and J=True results '-R' and '-J' being passed
# to the GMT program without any arguments. (Doing so causes GMT to
# repeat the previous values.)

gmt.psxy(R=True, J=True, W='1p,black', in_columns=data_as_columns)
gmt.psxy(R=True, J=True, W='1p,red',   in_rows=data_as_rows)
gmt.psxy(R=True, J=True, W='1p,blue',  in_string=data_as_string)
gmt.psxy(R=True, J=True, W='1p,purple,dashed', in_filename=filename)

gmt.save('example2.pdf')
