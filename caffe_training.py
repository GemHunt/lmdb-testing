'''
This calls the caffe binary to train models
'''




command_line = '/home/pkrush/caffe/build/tools/caffe '
command_line += 'train '
command_line += '-solver /home/pkrush/lmdb-files/train/1351/solver.prototxt '
command_line += '-weights /home/pkrush/jobs/20161028-100120-e0d3/snapshot_iter_140580.caffemodel '

print command_line