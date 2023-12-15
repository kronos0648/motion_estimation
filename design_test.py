from convnet_design import visualize_architecture


visualize_architecture(
    fm_sizes= [(16, 3), (8, 2), (6, 1), (2, 1)],
    fm_depths= [3, 16, 6, 6],
    fm_kernel_sizes=[(2, 2), (3, 2), (3, 1)],
    fm_stride_sizes= [(2,1),(1,2),None],
    fm_texts= ['Convolution', 'Convolution', 'Max-pooling'],
    fc_units=[12, 96, 96, 3],
    fc_texts= ['Flatten', 'Fully connected', 'Dropout','Fully connected'],
    part='arm'
    )