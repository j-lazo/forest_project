import tensorflow as tf
from tensorflow.keras import layers, Model, Input
import numpy as np

def mlp_block(inputs, hidden_units, activation='relu', dropout_rate=0.0, is_training=True, name=None):
    """Functional MLP block with batch norm and dropout."""
    x = inputs
    for i, units in enumerate(hidden_units):
        x = layers.Dense(units, name=f'{name}_dense_{i}')(x)
        x = layers.BatchNormalization(name=f'{name}_bn_{i}')(x)
        x = layers.Activation(activation, name=f'{name}_act_{i}')(x)
        if dropout_rate > 0:
            x = layers.Dropout(dropout_rate, name=f'{name}_drop_{i}')(x)
    return x

def farthest_point_sample(input_points, npoint):
    """Functional FPS sampling (simplified version)."""
    batch_size = tf.shape(input_points)[0]
    indices = tf.random.uniform((batch_size, npoint), 0, tf.shape(input_points)[1], dtype=tf.int32)
    return indices

def query_ball_point(radius, nsample, xyz1, xyz2):
    """Functional ball query (simplified version)."""
    batch_size = tf.shape(xyz1)[0]
    m = tf.shape(xyz2)[1]
    indices = tf.random.uniform((batch_size, m, nsample), 0, tf.shape(xyz1)[1], dtype=tf.int32)
    return indices

def group_points(points, indices):
    """Functional point grouping."""
    batch_size = tf.shape(points)[0]
    batch_indices = tf.tile(
        tf.reshape(tf.range(batch_size), [-1, 1, 1, 1]),
        [1, tf.shape(indices)[1], tf.shape(indices)[2], 1]
    )
    indices = tf.concat([batch_indices, tf.expand_dims(indices, -1)], axis=-1)
    return tf.gather_nd(points, indices)

def set_abstraction_module(input_xyz, input_points, npoint, radius, nsample, mlp, mlp2=None, group_all=False, name=None):
    """Functional Set Abstraction module."""
    with tf.name_scope(name):
        if group_all:
            # Global grouping
            new_xyz = tf.reduce_mean(input_xyz, axis=1, keepdims=True)
            grouped_xyz = tf.expand_dims(input_xyz, axis=1)
            if input_points is not None:
                grouped_points = tf.expand_dims(input_points, axis=1)
            else:
                grouped_points = None
        else:
            # FPS sampling
            fps_indices = farthest_point_sample(input_xyz, npoint)
            new_xyz = tf.gather(input_xyz, fps_indices, batch_dims=1)
            
            # Ball query grouping
            group_indices = query_ball_point(radius, nsample, input_xyz, new_xyz)
            grouped_xyz = group_points(input_xyz, group_indices)
            grouped_xyz -= tf.expand_dims(new_xyz, axis=2)  # Center points
            
            if input_points is not None:
                grouped_points = group_points(input_points, group_indices)
            else:
                grouped_points = None
        
        # Feature learning
        if input_points is not None:
            new_points = tf.concat([grouped_xyz, grouped_points], axis=-1)
        else:
            new_points = grouped_xyz
        
        new_points = mlp_block(new_points, mlp, name=f'{name}_mlp')
        
        if mlp2 is not None:
            new_points = mlp_block(new_points, mlp2, name=f'{name}_mlp2')
        
        # Max pooling
        new_points = tf.reduce_max(new_points, axis=2)
        
        return new_xyz, new_points

def feature_propagation_module(xyz1, xyz2, points1, points2, mlp, name=None):
    """Functional Feature Propagation module."""
    with tf.name_scope(name):
        # Compute distances
        dist = tf.norm(tf.expand_dims(xyz1, 2) - tf.expand_dims(xyz2, 1), axis=-1)
        dist = tf.maximum(dist, 1e-10)
        
        # Find 3 nearest neighbors
        _, knn_indices = tf.nn.top_k(-dist, k=3)
        knn_dist = tf.gather_nd(dist, knn_indices, batch_dims=1)
        
        # Compute interpolation weights
        weights = 1.0 / knn_dist
        weights /= tf.reduce_sum(weights, axis=2, keepdims=True)
        
        # Interpolate features
        interpolated_points = tf.reduce_sum(
            tf.gather_nd(points2, knn_indices, batch_dims=1) * tf.expand_dims(weights, -1),
            axis=2)
        
        if points1 is not None:
            new_points = tf.concat([interpolated_points, points1], axis=-1)
        else:
            new_points = interpolated_points
        
        return mlp_block(new_points, mlp, name=f'{name}_mlp')

def build_pointnet2_classifier(num_classes, num_points=1024):
    """Build PointNet++ classifier using functional API."""
    # Input layer
    input_points = Input(shape=(num_points, 3), name='input_points')
    
    # Set Abstraction layers
    l0_xyz = input_points
    l0_points = None
    
    # SA1
    l1_xyz, l1_points = set_abstraction_module(l0_xyz, l0_points, npoint=512, radius=0.2, nsample=32, mlp=[64, 64, 128], name='sa1')
    
    # SA2
    l2_xyz, l2_points = set_abstraction_module(l1_xyz, l1_points, npoint=128, radius=0.4, nsample=64,mlp=[128, 128, 256], name='sa2')
    
    # SA3 (global features)
    l3_xyz, l3_points = set_abstraction_module(l2_xyz, l2_points, npoint=None, radius=None, nsample=None, mlp=[256, 512, 1024], group_all=True, name='sa3')
    
    # Fully connected layers
    x = layers.Reshape((1024,))(l3_points)
    x = mlp_block(x, [512, 256], dropout_rate=0.5, name='fc')
    
    # Output layer
    outputs = tf.keras.layers.Dense(num_classes, kernel_initializer='normal')(x)
    return Model(inputs=input_points, outputs=outputs, name='pointnet2_cls')


def build_pointnet2_segmenter(num_classes, num_points=1024):
    """Build PointNet++ segmenter using functional API."""
    # Input layer
    input_points = Input(shape=(num_points, 3), name='input_points')
    
    # Set Abstraction layers
    l0_xyz = input_points
    l0_points = None
    
    # SA1
    l1_xyz, l1_points = set_abstraction_module(
        l0_xyz, l0_points, npoint=512, radius=0.1, nsample=32,
        mlp=[32, 32, 64], name='sa1')
    
    # SA2
    l2_xyz, l2_points = set_abstraction_module(
        l1_xyz, l1_points, npoint=128, radius=0.2, nsample=64,
        mlp=[64, 64, 128], name='sa2')
    
    # SA3 (global features)
    l3_xyz, l3_points = set_abstraction_module(
        l2_xyz, l2_points, npoint=None, radius=None, nsample=None,
        mlp=[128, 256, 1024], group_all=True, name='sa3')
    
    # Feature Propagation layers
    l2_points = feature_propagation_module(
        l2_xyz, l3_xyz, l2_points, l3_points, [256, 256], name='fp1')
    
    l1_points = feature_propagation_module(
        l1_xyz, l2_xyz, l1_points, l2_points, [256, 128], name='fp2')
    
    l0_points = feature_propagation_module(
        l0_xyz, l1_xyz, l0_points, l1_points, [128, 128, 128], name='fp3')
    
    # Segmentation head
    x = mlp_block(l0_points, [128, 128], name='seg_head')
    outputs = layers.Dense(num_classes, activation='softmax', name='outputs')(x)
    
    return Model(inputs=input_points, outputs=outputs, name='pointnet2_seg')