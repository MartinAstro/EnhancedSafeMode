import tensorflow as tf


def generate_network(input_dim, output_dim, num_layers, num_units, name):
    activation = 'relu'
    initializer = 'glorot_uniform'
    dtype = tf.float32

    inputs = tf.keras.Input(shape=(input_dim,))
    for _ in range(num_layers):
        x = tf.keras.layers.Dense(
            units=num_units,
            activation=activation,
            kernel_initializer=initializer,
            dtype=dtype,
        )(x)
    outputs = tf.keras.layers.Dense(
        units=output_dim,
        activation="linear",
        kernel_initializer=initializer,
        dtype=dtype,
    )(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name=name)

def main():
    mu = 5E18
    radius = 17E3

    act_abs_size = 10
    obs_abs_size = 10
    initial_state = [5E3, 0.0, 0.0]
    rel_state = [5E3, 0.0, 0.0, 0.01, 0.01, 0.01]
    impulse = [0.0,0.0,0.0]
    env_state = [mu, radius]
    belief = tf.Variable(initial_value=initial_state, trainable=False, name='belief')
    action = tf.Variable(initial_value=impulse, trainable=False, name='action')
    observation = tf.Variable(initial_value=rel_state, trainable=False, name='observation')
    env_params = tf.Variable(initial_value=env_state, trainable=False, name='env_params')
    f_A = generate_network(input_dim=tf.shape(action)[0], output_dim=act_abs_size, num_layers=8, num_units=20, name='f_A')
    f_T = generate_network(input_dim=tf.shape(belief)[0], output_dim=f_A.output.shape[1], num_layers=8, num_units=20, name='f_T')
    f_Tp = generate_network(input_dim=tf.shape(belief)[0], output_dim=f_A.output.shape[1], num_layers=8, num_units=20, name='f_Tp')
    f_Z = generate_network(input_dim=tf.shape(env_params)[0], output_dim=obs_abs_size, num_layers=8, num_units=20, name='f_Z')
    f_O = generate_network(input_dim=tf.shape(observation)[0], output_dim=f_Z.output.shape[1], num_layers=8, num_units=20, name='f_O')
    f_R = generate_network(input_dim=tf.shape(belief)[0], output_dim=f_A.output.shape[1], num_layers=8, num_units=20, name='f_R')
    

if __name__ == "__main__":
    main()