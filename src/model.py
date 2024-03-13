import os
from keras.optimizers.legacy import Adam # Use legacy version of Adam to avoid warnings
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import MultiLabelBinarizer
from keras.layers import Input, Embedding, Dense, Conv1D, Dropout, Conv1D, MaxPooling1D, Flatten
from keras.models import Model, load_model
import numpy as np
import json

# Parameters
epochs = 1
time_pred_threshold = 0.8 # How confident the model has to be to predict a time
batch_size = 32
model_path = './model/model.keras'
max_sequence_length = None # The maximum sentence length of the input data
train_test_split = 0.8 # 80% train, 20% test

def load_data():
    with open('./dataset/preprocessed_data.json') as f:
        dataset = json.load(f)

    tasks = [item['task'] for item in dataset]
    locations = ['home', 'work', 'public']
    times = ['wd-evening', 'wd-morning', 'we-anytime', 'wd-anytime', 'we-evening', 'we-morning', 'we-afternoon', 'wd-afternoon', 'wd-night', 'we-night']

    location_labels = [locations.index(item['location'][0]) for item in dataset]
    time_labels = [[times.index(time) for time in item['time']] for item in dataset]

    tokenizer = Tokenizer(num_words=1000)
    tokenizer.fit_on_texts(tasks)
    sequences = tokenizer.texts_to_sequences(tasks)
    max_sequence_length = int(max(len(seq) for seq in sequences))
    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)
    

    mlb = MultiLabelBinarizer()
    time_labels = mlb.fit_transform(time_labels)

    split_index = int(len(padded_sequences) * train_test_split)  # 80% train, 20% test
    train_data, test_data = padded_sequences[:split_index], padded_sequences[split_index:]
    train_location_labels, test_location_labels = np.array(location_labels)[:split_index], np.array(location_labels)[split_index:]
    train_time_labels, test_time_labels = time_labels[:split_index], time_labels[split_index:]

    return (train_data, train_location_labels, train_time_labels, test_data, test_location_labels, test_time_labels, tokenizer, locations, times, max_sequence_length)

def build_model(input_shape, num_locations, num_times, learning_rate, dropout_rate, num_filters, kernel_size, embedding_dim):

    # Input Layer
    input_layer = Input(shape=input_shape,)

    # Embedding Layer
    embedding_layer = Embedding(input_dim=1000,  
                                output_dim=embedding_dim,
                                input_length=max_sequence_length)(input_layer)

    # Convolutional Layer
    conv_layer = Conv1D(filters=num_filters,
                        kernel_size=kernel_size,
                        activation='relu')(embedding_layer)

    # MaxPooling Layer
    max_pooling_layer = MaxPooling1D(pool_size=2)(conv_layer)

    # Flatten Layer
    flatten_layer = Flatten()(max_pooling_layer)

    # Dropout Layer
    dropout_layer = Dropout(dropout_rate)(flatten_layer)

    # Dense Layer for each output
    location_output = Dense(num_locations, activation='softmax', name='location_output')(dropout_layer)
    time_output = Dense(num_times, activation='sigmoid', name='time_output')(dropout_layer)

    # Model Assembly
    model = Model(inputs=input_layer, outputs=[location_output, time_output])

    # Compile Model
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                loss={'location_output': 'sparse_categorical_crossentropy', 'time_output': 'binary_crossentropy'},
                metrics={'location_output': 'accuracy', 'time_output': 'accuracy'})

    return model

def load_saved_model():
    print("Loading model...")
    model = load_model(model_path)
    _, _, _, _, _, _, tokenizer, locations, times, max_sequence_length = load_data()
    return model, tokenizer, locations, times, max_sequence_length
        
# def train_model(learning_rate, dropout_rate, num_filters, kernel_size, embedding_dim):
#     print("Training a new model...")

#     train_data, train_location_labels, train_time_labels, test_data, test_location_labels, test_time_labels, tokenizer, locations, times, max_sequence_length = load_data()
#     model = build_model(max_sequence_length, len(locations), len(times), learning_rate, dropout_rate, num_filters, kernel_size, embedding_dim)
#     model.fit(train_data, {'location_output': train_location_labels, 'time_output': train_time_labels},
#             batch_size=batch_size, epochs=epochs, validation_data=(test_data, {'location_output': test_location_labels, 'time_output': test_time_labels}))
#     model.save(model_path)
    
#     print("Model trained and saved.")
#     return model, tokenizer, locations, times, max_sequence_length


# def evaluate_model(hyperparameters):
#     # Unpack the hyperparameters
#     learning_rate, dropout_rate, num_filters, kernel_size, embedding_dim = hyperparameters
    
#     # Load and preprocess the data (assuming a function exists for this)
#     train_data, train_location_labels, train_time_labels, test_data, test_location_labels, test_time_labels, tokenizer, locations, times, max_sequence_length = load_data()

#     # Build the model with the given hyperparameters
#     model = build_model(input_shape=max_sequence_length, 
#                         num_locations=len(locations),  
#                         num_times=len(times),
#                         learning_rate=learning_rate, 
#                         dropout_rate=dropout_rate, 
#                         num_filters=num_filters, 
#                         kernel_size=kernel_size, 
#                         embedding_dim=embedding_dim)
    
#     # Train the model (consider adding validation_split to model.fit for internal validation)
#     model.fit(train_data, {'location_output': train_location_labels, 'time_output': train_time_labels},
#             batch_size=batch_size, epochs=epochs, validation_data=(test_data, {'location_output': test_location_labels, 'time_output': test_time_labels}))
    
#     # Evaluate the model on the test set
#     evaluation_results = model.evaluate(test_data, {'location_output': test_location_labels, 'time_output': test_time_labels}, batch_size=batch_size)
    
#     # Assuming evaluation_results contains loss and accuracy, and you want to maximize accuracy
#     accuracy = evaluation_results[1]  # This index might change based on how your model's metrics are structured
    
#     # Return the negative accuracy because the optimizer seeks to minimize the objective function
#     return -accuracy

def train_and_evaluate_model(hyperparameters):
    learning_rate, dropout_rate, num_filters, kernel_size, embedding_dim = hyperparameters

    train_data, train_location_labels, train_time_labels, test_data, test_location_labels, test_time_labels, tokenizer, locations, times, max_sequence_length = load_data()

    kernel_size = int(kernel_size)
    num_filters = int(num_filters)
    embedding_dim = int(embedding_dim)
    
    model = build_model(max_sequence_length, len(locations), len(times), learning_rate, dropout_rate, num_filters, kernel_size, embedding_dim)
    model.fit(train_data, {'location_output': train_location_labels, 'time_output': train_time_labels},
            batch_size=batch_size, epochs=epochs, validation_data=(test_data, {'location_output': test_location_labels, 'time_output': test_time_labels}))
    
    results = model.evaluate(test_data, {'location_output': test_location_labels, 'time_output': test_time_labels}, batch_size=batch_size)
    location_accuracy = results[3]
    time_accuracy = results[4]

    score = (location_accuracy + time_accuracy) / 2

    model.save(model_path)
    
    print("Model trained and saved.")
    return score


def prediction(model, tokenizer, locations, times, max_sequence_length, input_sentence):
    sequence = tokenizer.texts_to_sequences([input_sentence])
    padded_sequence = pad_sequences(sequence, maxlen=max_sequence_length)

    predictions = model.predict(padded_sequence, batch_size=1)
    location_pred, time_pred = predictions

    location_index = np.argmax(location_pred, axis=1)[0]
    predicted_location = locations[location_index]

    predicted_times = [times[idx] for idx, pred in enumerate(time_pred[0]) if pred > time_pred_threshold]

    print(f"Predicted Location: {predicted_location}")
    print(f"Predicted Times: {predicted_times}")

def main():

    if os.path.exists(model_path):
        print("A trained model already exists.")

    # user_input = input("Type 'load' to load the model or 'retrain' to train a new model: ").lower()
    # if user_input == 'load':
    #     model, tokenizer, locations, times, max_sequence_length = load_saved_model()
    #
        
    hyperparameters = [0.001, 0.1, 64, 3, 100]
    train_and_evaluate_model(hyperparameters)

    
    test_sentence = "dining room windows"
    # prediction(model, tokenizer, locations, times, max_sequence_length, test_sentence)

if __name__ == '__main__':
    main()
