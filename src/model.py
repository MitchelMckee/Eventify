import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import MultiLabelBinarizer
from keras.layers import Input, Embedding, Dense, Conv1D, GlobalMaxPooling1D, Dropout
from keras.models import Model, load_model
import numpy as np
import json

# Parameters
epochs = 10
time_pred_threshold = 0.8
batch_size = 32
model_path = './model/model.keras'
max_sequence_length = None # The maximum sentence length of the input data

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
    max_sequence_length = max(len(seq) for seq in sequences)
    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)
    

    mlb = MultiLabelBinarizer()
    time_labels = mlb.fit_transform(time_labels)

    split_index = int(len(padded_sequences) * 0.6)  # 80% train, 20% test
    train_data, test_data = padded_sequences[:split_index], padded_sequences[split_index:]
    train_location_labels, test_location_labels = np.array(location_labels)[:split_index], np.array(location_labels)[split_index:]
    train_time_labels, test_time_labels = time_labels[:split_index], time_labels[split_index:]

    return (train_data, train_location_labels, train_time_labels, test_data, test_location_labels, test_time_labels, tokenizer, locations, times, max_sequence_length)

def build_model(input_shape, num_locations, num_times):
    input_layer = Input(shape=(input_shape,))

    embedding = Embedding(input_dim=1000, output_dim=16)(input_layer)
    conv1d = Conv1D(filters=128, kernel_size=5, activation='relu')(embedding)
    global_max_pooling = GlobalMaxPooling1D()(conv1d)
    dropout = Dropout(0.5)(global_max_pooling)
    layers = dropout

    location_output = Dense(num_locations, activation='softmax', name='location_output')(layers)
    time_output = Dense(num_times, activation='sigmoid', name='time_output')(layers)  # Sigmoid for multi-label

    model = Model(inputs=input_layer, outputs=[location_output, time_output])
    model.compile(optimizer='adam',
                loss={'location_output': 'sparse_categorical_crossentropy',
                        'time_output': 'binary_crossentropy'},
                metrics={'location_output': 'accuracy', 'time_output': 'accuracy'})
    return model

def train_or_load_model():
    load_data()
    if os.path.exists(model_path):
        print("Found a saved model.")
        user_input = input("Type 'load' to load the model or 'retrain' to train a new model: ").lower()
        if user_input == 'load':
            print("Loading model...")
            model = load_model(model_path)
            _, _, _, _, _, _, tokenizer, locations, times, max_sequence_length = load_data()
            return model, tokenizer, locations, times, max_sequence_length
    # If retrain
    print("Training a new model...")
    train_data, train_location_labels, train_time_labels, test_data, test_location_labels, test_time_labels, tokenizer, locations, times, max_sequence_length = load_data()
    model = build_model(max_sequence_length, len(locations), len(times))
    model.fit(train_data, {'location_output': train_location_labels, 'time_output': train_time_labels},
            batch_size=batch_size, epochs=epochs, validation_data=(test_data, {'location_output': test_location_labels, 'time_output': test_time_labels}))
    model.save(model_path)
    print("Evaluating model on test set...")
    results = model.evaluate(test_data, {'location_output': test_location_labels, 'time_output': test_time_labels}, batch_size=batch_size)

    print(model.metrics_names)
    print(f"Location Accuracy: {results[3]*100:.2f}%")  
    print(f"Time Accuracy: {results[4]*100:.2f}%") 

    print("Model trained and saved.")
    return model, tokenizer, locations, times, max_sequence_length

def prediction(model, tokenizer, locations, times, max_sequence_length, input_sentence="dining room windows"):
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
    model, tokenizer, locations, times, max_sequence_length = train_or_load_model()
    prediction(model, tokenizer, locations, times, max_sequence_length)

if __name__ == '__main__':
    main()
