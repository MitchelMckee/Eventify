import os
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import MultiLabelBinarizer
from keras.layers import Input, Embedding, GlobalAveragePooling1D, Dense, Conv1D, GlobalMaxPooling1D, Dropout
from keras.models import Model, load_model
import numpy as np
import json

# Parameters
epochs = 10
batch_size = 32
time_pred_threshold = 0.8
model_path = './model/model.h5'

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
    padded_sequences = pad_sequences(sequences, maxlen=100)

    mlb = MultiLabelBinarizer()
    time_labels = mlb.fit_transform(time_labels)

    split_index = int(len(padded_sequences) * 0.8)  # 80% train, 20% test
    train_data, test_data = padded_sequences[:split_index], padded_sequences[split_index:]
    train_location_labels, test_location_labels = np.array(location_labels)[:split_index], np.array(location_labels)[split_index:]
    train_time_labels, test_time_labels = time_labels[:split_index], time_labels[split_index:]

    return (train_data, train_location_labels, train_time_labels, test_data, test_location_labels, test_time_labels, tokenizer, locations, times)

def build_model(input_shape, num_locations, num_times):
    input_layer = Input(shape=(input_shape,))

    embedding = Embedding(input_dim=1000, output_dim=16)(input_layer)
    conv1d = Conv1D(filters=128, kernel_size=5, activation='relu')(embedding)
    global_max_pooling = GlobalMaxPooling1D()(conv1d)

    location_output = Dense(num_locations, activation='softmax', name='location_output')(global_max_pooling)
    time_output = Dense(num_times, activation='sigmoid', name='time_output')(global_max_pooling)  # Sigmoid for multi-label

    model = Model(inputs=input_layer, outputs=[location_output, time_output])
    model.compile(optimizer='adam',
                loss={'location_output': 'sparse_categorical_crossentropy',
                        'time_output': 'binary_crossentropy'},
                metrics={'location_output': 'accuracy', 'time_output': 'accuracy'})
    return model

def train_or_load_model():
    if os.path.exists(model_path):
        print("Found a saved model.")
        user_input = input("Type 'load' to load the model or 'retrain' to train a new model: ").lower()
        if user_input == 'load':
            print("Loading model...")
            return load_model(model_path)
    # If retrain
    print("Training a new model...")
    train_data, train_location_labels, train_time_labels, test_data, test_location_labels, test_time_labels, tokenizer, locations, times = load_data()
    model = build_model(100, len(locations), len(times))
    model.fit(train_data, {'location_output': train_location_labels, 'time_output': train_time_labels},
              epochs=epochs, validation_data=(test_data, {'location_output': test_location_labels, 'time_output': test_time_labels}))
    model.save(model_path)
    print("Model trained and saved.")
    return model, tokenizer, locations, times

def prediction(model, tokenizer, locations, times, input_sentence="visit mark"):
    sequence = tokenizer.texts_to_sequences([input_sentence])
    padded_sequence = pad_sequences(sequence, maxlen=100)

    predictions = model.predict(padded_sequence)
    location_pred, time_pred = predictions

    location_index = np.argmax(location_pred, axis=1)[0]
    predicted_location = locations[location_index]

    predicted_times = [times[idx] for idx, pred in enumerate(time_pred[0]) if pred > time_pred_threshold]

    print(f"Predicted Location: {predicted_location}")
    print(f"Predicted Times: {predicted_times}")

def main():
    model, tokenizer, locations, times = train_or_load_model()
    prediction(model, tokenizer, locations, times)

if __name__ == '__main__':
    main()
