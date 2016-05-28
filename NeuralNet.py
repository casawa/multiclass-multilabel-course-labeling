from keras.layers import Input, Dense
from keras.models import Model

OUTPUT_DIM = 40

def main():
    
    inputs = Inputs(shape=(,))

    first_layer = Dense(OUTPUT_DIM, activation='relu')(inputs)
    second_layer = Dense(OUTPUT_DIM, activation='relu')(first_layer)
    predictions = Dense(OUTPUT_DIM, activation='softmax')(second_layer)

    model = Model(input=inputs, output=predictions)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(data, labels)



if __name__ == '__main__':
    main()

