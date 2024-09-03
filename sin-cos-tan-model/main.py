from torch import nn, optim, tensor

import math

import matplotlib.pyplot as plot


# data model. handles all the necessary data related funtions
class DataModel:
    def __init__(self) -> None:
        pass

    def create_sin_input_outputs(self,
                                range_to_generate:int,
                                divide_by:int):

        the_range = range(0, range_to_generate+1)

        inputs = []
        outputs = []

        for integer in the_range:
            if divide_by != 0:
                inputs.append(integer/divide_by)
                outputs.append(math.sin(integer/divide_by))
            else:
                inputs.append(integer)
                outputs.append(math.sin(integer))

        return inputs, outputs
    
    def create_cos_input_outputs(self,
                                range_to_generate:int,
                                divide_by:int):

        the_range = range(0, range_to_generate+1)

        inputs = []
        outputs = []

        for integer in the_range:
            if divide_by != 0:
                inputs.append(integer/divide_by)
                outputs.append(math.cos(integer/divide_by))
            else:
                inputs.append(integer)
                outputs.append(math.cos(integer))

        return inputs, outputs
    
    def create_tan_input_outputs(self,
                                range_to_generate:int,
                                divide_by:int):

        the_range = range(0, range_to_generate+1)

        inputs = []
        outputs = []

        for integer in the_range:
            if divide_by != 0:
                inputs.append(integer/divide_by)
                outputs.append(math.tan(integer/divide_by))
            else:
                inputs.append(integer)
                outputs.append(math.tan(integer))

        return inputs, outputs
    
    def plot_data(self, datax:list,
                        datay:list,
                        ai_datay:list):
        
        plot.plot(datax, datay, color='green', label='Original')
        plot.plot(datax, ai_datay, color='red', label='Prediction')

        plot.legend(["Original", "Prediction"])

        plot.show()
    
    def plot_normal(self,
                    datax:list,
                    datay:list):
        
        plot.plot(datax, datay, color='green', label=f'Original: {function_to_train} function')
        plot.show()


# main neural network model.
class MainModel(nn.Module):
    def __init__(self,
                 inputs:int,
                 hidden:int,
                 outputs:int) -> None:
        super().__init__()

        self.input_neurons = nn.Linear(inputs, hidden)
        self.hidden_neurons = nn.Linear(hidden, hidden)
        self.output_neurons = nn.Linear(hidden, outputs)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.input_neurons(x)
        x = self.relu(x)

        x = self.hidden_neurons(x)
        x = self.relu(x)

        x = self.hidden_neurons(x)
        x = self.relu(x)

        x = self.hidden_neurons(x)
        x = self.relu(x)

        x = self.output_neurons(x)

        return x
    
# the function to train the model on
function_to_train = 'cos'
# initialize the variables
input_size = 1
hidden_size = 35
output_size = 1
learning_rate = 0.0001
epochs = 500

# create the model
model = MainModel(inputs=input_size,
                  hidden=hidden_size,
                  outputs=output_size)

data_model = DataModel()

# create the input and output data
if function_to_train == 'sin':
    data_in, data_out = data_model.create_sin_input_outputs(100, 10)
elif function_to_train == 'cos':
    data_in, data_out = data_model.create_cos_input_outputs(100, 10)
elif function_to_train == 'tan':
    data_in, data_out = data_model.create_tan_input_outputs(100, 10)

# create optimizer and loss functions
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
losser = nn.MSELoss()

if __name__ == "__main__":
    # initialize an empty list (filled with zeros so we can replace them later with predicted values)
    predictions = [0 for i in range(len(data_out))]
    # train for epochs
    for e in range(epochs):
        for x in data_in:
            # convert integer to 1D tensor
            x = tensor([x])

            # get the actual value for y, change that into tensor aswell
            index_for_y = data_in.index(x)

            y_actual = tensor([data_out[index_for_y]])

            # pass x to get y from model
            y = model(x)

            # calculate loss
            loss = losser(y, y_actual)

            # back propagate
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # replace the last predicted value for x with new y for x
            predictions[index_for_y] = y.tolist()[0]

        print(f"Epoch {e+1}/{epochs}: completed. Loss: {loss}")

    data_model.plot_data(
        data_in,
        data_out,
        predictions
    )