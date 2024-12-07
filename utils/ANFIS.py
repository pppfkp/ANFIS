import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class ANFISMembershipFunctionLayer(nn.Module):
    def __init__(self, input_dim, num_membership_functions, mean = None, sigma = None):
        super(ANFISMembershipFunctionLayer, self).__init__()

        self.mean = nn.Parameter(torch.rand(input_dim, num_membership_functions)) if mean == None else nn.Parameter(mean)
        self.sigma = nn.Parameter(torch.rand(input_dim, num_membership_functions)) if sigma == None else nn.Parameter(sigma)

    def gauss(self, x, mean, sigma):
        return torch.exp(-(((x - mean) / sigma) ** 2))
    
    def forward(self, x):
        membership_functions = self.gauss(x.unsqueeze(2), self.mean, self.sigma)
        return membership_functions
    
class ANFISRuleStrengthLayer(nn.Module):
    def __init__(self, number_of_inputs, number_of_membership_functions):
        super(ANFISRuleStrengthLayer, self).__init__()
        self.number_of_inputs = number_of_inputs
        self.number_of_membership_functions = number_of_membership_functions
        self.consequent_layer_size = number_of_membership_functions**number_of_inputs

    def forward(self, x):
        batch_size = x.size(0)

        firing_strengths = []

        for input_number in range(batch_size):
            firing_strength = torch.zeros(self.consequent_layer_size)
            index = 0

            for i1 in range(self.number_of_inputs):
                for j1 in range(self.number_of_membership_functions):
                    for i2 in range(i1 + 1, self.number_of_inputs):
                        for j2 in range(self.number_of_membership_functions):
                            firing_strength[index] = torch.mul(x[input_number][i1][j1], x[input_number][i2][j2])
                            index += 1

            firing_strengths.append(firing_strength)

        firing_strengths = torch.stack(firing_strengths)
        return firing_strengths

class ANFISRuleStrengthNormalizationLayer(nn.Module):
    def __init__(self):
        super(ANFISRuleStrengthNormalizationLayer, self).__init__()

    
    def forward(self, x):
        firing_strengths_sum = torch.sum(x, dim=1)
        normalized_firing_strengths = x / firing_strengths_sum.unsqueeze(1)

        return normalized_firing_strengths
    
class ANFISConsequentLayer(nn.Module):
    def __init__(self, consequent_layer_size, number_of_inputs, consequent_weights=None, consequent_biases=None):
        super(ANFISConsequentLayer, self).__init__()

        self.consequent_weights = nn.Parameter(torch.rand(consequent_layer_size, number_of_inputs)) if consequent_weights == None else nn.Parameter(consequent_weights)
        self.consequent_biases = nn.Parameter(torch.rand(consequent_layer_size, 1)) if consequent_biases == None else nn.Parameter(consequent_biases)

    def forward(self, x, plain_input):
        outputs = plain_input.unsqueeze(1) * self.consequent_weights.unsqueeze(0)
        outputs = torch.sum(outputs, dim=2).unsqueeze(2)
        outputs = outputs + self.consequent_biases.unsqueeze(0)
        outputs = outputs.squeeze()

        outputs_weighted_by_strengths = outputs * x

        return outputs_weighted_by_strengths

    
class ANFISOutputSingle(nn.Module):
    def __init__(self, consequent_layer_size):
        super(ANFISOutputSingle, self).__init__()

        self.consequent_layer_size = consequent_layer_size

    def forward(self, x):
        summed_output = torch.sum(x, dim=1)
        return summed_output
    
class ANFIS(nn.Module):
    def __init__(self, number_of_features=2, number_of_membership_functions=3, number_of_outputs=1, mean=None, sigma=None, consequent_weights=None, consequent_biases=None):
        super(ANFIS, self).__init__()
        self.number_of_features = number_of_features
        self.consequent_layer_size = number_of_membership_functions**number_of_features
        self.membership_function_layer = ANFISMembershipFunctionLayer(number_of_features, number_of_membership_functions, mean, sigma)
        self.rule_strength_layer = ANFISRuleStrengthLayer(number_of_features, number_of_membership_functions)
        self.rule_strength_normalization_layer = ANFISRuleStrengthNormalizationLayer()
        self.consequent_layer = ANFISConsequentLayer(self.consequent_layer_size, number_of_features, consequent_weights, consequent_biases)
        self.output = ANFISOutputSingle(self.consequent_layer_size)
        

    def forward(self, x):
        membership_functions = self.membership_function_layer(x)
        rule_strengths = self.rule_strength_layer(membership_functions)
        normalized_rule_strengths = self.rule_strength_normalization_layer(rule_strengths)
        consequent_outputs = self.consequent_layer(normalized_rule_strengths, x)
        output = self.output(consequent_outputs)
        return output.unsqueeze(-1)
    
    def gauss(self, x, mean, sigma):
        return torch.exp(-(((x - mean) / sigma) ** 2))
    
    def plot_gaussian_with_values(self, linspace=torch.linspace(-3,3,1000)):
        def gauss(x, mean, sigma):
            return torch.exp(-(((x - mean) / sigma) ** 2))
    
        means = self.membership_function_layer.mean
        sigmas = self.membership_function_layer.sigma
        for input_number in range(self.number_of_features):
            # Plot each membership function for the specified input
            for i in range(means.size(1)):
                y = gauss(linspace, means[input_number, i], sigmas[input_number, i])
                plt.plot(linspace.numpy(), y.detach().numpy(), label=f'MF {i+1}')

            plt.title(f'Gaussian Membership Functions for Input {input_number + 1}')
            plt.xlabel('x')
            plt.ylabel('MF value')
            plt.legend()
            plt.grid(True)
            plt.show()

    

            plt.show()

    
