"""Train and test a simple RNN for language detection."""
import torch, torch.nn as nn, time, math, random, string
from prep import get_data, get_data_test, all_categories

torch.manual_seed(2)
all_letters = string.ascii_letters + " .,;'-"
n_letters = len(all_letters)
n_categories = len(all_categories)

class RNN(nn.Module):
    def __init__(self, n_letters, n_categories, hidden_size=56):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(n_letters + hidden_size, hidden_size)
        self.i2o = nn.Linear(n_letters + hidden_size, n_categories)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        return self.softmax(output), hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

def wtotensor(word):
    """
    Encode a word as a tensor using a standard alphabet (defined in all_letters)
    For example:
    Give our alphabet:
    abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ .,;'-

    Each lettter has a uniqur position:
    0 -> a
    1 -> b
    etc.
    15 -> o
    So, if we want to encode the word 'oro' we will encode each letter
    by including 1 in it's position and left the other positions as 0:

    oro->
                           o is in 15th position in the alphabet--V
tensor([[[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
          0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
          0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
          0., 0., 0., 0., 0., 0., 0.]],

         [[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
r in 18th->1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0.]],
                 and again o is in 15th position in the alphabet--V
        [[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
          0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
          0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
          0., 0., 0., 0., 0., 0., 0.]]])

    """
    tensor = torch.zeros(len(word), 1, n_letters)
    for li, letter in enumerate(word):
        tensor[li][0][all_letters.find(letter)] = 1
    return tensor

def random_value(d):
    """Get random value from dictionary"""
    return d[random.randint(0, len(d) - 1)]

def get_tensorw(all_categories, words, category=None, word=None):
    """Get category and word tensors (random if not specified)"""
    if category is None and word is None:
        category = random_value(all_categories)
        word = random_value(words[category])
    category_tensor = torch.LongTensor([all_categories.index(category)])
    return category, word, category_tensor, wtotensor(word)

def get_category(output, categories):
    """Return most probable category from output tensor"""
    top_n, top_i = output.data.topk(1)
    category_i = top_i[0][0]
    return categories[category_i], category_i

def train(rnn, optimizer, loss_function, w_epochs, categories, words):
    """Train RNN model on random words from each language"""
    print('Starting training...')
    current_loss, wordi = 0, 0
    stats_total = {}.fromkeys(categories, 0)
    
    for w_epoch in range(1, w_epochs + 1):
        wordi += 1
        category, word, category_tensor, word_tensor = get_tensorw(categories, words)
        stats_total[category] += 1
        
        hidden = rnn.initHidden()
        for i in range(word_tensor.size()[0]):
            output, hidden = rnn(word_tensor[i], hidden)

        loss = loss_function(output, category_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        current_loss += loss.data.item()
        if wordi % 1000 == 0:
            guess, _ = get_category(output, categories)
            msg = 'V' if guess == category else f'X ({category})'
            print(f'{w_epoch} {w_epoch/w_epochs*100:.0f}% {word.ljust(20)} {guess} {msg.ljust(8)} {current_loss/1000:.6f}')
            current_loss = 0.0
            
    print(f'Finished training on {wordi} words')
    for c in categories:
        print(f'Trained on {stats_total[c]} words for {c}')

def test(rnn, optimizer, categories, test_words):
    """Test model accuracy on unseen words"""
    stats_correct = {}.fromkeys(categories, 0)
    stats_total = {}.fromkeys(categories, 0)
    print('Starting testing...')
    
    with torch.no_grad():
        for cat in categories:
            for w in test_words[cat]:
                _, _, category_tensor, word_tensor = get_tensorw(categories, test_words, cat, w)
                hidden = rnn.initHidden()
                
                for i in range(word_tensor.size()[0]):
                    output, hidden = rnn(word_tensor[i], hidden)
                    
                guess, _ = get_category(output, categories)
                stats_total[cat] += 1
                if guess == cat:
                    stats_correct[cat] += 1
                    
        for c in categories:
            accuracy = 100 * stats_correct[c] / stats_total[c]
            print(f'Test accuracy for {c} on {stats_total[c]} ({stats_correct[c]} correct) words: {accuracy:.0f}%')

if __name__ == '__main__':
    rnn = RNN(n_letters, n_categories)
    optimizer = torch.optim.Adam(rnn.parameters())
    loss_function = nn.CrossEntropyLoss()
    
    print('Getting training data...')
    categories, train_words = get_data()
    train(rnn, optimizer, loss_function, 10000, categories, train_words)
    
    print('Getting test data...')
    test_categories, test_words = get_data_test(exclude_words=[train_words[c] for c in all_categories])
    test(rnn, optimizer, test_categories, test_words)
    torch.save(rnn.state_dict(), 'model.ckpt')