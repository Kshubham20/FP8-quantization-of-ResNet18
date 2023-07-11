#supplementary function
# def ZerO_Init_on_matrix(matrix_tensor):
#     # Algorithm 1 in the paper.
    
#     m = matrix_tensor.size(0)
#     n = matrix_tensor.size(1)
    
#     if m <= n:
#         init_matrix = torch.nn.init.eye_(torch.empty(m, n))
#     elif m > n:
#         clog_m = math.ceil(math.log2(m))
#         p = 2**(clog_m)
#         init_matrix = torch.nn.init.eye_(torch.empty(m, p)) @ (torch.tensor(hadamard(p)).float()/(2**(clog_m/2))) @ torch.nn.init.eye_(torch.empty(p, n))
    
#     return init_matrix

# def Identity_Init_on_matrix(matrix_tensor):
#     # Definition 1 in the paper
#     # See https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.eye_ for details. Preserves the identity of the inputs in Linear layers, where as many inputs are preserved as possible, the same as partial identity matrix.
    
#     m = matrix_tensor.size(0)
#     n = matrix_tensor.size(1)
    
#     init_matrix = torch.nn.init.eye_(torch.empty(m, n))
    
#     return init_matrix


def init_sub_identity_conv1x1(weight):
    tensor = weight.data
    out_dim = tensor.size()[0]
    in_dim = tensor.size()[1]
    ori_dim = tensor.size()
    assert tensor.size()[2] == 1 and tensor.size()[3] == 1
    if out_dim<in_dim:
        i = torch.eye(out_dim).type_as(tensor)
        j = torch.zeros(out_dim,(in_dim-out_dim)).type_as(tensor)
        k = torch.cat((i,j),1)
    elif out_dim>in_dim:
        i = torch.eye(in_dim).type_as(tensor)
        j = torch.zeros((out_dim-in_dim),in_dim).type_as(tensor)
        k = torch.cat((i,j),0)
    else:
        k = torch.eye(out_dim).type_as(tensor)
    k.unsqueeze_(2)
    k.unsqueeze_(3)
    assert k.size() == ori_dim
    
    weight.data = k

class Hadamard_Transform(nn.Module):

    def __init__(self, dim_in, dim_out):
        super(Hadamard_Transform, self).__init__()
        if dim_in != dim_out:
            raise RuntimeError('orthogonal transform not supports dim_in != dim_out currently')
        hadamard_matrix = hadamard(dim_in)
        hadamard_matrix = torch.Tensor(hadamard_matrix)

        n = int(np.log2(dim_in))
        normalized_hadamard_matrix = hadamard_matrix / (2**(n / 2))

        self.hadamard_matrix = nn.Parameter(normalized_hadamard_matrix, requires_grad=False)


    def forward(self, x):
        # input is a B x C x N x M
        
        return torch.matmul(x.permute(0,2,3,1), self.hadamard_matrix).permute(0,3,1,2)
class SkipConnection(nn.Module):

    def __init__(self, scale=1):
        super(SkipConnection, self).__init__()
        self.scale = scale
    def _shortcut(self, input):
        #needs to be implemented

        return input

    def forward(self, x):
        # with torch.no_grad():
        identity = self._shortcut(x)
        return identity * self.scale

class ChannelPaddingSkip(SkipConnection):

    def __init__(self, num_expand_channels_left, num_expand_channels_right, scale=1):
        super(ChannelPaddingSkip, self).__init__(scale)
        self.num_expand_channels_left = num_expand_channels_left
        self.num_expand_channels_right = num_expand_channels_right
    
    def _shortcut(self, input):
        # input is (N, C, H, M)
        # and return is (N, C + num_left + num_right, H, M)
        
        return F.pad(input, (0, 0, 0, 0, self.num_expand_channels_left, self.num_expand_channels_right) , "constant", 0) 
class Zero_Relu(Function):
        
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = input.clamp(min=0)
        return output    
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input
    
zero_relu = Zero_Relu.apply

def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

# def init_params(net):
#     '''Init layer parameters.'''
#     for m in net.modules():
#         if isinstance(m, nn.Conv2d):
#             init.kaiming_normal(m.weight, mode='fan_out')
#             if m.bias:
#                 init.constant(m.bias, 0)
#         elif isinstance(m, nn.BatchNorm2d):
#             init.constant(m.weight, 1)
#             init.constant(m.bias, 0)
#         elif isinstance(m, nn.Linear):
#             init.normal(m.weight, std=1e-3)
#             if m.bias:
#                 init.constant(m.bias, 0)


# _, term_width = os.popen('stty size', 'r').read().split()
# term_width = int(term_width)
term_width=80

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f
# dataset loader 
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

#training code

def train(epoch,model): 
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    print('Train Loss:',(train_loss/(batch_idx+1)), 'Acc: ',100.*correct/total,'correct',correct,'total',total)


def test(epoch,model):
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
        print('Test Loss:',(test_loss/(batch_idx+1)),'Acc:',100.*correct/total,'Correct',correct,'total',total)

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        # print('Saving..')
        # state = {
        #     'net': net.state_dict(),
        #     'acc': acc,
        #     'epoch': epoch,
        # }
        # if not os.path.isdir('checkpoint'):
        #     os.mkdir('checkpoint')
        # torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc
    print(best_acc)
#training
# parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
# parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
# parser.add_argument('--resume', '-r', action='store_true',
#                     help='resume from checkpoint')
# args = parser.parse_args()

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
# Model
print('==> Building model..')
net = identity_resnet18()
net = net.to(device)
# if device == 'cuda':
#     net = torch.nn.DataParallel(net)
#     cudnn.benchmark = True

# if args.resume:
#     # Load checkpoint.
#     print('==> Resuming from checkpoint..')
#     assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
#     checkpoint = torch.load('./checkpoint/ckpt.pth')
#     net.load_state_dict(checkpoint['net'])
#     best_acc = checkpoint['acc']
#     start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


for epoch in range(start_epoch, start_epoch+100):
    train(epoch,net)
    test(epoch,net)
    scheduler.step()