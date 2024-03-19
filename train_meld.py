import numpy as np, argparse, time, pickle, random, math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import functional as F
from dataloader import MELDRobertaCometDataset
from model import MaskedNLLLoss
from commonsense_model import CommonsenseGRUModel
from sklearn.metrics import f1_score, accuracy_score
from losses2 import ESCL, CKSCL

import warnings
warnings.filterwarnings("ignore")

lamda=torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True).cuda() #the size ilamda.data.fill_(0.5)

def create_class_weight(mu=1):
    unique = [0, 1, 2, 3, 4, 5, 6]
    labels_dict = {0: 6436, 1: 1636, 2: 358, 3: 1002, 4: 2308, 5: 361, 6: 1607}        
    total = np.sum(list(labels_dict.values()))
    weights = []
    for key in unique:
        score = math.log(mu*total/labels_dict[key])
        weights.append(score)
    return weights

def create_class_weight_SCL(label):
    unique = [0, 1]
    one = sum(label)
    labels_dict = {0 : len(label) - one, 1: one}
    total = sum(list(labels_dict.values()))
    weights = []
    for key in unique:
        score = math.log(total/labels_dict[key])
        weights.append(score)
    return weights

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_MELD_loaders(batch_size=32, classify='emotion', num_workers=0, pin_memory=False):
    trainset = MELDRobertaCometDataset('train', classify)
    validset = MELDRobertaCometDataset('valid', classify)
    testset = MELDRobertaCometDataset('test', classify)

    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    valid_loader = DataLoader(validset,
                              batch_size=batch_size,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader

def train_or_eval_model2(model, model2, model3, loss_function, ESCL, CKSCL, dataloader, epoch, optimizer=None, train=False):
    losses, preds, labels, masks, losses_sense  = [], [], [], [], []
    alphas, alphas_f, alphas_b, vids = [], [], [], []
    max_sequence_len = []

    #model = torch#.load('\model_meld_emo3.pkl')

    assert not train or optimizer!=None
    if train:
        model.train()
        model2.eval()
        model3.eval()
    else:
        model.eval()
        model2.eval()
        model3.eval()

    seed_everything(seed)
    for data in dataloader:
        if train:
            optimizer.zero_grad()

        r1, r2, r3, r4, \
        x1, x2, x3, x4, x5, x6, \
        o1, o2, o3, \
        qmask, umask, label = [d.cuda() for d in data[:-1]] if cuda else data[:-1]
       
        log_prob, _, alpha, alpha_f, alpha_b, _ = model(r1, r2, r3, r4, x5, x6, x1, o2, o3, qmask, umask, False, False, att2=False) 
        
        # For CKSCL #######################################################################################################
        model2.load_state_dict(model.state_dict())
        model3.load_state_dict(model.state_dict())
        log_prob2, _, _, _, _, _ = model2(r1, r2, r3, r4, x5, x6, x1, o2, o3, qmask, umask, False, True, att2=False) # context
        log_prob3, _, _, _, _, _ = model3(r1, r2, r3, r4, x5, x6, x1, o2, o3, qmask, umask, True, False, att2=False) # kb
        #############################################################################################################

        lp_ = log_prob.transpose(0,1).contiguous().view(-1, log_prob.size()[2]) # batch*seq_len, n_classes
        labels_ = label.view(-1) # batch*seq_len
        loss = loss_function(lp_, labels_, umask)
        pred_ = torch.argmax(lp_,1) # batch*seq_len

        preds.append(pred_.data.cpu().numpy())     
        labels.append(labels_.data.cpu().numpy())
        masks.append(umask.view(-1).cpu().numpy())
        losses.append(loss.item()*masks[-1].sum())

        # SCL #######################################################################################################
        index = umask.view(-1).bool()
        lp_scl = lp_[index]
        labels_scl = labels_[index]
        # ESCL
        loss2 = ESCL(lp_, labels_) # emo
        # CKSCL
        lp2_ = log_prob2.transpose(0,1).contiguous().view(-1, log_prob2.size()[2]) # batch*seq_len, n_classes
        lp3_ = log_prob3.transpose(0,1).contiguous().view(-1, log_prob3.size()[2]) # batch*seq_len, n_classes
        pred2_ = torch.argmax(lp2_, 1)
        pred3_ = torch.argmax(lp3_, 1)
        lac_ = (pred2_ == pred_).int()
        lac2_ = (pred3_ == pred_).int()
        lac_scl = lac_[index]
        lac2_scl = lac2_[index]
        loss3 = CKSCL(lp_, lac_, weight=create_class_weight_SCL(lac_scl)) #context
        loss4 = CKSCL(lp_, lac2_, weight=create_class_weight_SCL(lac2_scl)) # kb
        #############################################################################################################

        #train = False
        if train:
            total_loss = loss + 0.2 * loss2 + 0.1 * loss3 + 0.3 * loss4
            total_loss.backward()
            if args.tensorboard:
                for param in model.named_parameters():
                    writer.add_histogram(param[0], param[1].grad, epoch)
            optimizer.step()
        else:
            alphas += alpha
            alphas_f += alpha_f
            alphas_b += alpha_b
            vids += data[-1]

    if preds!=[]:
        preds  = np.concatenate(preds)
        labels = np.concatenate(labels)
        masks  = np.concatenate(masks)
    else:
        return float('nan'), float('nan'), float('nan'), [], [], [], float('nan'),[]

    avg_loss = round(np.sum(losses)/np.sum(masks), 4)
    avg_sense_loss = round(np.sum(losses_sense)/np.sum(masks), 4)

    avg_accuracy = round(accuracy_score(labels,preds, sample_weight=masks)*100, 2)
    avg_fscore = round(f1_score(labels, preds, sample_weight=masks, average='weighted')*100, 2)
    return avg_loss, avg_accuracy, labels, preds, masks, [avg_fscore], [alphas, alphas_f, alphas_b, vids]

#hmodel = torch.load('\modelc.pkl')
#hmodel = torch.load('\model_meld.pkl')

def train_or_eval_model(model, model2, model3, loss_function, ESCL, CKSCL, dataloader, epoch, optimizer=None, train=False):
    losses, preds, labels, masks, losses_sense  = [], [], [], [], []
    alphas, alphas_f, alphas_b, vids = [], [], [], []
    max_sequence_len = []

    #model = torch#.load('\model_meld_emo3.pkl')

    assert not train or optimizer!=None
    if train:
        model.train()
    else:
        model.eval()

    seed_everything(seed)
    for data in dataloader:
        if train:
            optimizer.zero_grad()

        r1, r2, r3, r4, \
        x1, x2, x3, x4, x5, x6, \
        o1, o2, o3, \
        qmask, umask, label = [d.cuda() for d in data[:-1]] if cuda else data[:-1]
       
        #print(123)

        # orgin
        #if random.random() > 0.9:
        #log_prob, _, alpha, alpha_f, alpha_b, _ = model(r1, r2, r3, r4, x5, x6, x1, o2, o3, qmask, umask, att2=False)
        #else:
            #log_prob, _, alpha, alpha_f, alpha_b, _ = model(r1, r2, r3, r4, x5, x6, x1, o2, o3, qmask, umask, False, True, att2=False) # context
            #log_prob, _, alpha, alpha_f, alpha_b, _ = model(r1, r2, r3, r4, x5, x6, x1, o2, o3, qmask, umask, True, False, att2=False) # kb
        log_prob, _, alpha, alpha_f, alpha_b, _ = model(r1, r2, r3, r4, x5, x6, x1, o2, o3, qmask, umask, False, False, att2=False) 
        lp_ = log_prob.transpose(0,1).contiguous().view(-1, log_prob.size()[2]) # batch*seq_len, n_classes
        labels_ = label.view(-1) # batch*seq_len
        #print(lp_.shape, labels_.shape, umask.shape)
        loss = loss_function(lp_, labels_, umask)
        pred_ = torch.argmax(lp_,1) # batch*seq_len

        preds.append(pred_.data.cpu().numpy())        
        labels.append(labels_.data.cpu().numpy())
        masks.append(umask.view(-1).cpu().numpy())
        losses.append(loss.item()*masks[-1].sum())

        # index = umask.view(-1).bool()
        # lp_scl = lp_[index]
        # labels_scl = labels_[index]
        # loss2 = ESCL(lp_scl, labels_scl) # emo

        #train = False
        if train:
            total_loss = loss
            total_loss.backward()
            if args.tensorboard:
                for param in model.named_parameters():
                    writer.add_histogram(param[0], param[1].grad, epoch)
            optimizer.step()
        else:
            alphas += alpha
            alphas_f += alpha_f
            alphas_b += alpha_b
            vids += data[-1]

    if preds!=[]:
        preds  = np.concatenate(preds)
        labels = np.concatenate(labels)
        masks  = np.concatenate(masks)
    else:
        return float('nan'), float('nan'), float('nan'), [], [], [], float('nan'),[]

    avg_loss = round(np.sum(losses)/np.sum(masks), 4)
    avg_sense_loss = round(np.sum(losses_sense)/np.sum(masks), 4)

    avg_accuracy = round(accuracy_score(labels,preds, sample_weight=masks)*100, 2)
    avg_fscore = round(f1_score(labels, preds, sample_weight=masks, average='weighted')*100, 2)
    return avg_loss, avg_accuracy, labels, preds, masks, [avg_fscore], [alphas, alphas_f, alphas_b, vids]

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--no-cuda', action='store_true', default=False, help='does not use GPU')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate')
    parser.add_argument('--l2', type=float, default=0.0003, metavar='L2', help='L2 regularization weight')
    parser.add_argument('--rec-dropout', type=float, default=0.5, metavar='rec_dropout', help='rec_dropout rate')
    parser.add_argument('--dropout', type=float, default=0.5, metavar='dropout', help='dropout rate')
    parser.add_argument('--batch-size', type=int, default=8, metavar='BS', help='batch size')
    parser.add_argument('--epochs', type=int, default=40, metavar='E', help='number of epochs')
    parser.add_argument('--class-weight', action='store_true', default=False, help='use class weights')
    parser.add_argument('--active-listener', action='store_true', default=False, help='active listener')
    parser.add_argument('--attention', default='simple', help='Attention type in context GRU')
    parser.add_argument('--tensorboard', action='store_true', default=False, help='Enables tensorboard log')
    parser.add_argument('--mode1', type=int, default=2, help='Roberta features to use')
    parser.add_argument('--seed', type=int, default=100, metavar='seed', help='seed')
    parser.add_argument('--norm', type=int, default=0, help='normalization strategy')
    parser.add_argument('--mu', type=float, default=0, help='class_weight_mu')
    parser.add_argument('--classify', default='emotion')
    parser.add_argument('--residual', action='store_true', default=False, help='use residual connection')

    args = parser.parse_args()
    print(args)

    args.cuda = torch.cuda.is_available() and not args.no_cuda
    if args.cuda:
        print('Running on GPU')
    else:
        print('Running on CPU')

    if args.tensorboard:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter()

    emo_gru = True
    if args.classify == 'emotion':
        n_classes  = 7
    elif args.classify == 'sentiment':
        n_classes  = 3
    cuda       = args.cuda
    n_epochs   = args.epochs
    batch_size = args.batch_size

    global  D_s

    D_m = 1024
    D_s = 768
    D_g = 150
    D_p = 150
    D_r = 150
    D_i = 150
    D_h = 100
    D_a = 100

    D_e = D_p + D_r + D_i

    global seed
    seed = 100
    #seed_everything(seed)
    
    model = CommonsenseGRUModel(D_m, D_s, D_g, D_p, D_r, D_i, D_e, D_h, D_a,
                                n_classes=n_classes,
                                listener_state=args.active_listener,
                                context_attention=args.attention,
                                dropout_rec=args.rec_dropout,
                                dropout=args.dropout,
                                emo_gru=emo_gru,
                                mode1=args.mode1,
                                norm=args.norm,
                                residual=args.residual)

    print ('MELD COSMIC Model.')
    
    model2 = CommonsenseGRUModel(D_m, D_s, D_g, D_p, D_r, D_i, D_e, D_h, D_a,
                                n_classes=n_classes,
                                listener_state=args.active_listener,
                                context_attention=args.attention,
                                dropout_rec=args.rec_dropout,
                                dropout=args.dropout,
                                emo_gru=emo_gru,
                                mode1=args.mode1,
                                norm=args.norm,
                                residual=args.residual)

    model3 = CommonsenseGRUModel(D_m, D_s, D_g, D_p, D_r, D_i, D_e, D_h, D_a,
                                n_classes=n_classes,
                                listener_state=args.active_listener,
                                context_attention=args.attention,
                                dropout_rec=args.rec_dropout,
                                dropout=args.dropout,
                                emo_gru=emo_gru,
                                mode1=args.mode1,
                                norm=args.norm,
                                residual=args.residual)

    if cuda:
        model.cuda()
        model2.cuda()
        model3.cuda()
    #print(args.class_weight)
    if args.classify == 'emotion':
        if args.class_weight:
            if args.mu > 0:
                loss_weights = torch.FloatTensor(create_class_weight(args.mu))
            else:   
                loss_weights = torch.FloatTensor([0.30427062, 1.19699616, 5.47007183, 1.95437696, 
                0.84847735, 5.42461417, 1.21859721])
            loss_function  = MaskedNLLLoss(loss_weights.cuda() if cuda else loss_weights)
        else:
            loss_function = MaskedNLLLoss()
            
    else:
        loss_function = MaskedNLLLoss()

    loss_weights2 = torch.FloatTensor([0.30427062, 1.19699616, 5.47007183, 1.95437696,
                0.84847735, 5.42461417, 1.21859721])

    SCL1 = ESCL(weight=loss_weights2) # emo
    SCL2 = CKSCL()

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    
    if args.classify == 'emotion':
        lf = open('logs/cosmic_meld_emotion_logs.txt', 'a')
    elif args.classify == 'sentiment':
        lf = open('logs/cosmic_meld_sentiment_logs.txt', 'a')

    train_loader, valid_loader, test_loader = get_MELD_loaders(batch_size=batch_size, 
                                                               classify=args.classify,
                                                               num_workers=0)

    valid_losses, valid_fscores = [], []
    test_fscores, test_losses = [], []
    best_loss, best_label, best_pred, best_mask = None, None, None, None

    best_valid_f1 = 0
    best_test_f1 = 0

    for e in range(n_epochs):
        start_time = time.time()
        train_loss, train_acc, _, _, _, train_fscore, _ = train_or_eval_model2(model, model2, model3, loss_function, SCL1, SCL2, train_loader, e, optimizer, True)
        valid_loss, valid_acc, _, _, _, valid_fscore, _ = train_or_eval_model2(model, model2, model3, loss_function, SCL1, SCL2, valid_loader, e)
        test_loss, test_acc, test_label, test_pred, test_mask, test_fscore, attentions = train_or_eval_model2(model, model2, model3, loss_function, SCL1, SCL2, test_loader, e)       

        #print('no kb or context with true', np.sum(test_pred[test_mask==1] != test_label[test_mask==1]), len(test_pred[test_mask == 1]), test_pred[test_mask==1] != test_pred2[test_mask==1]) # 2610
        #print('no kb or context with origin', np.sum(test_pred[test_mask==1] != test_pred2[test_mask2==1]), len(test_pred[test_mask == 1])

        if valid_fscore[-1] > best_valid_f1:
            best_valid_f1 = valid_fscore[-1]
            #torch.save(model, '\model_meld_emo3.pkl')
 
        if test_fscore[-1] > best_test_f1:
            best_test_f1 = test_fscore[-1]

        valid_losses.append(valid_loss)
        valid_fscores.append(valid_fscore)
        test_losses.append(test_loss)
        test_fscores.append(test_fscore)       
        
        if args.tensorboard:
            writer.add_scalar('test: accuracy/loss', test_acc/test_loss, e)
            writer.add_scalar('train: accuracy/loss', train_acc/train_loss, e)
        
        x = 'epoch: {}, train_loss: {}, acc: {}, fscore: {}, valid_loss: {}, acc: {}, fscore: {}, test_loss: {}, acc: {}, fscore: {}, time: {} sec'.format(e+1, train_loss, train_acc, train_fscore, valid_loss, valid_acc, valid_fscore, test_loss, test_acc, test_fscore, round(time.time()-start_time, 2))
        
        print (x)
        lf.write(x + '\n')

    if args.tensorboard:
        writer.close()
        
    valid_fscores = np.array(valid_fscores).transpose()
    test_fscores = np.array(test_fscores).transpose()

    score1 = test_fscores[0][np.argmin(valid_losses)]
    score2 = test_fscores[0][np.argmax(valid_fscores[0])]    
    scores = [score1, score2]
    scores = [str(item) for item in scores]
    
    print ('Test Scores: Weighted F1')
    print('@Best Valid Loss: {}'.format(score1))
    print('@Best Valid F1: {}'.format(score2))
    print('@Best test F1:', best_test_f1)

    if args.classify == 'emotion':
        rf = open('results/cosmic_meld_emotion_results.txt', 'a')
    elif args.classify == 'sentiment':
        rf = open('results/cosmic_meld_sentiment_results.txt', 'a')
    
    rf.write('\t'.join(scores) + '\t' + str(args) + '\n')
    rf.close()

