import torchtext
from Transparency.common_code.common import *
from Transparency.Trainers.PlottingBC import generate_graphs, plot_adversarial_examples, plot_logodds_examples
from Transparency.configurations import configurations
from Transparency.Trainers.TrainerBC import Trainer, Evaluator
from Transparency.model.LR import LR
import Transparency.model.Binary_Classification as BC
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from temperature_scaling import ModelWithTemperature
from model.modelUtils import BatchHolder
import pickle
from common_code.common import pickle_to_file

def train_dataset(dataset, config='lstm') :
    try :
        config = configurations[config](dataset)
        trainer = Trainer(dataset, config=config, _type=dataset.trainer_type)
        trainer.train(dataset.train_data, dataset.dev_data, n_iters=8, save_on_metric=dataset.save_on_metric)
        evaluator = Evaluator(dataset, trainer.model.dirname, _type=dataset.trainer_type)
        _ = evaluator.evaluate(dataset.test_data, save_results=True)
        return trainer, evaluator
    except :
        return

def train_dataset_and_get_lime_explanations(dataset, encoders, num_iters=15):
    for e in encoders:
        config = configurations[e](dataset)
        trainer = Trainer(dataset, config=config,
                          _type=dataset.trainer_type)
        trainer.train(dataset.train_data, dataset.dev_data, n_iters=num_iters,
                      save_on_metric=dataset.save_on_metric)
        train_losses = trainer.model.train_losses

        lime_explanations = trainer.model.get_lime_explanations(
            dataset.test_data.X)
        return lime_explanations


def train_dataset_and_get_atn_map(dataset, encoders, num_iters=15):
    for e in encoders:
        config = configurations[e](dataset)
        trainer = Trainer(dataset, config=config,
                          _type=dataset.trainer_type)
        trainer.train(dataset.train_data, dataset.dev_data, n_iters=num_iters,
                      save_on_metric=dataset.save_on_metric)
        train_losses = trainer.model.train_losses

        evaluator = Evaluator(dataset, trainer.model.dirname,
                              _type=dataset.trainer_type)
        predictions, attentions = evaluator.evaluate(dataset.test_data,
                                                     save_results=True)

        # evaluator = Evaluator(dataset, trainer.model.last_epch_dirname,
        #                       _type=dataset.trainer_type)
        predictions_lst_epch, attentions_lst_epch = evaluator.evaluate(dataset.test_data)
        return predictions, attentions, predictions_lst_epch, attentions_lst_epch, train_losses

def train_dataset_and_temp_scale(dataset, encoders):
    for e in encoders:
        config = configurations[e](dataset)
        trainer = Trainer(dataset, config=config,
                          _type=dataset.trainer_type)
        trainer.train(dataset.train_data, dataset.dev_data, n_iters=8,
                      save_on_metric=dataset.save_on_metric)
        evaluator = Evaluator(dataset, trainer.model.dirname,
                              _type=dataset.trainer_type)

        print("Temperature-scaling..")
        orig_model = evaluator.model
        dev_x_tensor = BatchHolder(dataset.dev_data.X).seq
        dev_x_tensor_lengths = BatchHolder(dataset.dev_data.X).lengths
        dev_x_tensor_masks = BatchHolder(dataset.dev_data.X).masks
        valid_dataset = TensorDataset(dev_x_tensor, dev_x_tensor_lengths,
                                      dev_x_tensor_masks, torch.from_numpy(
                np.array(dataset.dev_data.y)))
        valid_loader = DataLoader(valid_dataset, batch_size=1)

        scaled_model = ModelWithTemperature(orig_model)
        scaled_model.set_temperature(valid_loader)

def train_dataset_on_encoders(dataset, encoders):
    for e in encoders :
        train_dataset(dataset, e)
        run_experiments_on_latest_model(dataset, e)

def generate_graphs_on_encoders(dataset, encoders) :
    for e in encoders :
        generate_graphs_on_latest_model(dataset, e)

def train_lr_on_dataset(dataset) :
    config = {
        'vocab' : dataset.vec,
        'stop_words' : True,
        'type' : dataset.trainer_type,
        'exp_name' : dataset.name
    }

    dataset.train_data.y = np.array(dataset.train_data.y)
    dataset.test_data.y = np.array(dataset.test_data.y)
    if len(dataset.train_data.y.shape) == 1 :
        dataset.train_data.y = dataset.train_data.y[:, None]
        dataset.test_data.y = dataset.test_data.y[:, None]
    lr = LR(config)
    lr.train(dataset.train_data)
    lr.evaluate(dataset.test_data, save_results=True)
    lr.save_estimator_logodds()
    return lr

def run_evaluator_on_latest_model(dataset, config='lstm') :
    config = configurations[config](dataset)
    latest_model = get_latest_model(os.path.join(config['training']['basepath'], config['training']['exp_dirname']))
    evaluator = Evaluator(dataset, latest_model, _type=dataset.trainer_type)
    _ = evaluator.evaluate(dataset.test_data, save_results=True)
    return evaluator

def run_evaluator_on_specific_model(dataset, model_path, config='lstm'):
    config = configurations[config](dataset)
    evaluator = Evaluator(dataset, model_path, _type=dataset.trainer_type)
    evaluator.model.temperature = config['training']['temperature']
    _ = evaluator.evaluate(dataset.test_data, save_results=True)
    return evaluator


def adding_params(net1, net2, alpha):
    for param1, param2 in zip(net1.parameters(), net2.parameters()):
        param1.data *= (1.0 - alpha)
        param1.data += param2.data * alpha


def divide_all_params(swa, n):
    for param in swa.parameters():
        param.data /= n


def eval_swa_model(dataset, top_lvl_models_dir):
    dirs = [d for d in os.listdir(top_lvl_models_dir) if
            'enc.th' in os.listdir(os.path.join(top_lvl_models_dir, d))]
    Model = BC.Model
    swa = Model.init_from_config(os.path.join(top_lvl_models_dir, dirs[0]))
    swa.dirname = os.path.join(top_lvl_models_dir, dirs[0])
    i = 1
    for new_model_dir in dirs[1:]:
        new_model = BC.Model.init_from_config(
            os.path.join(top_lvl_models_dir, new_model_dir))
        adding_params(swa.encoder, new_model.encoder, 1.0 / (i+1))
        adding_params(swa.decoder, new_model.decoder, 1.0 / (i+1))
        i += 1
    # divide_all_params(swa.encoder, len(dirs))
    # divide_all_params(swa.decoder, len(dirs))
    evaluator = Evaluator(dataset, os.path.join(top_lvl_models_dir, dirs[0]),
                          _type=dataset.trainer_type)
    evaluator.model = swa
    _ = evaluator.evaluate(dataset.test_data, save_results=True)
    return evaluator

def train_dataset_and_get_gradient(dataset, encoders, num_iters=15):
    for e in encoders:
        config = configurations[e](dataset)
        trainer = Trainer(dataset, config=config,
                          _type=dataset.trainer_type)
        trainer.train(dataset.train_data, dataset.dev_data, n_iters=num_iters,
                      save_on_metric=dataset.save_on_metric)

        evaluator = Evaluator(dataset, trainer.model.dirname,
                              _type=dataset.trainer_type)
        predictions, attentions = evaluator.evaluate(dataset.test_data,
                                                     save_results=True)
        grads = evaluator.gradient_experiment_get_grads(dataset.test_data)
        from Trainers.PlottingBC import process_grads
        process_grads(grads)
        return predictions, attentions, grads

def run_experiments_on_latest_model(dataset, config='lstm', force_run=True) :
    try :
        evaluator = run_evaluator_on_latest_model(dataset, config)
        test_data = dataset.test_data
        evaluator.gradient_experiment(test_data, force_run=force_run)
        evaluator.permutation_experiment(test_data, force_run=force_run)
        evaluator.adversarial_experiment(test_data, force_run=force_run)
#        evaluator.remove_and_run_experiment(test_data, force_run=force_run)
    except Exception as e:
        print(e)
        return

def generate_graphs_on_latest_model(dataset, config='lstm') :
    config = configurations[config](dataset)
    latest_model = get_latest_model(os.path.join(config['training']['basepath'], config['training']['exp_dirname']))
    evaluator = Evaluator(dataset, latest_model, _type=dataset.trainer_type)
    _ = evaluator.evaluate(dataset.test_data, save_results=False)
    generate_graphs(dataset, config['training']['exp_dirname'], evaluator.model, test_data=dataset.test_data)

def generate_adversarial_examples(dataset, config='lstm') :
    evaluator = run_evaluator_on_latest_model(dataset, config)
    config = configurations[config](dataset)
    plot_adversarial_examples(dataset, config['training']['exp_dirname'], evaluator.model, test_data=dataset.test_data)

def generate_logodds_examples(dataset, config='lstm') :
    evaluator = run_evaluator_on_latest_model(dataset, config)
    config = configurations[config](dataset)
    plot_logodds_examples(dataset, config['training']['exp_dirname'], evaluator.model, test_data=dataset.test_data)

def run_logodds_experiment(dataset, config='lstm') :
    model = get_latest_model(os.path.join('outputs', dataset.name, 'LR+TFIDF'))
    print(model)
    logodds = pickle.load(open(os.path.join(model, 'logodds.p'), 'rb'))
    evaluator = run_evaluator_on_latest_model(dataset, config)
    evaluator.logodds_attention_experiment(dataset.test_data, logodds, save_results=True)

def run_logodds_substitution_experiment(dataset) :
    model = get_latest_model(os.path.join('outputs', dataset.name, 'LR+TFIDF'))
    print(model)
    logodds = pickle.load(open(os.path.join(model, 'logodds.p'), 'rb'))
    evaluator = run_evaluator_on_latest_model(dataset)
    evaluator.logodds_substitution_experiment(dataset.test_data, logodds, save_results=True)

def get_top_words(dataset, config='lstm') :
    evaluator = run_evaluator_on_latest_model(dataset, config)
    test_data = dataset.test_data
    test_data.top_words_attn = find_top_words_in_all(dataset, test_data.X, test_data.attn_hat)

def get_results(path) :
    latest_model = get_latest_model(path)
    if latest_model is not None :
        evaluations = json.load(open(os.path.join(latest_model, 'evaluate.json'), 'r'))
        return evaluations
    else :
        raise LookupError("No Latest Model ... ")

names = {
    'vanilla_lstm':'LSTM',
    'lstm':'LSTM + Additive Attention',
    'logodds_lstm':'LSTM + Log Odds Attention',
    'lr' : 'LR + BoW',
    'logodds_lstm_post' : 'LSTM + Additive Attention (Log Odds at Test)'
}

def push_all_models(dataset, keys) :
    model_evals = {}
    for e in ['vanilla_lstm', 'lstm', 'logodds_lstm'] :
        config = configurations[e](dataset)
        path = os.path.join(config['training']['basepath'], config['training']['exp_dirname'])
        evals = get_results(path)
        model_evals[names[e]] = {keys[k]:evals[k] for k in keys}

    path = os.path.join('outputs', dataset.name, 'LR+TFIDF')
    evals = get_results(path)
    model_evals[names['lr']] = {keys[k]:evals[k] for k in keys}

    path = os.path.join('outputs', dataset.name, 'lstm+tanh+logodds(posthoc)')
    evals = get_results(path)
    model_evals[names['logodds_lstm_post']] = {keys[k]:evals[k] for k in keys}

    df = pd.DataFrame(model_evals).transpose()
    df['Model'] = df.index
    df = df.loc[[names[e] for e in ['lr', 'vanilla_lstm', 'lstm', 'logodds_lstm_post', 'logodds_lstm']]]

    os.makedirs(os.path.join('graph_outputs', 'evals'), exist_ok=True)
    df.to_csv(os.path.join('graph_outputs', 'evals', dataset.name + '+lstm+tanh.csv'), index=False)
    return df


