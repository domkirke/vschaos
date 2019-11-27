import torch, gc
from ..utils.onehot import fromOneHot
from ..utils.misc import sample_normalize, NaNError, apply_method, pdb, merge_dicts, accumulate_losses, normalize_losses


def recursive_add(obj, factor=1.):
    if issubclass(type(obj[0]), (list, tuple)):
        return [recursive_add([obj[j][i] for j in range(len(obj))], factor=factor) for i in range(len(obj[0]))]
    else:
        return sum(obj)*factor


def run(self, loader, preprocessing=None, epoch=None, optimize=True, schedule=False, period=None, plot=None, **kwargs):
    # train phase
    batch = 0; current_loss = 0;
    train_losses = []
    if preprocessing is None:
        preprocessing = self.preprocessing
    self.logger('start epoch')
    for x,y in loader:
        # forward
        #pdb.set_trace()
        self.logger('data loaded')
        if preprocessing is not None:
            x = preprocessing(x)
        if kwargs.get('sample_norm'):
            x = sample_normalize(x)
        self.logger('data preprocessed')
        try:
            out = self.models.forward(x, y=y, epoch=epoch)
            if self.reinforcers:
                out = self.reinforcers.forward(out, target=x, optimize=False)
                            #self.logger(log_dist("latent", out['z_params_enc'][-1]))
            #self.logger(log_dist("data", out['x_params']))
            # compute loss
            self.logger('data forwarded')
            #pdb.set_trace()
            batch_loss, losses = self.losses.loss(model=self.models, out=out, target=x, epoch=epoch, plot=plot and not batch, period=period)
            train_losses.append(losses)
        except NaNError:
            pdb.set_trace()

        # trace
        if self.trace_mode == "batch":
            if period is None:
                period = "train" if optimize else "test"
            apply_method(self.losses, "write", period, losses)
            apply_method(self.monitor, "update")
            self.logger("monitor updated")

        # learn
        self.logger('loss computed')
        if optimize:
            batch_loss.backward()
            self.optimize(self.models, batch_loss)

        if self.reinforcers:
            _, reinforcement_losses = self.reinforcers(out, target=x, epoch=epoch, optimize=optimize)

            self.logger('optimization done')
        # update loop
        named_losses = self.losses.get_named_losses(losses)
        if self.reinforcers:
            named_losses = {**named_losses, **self.reinforcers.get_named_losses(reinforcement_losses)}
        print("epoch %d / batch %d \nfull loss: %s / losses : %s "%(epoch, batch, batch_loss, named_losses))

        if kwargs.get('track_loss'):
            current_loss = current_loss + float(batch_loss)
        else:
            current_loss += float(batch_loss)
        batch += 1

        #if batch % 1 == 90:
        #    torch.cuda.empty_cache()

        del out; del x

    current_loss /= batch
    train_losses = recursive_add(train_losses, factor = 1/len(train_losses))
    # scheduling the training
    if schedule:
        apply_method(self.models, "schedule", current_loss)
    # cleaning cuda stuff
    gc.collect(); gc.collect()
    self.logger("cuda cleaning done")

    return current_loss, train_losses 


def run_accumulate(self, loader, preprocessing=None, epoch=None, optimize=True, schedule=False, period=None, plot=None, **kwargs):
    # train phase
    batch = 0; current_loss = 0; n_splits = kwargs.get('batch_split', None)
    assert n_splits, "when using accumulating runs the batch_split argument must be given"
    train_losses = None;
    if preprocessing is None:
        preprocessing = self.preprocessing
    self.logger('start epoch')
    for x,y in loader:
        # forward
        self.logger('data loaded')
        # split in equal parts
        x_split = torch.split(x, n_splits)
        if y != [] and y is not None:
            y_split = {k:torch.split(v, n_splits) for k,v in y.items()}
            y_split = [{k:v[i] for k, v in y_split.items()} for i in range(len(x_split))]
        else:
            y_split = [[]]*len(x_split)

        batch_loss = torch.tensor(0., requires_grad=True, device=next(self.models.parameters()).device)
        batch_losses = list()
        for i, x_tmp in enumerate(x_split):
            y_tmp = y_split[i]
            if preprocessing is not None:
                x_tmp = preprocessing(x_tmp)
            if kwargs.get('sample_norm'):
                x_tmp = sample_normalize(x_tmp)
            self.logger('data preprocessed')
            try:
                out = self.models.forward(x_tmp, y=y_tmp, epoch=epoch)
                if self.reinforcers:
                    out = self.reinforcers.forward(out)
                                #self.logger(log_dist("latent", out['z_params_enc'][-1]))
                #self.logger(log_dist("data", out['x_params']))
                # compute loss
                self.logger('data forwarded')
                loss_tmp, losses_tmp = self.losses.loss(model=self.models, out=out, target=x_tmp, epoch=epoch, plot=plot and not batch, period=period)
                loss_tmp = 1/len(x_split)*loss_tmp
                batch_loss = batch_loss + loss_tmp; batch_losses.append(losses_tmp)

                # accumulate reinforcers
                if self.reinforcers:
                     _, reinforcement_losses = self.reinforcers(out, target=x_tmp, epoch=epoch, optimize=False, retain_graph=True)

            except NaNError:
                pdb.set_trace()

            # learn
            self.logger('loss computed')
            if optimize:
                loss_tmp.backward(retain_graph=True)

        losses = recursive_add(batch_losses, factor=1/len(x))
        # trace
        if self.trace_mode == "batch":
            if period is None:
                period = "train" if optimize else "test"
            apply_method(self.losses, "write", period, losses)
            apply_method(self.monitor, "update")
            self.logger("monitor updated")

        if optimize:
            self.optimize(self.models, batch_loss)
            if self.reinforcers:
                self.reinforcers.step()

        self.logger('optimization done')

        # update loop
        named_losses = self.losses.get_named_losses(losses)
        if self.reinforcers:
            named_losses = {**named_losses, **self.reinforcers.get_named_losses(reinforcement_losses)}
        print("epoch %d / batch %d \nfull loss: %s / losses : %s "%(epoch, batch, batch_loss, named_losses))

        if kwargs.get('track_loss'):
            current_loss = current_loss + batch_loss
        else:
            current_loss += float(batch_loss)

        batch += 1
        del out; del x_tmp

    current_loss /= batch
    # scheduling the training
    if schedule:
        apply_method(self.models, "schedule", current_loss)
    # cleaning cuda stuff
    torch.cuda.empty_cache()
    gc.collect(); gc.collect()
    self.logger("cuda cleaning done")

    return current_loss, losses




def run_scan(self, loader, preprocessing=None, epoch=None, optimize=True, schedule=False, period=None, plot=None, **kwargs):
    # train phase
    batch = 0; current_loss = 0;
    train_losses = None;
    if preprocessing is None:
        preprocessing = self.preprocessing
    self.logger('start epoch')
    full_losses = None
    for x,y in loader:
        # prepare audio data
        self.logger('data loaded')
        if preprocessing[0] is not None:
            x = preprocessing[0](x)
        if kwargs.get('sample_norm'):
            x = sample_normalize(x)
        self.logger('data preprocessed')
        # prepare symbol data
        x_sym = [y.get(s) for s in self.symbols]
        if preprocessing[1] is not None:
            x_sym = preprocessing[1](x_sym)
        try:
            out_audio = self.models[0].forward(x, y=y, epoch=epoch)
            if self.reinforcers[0]:
                out_audio = self.reinforcers[0].forward(out_audio)

            out_symbol = self.models[1].forward(x_sym, y=y, epoch=epoch)
                            #self.logger(log_dist("latent", out['z_params_enc'][-1]))
            #self.logger(log_dist("data", out['x_params']))
            # compute loss
            self.logger('data forwarded')
            audio_batch_loss, audio_losses = self.losses[0].loss(model=self.models[0], out=out_audio, target=x, epoch=epoch, plot=plot and not batch, period=period)
            symbol_batch_loss, symbol_losses = self.losses[1].loss(model=self.models[1], out=out_symbol, target=[fromOneHot(x) for x in x_sym], epoch=epoch, plot=plot and not batch, period=period)
            # compute transfer loss
            transfer_loss, transfer_losses = self.get_transfer_loss(out_audio, out_symbol)

            batch_loss = audio_batch_loss + symbol_batch_loss + transfer_loss
            losses = [audio_losses, symbol_losses, transfer_losses]
            train_losses = losses if train_losses is None else train_losses + losses
        except NaNError:
            pdb.set_trace()

        # trace
        if self.trace_mode == "batch":
            if period is None:
                period = "train" if optimize else "test"
            apply_method(self.losses, "write", period, losses)
            apply_method(self.monitor, "update")
            self.logger("monitor updated")

        # learn
        self.logger('loss computed')
        if optimize:
            batch_loss.backward()
            self.optimize(self.models, batch_loss)
        if self.reinforcers[0]:
            _, reinforcement_losses = self.reinforcers[0](out_audio, target=x, epoch=epoch, optimize=optimize)

        self.logger('optimization done')
        # update loop
        named_losses_audio = self.losses[0].get_named_losses(losses[0])
        named_losses_symbol = self.losses[1].get_named_losses(losses[1])
        named_losses_transfer = self.transfer_loss.get_named_losses(losses[2])

        print("epoch %d / batch %d \nfull loss: %s"%(epoch, batch, batch_loss))
        print('audio : %s, %s'%(audio_batch_loss, named_losses_audio))
        print('symbol : %s, %s'%(symbol_batch_loss, named_losses_symbol))
        print('transfer : %s, %s'%(transfer_loss, named_losses_transfer))

        if kwargs.get('track_loss'):
            current_loss = current_loss + batch_loss
        else:
            current_loss += float(batch_loss)
        batch += 1
        if full_losses is None:
            full_losses = losses
        else:
            full_losses = accumulate_losses([full_losses, losses])
        del out_audio; del out_symbol; del x

    current_loss /= batch
    normalize_losses(full_losses, batch)
    # scheduling the training
    if schedule:
        apply_method(self.models, "schedule", current_loss)
    # cleaning cuda stuff
    torch.cuda.empty_cache()
    gc.collect(); gc.collect()
    self.logger("cuda cleaning done")

    return current_loss, full_losses
