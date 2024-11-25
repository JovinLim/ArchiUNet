def train(config):
    strategy = config['strategy']
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    resume = config.get('resume', False)
    baseline = config.get('baseline', False)
    classes = ['bg'] + config['classes']
    model = config['model']
    backbone = config['backbone']
    deep_supervision = config.get('deep_supervision', False)

    # filters = [16, 32, 64, 128, 256] if 'filters' not in config else config['filters']
    filters = [32, 64, 128, 256, 512] if 'filters' not in config else config['filters']
    # filters = [64, 128, 256, 512, 1024] if 'filters' not in config else config['filters']

    heatmap = config.get('heatmap', False)
    aaf = config.get('aaf', [])
    aaf_count = len(aaf)

    epochs = 200

    log_dir = 'models/' + '_'.join(
        [model, config['name'], str(backbone or ''), ','.join(map(str, filters)), ','.join(config['datasets']),
         time.strftime("%Y%m%d-%H%M%S")])

    with strategy.scope():
        if resume:
            print('Resuming training')
            if model == 'zeng':
                custom_objects = {'loss_function': loss_functions.balanced_entropy(len(classes))}
            else:
                custom_objects = {'loss_function': loss_functions.asym_unified_focal_loss(len(classes))}

            unet_model = tf.keras.models.load_model('models/zeng_final3', custom_objects=custom_objects)
        else:
            LEARNING_RATE = 1e-4

        loss_funcs = [loss_functions.asym_unified_focal_loss(len(classes))]
        names = ['asym_unified_focal_loss']
        inds = [0]
        dec = [False]
        if heatmap:
            loss_funcs.append(loss_functions.heatmap_regression_loss(len(classes), config['opening_inds']))
            names.append('Heatmap')
            inds.append(len(inds))
            dec.append(False)

        if aaf_count > 0:
            init_w = tf.constant_initializer(1 / aaf_count)
            w_edge = tf.Variable(
                name='edge_w',
                initial_value=init_w(shape=(1, 1, 1, len(classes), 1, aaf_count)),
                dtype=tf.float32,
                trainable=True)
            w_not_edge = tf.Variable(
                name='nonedge_w',
                initial_value=init_w(shape=(1, 1, 1, len(classes), 1, aaf_count)),
                dtype=tf.float32,
                trainable=True)
            aaf_ind = len(inds)

            for a, s in enumerate(aaf):
                loss_funcs.append(
                    loss_functions.adaptive_affinity_loss(size=s, k=a, num_classes=len(classes),
                                                            w_edge=w_edge,
                                                            w_not_edge=w_not_edge))
                names.append('AAF({0}x{0})'.format(2 * s + 1))
                inds.append(aaf_ind)
                dec.append(True)

        unet_model = cab2((None, None, 3), n_labels=len(classes), backbone=backbone,
                            filter_num_down=filters,
                            output_activation='Softmax',
                            batch_norm=True,
                            deep_supervision=deep_supervision,
                            aaf=(aaf_count > 0),
                            use_hhdc=config.get('hhdc', False),
                            use_cam=config.get('cam', True))

        automatic_loss = AutomaticWeightedLoss(loss_funcs, names, inds, dec, epochs, log_dir)
        unet_model.automatic_loss = automatic_loss
        unet_model.loss_sigmas = automatic_loss.sigmas
        if aaf_count > 0:
            unet_model.w_edge = w_edge
            unet_model.w_not_edge = w_not_edge
        unet_model.compile(loss=automatic_loss.combined_loss(),
                            optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                            metrics=[metrics.CategoricalAccuracy()],
                            run_eagerly=False)

    # NORMALIZE=False IF EFFICIENTNET IS USED
    # Otherwise=True
    normalize = config['normalize']
    print(config)
    print("Backbone: {0}, normalize: {1}".format(backbone, normalize))
    train_dataset, validation_dataset, test_dataset = floorplans.load_train_data(classes, config['datasets'],
                                                                                 normalize=normalize)

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, verbose=1)

    callbacks = [early_stop]
    if model == 'unetpp' or model == 'unet3p' or model == 'ours_multi' or model == 'cab1' or model == 'cab2':
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-5,
                                                         verbose=1)
        callbacks.append(reduce_lr)
    if hasattr(unet_model, 'automatic_loss'):
        callbacks.append(AutomaticWeightedLossCallback(heatmap, aaf_count > 0))
    trainer = Trainer(checkpoint_callback=True, learning_rate_scheduler=None, tensorboard_images_callback=False,
                      callbacks=callbacks, log_dir_path=log_dir)
    trainer.fit(unet_model,
                train_dataset,
                validation_dataset,
                epochs=epochs,
                batch_size=2,
                verbose=2)

    del unet_model
    tf.keras.backend.clear_session()
    print('========== DONE ==========')
