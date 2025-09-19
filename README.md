# Saliency based He for IoT 

Based on https://github.com/csjunjun/RIC and https://dl.acm.org/doi/10.1145/3595916.3626741

## Setting up
Add to RIC: http://richzhang.github.io/PerceptualSimilarity/
Add to CNN_compact: https://github.com/aaron-xichen/pytorch-playground
Data Set will download automatically (svhn and stl10)

## Example  Test Run
check the hook in the code! In Server.py line 539 and 540:
```
    for idx, train_batch in enumerate(test_loader):

            if idx > 2:
                break
```
Currently breaking off after 3 examples for testing purposes, delete these two lines if you wish to test the whole dataset.

### Autoamtic
For the first run it is advised to do a manual run because the dataset will download on the first run, which the pipeline is not build for!

in run_pipeline2.py at your paths in line 18 and 19:
```
    BASE_PROJECT_PATH = "/your/path/here"
    CLIENT_BASE_PATH = '/your/path/here' 
```
the path should lead to the local save point of this github directory.

After line 196 in run_pipeline2.py add whatever configuration you want to run:

```
    if __name__ == "__main__":
        sal_for_he = False
        compress = True
        sal_for_cs = False
        encrypt = True
        share_guide = True
        
        # Run the full pipeline (train and test) with the initial configuration
        run_pipeline_for_config(sal_for_he, compress, sal_for_cs, encrypt, share_guide, train_mode=False)
```
In this example: Pipeline to be conducted with compression and encryption both executed based on feature extraction.

run the script run_pipeline2.py. You will see:
```
    --- Starting Testing with Flags: stl10_no_sal_he_no_sal_cs_compressed_encrypted_share_guide_color ---
    Output directory (Old files of the same flags will be overwritten!): /your/path/here/output/result2/stl10_no_sal_he_no_sal_cs_compressed_encrypted_share_guide_color/
    Suche nach blockierendem Prozess auf Port 5001...
    Fehler: 'lsof' Befehl fehlgeschlagen. Möglicherweise keine Berechtigung oder 'lsof' ist nicht installiert: Command '['sudo', 'lsof', '-t', '-i', ':5001']' returned non-zero exit status 1.
    Starting '/your/path/here/RIC/Server.py' with Arguments: ['--base_project_path=/your/path/here', '--test', '--compress', '--encrypt', '--share_guide']
    Wait 100 Seconds, until Server is ready...
    Starting '/your/path/here/RIC/IoT_client.py' with Arguments: []
    Test completed.
    Pipeline concluded.
```
as of the output, you can see the logs of client and server at: /your/path/here/output/result2/stl10_no_sal_he_no_sal_cs_compressed_encrypted_share_guide_color/ --> directory changes with changed settings

the server log will specify the Arguments again:
```
    Mode: Test
    Project's path: /your/path/here/
    Compression: Yes
    Compression guided by: Feature extraction
    Encryption: Yes
    Encryption guided by: Feature extraction
    Shared Guidance: Yes
```

and also an output directory for generated files:
```
    Output directory (Old files of the same flags will be overwritten!): /your/path/here/output/experiments/stl10_encrypted_feature extraction_compressed_feature extraction_shared_color/
```

look in there, you can see the generated images under testimages and the used weights under model.
![img](/output/experiments/stl10_encrypted_feature%20extraction_compressed_feature%20extraction_shared_color/testimages/0.png)

to be read as in the paper.

In the Server log, also keep an eye open for correct connection to the client and correct loading of the weights! It should look similar to this:
```
    Connecting...
    Connected to ('127.0.0.1', 48566)
    DEBUG: loading model from:  /your/path/here/
    => Loading resume checkpoint '/your/path/here/output/experiments/stl10_encrypted_feature extraction_compressed_feature extraction_shared_color/model/reveal_network_weights.pth'
    => Resumed from epoch 0 and model moved to cuda:0
```

At the end you should see the quality measurements specified in the paper and a message that the server is finished:

```
    Server: Data saved through compression 2: 0.0000 MB
    Server: Client-Iterationtime for Batch 2: 0.1762s
    Server: Client-Memory-measure 2: 0.0198 MB
    Server: Client-Peak-Memory-measure 2: 2.2169 MB
    classification success rate:30/30=1.000000
    avg. l2loss_secrets:109.410602
    avg. l2loss_covers:27.005980
    avg. psnr_secrets:3.651607
    avg. ssim_secrets:0.027889
    avg. mean_pixel_errors:135.364840
    avg. lpips_errors:1.099152
    avg. lpips_scs:1.137767
    avg. mse_scs:54.203158
    avg. psnr_scs:9.789728
    avg. ssim_scs:0.024190
    avg. duration on client side: 0.356160s
    avg. memory usage on client side: 0.031177MB
    avg. peak memory usage on client side: 2.216051MB
    avg. data saved through compression: 0.000000MB

    Server closed - finished.
```
please note, that these values are averages over the until then conducted data points. each measurement output includes information of the measurement before. (resets each rerun of course)

In the client log, also keep an eye open for correct connection and model weights handling:
```
    Connecting to Server...
    Connected to 127.0.0.1:5001
    => Loading resume checkpoint '/your/path/here/output/experiments/stl10_encrypted_feature extraction_compressed_feature extraction_shared_color/model/hide_network_weights.pth'
    => Resumed from epoch 0 and model moved to cuda.
    => Loading resume checkpoint '/your/path/here/output/experiments/stl10_encrypted_feature extraction_compressed_feature extraction_shared_color/model/prep_network_weights.pth'
    => Resumed from epoch 0 and model moved to cuda.
```

also, the client always ends with:
```
    Server connection lost or no data left to compute.
    Client shut down.
```
this either means the pipeline ran through correctly or that there was an error on server side - if the server log looks good, it is the former.

### Manual
run the server.py with arguments like:
```
    /usr/bin/python3 "/your/path/here/RIC/Ser
    ver.py" --base_project_path="/your/path/here/" --test --compress --encrypt --sal_for_he --sal_for_cs --share
```
In this example we run a test with compression and encryption using both saliency maps as guidance. check the arguments again in the server output:
```
    Mode: Test
    Project's path: /your/path/here/
    Compression: Yes
    Compression guided by: Saliency map
    Encryption: Yes
    Encryption guided by: Saliency map
    Shared Guidance: Yes
```
Also take note of the output path:
```
    Output directory (Old files of the same flags will be overwritten!): /your/path/here/output/experiments/stl10_encrypted_saliency_compressed_saliency_shared_color/
```
If it is your first run, the dataset might need to be downloaded, that might take a while. Otherwise you will see:
```
    Building STL10 data loader with 1 workers
    Files already downloaded and verified
```

Eventually, you will see:
```
    Connecting...
```
and nothing happening anymore. This means, the Server is initialized and searching for a client to connect to. Now start the client in a different window like this:
```
    /usr/bin/python3 "/your/path/here/RIC/IoT_client.py" --base_project_path='/your/path/here'
```
It will start connecting to the Server:
```
    Connecting to Server...
    Connected to 127.0.0.1:5001
```
check that the weights are loaded correctly!
```
    => Loading resume checkpoint '/your/path/here/output/experiments/stl10_encrypted_saliency_compressed_saliency_shared_color/model/hide_network_weights.pth'
```
If the client ran through without issues it will close with:
```
Server connection lost or no data left to compute.
Client shut down.
```
this either  means the scripts finished correclty or there is an error on the Server's side. check the Server script again. Watch if it connected successfully:
```
    Connected to ('127.0.0.1', 32900)
```
Also look if it loaded the model weights correctly:
```
    => Loading resume checkpoint '/your/path/here/output/experiments/stl10_encrypted_saliency_compressed_saliency_shared_color/model/reveal_network_weights.pth'
```

At the end you should see the quality measurements specified in the paper and a message that the server is finished:
```
    Server: Data saved through compression 2: 0.0000 MB
    Server: Client-Iterationtime for Batch 2: 0.3769s
    Server: Client-Memory-measure 2: 0.0373 MB
    Server: Client-Peak-Memory-measure 2: 2.2144 MB
    classification success rate:30/30=1.000000
    avg. l2loss_secrets:42.589100
    avg. l2loss_covers:21.943601
    avg. psnr_secrets:11.857324
    avg. ssim_secrets:0.255814
    avg. mean_pixel_errors:49.740055
    avg. lpips_errors:0.707516
    avg. lpips_scs:1.186031
    avg. mse_scs:53.880683
    avg. psnr_scs:9.836046
    avg. ssim_scs:0.044057
    avg. duration on client side: 0.624634s
    avg. memory usage on client side: 0.168756MB
    avg. peak memory usage on client side: 2.214364MB
    avg. data saved through compression: 0.000000MB

    Server closed - finished.
```
please note, that these values are averages over the until then conducted data points. each measurement output includes information of the measurement before. (resets each rerun of course)

You can now also look at the path specified in the Server script output earlier (/your/path/here/output/experiments/stl10_encrypted_saliency_compressed_saliency_shared_color/) and see the generated pictures under testimages and the used weights under model.



## Example Train run
Check the number of epochs int he server.py script (here it's 3 for testing purposes):

```
    if train_mode:
        train(train_loader, beta, learning_rate, batch_size, 0, 3)
```
run the Server.py with desired arguments to train for in train mode:
```
    /usr/bin/python3 "your/path/here/RIC/Server.py" --base_project_path="your/path/here/" --train --encrypt --sal_for_he
```

check your choosen arguments again in the output:
```
  Mode: Train
  Project's path: your/path/here/
  Compression: No
  Compression guided by: Feature extraction
  Encryption: Yes
  Encryption guided by: Saliency map
  Shared Guidance: No
```

take note of the output directory:
```
    Output directory (Old files of the same flags will be overwritten!): your/path/here/output/experiments/svhn_encrypted_saliency_uncompressed_feature extraction_seperate_color/
```

the generated weights will be saved there in model_train_epochs. at the end of training the latest trained model will be split and saved to the model subfolder. Next time in testing, the newly trianed weights will be used automatically. 

During training you can monitor the loss per batch:
```
    Training: Batch 1/1145. Loss of -1.9469,secret loss of 0.2198, attack1_loss of -4.1449, loss_cover 0.01306, ssim_loss: -0.0043
```
But this printing makes the training process take even longer! comment it out in the code of Server.py line 457, if not needed.