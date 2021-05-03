from __future__ import print_function
import sys
import os
import numpy as np
import logging as log
from openvino.inference_engine import IECore


def predict(self, model, device, input):
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    model_xml = model
    model_bin = os.path.splitext(model_xml)[0] + ".bin"

    # Plugin initialization for specified device and load extensions library if specified
    log.info("Creating Inference Engine")
    ie = IECore()

    ### OUTDATED
    # if cpu_extension and 'CPU' in device:
    #    ie.add_extension(cpu_extension, "CPU")

    # Read IR
    log.info("Loading network files:\n\t{}\n\t{}".format(model_xml, model_bin))
    net = ie.read_network(model=model_xml, weights=model_bin)

    ### OUTDATED
    # if "CPU" in args.device:
    #     supported_layers = ie.query_network(net, "CPU")
    #     not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
    #     if len(not_supported_layers) != 0:
    #         log.error("Following layers are not supported by the plugin for specified device {}:\n {}".
    #                   format(args.device, ', '.join(not_supported_layers)))
    #         log.error("Please try to specify cpu extensions library path in sample's command line parameters using -l "
    #                   "or --cpu_extension command line argument")
    #         sys.exit(1)
    
    print(net.input_info['img'].input_data.shape)

    log.info("Preparing input blobs")

    net.batch_size = len(input)

    # Read and pre-process input images
    n, c, h, w = net.input_info['img'].input_data.shape
    images = np.ndarray(shape=(n, c, h, w))

    # Loading model to the plugin
    log.info("Loading model to the plugin")
    exec_net = ie.load_network(network=net, device_name=device)

    # Start sync inference
    log.info("Starting inference in synchronous mode")
    res = exec_net.infer(inputs={'img': images})

    return res['StatefulPartitionedCall/model/dense_1/Softmax']

