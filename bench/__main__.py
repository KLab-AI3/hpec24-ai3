from bench import models, layers, swap_conv2d

if __name__ == "__main__":
    print('BENCH')
    layers.run()
    models.run()
    swap_conv2d.run()
