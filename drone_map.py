import numpy as np
import itertools

#ws = [783,931]
ws = [780,930]

class MovingMapper():

    def __init__(self, window_size, object_x, object_y, block_size):
        self.window_size = window_size
        self.block_size = block_size
        self.object_x = object_x
        self.object_y = object_y
        self.mapped = self._create_map()

    def _create_map(self):
        '''
        This function will return the map flagging pixel like objects like the drone position.
        :return: numpy array of zeroes and 1.
        '''
        mask = np.zeros(self.window_size)

        for r in itertools.product([a for a in range(self.object_x, self.object_x + self.block_size)],
                                   [b for b in range(self.object_y, self.object_y + self.block_size)]):
            try:
                mask[r[0], r[1]] = 1
            except IndexError:
                pass
            else:
                pass
        return mask

# The no fly zones here show up differently than in the other code as their xy orientation is inverted.
nfz_heyshan_T = [750,575,100]
nfz_cark_T = [400,500,80]
nfz_barrow_T = [520,70,60]

class NoFlyMapper():

    def __init__(self, window_size=ws, nfz_xyr=[nfz_heyshan_T,nfz_barrow_T,nfz_cark_T]):

        '''
        :param window_size: the total window size of the game.
        :param nfz_xyr: a list containing [center position x, center position y, radius] of every no fly zone.
        '''
        self.window_size = window_size
        self.nfz_xyr = nfz_xyr
        self.no_fly_zone = self._no_fly_zone()

    def _no_fly_zone(self):
        '''
        This function will map the no fly zones
        :return: numpy array of zeroes and 1.
        '''
        mask = np.zeros(self.window_size)

        for xyr in self.nfz_xyr:
            for r in itertools.product([a for a in range(xyr[0]-xyr[2], xyr[0] + xyr[2])],
                                       [b for b in range(xyr[1]-xyr[2], xyr[1] + xyr[2])]):
                try:
                    delta_x = r[0]-xyr[0]
                    delta_y = r[1]-xyr[1]
                    if r[0] > 0 and r[1] > 0 and np.sqrt((delta_x)**2+(delta_y)**2) < xyr[2]:
                        mask[r[0], r[1]] = 1
                except IndexError:
                    pass
                else:
                    pass

        return mask


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    #a = NoFlyMapper()
    dronemap = MovingMapper([783,931],400,200,5)
    print(type(dronemap.mapped))
    plt.imshow(dronemap.mapped)
    #nfz_heyshan = [575, 750,100]
    #nfz_cark = [500, 400,80]
    #nfz_barrow = [70, 520,60]


    #nfz_map = NoFlyMapper([783,931],[nfz_heyshan,nfz_barrow,nfz_cark])
    #plt.imshow(nfz_map.no_fly_zone)
    plt.show()