class Logger():
    def __init__(self, *args):
        self.args = args
        self.val = [0.]*len(args)
        self.counter = 0

    def accum(self, new_val):
        if type(new_val) is list:
            new_val = [x for x in new_val if x is not None]
            assert(len(new_val) == len(self.val))
            for i in range(len(self.val)):
                self.val[i] += new_val[i]
        else:
            assert(len(self.val) == 1)
            self.val[0] += new_val
        self.counter += 1

    def clear(self):
        self.val = [0.]*len(self.val)
        self.counter = 0

    def get_status(self, epoch, time, it=None):
        if it is None:
            line = 'Epoch %d (%.3f secs)' %(epoch, time)
        else:
            line = 'Epoch %d iter %d (%.3f secs)' %(epoch, it, time)
        for i in range(len(self.val)):
            line += ', %s %f' % (self.args[i], self.val[i]/self.counter)
        return line

    def get_status_no_header(self, nocomma=False):
        line = '%s %f' % (self.args[0], self.val[0]/self.counter) \
                if nocomma else \
                ', %s %f' % (self.args[0], self.val[0]/self.counter)
        for i in range(1, len(self.val)):
            line += ', %s %f' % (self.args[i], self.val[i]/self.counter)
        return line
