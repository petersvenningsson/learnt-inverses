import sample

max_components = 10
parameters = ('dpar','dperp','theta','phi','d0','rpar','rperp','r1','r2')

parameter_interval = {
'dpar' : (sample.opt['dmin'], sample.opt['dmax']),
'dperp': (sample.opt['dmin'], sample.opt['dmax']),
'd0'   : (sample.opt['dmin'], sample.opt['dmax']),
'rpar' : (sample.opt['rmin'], sample.opt['rmax']),
'rperp': (sample.opt['rmin'], sample.opt['rmax']),
'r1'   : (sample.opt['r1min'], sample.opt['r1max']),
'r2'   : (sample.opt['r2min'], sample.opt['r2max']),
}