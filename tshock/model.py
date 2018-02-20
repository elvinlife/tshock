"""    
tshock:model
author:@elvinlife
time:since 2018/2/7
"""
from .widget import Widget
from .private import _make_tuple

class Slot(object):
    def __init__(self,
                 sess=None,
                 inputs = None,
                 outputs=None,
                 updates=None,
                 ostream=None
                 ):
        """
        :param sess:
        :param inputs: The placeholders need to fill in
        :param outputs: The output tensors
        :param updates: The update tensors
        """
        if sess is None:
            raise ValueError('Slot must be provided with a session')
        self._sess = sess
        self._inputs = _make_tuple(inputs)
        self._outputs = _make_tuple(outputs)
        self._updates = _make_tuple(updates)
        self._runnables = _make_tuple([i for i in self._outputs] + [j for j in self._updates])
        self._outputs_len = len(self._outputs)
        print(self._runnables)

    def run(self, *args):
        """
        :param args:The feed dict value for all inputs.
        :param kwargs:No use
        :return:
        """
        if len(args) != len(self._inputs):
            print((len(args), len(self._inputs)))
            raise ValueError('The data number and the placeholder number must be same.')
        outputs = self._sess.run(self._runnables,feed_dict = {input: args[i] for i, input in enumerate(self._inputs)})
        return outputs[:self._outputs_len]

class AbsModel(Widget):
    def __init__(self,
                 name,
                 session,
                 gpu_opiton_allow_growth = True,
                 log_device_placement = False,
                 allow_soft_placement = True,
                 tensorboard_port=6066):
        """
        Model derive from widget(to manage variable scope) and is automatically set up.
        You must design your own Model derived from AbsModel by overriding _setup() method.
        You can add slot during _setup() to help you manage the Model.
        """
        self._sess = session
        self._slots = {}
        super(AbsModel, self).__init__(name)

    def _build(self):
        """
        Build the system
        """
        self._setup()

    def _setup(self):
        """
        The concrete structure of the system
        :return:
        """
        raise NotImplementedError()

    def  _add_slot(self,
                  name,
                  inputs=None,
                  outputs=None,
                  updates=None,
                  givens=None
                  ):
        """
        All param must be tuple/list type
        :param name:
        :param inputs:
        :param outputs:
        :param updates:
        :param givens:
        :return:
        """
        if name in self._slots:
            raise ValueError('Slot {} already exists.'.format(name))
        self._slots[name] = Slot(
            sess=self._sess,
            inputs=inputs,
            outputs=outputs,
            updates=updates,
        )

    def get_slot(self, name=None):
        if name not in self._slots:
            raise ValueError('Slot {} not exists.'.format(name))
        else:
            return self._slots[name]