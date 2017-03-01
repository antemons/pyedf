import numpy as np
try:
    import pylab as plt
except ImportError:
    print("pylab could not be imported, plotting will cause errors")


class Bunch(object):
    """ The Bunch design pattern """
    def __init__(self, **kwargs):
        self.__dict__ = kwargs


def cached_property(function):
    """ decorator to strore the results of a method
    """
    attr_name = '_chached_' + function.__name__

    @property
    def _chached_property(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, function(self))
        return getattr(self, attr_name)
    return _chached_property


class Signal:

    def __init__(self, max_time=10, sampling_rate=10):
        self._sampling_rate = sampling_rate
        self._time = np.linspace(0, max_time, max_time * self._sampling_rate)
        self._max_time = max(self._time)

    def __call__(self):
        """ generate the new tuple(time, input, target)
        """
        raise NotImplemented

    def show_single_instance(self):
        time, input_, target = self()
        plt.plot(time, input_)
        plt.plot(time, target)
        plt.show()


class FindLastMax():
    """ Generate input (random events) and target (height of last event) """

    def __init__(self, mean_eventrate, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._eventrate = mean_eventrate

    def __call__(self):
        input_ = np.zeros_like(self._time)
        target = np.zeros_like(self._time)
        last_event_time = 0
        while True:
            event_time = (last_event_time +
                          np.random.exponential(1/self._eventrate))
            value = np.random.random()
            idx = np.searchsorted(self._time, event_time)
            if idx == len(self._time):
                break
            input_[idx] = value
            target[idx:] = value
            last_event_time = event_time

        return self._time, input_, target


class EKG(Signal):
    """ Generate syntetic pais of ECG and respiration

    example:
        EKG().show_single_instance()
    """

    def __init__(self,
                 heart_rate=60/60, respiration_rate=15/60,
                 heart_fluctuations=0.1, respiration_fluctuations=0.1,
                 esk_strength=0.1, rsa_strength=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._heart_rate = heart_rate
        self._respiration_rate = respiration_rate
        self._heart_fluctuations = heart_fluctuations
        self._respiration_fluctuations = respiration_fluctuations
        self._esk_strength = esk_strength
        self._rsa_strength = rsa_strength

    def _random_frequency(self, mean_rate):
        return mean_rate * (0.9 + 0.2 * np.random.random())

    @property
    def _time_diff(self):
        return 1 / self._sampling_rate

    def _gen_phase_heartbeat(self):
        return (np.random.random() +
                self._time * self._random_frequency(self._heart_rate) +
                self._heart_fluctuations * self._random_walk())

    def _gen_phase_respiration(self):
        return (np.random.random() +
                self._time * self._random_frequency(self._respiration_rate) +
                self._respiration_fluctuations * self._random_walk())

    def _gen_respiration(self, phase):
        return np.sin(2 * np.pi * phase)

    def _coupled_via_esk(self, heartbeat, respiration):
        return heartbeat * (1 + self._esk_strength * respiration)

    def _couple_via_rsa(self, phase_heartbeat, phase_respiration):
        """ including respiratory sinus arrhythmia
        """
        phase_heartbeat[:] += (self._time_diff * self._rsa_strength *
                               np.sin(2 * np.pi * phase_respiration)).cumsum()

    def _random_walk(self):
        return (np.sqrt(self._time_diff) *
                np.random.randn(len(self._time)).cumsum())

    def _gen_heartbeat(self, phase):
        WIDTH = 0.06
        theta = phase % 1
        delta_theta = theta - 0.5
        return np.exp(-delta_theta**2 / (2 * WIDTH))

    def __call__(self):
        phase_respiration = self._gen_phase_respiration()
        phase_heartbeat = self._gen_phase_heartbeat()
        self._couple_via_rsa(phase_heartbeat, phase_respiration)
        heartbeat = self._gen_heartbeat(phase_heartbeat)
        respiration = self._gen_respiration(phase_respiration)
        heartbeat_signal = self._coupled_via_esk(heartbeat, respiration)
        return self._time, heartbeat_signal, respiration
