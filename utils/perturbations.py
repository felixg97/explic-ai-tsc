import numpy as np

class UTSPerturbations:

    def __init__(self):
        pass

    def apply_perturbation(self, timeseries_instance, start_idx, end_idx, 
        perturbation='occlusion', reference_timeseries=None):
        timeseries = timeseries_instance.copy()
        if perturbation == 'occlusion':
            self.perturb_occlusion(timeseries, start_idx, end_idx)
        # elif perturbation == 'mean':
            # self.perturb_mean(timeseries, start_idx, end_idx)
        elif perturbation == 'mean':
            self.perturb_total_mean(timeseries, start_idx, end_idx)
        elif perturbation == 'noise':
            self.perturb_noise(timeseries, start_idx, end_idx)
        elif perturbation == 'value_inversion':
            self.perturb_value_inversion(timeseries, start_idx, end_idx, reference_timeseries)
        elif perturbation == 'sequence_swap': # or flipping
            self.perturb_sequence_swap(timeseries, start_idx, end_idx)
        
        return timeseries


    def perturb_occlusion(self, timeseries_instance, start_idx, end_idx):
        if start_idx == end_idx:
            timeseries_instance[start_idx] = 0
        else:
            timeseries_instance[start_idx:end_idx+1] = 0
        return timeseries_instance


    ## FIXME: sequence mean too many indices -> ???
    def perturb_mean(self, timeseries_instance, start_idx, end_idx):
        if start_idx == end_idx:
            timeseries_instance[start_idx] = np.mean(timeseries_instance)
        else:
            timeseries_instance[start_idx:end_idx+1] = np.mean(timeseries_instance[start_idx, end_idx])
        return timeseries_instance


    def perturb_total_mean(self, timeseries_instance, start_idx, end_idx):
        if start_idx == end_idx:
            timeseries_instance[start_idx] = np.mean(timeseries_instance)
        else:
            timeseries_instance[start_idx:end_idx+1] = np.mean(timeseries_instance)
        return timeseries_instance


    def perturb_noise(self, timeseries_instance, start_idx, end_idx):
        timeseries = np.array(timeseries_instance)
        if start_idx == end_idx:
            timeseries[start_idx] = np.random.uniform(timeseries.min(), 
                timeseries.max(), 1)
        else:
            for idx in range(start_idx, end_idx+1):
                timeseries[idx] = np.random.uniform(timeseries.min(), 
                    timeseries.max(), 1)
        return timeseries

    # WARN: Use `zero_division` parameter to control this behavior. 
    #       When used for sequences
    def perturb_value_inversion(self, timeseries_instance, start_idx, end_idx, 
        reference_timeseries):
        if reference_timeseries is None:
            return 0
        if start_idx == end_idx:
            timeseries_instance[start_idx] = (reference_timeseries.max() - timeseries_instance[start_idx])
            pass
        else:
            for idx in range(start_idx, end_idx+1):
                timeseries_instance[idx] = (reference_timeseries.max() - timeseries_instance[idx])
        return timeseries_instance

    def perturb_sequence_swap(self, timeseries_instance, start_idx, end_idx):
        _timeseries_instance = np.array(timeseries_instance)
        timeseries = np.array(_timeseries_instance)
        if start_idx == end_idx:
            timeseries = timeseries
        else:
            start = start_idx if start_idx > 0 else 0
            end = (end_idx+1) if (end_idx+1)  else time_series.shape
            subsequence_start = np.array([el for el in _timeseries_instance[0:start]])
            subsequence_mid = np.flip(np.array([el for el in _timeseries_instance[start:end]]))
            subsequence_end = np.array([el for el in _timeseries_instance[end:-1]])
            timeseries = np.concatenate([subsequence_start, subsequence_mid, subsequence_end])
        return timeseries



####### Testing #######
# p = UTSPerturbations()

# arr = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# ts = p.perturb_sequence_inversion(arr, 7, 10)

# print(arr)
# print(ts.shape)
# print(ts)

# print([i for i in range(2, 4)])


