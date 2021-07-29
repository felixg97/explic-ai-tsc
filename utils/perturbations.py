import numpy as np

class UTSPerturbations:

    def __init__(self):
        pass

    def apply_perturbation(self, timeseries_instance, start_idx, end_idx, perturbation='occlusion'):
        timeseries = timeseries_instance.copy()
        if perturbation == 'occlusion':
            self.perturb_occlusion(timeseries, start_idx, end_idx)
        elif perturbation == 'mean':
            self.perturb_mean(timeseries, start_idx, end_idx)
        elif perturbation == 'total_mean':
            self.perturb_total_mean(timeseries, start_idx, end_idx)
        elif perturbation == 'noise':
            self.perturb_noise(timeseries, start_idx, end_idx)
        return timeseries


    def perturb_occlusion(self, timeseries_instance, start_idx, end_idx):
        if start_idx == end_idx:
            timeseries_instance[start_idx] = 0
        else:
            timeseries_instance[start_idx:end_idx+1] = 0
        return timeseries_instance


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
            timeseries_instance[start_idx:end_idx+1] = np.mean(timeseries_instance[start_idx:end_idx])
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


####### Testing #######
# p = UTSPerturbations()

# arr = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# ts = p.perturb_noise(arr, 2, 4)

# print(ts)

# print([i for i in range(2, 4)])


