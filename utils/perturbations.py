class UTSPerturbations:

    def __init__(self):
        pass

    def apply_perturbation(self, timeseries_instance, start_idx, end_idx, perturbation='occlusion'):
        if perturbation == 'occlusion':
            self.perturb_occlusion(timeseries_instance, start_idx, end_idx)
        elif perturbation == 'mean':
            self.perturb_mean(timeseries_instance, start_idx, end_idx)
        elif perturbation == 'total_mean':
            self.perturb_total_mean(timeseries_instance, start_idx, end_idx)
        elif perturbation == 'noise':
            self.perturb_noise(timeseries_instance, start_idx, end_idx)
        return timeseries_instance

    def perturb_occlusion(self, timeseries_instance, start_idx, end_idx):
        if start_idx == end_idx:
            timeseries_instance[start_idx] = 0
        else:
            timeseries_instance[start_idx:end_idx] = 0
        return timeseries_instance


    def perturb_mean(self, timeseries_instance, start_idx, end_idx):
        if start_idx == end_idx:
            timeseries_instance[start_idx] = np.mean(timeseries_instance)
        else:
            timeseries_instance[start_idx:end_idx] = np.mean(timeseries_instance[start_idx, end_idx])
        return timeseries_instance


    def perturb_total_mean(self, timeseries_instance, start_idx, end_idx):
        if start_idx == end_idx:
            timeseries_instance[start_idx] = np.mean(timeseries_instance)
        else:
            timeseries_instance[start_idx:end_idx] = np.mean(timeseries_instance[start_idx:end_idx])
        return timeseries_instance


    def perturb_noise(self, timeseries_instance, start_idx, end_idx):
        if start_idx == end_idx:
            timeseries_instance[start_idx] = np.random.uniform(timeseries_instance.min(), 
                timeseries_instance.max(), 1)
        else:
            for idx in range(start_idx, end_idx):
                timeseries_instance[idx] = np.random.uniform(timeseries_instance.min(), 
                    timeseries_instance.max(), 1)
        return timeseries_instance