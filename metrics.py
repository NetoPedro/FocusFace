from pyeer.eer_info import get_eer_stats
from pyeer.report import generate_eer_report, export_error_rates
from pyeer.plot import plot_eer_stats



def calculate_metrics(gen_scores,fake_scores,epoch):
    metrics = get_eer_stats(gen_scores, fake_scores)
    generate_eer_report([metrics], ['A'], 'pyeer_report_'+str(epoch)+'.html')
    return metrics.fmr0,metrics.fmr100,metrics.fmr1000,metrics.gmean,metrics.imean,metrics.auc
