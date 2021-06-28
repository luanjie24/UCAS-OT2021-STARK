from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = 'D:\\video_project\demo\data\got10k_lmdb'
    settings.got10k_path = 'D:\\video_project\demo\data\got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.lasot_lmdb_path = 'D:\\video_project\demo\data\lasot_lmdb'
    settings.lasot_path = 'D:\\video_project\demo\data\lasot'
    settings.network_path = 'D:\\video_project\demo\test/networks'    # Where tracking networks are stored.
    settings.nfs_path = ''
    settings.otb_path = ''
    settings.prj_dir = 'D:\\video_project\demo'
    settings.result_plot_path = 'D:\\video_project\demo/test/result_plots'
    settings.results_path = 'D:\\video_project\demo/test/tracking_results'    # Where to store tracking results
    settings.save_dir = 'D:\\video_project\demo'
    settings.segmentation_path = 'D:\\video_project\demo\test/segmentation_results'
    settings.tn_packed_results_path = ''
    settings.tpl_path = ''
    settings.trackingnet_path = 'D:\\video_project\demo\data\trackingNet'
    settings.uav_path = ''
    settings.vot_path = 'D:\\video_project\demo\data\VOT2019'
    settings.youtubevos_dir = ''

    return settings
 
